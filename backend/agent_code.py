# =============================================================================
# SECTION A: Imports, Global Setup, and Speed Switches
# =============================================================================

import os
import sys
import random
import time
import json
import pickle
import re
import traceback
import warnings
from datetime import datetime
from hashlib import md5
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Union

# --- Scientific & ML Libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from torchvision import transforms
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import cv2


# --- NLP, RAG, and Embeddings
from rank_bm25 import BM25Okapi
from langchain_huggingface import HuggingFaceEmbeddings
from ragatouille import RAGPretrainedModel
from transformers import AutoTokenizer
from langgraph.graph import StateGraph, END


from transformers import AutoTokenizer
from duckduckgo_search import DDGS
from langchain_huggingface import HuggingFaceEmbeddings
from ragatouille import RAGPretrainedModel
from rank_bm25 import BM25Okapi
import faiss

from sentence_transformers import CrossEncoder

# --- FAISS, Agent, and Others
import faiss

# --- Utility/Environment
from dotenv import load_dotenv  # Reads .env file (optional)
from IPython.display import Image, display

# --- Optional Tracing/Debugging
from langsmith import traceable  # Traces runs to LangSmith (optional)

# --- FastAPI, etc. will be imported in backend section

# --- Load environment variables from .env if present
load_dotenv()

from doctor2_captum_helper import (
    doctor2_model,
    doctor2_tokenizer,
    run_full_llm_attribution,
)

# =============================================================================
# SPEED / RESOURCE SWITCHES
# =============================================================================
SKIP_CLI = os.getenv("AGENT_IMPORT_ONLY") == "1"

# --- Embeddings and Data Loading
EMBED_PATH = "visit_patient_emb.npz"
data = np.load(EMBED_PATH)
ids = np.array(data["visit2pat"])

# --- Speed/Accuracy Switches (Tune for development vs. production)
FAST_MODE = True  # Flip to False for full accuracy

MAX_WORKERS_LLM        = 2 if FAST_MODE else 6
MAX_CLUSTERS_PER_VISIT = 1 if FAST_MODE else 3   # Summarize only the first N clusters
FAISS_K                = 15 if FAST_MODE else 30 # ANN neighbours to retrieve
FINAL_RAG_DOCS         = 8 if FAST_MODE else 15  # Documents fed to the LLM
OLLAMA_PARALLEL        = 2 if FAST_MODE else 8   # Ollama server threads

os.environ["OLLAMA_NUM_PARALLEL"] = str(OLLAMA_PARALLEL)

# --- GPU/CPU Device Setup
os.environ.pop("CUDA_VISIBLE_DEVICES", None)  # let PyTorch pick the best device
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# =============================================================================
# SECTION B: Custom Keras Layers, Model Utility, and Constants
# =============================================================================

import os
import sys
import cv2
import torch
import keras
import numpy as np
import pandas as pd
from datetime import datetime
from tensorflow.keras import layers

# ===== Custom Layer: SpatialAttention =====
@keras.utils.register_keras_serializable(package="Custom")
class SpatialAttention(layers.Layer):
    def __init__(self, kernel_size=7, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
        self.concat = layers.Concatenate(axis=-1)
        self.conv = layers.Conv2D(1, kernel_size=kernel_size, padding="same", activation="sigmoid")
        self.multiply = layers.Multiply()
    def call(self, inputs):
        avg_pool = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=-1, keepdims=True)
        concat = self.concat([avg_pool, max_pool])
        attention = self.conv(concat)
        return self.multiply([inputs, attention])

# ===== Dice coefficient metric (for segmentation) =====
def dice_coef(y_true, y_pred, smooth=1):
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(tf.round(y_pred), [-1])
    inter = tf.reduce_sum(y_true * y_pred)
    return (2. * inter + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

# ===== Model and Data Paths =====
_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_BACKEND_DIR, "data", "image")
SA_UNET_WEIGHTS = os.path.join(DATA_DIR, "sa_unet_savedmodel")
RESNET_WEIGHTS = os.path.join(DATA_DIR, "best_model.pth")
LABEL_CSV_PATH = os.path.join(DATA_DIR, "train.csv")
IMG_SIZE_SEG = (256, 256)
IMG_SIZE_CLS = (224, 224)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

def get_label_cols(csv_path: str):
    df = pd.read_csv(csv_path, nrows=0)
    drop = {"abs_path", "image", "original_idx", "is_augmented", "aug_idx", "image_path"}
    return [c for c in df.columns if c not in drop]

from resnet_base_code import create_model

def load_sa_unet(weights_path: str):
    print(f"[DEBUG] Loading SA-UNet model from: {weights_path}")
    if not os.path.exists(weights_path):
        print(f"[SA-UNet] Weights not found at {weights_path}", file=sys.stderr)
        return None
    keras.backend.clear_session()
    try:
        model = keras.models.load_model(
            weights_path,
            custom_objects={"SpatialAttention": SpatialAttention, "dice_coef": dice_coef},
            compile=False
        )
        print(f"[SA-UNet] Loaded model OK from {weights_path}", file=sys.stderr)
        return model
    except Exception as e:
        print(f"[SA-UNet] Failed to load model from {weights_path}", file=sys.stderr)
        import traceback; traceback.print_exc()
        print("Exception type:", type(e).__name__, "| Exception:", e)
        return None

def load_resnet(weights_path: str, label_cols):
    print(f"[DEBUG] Loading ResNet model from: {weights_path}")
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"[ResNet] Weights file not found: {weights_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(len(label_cols), pretrained=False).to(device)
    try:
        ckpt = torch.load(weights_path, map_location=device)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
            thresholds = np.array(ckpt.get("best_thresholds", [0.5] * len(label_cols)))
        else:
            model.load_state_dict(ckpt)
            thresholds = np.array([0.5] * len(label_cols))
        model.eval()
        print(f"[ResNet] Successfully loaded model from {weights_path}", file=sys.stderr)
        return model, thresholds, device
    except Exception:
        print(f"[ResNet] Failed to load model from {weights_path}", file=sys.stderr)
        import traceback; traceback.print_exc(file=sys.stderr)
        raise

label_cols = get_label_cols(LABEL_CSV_PATH)
print("LABEL COLS:", label_cols)
sa_unet = load_sa_unet(SA_UNET_WEIGHTS)
print("SA-UNet model after load:", sa_unet)
assert sa_unet is not None, "SA-UNet did not load!"
resnet, thresholds, device_torch = load_resnet(RESNET_WEIGHTS, label_cols)
print("ResNet model after load:", resnet)

def sa_unet_predict(model, img_rgb: np.ndarray, target_size=(256,256)) -> np.ndarray:
    """Predicts lung mask from input RGB image using SA-UNet model."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, target_size)
    gray = gray.astype("float32") / 255.0
    batch = gray[None, ..., None]
    pred = model.predict(batch, verbose=0)[0, ..., 0]
    bin_mask = (pred > 0.5).astype("float32")
    full_mask = cv2.resize(
        bin_mask,
        (img_rgb.shape[1], img_rgb.shape[0]),
        interpolation=cv2.INTER_NEAREST
    )
    return full_mask

def apply_mask(original_rgb, mask):
    """Returns (masked_lung_image, blended_overlay_with_green_lung) from original RGB and mask."""
    lung = (original_rgb * mask[..., None]).astype("uint8")
    overlay = original_rgb.copy()
    green_mask = np.zeros_like(original_rgb)
    green_mask[..., 1] = 255
    overlay = np.where(mask[..., None] > 0.5, green_mask, overlay)
    alpha = 0.35
    blended = cv2.addWeighted(original_rgb, 1 - alpha, overlay, alpha, 0)
    return lung, blended

from torchvision import transforms
RESNET_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(IMG_SIZE_CLS),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])

def resnet_predict(model_pt, img_rgb, mask, thresholds, device):
    """Runs prediction on masked lung image with ResNet, returns preds, probs, tensor."""
    masked = (img_rgb * mask[..., None]).astype("uint8")
    tensor = RESNET_TRANSFORM(masked).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model_pt(tensor)
        probs = torch.sigmoid(logits).cpu().numpy()[0]
        preds = (probs > thresholds).astype(int)
    return preds, probs, tensor

def grad_cam_torch(model, image_tensor, target_class_idx, target_layer='layer4'):
    """
    Returns Grad-CAM heatmap for ResNet.
    """
    model.eval()
    activations = {}
    gradients = {}
    def forward_hook(module, input, output):
        activations['value'] = output.detach()
    def backward_hook(module, grad_input, grad_output):
        gradients['value'] = grad_output[0].detach()
    handle_f, handle_b = None, None
    for name, module in model.named_modules():
        if name == target_layer:
            handle_f = module.register_forward_hook(forward_hook)
            handle_b = module.register_backward_hook(backward_hook)
            break
    output = model(image_tensor)
    pred_score = output[0, target_class_idx]
    model.zero_grad()
    pred_score.backward(retain_graph=True)
    acts = activations['value'][0]
    grads = gradients['value'][0]
    weights = grads.mean(dim=(1, 2))
    cam = (weights[:, None, None] * acts).sum(dim=0)
    cam = torch.relu(cam).cpu().numpy()
    if handle_f: handle_f.remove()
    if handle_b: handle_b.remove()
    return cam

def get_user_output_dir(user_id):
    """
    Returns output dir for saving images per user/session, ensures it exists.
    """
    out_dir = os.path.join("outputs", str(user_id))
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

def detect_chest_xray(
    image_or_path,
    sa_unet,
    resnet,
    thresholds,
    label_cols,
    device="cpu",
    return_explainability=True,
    user_id=None
):
    """
    Pipeline for X-ray segmentation, classification, and explainability.
    Returns a dict with findings, output image paths, and GradCAM info.
    """
    out_dir = get_user_output_dir(user_id or "default")
    print("\n===== [Image Debug] detect_chest_xray() =====")

    # 1. Load image
    if isinstance(image_or_path, str):
        img_arr = cv2.imread(image_or_path)
        if img_arr is None:
            print(f"[Error] Could not load image: {image_or_path}")
            raise FileNotFoundError(f"Could not load image: {image_or_path}")
        img_rgb = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
        fname = os.path.splitext(os.path.basename(image_or_path))[0]
        print(f"[Debug] Loaded image from {image_or_path} | shape={img_rgb.shape}")
    elif isinstance(image_or_path, np.ndarray):
        img_rgb = image_or_path
        fname = f"xray_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        print(f"[Debug] Loaded image from array | shape={img_rgb.shape}")
    else:
        print("[Error] Input is neither path nor numpy array.")
        raise ValueError("Input must be file path or numpy array")

    # 2. Segmentation
    mask = sa_unet_predict(sa_unet, img_rgb)
    print(f"[Debug] Segmentation mask: shape={mask.shape}, unique={np.unique(mask)}")
    mask_img = (mask * 255).astype(np.uint8)
    mask_path = os.path.join(out_dir, f"{fname}_segmask.png")
    cv2.imwrite(mask_path, mask_img)
    print(f"[Debug] Mask image saved as {mask_path}")

    # 3. Classification (on masked image only)
    preds, probs, tensor = resnet_predict(resnet, img_rgb, mask, thresholds, device)
    print("[Debug] Classification results per label:")
    for label, prob, pred in zip(label_cols, probs, preds):
        print(f"  - {label:22} | prob={prob:.3f} | pred={pred}")
    findings = [(label, float(prob)) for label, prob, pred in zip(label_cols, probs, preds) if pred]
    findings = sorted(findings, key=lambda x: x[1], reverse=True)
    print(f"[Debug] Findings (sorted): {findings if findings else 'No Finding'}")

    # Save original image
    orig_path = os.path.join(out_dir, f"{fname}_original.png")
    cv2.imwrite(orig_path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))

    gradcam_info = {
        "label": None,
        "heatmap": None,
        "gradcam_overlay": None,
        "gradcam_segmented": None,
    }

    heatmap_path = None
    gradcam_overlay_path = None
    gradcam_segmented_path = None

    # 4. Grad-CAM (Explainability)
    if return_explainability and findings:
        top_label, _ = findings[0]
        top_idx = label_cols.index(top_label)
        cam = grad_cam_torch(resnet, tensor, target_class_idx=top_idx, target_layer='layer4')
        cam = cv2.resize(cam, IMG_SIZE_CLS, interpolation=cv2.INTER_LINEAR)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        heatmap_path = os.path.join(out_dir, f"{fname}_gradcam_heatmap.png")
        cv2.imwrite(heatmap_path, cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR))
        print(f"[Debug] GradCAM heatmap saved as {heatmap_path} for label '{top_label}'")

        # -- OVERLAY ON ORIGINAL XRAY (BEST PRACTICE) --
        # Resize X-ray to classification input size
        orig_resized = cv2.resize(img_rgb, IMG_SIZE_CLS, interpolation=cv2.INTER_AREA)
        gradcam_overlay = cv2.addWeighted(orig_resized, 0.5, heatmap, 0.5, 0)
        gradcam_overlay_path = os.path.join(out_dir, f"{fname}_gradcam_overlay.png")
        cv2.imwrite(gradcam_overlay_path, cv2.cvtColor(gradcam_overlay, cv2.COLOR_RGB2BGR))
        print(f"[Debug] GradCAM overlay on original X-ray saved as {gradcam_overlay_path}")

        # -- (OPTIONAL) OVERLAY ONLY INSIDE LUNG MASK --
        mask_resized = cv2.resize(mask, IMG_SIZE_CLS, interpolation=cv2.INTER_NEAREST)
        mask3 = np.repeat(mask_resized[..., None], 3, axis=-1).astype(np.float32)
        masked_img = (orig_resized * mask3).astype(np.uint8)
        heatmap_masked = (heatmap * mask3).astype(np.uint8)
        alpha = 0.4
        gradcam_segmented = cv2.addWeighted(masked_img, 1 - alpha, heatmap_masked, alpha, 0)
        gradcam_segmented_path = os.path.join(out_dir, f"{fname}_gradcam_segmented.png")
        cv2.imwrite(gradcam_segmented_path, cv2.cvtColor(gradcam_segmented, cv2.COLOR_RGB2BGR))
        print(f"[Debug] GradCAM overlay (lung only) saved as {gradcam_segmented_path}")

        gradcam_info = {
            "label": top_label,
            "heatmap": heatmap_path,
            "gradcam_overlay": gradcam_overlay_path,
            "gradcam_segmented": gradcam_segmented_path,
        }

    print("===== [Image Debug] detect_chest_xray() END =====\n")

    # === Return findings, file paths, and explainability info ===
    return {
        "findings": findings if findings else ["No Finding"],
        "paths": {
            "original": orig_path,
            "mask": mask_path,
            "gradcam_heatmap": heatmap_path,
            "gradcam_overlay": gradcam_overlay_path,
            "gradcam_segmented": gradcam_segmented_path,
        },
        "gradcam": gradcam_info,
    }



# =============================================================================
# SECTION C: Tokenizer, LLMs, Embeddings, RAG, and LLM Cache
# =============================================================================


# LLM chat API for local models (doctor2 via Ollama)
from ollama import chat
from langchain.llms.base import LLM

# ---------- TOKENIZER & CONSTANTS ----------
tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-hf")
MAX_TOKENS = 2048

TRUSTED_HEALTHCARE_DOMAINS = [
    ".gov", "nih.gov", "mayoclinic.org", "who.int",
    "medscape.com", "medlineplus.gov", "cdc.gov", "uptodate.com",
    "clevelandclinic.org", "emedicinehealth.com", "msdmanuals.com",
    "cancer.gov", "nhs.uk", "health.harvard.edu", "aacn.org",
    "jamanetwork.com", "bmj.com", "livescience.com"
]

# ---------- WEB SEARCH TOOL ----------
from urllib.parse import urlparse

def is_trusted_domain(url: str) -> bool:
    """Check if the URL is from a trusted healthcare domain."""
    host = urlparse(url).netloc.lower()
    for domain in TRUSTED_HEALTHCARE_DOMAINS:
        if host.endswith(domain) or domain in host:
            return True
    return False

def web_search_tool(query: str, num_results: int = 3) -> list:
    print(f"[DEBUG] Performing web search: {query}")
    try:
        with DDGS() as ddgs:
            results = []
            for r in ddgs.text(query, max_results=num_results * 6):
                url = r.get("href", "")
                snippet = r.get("body", "").strip()
                # Filter for trusted domains only
                if snippet and is_trusted_domain(url):
                    results.append(snippet)
                if len(results) >= num_results:
                    break
        print(f"[DEBUG] Web search got {len(results)} results (from trusted domains).")
        return results or ["No results from trusted healthcare sources."]
    except Exception as e:
        print(f"[DEBUG] Web search failed: {e}")
        return [f"❌ Web search failed: {e}"]

def extract_keywords_llm(patient_text: str) -> list:
    """
    Uses a Large Language Model (LLM) to extract the main clinical symptoms as structured keywords from a patient description.
    The prompt is explicitly framed with professional instructions to ensure consistent, clinically relevant output.
    """
    prompt = (
        "You are a board-certified physician and expert medical writer.\n"
        "Below is an instruction that describes a task, followed by the patient's description.\n\n"
        "### Instruction:\n"
        "Extract the core medical symptoms and relevant clinical findings described in the patient’s text. "
        "Return only a concise, comma-separated list of symptom keywords, each in clear, objective, clinical language. "
        "Avoid including demographic information, social context, or subjective impressions unrelated to symptoms. "
        "Do not reference the doctor or make assumptions beyond the patient’s explicit description. "
        "Do not include full sentences, explanations, or diagnostic reasoning—just the core symptoms or findings as single keywords or short phrases.\n\n"
        f"### Patient Description:\n{patient_text}\n\n"
        "### Extracted Symptom Keywords:"
    )
    response = llm_generate(prompt)
    keywords = [kw.strip() for kw in response.split(',') if kw.strip()]
    print(f"[DEBUG] LLM extracted keywords: {keywords}")
    return keywords


# ---------- TOKEN/TRUNCATION UTILS ----------
def truncate_prompt_tokens(prompt, max_tokens=MAX_TOKENS):
    tokens = tokenizer.encode(prompt)
    if len(tokens) > max_tokens:
        truncated = tokenizer.decode(tokens[:max_tokens], skip_special_tokens=True)
        print(f"⚠️  Prompt truncated from {len(tokens)} to {max_tokens} tokens.")
        return truncated
    return prompt

def truncate_prompt(prompt: str, limit: int = MAX_TOKENS) -> str:
    return truncate_prompt_tokens(prompt, max_tokens=limit)

def chunk_text_by_tokens(text, tokenizer, max_tokens=MAX_TOKENS):
    tokens = tokenizer.encode(text)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i:i+max_tokens]
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        chunks.append(chunk_text)
    return chunks

def symptom_in_first_sentence(symptom: str, text: str, max_tokens: int = 15) -> bool:
    """
    Checks if 'symptom' (e.g., 'headache') appears in the first max_tokens tokens of the first sentence.
    """
    symptom = symptom.lower().strip()
    first_sentence = re.split(r'[.!?]\s', text.strip(), maxsplit=1)[0]
    first_tokens = first_sentence.lower().split()[:max_tokens]
    return any(symptom == tok.strip(" ,;:") for tok in first_tokens)

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------- LLM CACHE FOR FAST REPEAT RUNS ----------
CACHE_DIR = "cache/llm"
os.makedirs(CACHE_DIR, exist_ok=True)

def _cached_llm_call(prompt: str, model) -> str:
    h = md5(prompt.encode()).hexdigest()
    f = os.path.join(CACHE_DIR, h + ".txt")
    if os.path.exists(f):
        with open(f) as fp: 
            out = fp.read()
            print(f"[LLM CACHE] Returning cached output for prompt hash {h}")
            return out
    print(f"[LLM CACHE] No cache found, calling LLM for prompt hash {h}")
    out = model._call(prompt)
    with open(f, "w") as fp: fp.write(out)
    return out

# ---------- PARALLEL PARTIAL REASONING ----------
import concurrent.futures

def parallel_partial_reasoning(
    llm, big_text, user_query, tokenizer,
    max_tokens=MAX_TOKENS, prompt_template=None, max_workers=2
):
    blocks = chunk_text_by_tokens(big_text, tokenizer, max_tokens)
    print(f"[DEBUG] Running parallel reasoning on {len(blocks)} blocks with {max_workers} workers.")
    def single_reasoning(chunk):
        if prompt_template:
            prompt = prompt_template(chunk, user_query)
        else:
            prompt = (
                f"Given the following patient history block, provide a clinical summary relevant to the query: '{user_query}'. "
                f"Only include relevant findings, symptoms, diagnoses.\n\nHistory block:\n{chunk}"
            )
        prompt = truncate_prompt(prompt, max_tokens)
        return _cached_llm_call(prompt, llm)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(single_reasoning, blocks))
    return results

# ---------- LLM WRAPPER FOR OLLAMA ----------
class OllamaLLM(LLM):
    model: str = "doctor2"
    def _call(self, prompt: str, stop=None) -> str:
        prompt = truncate_prompt(prompt, MAX_TOKENS)
        response = chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.get("message", {}).get("content", "❌ No response generated.")
    @property
    def _llm_type(self) -> str:
        return "doctor2"

llm = OllamaLLM(model="doctor2")

# ---------- SPACY/SCIBERT FOR CLEAN QUERY ----------
import en_core_sci_scibert
nlp = en_core_sci_scibert.load()

# ---------- EMBEDDINGS & MODELS ----------
os.environ["OLLAMA_HOST"] = "http://127.0.0.1:11500"

primary_embeddings_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5",
    model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)
alternative_embeddings_model = HuggingFaceEmbeddings(
    model_name="Zybg/synthetic-clinical-embedding-model",
    model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)
cross_encoder = CrossEncoder("pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb")

# ---------- LOAD FAISS, CHUNKS, BM25, COLBERT ----------
faiss_index_path = "faiss_index_D.idx"
chunks_path = "chunks.pkl"
for p in [faiss_index_path, chunks_path]:
    if not os.path.exists(p):
        raise FileNotFoundError(f"❌ File not found: {p}")

faiss_index = faiss.read_index(faiss_index_path)
with open(chunks_path, "rb") as f:
    chunks = pickle.load(f)
corpus = [chunk.page_content for chunk in chunks]
tokenized_corpus = [doc.split() for doc in corpus]
bm25 = BM25Okapi(tokenized_corpus)
colbert_reranker = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")

# ---------- QUERY PROCESSING & RAG SEARCH ----------
def clean_query(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return " ".join(tokens)

def compute_query_embedding(query_text):
    emb1 = np.asarray(primary_embeddings_model.embed_query(query_text), dtype="float32")
    emb2 = np.asarray(alternative_embeddings_model.embed_query(query_text), dtype="float32")
    if emb1.shape != emb2.shape:
        return emb1.reshape(1, -1)
    return ((emb1 + emb2) / 2.0).reshape(1, -1)

def truncate_text(text, tokenizer, max_length=512):
    tokens = tokenizer.tokenize(text)
    if len(tokens) > max_length:
        tokens = tokens[:max_length]
    return tokenizer.convert_tokens_to_string(tokens)

def search(query: str, embedding=None, k_faiss: int = 15, k_final: int = 8) -> Tuple[str, List[str]]:
    print(f"[DEBUG] Performing RAG search for: {query}")
    cleaned = clean_query(query)
    emb = embedding if embedding is not None else compute_query_embedding(cleaned)
    distances, indices = faiss_index.search(emb, k_faiss)
    bm25_scores = bm25.get_scores(cleaned.split())
    fused: List[Tuple[int, float]] = []
    for idx, faiss_dist in zip(indices[0], distances[0]):
        if 0 <= idx < len(chunks):
            fused.append((idx, -faiss_dist + bm25_scores[idx]))
    if not fused:
        return "", []
    fused = sorted(fused, key=lambda x: x[1], reverse=True)[:k_final * 2]
    candidate_texts = [chunks[idx].page_content for idx, _ in fused]
    # ColBERT reranking
    colbert_results = colbert_reranker.rerank(query, candidate_texts, k=len(candidate_texts))
    colbert_candidates = [candidate_texts[r["result_index"]] for r in colbert_results]
    # Final top-k selection
    top_docs = colbert_candidates[:k_final]
    joined_context = "\n".join(top_docs)
    print(f"[DEBUG] RAG returned {len(top_docs)} docs.")
    return joined_context, top_docs


# =============================================================================
# SECTION D: Personalized Patient History Retrieval via Embeddings
# =============================================================================

# === LOAD NODES AND VISIT INDICES ===
_DATA_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
with open(os.path.join(_DATA_BASE, "nodes_200k.pkl"), "rb") as f:
    nodes_list = pickle.load(f)
visit_stay_ids = [n["stay_id"] for n in nodes_list if n["type"] == "Visit"]
visit_subject_ids = [n["subject_id"] for n in nodes_list if n["type"] == "Visit"]
visit_datetimes = [n.get("intime", None) for n in nodes_list if n["type"] == "Visit"]

# === LOAD TABLES ===
edstays_df = pd.read_pickle(os.path.join(_DATA_BASE, "cleaned_edstays.pkl"))
diagnosis_df = pd.read_pickle(os.path.join(_DATA_BASE, "cleaned_diagnosis.pkl"))
triage_df = pd.read_pickle(os.path.join(_DATA_BASE, "cleaned_triage.pkl"))
vitalsign_df = pd.read_pickle(os.path.join(_DATA_BASE, "cleaned_vitalsign.pkl"))

# ---- Retrieve all metadata and summary for a visit ----

def get_visit_metadata_by_index(idx):
    stay_id = visit_stay_ids[idx]
    subject_id = visit_subject_ids[idx]
    intime = visit_datetimes[idx]

    # --- edstays: demographics, arrival/disposition, age ---
    stay_row = edstays_df[edstays_df["stay_id"] == stay_id]
    meta = {
        "index": idx,
        "stay_id": stay_id,
        "subject_id": subject_id,
        "intime": intime
    }
    if not stay_row.empty:
        s = stay_row.iloc[0]
        meta.update({
            "gender": str(s.get("gender")),
            "race": str(s.get("race")),
            "disposition": str(s.get("disposition")),
            "arrival_transport": str(s.get("arrival_transport")),
            "outtime": str(s.get("outtime")) if "outtime" in s else None,
            "age": float(s.get("anchor_age")) if "anchor_age" in s else None,
            "insurance": str(s.get("insurance")) if "insurance" in s else None
        })
    # --- triage: chief complaint, triage values ---
    triage_row = triage_df[triage_df["stay_id"] == stay_id]
    if not triage_row.empty:
        tri = triage_row.iloc[0]
        meta["chiefcomplaint"] = tri.get("chiefcomplaint", None)
        for v in ["heartrate", "temperature", "resprate", "o2sat", "sbp", "dbp", "pain"]:
            if v in tri and not pd.isna(tri[v]):
                meta[v] = tri[v]
        # add acuity if available
        if "acuity" in tri and not pd.isna(tri["acuity"]):
            meta["acuity"] = tri["acuity"]
    # --- diagnosis: πρώτη διάγνωση ---
    diag_rows = diagnosis_df[diagnosis_df["stay_id"] == stay_id].sort_values("seq_num")
    if not diag_rows.empty:
        first_diag = diag_rows.iloc[0]
        meta["diagnosis_icd"] = first_diag["icd_code"]
        meta["diagnosis_title"] = first_diag["icd_title"]
    # --- vitals: τελευταία μέτρηση (discharge) ---
    vs_rows = vitalsign_df[vitalsign_df["stay_id"] == stay_id]
    if not vs_rows.empty:
        vs = vs_rows.iloc[-1]
        for v in ["heartrate", "temperature", "resprate", "o2sat", "sbp", "dbp"]:
            if v in vs and not pd.isna(vs[v]):
                meta[f"last_{v}"] = vs[v]
        if "rhythm" in vs and pd.notna(vs["rhythm"]):
            meta["last_rhythm"] = vs["rhythm"]
    return meta

def get_visit_note_by_index(idx):
    stay_id = visit_stay_ids[idx]
    note_lines = []
    # Chief complaint
    triage_row = triage_df[triage_df["stay_id"] == stay_id]
    chiefcomplaint = triage_row["chiefcomplaint"].values[0] if not triage_row.empty else None
    if chiefcomplaint:
        note_lines.append(f"Chief Complaint: {chiefcomplaint}")
    # Diagnosis
    diag_rows = diagnosis_df[diagnosis_df["stay_id"] == stay_id].sort_values("seq_num")
    if not diag_rows.empty:
        first_diag = diag_rows.iloc[0]
        note_lines.append(f"Diagnosis: {first_diag['icd_title']} (ICD: {first_diag['icd_code']})")
    # Triage Vitals
    if not triage_row.empty:
        tri = triage_row.iloc[0]
        vitals = []
        for v in ["heartrate", "temperature", "resprate", "o2sat", "sbp", "dbp"]:
            if v in tri and not pd.isna(tri[v]):
                vitals.append(f"{v.capitalize()}: {tri[v]}")
        if vitals:
            note_lines.append("Triage Vitals: " + ", ".join(vitals))
    # Summary from edstays
    stay_row = edstays_df[edstays_df["stay_id"] == stay_id]
    if not stay_row.empty:
        s = stay_row.iloc[0]
        note_lines.append(f"Admitted: {s['intime']} - Discharged: {s['outtime']} (Disposition: {s['disposition']})")
    return "\n".join(note_lines) if note_lines else f"No clinical note found for stay_id {stay_id}"

# --------- (1) TEXT EMBEDDINGS AND FUSION ---------
def get_text_embedding(text, embeddings_model):
    return np.asarray(embeddings_model.embed_query(text), dtype=np.float32)

def fuse_embeddings(query_emb, label_emb, alpha=0.7):
    return alpha * query_emb + (1 - alpha) * label_emb

# --------- (2) FAISS SEARCH (PATIENT-SPECIFIC) ---------
FAISS_INDEX_PATH = "faiss_patient_index.bin"
MAPPING_PATH = "faiss_patient_mapping.pkl"

FAISS_INDEX = faiss.read_index(FAISS_INDEX_PATH)
with open(MAPPING_PATH, "rb") as f:
    INDEX_MAP = pickle.load(f)
ALL_EMB = FAISS_INDEX.reconstruct_n(0, FAISS_INDEX.ntotal)  # O(1) load στη RAM

def search_patient_visits_faiss_v2(
    fused_emb,
    patient_id,
    k,
):
    """
    Efficient FAISS patient-specific search (NO reconstruction/copy per request).
    """
    global ALL_EMB, INDEX_MAP

    # Ensure patient_id is int
    if isinstance(patient_id, str) and patient_id.isdigit():
        patient_id = int(patient_id)

    # Indices for the specific patient
    indices = [i for i, pid in INDEX_MAP.items() if pid == patient_id]
    if len(indices) == 0:
        print(f"No embeddings found for patient_id={patient_id}")
        return [], []

    # Numpy indexing, no new allocation
    sub_emb = ALL_EMB[indices]
    d = sub_emb.shape[1]

    fused_emb = np.asarray(fused_emb).reshape(1, -1)
    if fused_emb.shape[1] != d:
        raise ValueError(
            f"[ERROR] Dimension mismatch: fused_emb.shape={fused_emb.shape}, FAISS index dim={d}."
        )

    sub_index = faiss.IndexFlatL2(d)
    sub_index.add(sub_emb)
    D, I = sub_index.search(fused_emb, k)
    neighbor_global_idx = [indices[i] for i in I[0]]
    return neighbor_global_idx, D[0]

# --------- (3) FULL METADATA COLLECTION ---------

def collect_visit_metadata_and_texts(global_indices):
    visit_metadata_list = []
    visit_texts_list = []
    for idx in global_indices:
        meta = get_visit_metadata_by_index(idx)
        note = get_visit_note_by_index(idx)
        visit_metadata_list.append(meta)
        visit_texts_list.append(note)
    return visit_metadata_list, visit_texts_list

# --------- (4) CONTEXT CREATION FOR LLM ---------

def create_personalized_llm_context(query, image_labels, relevant_visits, extra_notes=""):
    context = (
        f"User Query: {query}\n"
        f"Image Predicted Labels: {', '.join(image_labels)}\n"
        f"{extra_notes}\n"
        "Most Relevant Patient History Visits (with metadata):\n"
    )
    for i, visit in enumerate(relevant_visits, 1):
        context += f"\nVisit {i}: {json.dumps(visit, ensure_ascii=False)}"
    context += "\n\nGenerate a personalized, structured clinical report (assessment, reasoning, diagnosis)."
    return context

# --------- (5) PIPELINE FUNCTION (ENTRY POINT) ---------

def personalized_patient_history_workflow_with_texts(
    query_text,
    image_labels,
    patient_id,
    embeddings_model,
    k=5,
    alpha=0.6,
    extra_notes="",
):
    # 1. Compute embeddings for query and image labels
    query_emb = get_text_embedding(query_text, embeddings_model)
    labels_text = " ".join(image_labels)
    label_emb = get_text_embedding(labels_text, embeddings_model)
    # 2. Fuse embeddings
    fused_emb = fuse_embeddings(query_emb, label_emb, alpha=alpha)
    # 3. Search
    top_global_indices, dists = search_patient_visits_faiss_v2(
        fused_emb, patient_id, k
    )
    # 4. Retrieve full metadata and note texts for each visit
    relevant_visits, visit_texts = collect_visit_metadata_and_texts(top_global_indices)
    # 5. Build LLM context (if required)
    llm_context = create_personalized_llm_context(
        query_text, image_labels, relevant_visits, extra_notes=extra_notes
    )
    return relevant_visits, llm_context, visit_texts, dists



# =============================================================================
# SECTION E: Agent Logic (Planner, Nodes, Reasoning Chain)
# =============================================================================

from operator import add
from typing import List, Dict, Optional, Union, Any
from pydantic import BaseModel

# --- Data Classes ---
class AgentAction(BaseModel):
    tool_name: str
    tool_input: dict
    tool_output: Optional[Union[str, dict]] = None
    reasoning: Optional[str] = None

class AgentState(BaseModel):
    input: str
    chat_history: List[dict]
    intermediate_steps: List[AgentAction]
    agent_outcome: Union[str, dict, None] = None
    plan: List[str]
    metadata: Dict[str, Union[str, int, float]]
    self_refine_iter: int
    context_cache: Dict[str, Any]
    reflect_decision: Optional[str] = None 

# --- Helper Functions ---
def findings_to_sentence(xfind):
    """Converts findings list to a single clinical sentence."""
    if xfind and isinstance(xfind, (list, tuple)) and any(str(f).lower() not in ['no finding', 'none', '', ' '] for f in xfind):
        valid = [str(f) for f in xfind if str(f).lower() not in ['no finding', 'none', '', ' ']]
        return f"The following chest X-ray findings were detected: {', '.join(valid)}."
    else:
        return "No abnormal findings were reported on the chest X-ray."

def truncate_prompt(prompt, max_tokens=2048):
    """Truncates prompt for safety (chars, not tokens)."""
    if len(prompt) > max_tokens * 4:
        print(f"\n⚠️ Prompt truncated from {len(prompt)} to {max_tokens*4} chars")
        return prompt[:max_tokens*4]
    return prompt

def flatten_history_texts(history_texts):
    """Flattens list of history texts or handles None."""
    if isinstance(history_texts, list):
        return "\n---\n".join(history_texts[:2])
    elif history_texts is None:
        return "No prior history available."
    else:
        return str(history_texts)

def create_combined_query(query, xray_findings, history_texts):
    """Combines query, findings, and history into a single LLM prompt."""
    findings_sentence = findings_to_sentence(xray_findings)
    history_block = flatten_history_texts(history_texts)
    combined = (
        f"Patient Query: {query}\n"
        f"CHEST XRAY Findings: {findings_sentence}\n"
        f"Relevant Patient History:\n{history_block}\n"
    )
    return combined

def llm_generate(prompt: str) -> str:
    """Calls the LLM and prints prompt/response."""
    print("\n========== [LLM PROMPT - FULL] ==========\n" + prompt)
    result = llm._call(prompt)
    print("\n========== [LLM RESPONSE] ==========\n" + str(result)[:700] + '\n...')
    return result

# =========================
# PROMPT GENERATORS
# =========================

def generate_initial_prompt(query, rag=None, xray_findings=None, patient_history=None, web_context=None):
    instruction = (
        "You are a board-certified physician and expert medical writer.\n"
        "Below is an instruction that describes a task, followed by a patient query and the corresponding Diagnosis Report output.\n\n"
        "### Instruction:\n"
        "When provided with a patient’s query, generate a structured and concise Diagnosis Report consisting of:\n"
        "- Assessment: Synthesis of the patient's presentation and relevant clinical findings\n"
        "- Differential Diagnosis: Prioritized list of possible conditions, each with a clear and concise justification based only on the information provided\n"
        "- Final Diagnosis: The most likely diagnosis stated explicitly\n"
        "- Explanation of Final Diagnosis: A brief, clinically sound rationale for the diagnosis, linking it directly to the patient's symptoms\n\n"
        "Do not reference the doctor or their statements. Do not use phrases such as \"the doctor said,\" \"according to the physician,\" or similar. "
        "Avoid speculation, assumptions, or inferred reasoning beyond what is explicitly stated. "
        "Use clear, objective, and professional clinical language only."
    )
    prompt = (
        f"{instruction}\n\n"
        f"### Patient Query:\n{query}\n\n"
        f"### Diagnosis Report:\n"
    )
    return prompt

def generate_rag_web_enrichment_prompt(prev_diag, rag, web_context, user_query):
    return (
        "You are a board-certified physician and expert medical writer.\n"
        "Below is an instruction that describes a task, followed by context and the corresponding Diagnosis Report output.\n\n"
        "### Instruction:\n"
        "Given the previous Diagnosis Report and the additional external evidence below, gathered from literature and web sources:\n"
        "- Treat all external evidence as general information. Do NOT incorporate it into the patient’s past medical history. It is NOT the patient’s personal history.\n"
        "- Use clear, objective, and professional clinical language only.\n"
        "- Do not reference the doctor or their statements. Do not use phrases such as \"the doctor said,\" \"according to the physician,\" or similar.\n"
        "Think step by step:\n"
        "1. First, interpret and summarize the new external evidence.\n"
        "2. Next, explicitly reason whether this evidence justifies any change to the diagnosis, or whether it supports, refutes, or does not affect the prior diagnosis.\n"
        "3. Only revise the Diagnosis Report if truly justified; otherwise, explain why no changes are needed.\n"
        "4. Clearly state what (if anything) changes and why—then present your full updated Diagnosis Report, strictly in the structured format below:\n"
        "- Assessment: Synthesis of the patient's presentation and relevant clinical findings\n"
        "- Differential Diagnosis: Prioritized list of possible conditions, each with a clear and concise justification based only on the information provided\n"
        "- Final Diagnosis: The most likely diagnosis stated explicitly\n"
        "- Explanation of Final Diagnosis: A brief, clinically sound rationale for the diagnosis, linking it directly to the patient's symptoms\n\n"
        f"### Previous Diagnosis Report:\n{prev_diag}\n\n"
        f"### External Evidence (literature, web):\n{rag}\n{web_context}\n\n"
        f"### Patient Query:\n{user_query}\n\n"
        "### Diagnosis Report:\n"
    )


def generate_consistency_prompt(final_diag, query, xray_findings, history):
    prompt = (
        "You are a board-certified physician and expert medical writer.\n"
        "Below is an instruction that describes a task, followed by a patient query, relevant imaging findings, patient history, and the corresponding Diagnosis Report output.\n\n"
        "### Instruction:\n"
        "You are required to generate a structured and concise Diagnosis Report with the following sections:\n"
        "- Assessment: Provide a synthesis of the patient's presentation and all relevant clinical findings.\n"
        "- Differential Diagnosis: Present a prioritized list of possible conditions, each with a clear and concise justification based solely on the information provided.\n"
        "- Final Diagnosis: State explicitly the single most likely diagnosis.\n"
        "- Explanation of Final Diagnosis: Offer a brief, clinically sound rationale for the diagnosis, directly linking it to the patient's symptoms and findings.\n\n"
        "Do not reference the physician or their statements. Do not use phrases such as \"the doctor said,\" \"according to the physician,\" or similar expressions.\n"
        "Avoid speculation, unwarranted assumptions, or reasoning that is not fully supported by the information presented.\n"
        "Use clear, objective, and professional clinical language at all times.\n"
        "You MUST explicitly address and reference all findings, imaging results, and history elements provided below.\n"
        "- If any disease, finding, or history element is determined to be irrelevant, clearly state this in the appropriate section—do not omit it.\n"
        "- Do not leave any section blank. Use the structured format exactly once in your response.\n\n"
        f"### Diagnosis Report:\n{final_diag}\n\n"
        f"### Patient Query:\n{query}\n\n"
        f"### Imaging Findings:\n{findings_to_sentence(xray_findings) or 'None'}\n\n"
        f"### Patient History:\n{flatten_history_texts(history)}\n\n"
        "### Diagnosis Report (REVISED):\n"
    )
    return prompt

def generate_image_enrichment_prompt(prev_diag, xray_findings, user_query):
    return (
        "You are a board-certified physician and expert medical writer.\n"
        "Below is an instruction that describes a task, followed by relevant context and the corresponding Diagnosis Report output.\n\n"
        "### Instruction:\n"
        "Given the previous Diagnosis Report and the new imaging findings provided below:\n"
        "- Use clear, objective, and professional clinical language only.\n"
        "- Do not reference the doctor or their statements. Do not use phrases such as \"the doctor said,\" \"according to the physician,\" or similar.\n"
        "- Avoid speculation, assumptions, or inferred reasoning beyond what is explicitly stated.\n"
        "Think step by step:\n"
        "1. List and briefly explain each imaging finding in clear clinical terms.\n"
        "2. Explicitly reason how each finding supports, contradicts, or does not affect the previous diagnosis.\n"
        "3. Revise, confirm, or refute the previous diagnosis as necessary, providing a clear and logical explanation at each step.\n"
        "4. Present your final, updated Diagnosis Report strictly in the following structured format:\n"
        "- Assessment: Synthesis of the patient's presentation and all relevant clinical findings\n"
        "- Differential Diagnosis: Prioritized list of possible conditions, each with a clear and concise justification based only on the information provided\n"
        "- Final Diagnosis: The most likely diagnosis stated explicitly\n"
        "- Explanation of Final Diagnosis: A brief, clinically sound rationale for the diagnosis, linking it directly to the patient's symptoms and imaging findings\n\n"
        f"### Previous Diagnosis Report:\n{prev_diag}\n\n"
        f"### Imaging Findings:\n{findings_to_sentence(xray_findings) or 'None'}\n\n"
        f"### Patient Query:\n{user_query}\n\n"
        "### Diagnosis Report:\n"
    )


def generate_image_rag_web_enrichment_prompt(prev_diag, rag, web_context, xray_findings, user_query):
    return (
        "You are a board-certified physician and expert medical writer.\n"
        "Below is an instruction that describes a task, followed by relevant context and the corresponding Diagnosis Report output.\n\n"
        "### Instruction:\n"
        "Given the previous Diagnosis Report, the new imaging findings, and the additional external evidence below (gathered from literature and web sources):\n"
        "- Treat all external evidence as general information. Do NOT incorporate it into the patient’s past medical history. It is NOT the patient’s personal history.\n"
        "- Use clear, objective, and professional clinical language only.\n"
        "- Do not reference the doctor or their statements. Do not use phrases such as \"the doctor said,\" \"according to the physician,\" or similar.\n"
        "- Avoid speculation, assumptions, or inferred reasoning beyond what is explicitly stated.\n"
        "Think step by step:\n"
        "1. Interpret the imaging findings in detail and summarize the new external evidence.\n"
        "2. Explicitly reason how, if at all, this new evidence changes your understanding of the findings or the diagnosis.\n"
        "3. Only revise the Diagnosis Report if truly justified—otherwise, explain why no changes are needed.\n"
        "4. Clearly state what (if anything) changes and why—then present your full, updated Diagnosis Report strictly in the following structured format:\n"
        "- Assessment: Synthesis of the patient's presentation and all relevant clinical findings\n"
        "- Differential Diagnosis: Prioritized list of possible conditions, each with a clear and concise justification based only on the information provided\n"
        "- Final Diagnosis: The most likely diagnosis stated explicitly\n"
        "- Explanation of Final Diagnosis: A brief, clinically sound rationale for the diagnosis, linking it directly to the patient's symptoms and findings\n\n"
        f"### Previous Diagnosis Report:\n{prev_diag}\n\n"
        f"### Imaging Findings:\n{findings_to_sentence(xray_findings) or 'None'}\n\n"
        f"### External Evidence (literature, web):\n{rag}\n{web_context}\n\n"
        f"### Patient Query:\n{user_query}\n\n"
        "### Diagnosis Report:\n"
    )


def generate_history_enrichment_prompt(prev_diag, history_text, user_query, xray_findings=None):
    return (
        "You are a board-certified physician and expert medical writer.\n"
        "Below is an instruction that describes a task, followed by relevant context and the corresponding Diagnosis Report output.\n\n"
        "### Instruction:\n"
        "Given the previous Diagnosis Report and the new patient history provided below:\n"
        "- Use clear, objective, and professional clinical language only.\n"
        "- Do not reference the doctor or their statements. Do not use phrases such as \"the doctor said,\" \"according to the physician,\" or similar.\n"
        "- Avoid speculation, assumptions, or inferred reasoning beyond what is explicitly stated.\n"
        "Think step by step:\n"
        "1. Summarize the key new points from the patient history in clear, clinical terms.\n"
        "2. Explicitly reason how these history details support, contradict, or add to the prior diagnosis and findings.\n"
        "3. Clearly state what (if anything) changes and why—then present your full, updated Diagnosis Report strictly in the following structured format:\n"
        "- Assessment: Synthesis of the patient's presentation and all relevant clinical findings\n"
        "- Differential Diagnosis: Prioritized list of possible conditions, each with a clear and concise justification based only on the information provided\n"
        "- Final Diagnosis: The most likely diagnosis stated explicitly\n"
        "- Explanation of Final Diagnosis: A brief, clinically sound rationale for the diagnosis, linking it directly to the patient's symptoms and findings\n\n"
        f"### Previous Diagnosis Report:\n{prev_diag}\n\n"
        f"### Imaging Findings:\n{findings_to_sentence(xray_findings) if xray_findings else 'None'}\n\n"
        f"### Patient History:\n{flatten_history_texts(history_text)}\n\n"
        f"### Patient Query:\n{user_query}\n\n"
        "### Diagnosis Report:\n"
    )


def generate_history_rag_web_enrichment_prompt(prev_diag, rag, web_context, history_text, user_query, xray_findings=None):
    return (
        "You are a board-certified physician and expert medical writer.\n"
        "Below is an instruction that describes a task, followed by relevant context and the corresponding Diagnosis Report output.\n\n"
        "### Instruction:\n"
        "Given the previous Diagnosis Report, the new patient history, and the additional external evidence below (gathered from literature and web sources):\n"
        "- Treat all external evidence as general information. Do NOT incorporate it into the patient’s past medical history. It is NOT the patient’s personal history.\n"
        "- Use clear, objective, and professional clinical language only.\n"
        "- Do not reference the doctor or their statements. Do not use phrases such as \"the doctor said,\" \"according to the physician,\" or similar.\n"
        "- Avoid speculation, assumptions, or inferred reasoning beyond what is explicitly stated.\n"
        "Think step by step:\n"
        "1. Summarize any new information from the patient history and the external sources in clear, clinical terms.\n"
        "2. Explicitly reason how each piece of information supports, contradicts, or adds to your previous reasoning and diagnosis.\n"
        "3. Only update the Diagnosis Report if the new history or evidence justifies it—clearly state exactly what changed and why.\n"
        "4. Present your full, updated Diagnosis Report strictly in the following structured format:\n"
        "- Assessment: Synthesis of the patient's presentation and all relevant clinical findings\n"
        "- Differential Diagnosis: Prioritized list of possible conditions, each with a clear and concise justification based only on the information provided\n"
        "- Final Diagnosis: The most likely diagnosis stated explicitly\n"
        "- Explanation of Final Diagnosis: A brief, clinically sound rationale for the diagnosis, linking it directly to the patient's symptoms and findings\n\n"
        f"### Previous Diagnosis Report:\n{prev_diag}\n\n"
        f"### Imaging Findings:\n{findings_to_sentence(xray_findings) if xray_findings else 'None'}\n\n"
        f"### Patient History:\n{flatten_history_texts(history_text)}\n\n"
        f"### External Evidence (literature, web):\n{rag}\n{web_context}\n\n"
        f"### Patient Query:\n{user_query}\n\n"
        "### Diagnosis Report:\n"
    )

# =============================================================================
# SECTION: Agent Nodes (Planner, Chain, Reflector, Refinement)
# =============================================================================

MAX_SELF_REFINE = 2

# --- PLANNER NODE ---
def planner_node(state: AgentState) -> dict:
    plan = ["query_diag"]
    plan.append("query_rag_web_enrichment")
    if state.metadata.get("image_path"):
        plan.append("image_diag")
        plan.append("image_rag_web_enrichment")
    if state.metadata.get("patient_id"):
        plan.append("history_diag")
        plan.append("history_rag_web_enrichment")
    plan.append("consistency_check")
    return {"plan": plan}

# --- GET LAST NON-EMPTY REPORT ---
def get_last_report(state):
    for step in reversed(state.intermediate_steps):
        out = getattr(step, "tool_output", None)
        if out and len(str(out).strip()) > 30:
            return out
    return None

# --- MAIN REACT AGENT NODE ---
def react_agent_node(state: AgentState) -> dict:
    plan = state.plan
    steps_taken = [step.tool_name for step in state.intermediate_steps]
    next_tool = next((step for step in plan if step not in steps_taken), None)
    if not next_tool:
        return {}

    cache = state.context_cache.copy()
    user_query = state.input
    metadata = state.metadata
    xray_image = metadata.get("image_path")
    patient_id = metadata.get("patient_id")

    print(f"\n[Agent Debug] [react_agent_node] Step: {next_tool} | Context cache keys: {list(cache.keys())}")

    tool_input = {}

    # --- 1️⃣ QUERY DIAG ---
    if next_tool == "query_diag":
        diagnosis_prompt = generate_initial_prompt(user_query)
        diagnosis_prompt = truncate_prompt(diagnosis_prompt, 2048)
        diagnosis_report = llm_generate(diagnosis_prompt)
        new_action = AgentAction(
            tool_name=next_tool,
            tool_input={"query": user_query},
            tool_output=diagnosis_report
        )
        return {
            "intermediate_steps": state.intermediate_steps + [new_action],
            "context_cache": cache,
        }

    # --- 2️⃣ QUERY RAG/WEB ENRICHMENT ---
    if next_tool == "query_rag_web_enrichment":
        prev_diag = get_last_report(state)
        keywords = extract_keywords_llm(user_query)
        keywords_query = ", ".join(keywords) if keywords else user_query
        web_results = web_search_tool(keywords_query, num_results=4)
        cache["web_context_query"] = "\n".join(web_results)
        rag, _ = search(user_query)
        rag = str(rag)[:1000]
        cache["rag_query"] = rag
        tool_input = {
            "prev_diag": prev_diag,
            "rag": rag,
            "web_context": cache.get("web_context_query"),
            "query": user_query,
        }
        diagnosis_prompt = generate_rag_web_enrichment_prompt(
            prev_diag=prev_diag,
            rag=rag,
            web_context=cache["web_context_query"],
            user_query=user_query,
        )
        diagnosis_prompt = truncate_prompt(diagnosis_prompt, 2048)
        diagnosis_report = llm_generate(diagnosis_prompt)
        new_action = AgentAction(
            tool_name=next_tool,
            tool_input=tool_input,
            tool_output=diagnosis_report
        )
        return {
            "intermediate_steps": state.intermediate_steps + [new_action],
            "context_cache": cache,
        }

    # --- 3️⃣ IMAGE DIAG ---
    if next_tool == "image_diag":
        if not xray_image:
            completed_action = AgentAction(
                tool_name=next_tool,
                tool_input={},
                tool_output=None,
                reasoning="Skipped image_diag as no image was provided."
            )
            return {
                "intermediate_steps": state.intermediate_steps + [completed_action],
                "context_cache": cache,
            }
        result = detect_chest_xray(
            xray_image,
            sa_unet,
            resnet,
            thresholds,
            label_cols,
            device=device_torch,
            return_explainability=True,
            user_id=patient_id or "default"
        )
        findings_raw = result.get("findings", [])
        # Get labels (diagnoses) and confidences
        xray_findings = [f[0] for f in findings_raw if isinstance(f, tuple) and str(f[0]).lower() not in ["no finding", "none", "", " "]]
        xray_confidences = [float(f[1]) for f in findings_raw if isinstance(f, tuple) and str(f[0]).lower() not in ["no finding", "none", "", " "]]
        # Store as list of dicts for UI/XAI compatibility
        xray_labels_conf = [
            {"label": str(lab), "confidence": float(conf)}
            for lab, conf in zip(xray_findings, xray_confidences)
        ]

        cache["xray_findings"] = xray_findings
        cache["xray_confidences"] = xray_confidences
        cache["xray_labels_conf"] = xray_labels_conf
        cache["xray_classification"] = findings_raw
        paths = result.get("paths", {})
        cache["original_xray_path"] = paths.get("original")
        cache["mask_path"] = paths.get("mask")
        cache["gradcam_heatmap_path"] = paths.get("gradcam_heatmap")
        cache["gradcam_overlay_path"] = paths.get("gradcam_segmented")
        cache["gradcam_info"] = result.get("gradcam", {})

        prev_diag = get_last_report(state)
        tool_input = {
            "prev_diag": prev_diag,
            "xray_findings": xray_findings,
            "user_query": user_query,
            "xray_labels_conf": xray_labels_conf,
        }
        diagnosis_prompt = generate_image_enrichment_prompt(
            prev_diag=prev_diag,
            xray_findings=xray_findings,
            user_query=user_query,
        )
        diagnosis_prompt = truncate_prompt(diagnosis_prompt, 2048)
        diagnosis_report = llm_generate(diagnosis_prompt)
        new_action = AgentAction(
            tool_name=next_tool,
            tool_input=tool_input,
            tool_output=diagnosis_report
        )
        return {
            "intermediate_steps": state.intermediate_steps + [new_action],
            "context_cache": cache,
        }

    # --- 4️⃣ IMAGE RAG/WEB ENRICHMENT ---
    if next_tool == "image_rag_web_enrichment":
        xray_findings = cache.get("xray_findings", [])
        findings_str = ", ".join(xray_findings) if xray_findings else ""
        if findings_str:
            search_query = f"chest X-ray shows {findings_str}"
            web_results = web_search_tool(search_query, num_results=4)
        else:
            web_results = []
        cache["web_context_image"] = "\n".join(web_results)
        rag, _ = search(findings_str) if findings_str else ("", None)
        rag = str(rag)[:1000] if rag else ""
        cache["rag_image"] = rag
        prev_diag = get_last_report(state)
        tool_input = {
            "prev_diag": prev_diag,
            "rag": rag,
            "web_context": cache.get("web_context_image"),
            "xray_findings": xray_findings,
            "user_query": user_query,
        }
        diagnosis_prompt = generate_image_rag_web_enrichment_prompt(
            prev_diag=prev_diag,
            rag=rag,
            web_context=cache["web_context_image"],
            xray_findings=xray_findings,
            user_query=user_query,
        )
        diagnosis_prompt = truncate_prompt(diagnosis_prompt, 2048)
        diagnosis_report = llm_generate(diagnosis_prompt)
        new_action = AgentAction(
            tool_name=next_tool,
            tool_input=tool_input,
            tool_output=diagnosis_report
        )
        return {
            "intermediate_steps": state.intermediate_steps + [new_action],
            "context_cache": cache,
        }

    # --- 5️⃣ HISTORY DIAG ---
    if next_tool == "history_diag":
        if not patient_id:
            completed_action = AgentAction(
                tool_name=next_tool,
                tool_input={},
                tool_output=None,
                reasoning="Skipped history_diag as no patient_id provided."
            )
            return {
                "intermediate_steps": state.intermediate_steps + [completed_action],
                "context_cache": cache,
            }
        image_labels = cache.get("xray_findings", [])
        relevant_visits, llm_context, visit_texts, dists = personalized_patient_history_workflow_with_texts(
            query_text=user_query,
            image_labels=image_labels,
            patient_id=patient_id,
            embeddings_model=primary_embeddings_model,
            k=5,
            alpha=0.6,
            extra_notes="",
        )
        print("\n=== [FAISS SIMILARITY SCORES for PATIENT HISTORY] ===")
        for i, (visit, dist) in enumerate(zip(relevant_visits, dists), 1):
        	print(f"Visit {i} (stay_id={visit.get('stay_id', '-')}, subject_id={visit.get('subject_id', '-')}) : Distance = {dist:.4f}")
        print("="*55)
        visit_texts = visit_texts[:2]
        patient_history_text = flatten_history_texts(visit_texts)
        cache["history_texts"] = visit_texts
        cache["history_text"] = patient_history_text
        prev_diag = get_last_report(state)
        tool_input = {
            "prev_diag": prev_diag,
            "history": patient_history_text,
            "user_query": user_query,
            "xray_findings": image_labels,
        }
        diagnosis_prompt = generate_history_enrichment_prompt(
            prev_diag=prev_diag,
            history_text=patient_history_text,
            user_query=user_query,
            xray_findings=image_labels,
        )
        diagnosis_prompt = truncate_prompt(diagnosis_prompt, 2048)
        diagnosis_report = llm_generate(diagnosis_prompt)
        new_action = AgentAction(
            tool_name=next_tool,
            tool_input=tool_input,
            tool_output=diagnosis_report
        )
        return {
            "intermediate_steps": state.intermediate_steps + [new_action],
            "context_cache": cache,
        }

    # --- 6️⃣ HISTORY RAG/WEB ENRICHMENT ---
    if next_tool == "history_rag_web_enrichment":
        history_text = cache.get("history_text", "")
        xray_findings = cache.get("xray_findings", [])
        prev_diag = get_last_report(state)
        web_results = web_search_tool(history_text, num_results=4) if history_text else []
        cache["web_context_history"] = "\n".join(web_results)
        rag, _ = search(history_text) if history_text else ("", None)
        rag = str(rag)[:1000] if rag else ""
        cache["rag_history"] = rag
        tool_input = {
            "prev_diag": prev_diag,
            "rag": rag,
            "web_context": cache.get("web_context_history"),
            "history_text": history_text,
            "user_query": user_query,
            "xray_findings": xray_findings,
        }
        diagnosis_prompt = generate_history_rag_web_enrichment_prompt(
            prev_diag=prev_diag,
            rag=rag,
            web_context=cache["web_context_history"],
            history_text=history_text,
            user_query=user_query,
            xray_findings=xray_findings,
        )
        diagnosis_prompt = truncate_prompt(diagnosis_prompt, 2048)
        diagnosis_report = llm_generate(diagnosis_prompt)
        new_action = AgentAction(
            tool_name=next_tool,
            tool_input=tool_input,
            tool_output=diagnosis_report
        )
        return {
            "intermediate_steps": state.intermediate_steps + [new_action],
            "context_cache": cache,
        }

    # --- 7️⃣ CONSISTENCY CHECK ---
    if next_tool == "consistency_check":
        final_diag = state.intermediate_steps[-1].tool_output if state.intermediate_steps else None
        tool_input = {
            "final_diag": final_diag,
            "query": user_query,
            "xray_findings": cache.get("xray_findings", []),
            "history": cache.get("history_text", ""),
        }
        consistency_prompt = generate_consistency_prompt(
            final_diag=final_diag,
            query=user_query,
            xray_findings=cache.get("xray_findings", []),
            history=cache.get("history_text", "")
        )
        consistency_prompt = truncate_prompt(consistency_prompt, 2048)
        revised_report = llm_generate(consistency_prompt)
        new_action = AgentAction(
            tool_name=next_tool,
            tool_input=tool_input,
            tool_output=revised_report
        )
        return {
            "intermediate_steps": state.intermediate_steps + [new_action],
            "context_cache": cache,
        }


# --- SELF-REFINEMENT DECISION FUNCTION ---
def needs_refine_fn(result: str, *, query_symptom: str = "") -> bool:
    if not result or len(result.strip()) < 100:
        return True
    lc = result.lower()
    negative_patterns = [
        "insufficient", "not found", "uncertain", "can't assist",
        "cannot assist", "cannot provide", "qualified healthcare professional",
        "disclaimer"
    ]
    if any(p in lc for p in negative_patterns):
        return True
    must_have = ["assessment", "differential diagnosis", "final diagnosis", "explanation"]
    if not all(h in lc for h in must_have):
        return True
    for sec in must_have:
        pattern = rf"{sec}(\s*[:\-–]?\s*)(no|not|none|n/a|empty|missing)?(\.|$)"
        if re.search(pattern, lc):
            return True
    if not re.search(r"differential diagnosis.*?(1\.\)|1\.)", lc, re.DOTALL):
        return True
    return False

def strengthen_prompt_for_refine(prompt, result):
    if all(h in result.lower() for h in ["assessment", "differential diagnosis", "final diagnosis", "explanation"]):
        return prompt
    structure = (
        "\n\nRewrite your answer as a **fully structured, clinical Diagnosis Report, DONT INCLUDE REPETATIONS**."
        "\nYou MUST include all of the following sections, clearly labeled and filled out:"
        "\n1. **Assessment** (summary of the clinical problem)"
        "\n2. **Differential Diagnosis** (numbered list, e.g., 1.) 2.) 3.))"
        "\n3. **Final Diagnosis** (with justification)"
        "\n4. **Explanation** (directly link symptoms, findings, and history to your diagnosis; don't leave empty)"
        "\n\nDo NOT include apologies, disclaimers, or refusals. If you lack information, explicitly write 'Not enough data', but do not leave any section blank."
        "\nYour answer should be detailed, medically precise, and ready for inclusion in a medical record."
    )
    return prompt + structure

# --- TOOL NODE: SELF-REFINEMENT LOOP ---
def run_tool_node(state: AgentState):
    action      = state.intermediate_steps[-1]
    tool_name   = action.tool_name
    tool_input  = action.tool_input
    user_query  = state.input

    if tool_name == "query_diag":
        prompt = generate_initial_prompt(tool_input.get("query", ""))
    elif tool_name == "query_rag_web_enrichment":
        prompt = generate_rag_web_enrichment_prompt(
            tool_input.get("prev_diag", ""),
            tool_input.get("rag", ""),
            tool_input.get("web_context", ""),
            user_query=tool_input.get("query", "")
        )
    elif tool_name == "image_diag":
        prompt = generate_image_enrichment_prompt(
            tool_input.get("prev_diag", ""),
            tool_input.get("xray_findings", []),
            user_query=user_query,
        )
    elif tool_name == "image_rag_web_enrichment":
        prompt = generate_image_rag_web_enrichment_prompt(
            tool_input.get("prev_diag", ""),
            tool_input.get("rag", ""),
            tool_input.get("web_context", ""),
            tool_input.get("xray_findings", []),
            user_query=user_query,
        )
    elif tool_name == "history_diag":
        prompt = generate_history_enrichment_prompt(
            tool_input.get("prev_diag", ""),
            tool_input.get("history", ""),
            user_query=user_query,
            xray_findings=tool_input.get("xray_findings", [])
        )
    elif tool_name == "history_rag_web_enrichment":
        prompt = generate_history_rag_web_enrichment_prompt(
            tool_input.get("prev_diag", ""),
            tool_input.get("rag", ""),
            tool_input.get("web_context", ""),
            tool_input.get("history_text", ""),
            user_query=user_query,
            xray_findings=tool_input.get("xray_findings", [])
        )
    elif tool_name == "consistency_check":
        prompt = generate_consistency_prompt(
            tool_input.get("final_diag", ""),
            tool_input.get("query", ""),
            tool_input.get("xray_findings", []),
            tool_input.get("history", "")
        )
    else:
        prompt = f"❌ Tool '{tool_name}' not implemented."

    prompt = truncate_prompt(prompt, 2048)
    print("\n========== [LLM PROMPT - FULL] ==========\n" + prompt)

    max_refine = 2
    ref_iter   = 0
    result     = llm._call(prompt)

    while needs_refine_fn(result, query_symptom=user_query) and ref_iter < max_refine:
        prompt  = strengthen_prompt_for_refine(prompt, result)
        prompt  = truncate_prompt(prompt, 2048)
        result  = llm._call(prompt)
        ref_iter += 1

    result = re.sub(r"```(python|py)?\n?|```", "", result, flags=re.DOTALL)

    completed_action = AgentAction(
        tool_name   = tool_name,
        tool_input  = tool_input,
        tool_output = result
    )

    return {
        "intermediate_steps": state.intermediate_steps[:-1] + [completed_action]
    }



# --- REFLECTOR NODE ---
def reflector_node(state: AgentState) -> dict:
    plan = state.plan
    intermediate_steps = state.intermediate_steps
    if not intermediate_steps:
        return {}
    steps_taken = [s.tool_name for s in intermediate_steps]
    all_steps_done = all(tool in steps_taken for tool in plan)
    last_action = intermediate_steps[-1]
    output = str(last_action.tool_output or "").lower()
    max_refine = state.self_refine_iter
    decision = "CONTINUE"
    critique = "Output looks acceptable. Continuing to next step in plan."

    needs_refine = (
        not output.strip() or
        any(p in output for p in [
            "insufficient", "not found", "uncertain",
            "can't assist", "cannot assist", "cannot provide",
            "qualified healthcare professional", "disclaimer"
        ])
    )
    just_did_consistency = (last_action.tool_name == "consistency_check")

    if (all_steps_done or just_did_consistency):
        if needs_refine and max_refine < MAX_SELF_REFINE:
            decision = "REVISE"
            critique = "Output was incomplete or too vague. Triggering enriched self-refinement."
        else:
            decision = "FINAL"
            critique = "All planned steps executed (or consistency_check done). Proceeding to final answer."
    elif needs_refine and max_refine < MAX_SELF_REFINE:
        decision = "REVISE"
        critique = "Output was incomplete or too vague. Triggering enriched self-refinement."

    last_action.reasoning = critique
    return {
        "reflect_decision": decision,
        "intermediate_steps": intermediate_steps[:-1] + [last_action],
        "self_refine_iter": max_refine + 1 if decision == "REVISE" else max_refine
    }

import glob

def enrich_with_captum_xai(
    answer: str,
    query: str,
    xray_findings: Optional[str] = None,         # optional
    history: Optional[str] = None,               # optional
    doctor2_model=None,
    doctor2_tokenizer=None,
    run_full_llm_attribution=None,
    user_id: Optional[Union[int, str]] = None,
    out_dir: str = "captum_attr_full",           # <<-- MUST be absolute per-session dir!
    top_k_words: int = 5,                        # keep only top-k words by |attribution|
    max_img_px: int = 3500,
    classification_results: Optional[List[Tuple[str, float]]] = None,
    original_xray_path: Optional[Union[str, List[str]]] = None,   # optional
    gradcam_overlay_path: Optional[Union[str, List[str]]] = None, # optional
    captum_img_path: Optional[Union[str, List[str]]] = None       # (compat)
) -> dict:
    """
    Run Captum ONLY for the inputs provided in this call and return ONLY the artifacts
    produced in the current run. All outputs are written to the directory passed in out_dir.
    """
    import sys, os, glob, re, unicodedata
    from datetime import datetime
    from tqdm import tqdm

    # ---------- prepare output dir ----------
    os.makedirs(out_dir, exist_ok=True)

    # ---------- tokens->words ----------
    def tokens_to_words(tokens, scores):
        words, word_scores = [], []
        curr_word, curr_score, count = '', 0.0, 0

        def _is_word_start(t):
            if t.startswith("##"): return False
            if t.startswith("▁"): return True
            if t[:1] in {" ", "\t", "\n"}: return True
            if not curr_word: return True
            return False

        for t, s in zip(tokens, scores):
            t_str = str(t)
            if t_str in ("[PAD]", "<pad>", "</s>", "[CLS]", "[SEP]", "[BOS]", "[EOS]"):
                continue
            if _is_word_start(t_str):
                if curr_word:
                    words.append(curr_word)
                    word_scores.append(curr_score / max(count, 1))
                curr_word = t_str.lstrip("▁").lstrip("##").lstrip()
                curr_score = float(s)
                count = 1
            else:
                part = t_str.lstrip("##")
                curr_word += part
                curr_score += float(s)
                count += 1

        if curr_word:
            words.append(curr_word)
            word_scores.append(curr_score / max(count, 1))

        pairs = []
        for w, sc in zip(words, word_scores):
            if w.strip() and any(unicodedata.category(ch)[0] in "LMN" for ch in w):
                pairs.append((w, float(sc)))
        pairs.sort(key=lambda x: -abs(x[1]))
        return pairs[:top_k_words]

    # ---------- decide exactly which components to run ----------
    inputs = []
    if isinstance(query, str) and query.strip():
        inputs.append(("query", query, "Query"))
    if isinstance(xray_findings, str) and xray_findings.strip():
        inputs.append(("imgfind", xray_findings, "Image Findings"))
    if isinstance(history, str) and history.strip():
        inputs.append(("history", history, "History"))

    ran_types = {tag for (tag, _, _) in inputs}  # e.g. {"query"} if no image/history
    top_words = {}

    # ---------- run Captum attribution for the selected inputs ----------
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if run_full_llm_attribution is not None and inputs:
        with tqdm(total=len(inputs), desc="LLM Attribution", ncols=80) as pbar:
            for tag, input_attr, comp_name in inputs:
                try:
                    artefacts = run_full_llm_attribution(
                        prompt=input_attr,
                        target_text=answer,
                        model=doctor2_model,
                        tokenizer=doctor2_tokenizer,
                        tag_prefix=f"{tag}",
                        out_dir=out_dir,      # <--- **write ONLY to out_dir**
                        num_trials=1,
                        save_json=True,
                        # timestamp=run_ts,   # (uncomment if needed)
                    )
                    tokens = (artefacts or {}).get("feature_ablation", {}).get("tokens", [])
                    scores = (artefacts or {}).get("feature_ablation", {}).get("scores", [])
                    if tokens and scores and len(tokens) == len(scores):
                        wa = tokens_to_words(tokens, scores)
                        if wa:
                            top_words[comp_name] = [(w, sc) for w, sc in wa]
                except Exception as err:
                    print(f"[Captum] {comp_name} attribution skipped: {err}", file=sys.stderr)
                pbar.update(1)

    # ---------- pick captum PNGs ONLY in the provided out_dir ----------
    captum_imgs = {}
    pattern = re.compile(r"^(?P<type>query|imgfind|history)_(?P<ts>\d{8}_\d{6})_(?P<mode>seq|tok)\.png$")
    for path in glob.glob(os.path.join(out_dir, "*.png")):
        fname = os.path.basename(path)
        m = pattern.match(fname)
        if not m:
            continue
        if m.group("type") not in ran_types:
            continue
        label = f"captum_{m.group('type')}_{m.group('ts')}_{m.group('mode')}"
        captum_imgs[label] = path

    # ---------- images (optional) ----------
    def first_img(img):
        if isinstance(img, (list, tuple)):
            return img[0] if img else None
        return img

    original_xray = first_img(original_xray_path)
    gradcam_overlay = first_img(gradcam_overlay_path)

    # ---------- build result ----------
    fallback_xai_text = (
        "No explainable AI (XAI) report is available for this case. "
        "The system did not produce salient findings or attributions."
    )
    explain_text = "Explainable AI outputs attached. See attribution images and tables."
    if not (original_xray or gradcam_overlay or top_words or captum_imgs):
        explain_text = fallback_xai_text

    result = {
        "classification_results": classification_results or [],
        "original_xray": original_xray,     # may be None
        "gradcam_overlay": gradcam_overlay, # may be None
        "top_words": top_words,             # may include only the components we ran
        "explain_text": explain_text,
        "run_ts": run_ts,
    }
    result.update(captum_imgs)

    if classification_results:
        result["classification_labels"] = [
            {"label": str(lab), "confidence": float(score)}
            for lab, score in sorted(classification_results, key=lambda x: -float(x[1]))
        ]

    return result



def final_answer_node(state: "Any") -> dict:
    """
    Builds the final agent output and prepares inputs for XAI.
    """

    import sys
    import re
    import os
    from datetime import datetime

    # ---------- CONFIG ----------
    INCLUDE_VITALS_IN_ATTR = False

    def vitals_abnormal(v):
        if not v: 
            return False
        hr = v.get("heartrate")
        temp = v.get("temperature")
        rr = v.get("resprate")
        spo2 = v.get("o2sat")
        sbp = v.get("sbp")
        dbp = v.get("dbp")
        if hr is not None and (hr < 50 or hr > 110): return True
        if temp is not None and (temp < 95.0 or temp >= 100.4): return True
        if rr is not None and (rr < 8 or rr > 22): return True
        if spo2 is not None and spo2 < 92: return True
        if sbp is not None and sbp < 90: return True
        if dbp is not None and dbp < 50: return True
        return False

    def _find(step: str) -> str:
        return next(
            (
                getattr(s, "tool_output", None)
                for s in reversed(getattr(state, "intermediate_steps", []))
                if getattr(s, "tool_name", "") == step and getattr(s, "tool_output", None)
            ),
            "",
        )

    refusal_kw = {
        "cannot provide a diagnosis",
        "qualified healthcare professional",
        "disclaimer",
        "refuse",
    }

    def parse_history_text(history_text: str):
        if not history_text or not isinstance(history_text, str):
            return []
        blocks = re.split(r"\n\s*---\s*\n|^\s*---\s*$", history_text, flags=re.MULTILINE)
        entries = []
        cc_pat = re.compile(r"^\s*Chief Complaint\s*:\s*(.+)$", re.IGNORECASE)
        dx_pat = re.compile(r"^\s*Diagnosis\s*:\s*(.+)$", re.IGNORECASE)
        vit_pat = re.compile(r"Triage Vitals\s*:\s*(.*)$", re.IGNORECASE)
        v_map = {
            "Heartrate": "heartrate",
            "Temperature": "temperature",
            "Resprate": "resprate",
            "O2sat": "o2sat",
            "Sbp": "sbp",
            "Dbp": "dbp",
        }
        kv_pat = re.compile(r"(Heartrate|Temperature|Resprate|O2sat|Sbp|Dbp)\s*:\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
        adm_pat = re.compile(
            r"Admitted\s*:\s*([0-9:\-\s]+)\s*-\s*Discharged\s*:\s*([0-9:\-\s]+)\s*\(Disposition\s*:\s*([A-Z]+)\)",
            re.IGNORECASE
        )
        for raw in blocks:
            if not raw or not raw.strip():
                continue
            chief, diag, vitals, admitted, discharged, disp = None, None, None, None, None, None
            for line in raw.splitlines():
                line = line.strip()
                if not line:
                    continue
                m = cc_pat.match(line)
                if m:
                    chief = m.group(1).strip()
                    continue
                m = dx_pat.match(line)
                if m:
                    diag = m.group(1).strip()
                    continue
                m = vit_pat.search(line)
                if m:
                    vit_str = m.group(1)
                    parsed = {}
                    for vm in kv_pat.finditer(vit_str):
                        key = v_map.get(vm.group(1).capitalize())
                        try:
                            parsed[key] = float(vm.group(2))
                        except Exception:
                            parsed[key] = None
                    if parsed:
                        vitals = parsed
                    continue
                m = adm_pat.search(line)
                if m:
                    admitted = m.group(1).strip()
                    discharged = m.group(2).strip()
                    disp = m.group(3).strip().upper()
            entries.append({
                "chief_complaint": chief,
                "diagnosis": diag,
                "vitals": vitals,
                "admitted": admitted,
                "discharged": discharged,
                "disposition": disp,
            })
        return entries

    def format_vitals_short(v):
        if not v:
            return ""
        parts = []
        if v.get("heartrate") is not None: parts.append(f"HR {int(round(v['heartrate']))}")
        if v.get("temperature") is not None: parts.append(f"T {v['temperature']:.1f}")
        if v.get("resprate") is not None: parts.append(f"RR {int(round(v['resprate']))}")
        if v.get("o2sat") is not None: parts.append(f"SpO₂ {int(round(v['o2sat']))}%")
        sbp = v.get("sbp"); dbp = v.get("dbp")
        if sbp is not None and dbp is not None: parts.append(f"BP {int(round(sbp))}/{int(round(dbp))}")
        return ", ".join(parts)

    def build_history_for_attribution(history_text: str) -> str:
        entries = parse_history_text(history_text)
        if not entries:
            return ""
        entries = list(reversed(entries))
        out_lines = []
        for e in entries:
            cc = e.get("chief_complaint")
            dx = e.get("diagnosis")
            v = e.get("vitals")
            line = []
            if cc: line.append(f"Chief Complaint: {cc}")
            if dx: line.append(f"Diagnosis: {dx}")
            if v and (INCLUDE_VITALS_IN_ATTR or vitals_abnormal(v)):
                vs = format_vitals_short(v)
                if vs:
                    line.append(f"Vitals: {vs}")
            if line:
                out_lines.append(" | ".join(line))
        return "\n".join(out_lines[:3])  # cap length a bit

    # ---------- pick best textual answer ----------
    answer = _find("consistency_check")
    if not answer or any(w in answer.lower() for w in refusal_kw):
        for fb in ("history_diag", "image_diag", "query_diag"):
            answer = _find(fb)
            if answer and not any(w in answer.lower() for w in refusal_kw):
                break
        else:
            answer = (
                "⚠️ **No structured clinical diagnosis could be generated.** "
                "Please review the case and available findings."
            )

    context_cache = getattr(state, "context_cache", {}) or {}

    def get_user_input_query():
        return getattr(state, "input", None) or getattr(state, "query", None) or context_cache.get("query", "")

    def get_user_input_xray_findings():
        xf = context_cache.get("xray_findings", None)
        if xf and isinstance(xf, list):
            return ", ".join([str(xx) for xx in xf])
        return xf or ""

    def get_user_input_history():
        history_raw = context_cache.get("history_text", "") or context_cache.get("history", "")
        return build_history_for_attribution(history_raw)

    classification_results = context_cache.get("xray_classification", None)
    original_xray_path = context_cache.get("original_xray_path", None)
    gradcam_overlay_path = context_cache.get("gradcam_overlay_path", None)
    captum_img_path = context_cache.get("captum_img_path", None)

    # ---------- Unique session dir creation ----------
    # Get user_id and create run_ts
    if hasattr(state, "metadata"):
        user_id = (
            getattr(state, "metadata", {}).get("patient_id")
            or getattr(state, "metadata", {}).get("user_id")
            or "default"
        )
    else:
        user_id = "default"

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    _be_dir = os.path.dirname(os.path.abspath(__file__))
    session_dir = os.path.join(_be_dir, "outputs", str(user_id), run_ts)
    os.makedirs(session_dir, exist_ok=True)

    # ---------- XAI enrichment ----------
    try:
        # doctor2_model / doctor2_tokenizer / run_full_llm_attribution must be in scope!
        print("doctor2_model:", type(doctor2_model))
        print("doctor2_tokenizer:", type(doctor2_tokenizer))
        xai_dict = enrich_with_captum_xai(
            classification_results=classification_results,
            original_xray_path=original_xray_path,
            gradcam_overlay_path=gradcam_overlay_path,
            captum_img_path=captum_img_path,
            query=get_user_input_query(),
            xray_findings=get_user_input_xray_findings(),
            history=get_user_input_history(),
            answer=answer,
            doctor2_model=doctor2_model,
            doctor2_tokenizer=doctor2_tokenizer,
            run_full_llm_attribution=run_full_llm_attribution,
            user_id=user_id,
            out_dir=session_dir      # <----- All artifacts in this folder!
        )
    except Exception as err:
        print(f"[Captum] attribution skipped: {err}", file=sys.stderr)
        xai_dict = {}

    # ---------- fallbacks ----------
    fallback_diag = (
        answer if answer and str(answer).strip()
        else (context_cache.get("xray_findings") or "[No diagnosis generated]")
    )
    fallback_xai = xai_dict.get("explain_text") or "No XAI explanation generated for this case."

    # ---------- final payload ----------
    agent_output = {
        "diagnosis": fallback_diag,
        "xai_report": fallback_xai,
        "classification_results": xai_dict.get("classification_results", []),
        "original_xray": xai_dict.get("original_xray"),
        "gradcam_overlay": xai_dict.get("gradcam_overlay"),
        "top_words": xai_dict.get("top_words", {}),
        "classification_labels": xai_dict.get("classification_labels", []),
        "sections_available": xai_dict.get("sections_available", {
            "Query": True,
            "Image Findings": bool(get_user_input_xray_findings().strip()),
            "History": bool(get_user_input_history().strip()),
        }),
    }

    for k, v in xai_dict.items():
        if k.startswith("captum_"):
            agent_output[k] = v

    print("\n[Agent DEBUG] Output dict:", agent_output, "\n")
    return {"agent_outcome": agent_output}





# --------- COMPILE GRAPH --------- 
graph = StateGraph(AgentState)
graph.add_node("planner",        planner_node)
graph.add_node("react_agent",    react_agent_node)
graph.add_node("run_tool",       run_tool_node)
graph.add_node("reflector",      reflector_node)
graph.add_node("final_answer",   final_answer_node)
graph.add_edge("planner",      "react_agent")
graph.add_edge("react_agent",  "run_tool")
graph.add_edge("run_tool",     "reflector")
graph.add_conditional_edges(
    "reflector",
    path=lambda state: (
        "react_agent" if getattr(state, "reflect_decision", None) in ("REVISE", "CONTINUE") else "final_answer"
    )
)
graph.add_edge("final_answer", END)
graph.set_entry_point("planner")
runnable = graph.compile()

# --------- Visualize the pipeline (optional) ---------
png_bytes = runnable.get_graph().draw_mermaid_png()
with open("graph.png", "wb") as f:
    f.write(png_bytes)
print("✅ Το διάγραμμα γράφτηκε στο graph.png")

# Open image (Linux/Mac/Windows)
import subprocess, platform, os
if platform.system() == "Darwin":
    subprocess.run(["open", "graph.png"])
elif platform.system() == "Linux":
    subprocess.run(["xdg-open", "graph.png"])
elif platform.system() == "Windows":
    os.startfile("graph.png")

# --------- MEMORY / EXPORT UTILITIES ---------
class ConfigSpec:
    def __init__(self, id: str, description: str = ""):
        self.id = id
        self.description = description

class CustomMemorySaver:
    config_specs = [ConfigSpec(id="filename", description="Path to agent memory file.")]
    def __init__(self, filename="agent_memory.json"):
        self.filename = filename
    @property
    def config(self):
        return {"filename": self.filename}
    def save(self, state: dict) -> None:
        def pydantic_to_dict(o):
            if isinstance(o, BaseModel):
                return o.model_dump() if hasattr(o, "model_dump") else o.dict()
            return str(o)
        os.makedirs(os.path.dirname(self.filename) or ".", exist_ok=True)
        with open(self.filename, "w") as f:
            json.dump(state, f, default=pydantic_to_dict, indent=2)
        print(f"✅ State saved to {self.filename}")
    def load(self) -> Union[dict, None]:
        if os.path.exists(self.filename):
            with open(self.filename, "r") as f:
                return json.load(f)
        return None

# --------- MAIN EXECUTION & CLI ---------
if __name__ == "__main__" and not SKIP_CLI:
    import time, json

    xray_image  = input("\n🖼️ Enter chest X‑ray image file path (or press Enter to skip): ").strip()
    user_query  = input("\n📝 Enter your clinical query: ").strip()
    if not user_query:
        raise ValueError("❌ Query cannot be empty!")
    patient_id  = input("\n🔍 Enter Patient ID (or press Enter to skip): ").strip()

    metadata = {}
    if xray_image:
        metadata["image_path"] = xray_image
    if patient_id:
        metadata["patient_id"] = patient_id

    agent_state = {
        "input":              user_query,
        "chat_history":       [],
        "intermediate_steps": [],
        "agent_outcome":      "",
        "plan":               [],
        "metadata":           metadata,
        "self_refine_iter":   0,
        "context_cache":      {},
    }

    start_time = time.time()
    print("\n⏳ Επεξεργασία…")

    graph_output = runnable.invoke(agent_state, config={"recursion_limit": 50})
    elapsed = time.time() - start_time

    print("\nFinal Output from Agent:\n", graph_output.get("agent_outcome", ""))
    print(f"\n🕒 Χρόνος εκτέλεσης: {elapsed/60:.2f} min  ({elapsed:.1f} sec)")

    # (προαιρετικά) αποθήκευση μνήμης
    memory_saver = CustomMemorySaver(filename="agent_memory.json")
    memory_saver.save(graph_output)
    print("✅ Reasoning chain exported.")
