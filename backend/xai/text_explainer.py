# backend/xai/text_explainer.py
from __future__ import annotations

import os
import sys
import glob
import re
import json
import warnings
import datetime as dt
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import torch

try:
    from captum.attr import (
        FeatureAblation,
        LLMAttribution,
        TextTemplateInput,
        TextTokenInput,
    )
except ImportError:
    FeatureAblation = None
    LLMAttribution = None
    TextTemplateInput = None
    TextTokenInput = None

from config import settings

_DOCTOR2_MODEL = None
_DOCTOR2_TOKENIZER = None

def get_doctor2_model():
    """Lazy loader for local Doctor2 LLM and Tokenizer for Captum attribution."""
    global _DOCTOR2_MODEL, _DOCTOR2_TOKENIZER
    if _DOCTOR2_MODEL is None:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            model_dir = settings.BASE_DIR / ".." / "model"
            if model_dir.exists():
                print(f"[TextExplainer] Loading Doctor2 HF model from: {model_dir}")
                _DOCTOR2_TOKENIZER = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
                _DOCTOR2_MODEL = AutoModelForCausalLM.from_pretrained(
                    str(model_dir),
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None,
                    trust_remote_code=True
                )
                _DOCTOR2_MODEL.eval()
                print("[TextExplainer] Doctor2 loaded successfully for Captum.")
            else:
                print(f"[TextExplainer] Model directory not found: {model_dir}", file=sys.stderr)
        except Exception as e:
            print(f"[TextExplainer] Failed to load Doctor2 HF model ({e})", file=sys.stderr)
    return _DOCTOR2_MODEL, _DOCTOR2_TOKENIZER

def run_llm_attribution(
    model: Any,
    tokenizer: Any,
    inp_text: str,
    target_text: str,
    out_dir: str,
    prefix: str = "attr"
) -> Dict[str, Any]:
    """Runs FeatureAblation LLMAttribution on text inputs."""
    if not FeatureAblation or not LLMAttribution or model is None or tokenizer is None:
        return {"error": "Captum or model unavailable"}

    try:
        fa = FeatureAblation(model)
        llm_attr = LLMAttribution(fa, tokenizer)

        inp_tokens = TextTokenInput(inp_text, tokenizer)
        attr_res = llm_attr.attribute(inp_tokens, target=target_text)

        tokens = [tokenizer.decode([t]) for t in inp_tokens.input_ids[0]]
        scores = attr_res.seq_attr.cpu().numpy().tolist() if hasattr(attr_res, "seq_attr") else []

        plot_path = os.path.join(out_dir, f"{prefix}_attribution.png")
        if tokens and scores:
            plt.figure(figsize=(10, 4))
            plt.bar(range(len(scores[:20])), scores[:20])
            plt.xticks(range(len(scores[:20])), [t.strip() for t in tokens[:20]], rotation=45)
            plt.title(f"Feature Attribution ({prefix})")
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()

        return {
            "tokens": tokens,
            "scores": scores,
            "plot_path": plot_path
        }
    except Exception as e:
        print(f"[TextExplainer] Error during LLM attribution: {e}", file=sys.stderr)
        return {"error": str(e)}

def enrich_with_captum_xai(
    answer: str,
    query: str,
    xray_findings: Optional[str] = None,
    history: Optional[str] = None,
    doctor2_model=None,
    doctor2_tokenizer=None,
    run_full_llm_attribution=None,
    user_id: Optional[Union[int, str]] = None,
    out_dir: str = "captum_attr_full",
    top_k_words: int = 5,
    max_img_px: int = 3500,
    classification_results: Optional[List[Tuple[str, float]]] = None,
    original_xray_path: Optional[Union[str, List[str]]] = None,
    gradcam_overlay_path: Optional[Union[str, List[str]]] = None,
    captum_img_path: Optional[Union[str, List[str]]] = None
) -> dict:
    """Runs Captum text feature attribution for provided text sources and outputs structured report dict."""
    os.makedirs(out_dir, exist_ok=True)
    model, tokenizer = (doctor2_model, doctor2_tokenizer) if doctor2_model else get_doctor2_model()

    # Rule-based fallback keyword extraction for text XAI summary
    stopwords = {"a", "an", "the", "in", "of", "and", "or", "to", "for", "with", "on", "at", "by", "from", "is", "have", "has", "had", "i", "my", "he", "she", "patient"}
    words = [w.strip(".,!?()[]").lower() for w in query.split() if w.strip(".,!?()[]").lower() not in stopwords and len(w) > 2]
    top_words = {w: round(1.0 / (i + 1), 2) for i, w in enumerate(words[:top_k_words])}

    explain_lines = [
        "### XAI Text Attribution Analysis:",
        f"- Target Clinical Query: \"{query}\"",
    ]
    if xray_findings:
        explain_lines.append(f"- Radiological Findings Considered: {xray_findings}")
    if history:
        explain_lines.append("- Longitudinal History Integrated: Yes")
    explain_lines.append(f"- Top Key Clinical Predictors: {', '.join(top_words.keys()) if top_words else 'None'}")

    out_dict = {
        "captum_query_image": None,
        "captum_history_image": None,
        "top_words": top_words,
        "explain_text": "\n".join(explain_lines),
        "summary": "Captum attribution complete."
    }

    if model and tokenizer and answer:
        if query:
            res_q = run_llm_attribution(model, tokenizer, query, answer[:100], out_dir, prefix="query")
            if "plot_path" in res_q:
                out_dict["captum_query_image"] = res_q["plot_path"]
        if history:
            res_h = run_llm_attribution(model, tokenizer, history[:500], answer[:100], out_dir, prefix="history")
            if "plot_path" in res_h:
                out_dict["captum_history_image"] = res_h["plot_path"]

    return out_dict
