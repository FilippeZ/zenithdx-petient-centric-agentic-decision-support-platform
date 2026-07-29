#!/usr/bin/env python3 
"""
doctor2_captum_helper.py
=========================
... (header unchanged)
"""

from __future__ import annotations

import json
import os
import datetime as dt
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import torch
from captum.attr import (
    FeatureAblation,
    LLMAttribution,
    TextTemplateInput,
    TextTokenInput,
)
from transformers import AutoModelForCausalLM, AutoTokenizer

from tqdm import tqdm  # <-- Progress bar

###############################################################################
# ─────────────────────────────── EDIT THIS PATH ──────────────────────────────
###############################################################################
_HERE = Path(__file__).resolve().parent
DOCTOR2_HF_PATH: str | Path = _HERE / ".." / "model"  # ZenithDx_Final/model/
###############################################################################

def _hf_is_valid(folder: str | Path) -> bool:
    p = Path(folder)
    return (
        p.is_dir()
        and (p / "config.json").is_file()
        and (any(p.glob("*.bin")) or any(p.glob("*.safetensors")))
    )

try:
    from transformers import BitsAndBytesConfig  # type: ignore
    _HAVE_BNB = True
except Exception:
    _HAVE_BNB = False
    class BitsAndBytesConfig:  # type: ignore
        def __init__(self, *_, **__):
            raise RuntimeError("bitsandbytes not installed")

def _bnb_cfg() -> Optional["BitsAndBytesConfig"]:
    if not _HAVE_BNB:
        return None
    try:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    except Exception:
        return None

def _load_model(
    base: str | Path,
    *,
    try_4bit: bool = True,
) -> tuple[Optional[torch.nn.Module], Optional[object]]:
    print(f"[doctor2] Trying to load model from: {base}")
    quant_cfg = _bnb_cfg() if (try_4bit and _HAVE_BNB) else None
    kwargs: Dict[str, Any] = {"trust_remote_code": True, "device_map": "auto"}
    if quant_cfg:
        kwargs["quantization_config"] = quant_cfg

    try:
        model = AutoModelForCausalLM.from_pretrained(base, **kwargs)
        print("[doctor2] Model loaded successfully.")
        tokenizer = AutoTokenizer.from_pretrained(
            base, use_fast=True, trust_remote_code=True
        )
        print("[doctor2] Tokenizer loaded successfully.")
        tokenizer.pad_token = tokenizer.eos_token

        model.eval().requires_grad_(False)
        model.config.use_cache = False

        mode = "4-bit" if quant_cfg else "fp16/fp32"
        print(f"[doctor2] ✔ loaded HF checkpoint ({mode})")
        print(f"[doctor2] Model type: {type(model)} | Tokenizer type: {type(tokenizer)}")
        return model, tokenizer

    except Exception as err:
        print(f"[doctor2] ERROR during model/tokenizer load: {err}")
        if quant_cfg is not None:
            warnings.warn(f"[doctor2] load failed (4-bit): {err}; retrying full-precision…")
            return _load_model(base, try_4bit=False)
        warnings.warn(f"[doctor2] load failed (full): {err}")
        return None, None

# ====================== LOAD MODEL AT MODULE IMPORT ===========================
doctor2_model: Optional[torch.nn.Module] = None
doctor2_tokenizer: Optional[object] = None

print("Checking HF validity for:", DOCTOR2_HF_PATH)
print("Path exists:", Path(DOCTOR2_HF_PATH).exists())
print("Is dir:", Path(DOCTOR2_HF_PATH).is_dir())
print("Has config.json:", (Path(DOCTOR2_HF_PATH) / "config.json").is_file())
print("Has *.bin:", any(Path(DOCTOR2_HF_PATH).glob("*.bin")))
print("Has *.safetensors:", any(Path(DOCTOR2_HF_PATH).glob("*.safetensors")))

if _hf_is_valid(DOCTOR2_HF_PATH):
    print("[doctor2] HF model path is valid. Proceeding to load...")
    doctor2_model, doctor2_tokenizer = _load_model(DOCTOR2_HF_PATH)
    print("Model loaded: ", type(doctor2_model))
    print("Tokenizer loaded: ", type(doctor2_tokenizer))
else:
    warnings.warn(f"[doctor2] ❌ HF model path not valid: {DOCTOR2_HF_PATH}")

if doctor2_model is None or doctor2_tokenizer is None:
    warnings.warn("[doctor2] ❌ could NOT load model or tokenizer – Captum will be skipped")
    print(f"doctor2_model: {doctor2_model}")
    print(f"doctor2_tokenizer: {doctor2_tokenizer}")

warnings.filterwarnings("ignore", ".*past_key_values.*")
warnings.filterwarnings("ignore", ".*Skipping this token.*")

def _plot_and_save(attr_obj, out_dir: str, tag: str, input_text: str = None) -> tuple[str, str]:
    """Save sequence-level bar plot and token-token heatmap to *out_dir*.
    If input_text is given, display it as the X-axis label.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Robust: support both old/new Captum
    if hasattr(attr_obj, "sequence_attributions"):
        attributions = attr_obj.sequence_attributions
    elif hasattr(attr_obj, "attributions"):
        attributions = attr_obj.attributions
    else:
        attributions = None

    # --- Bar chart (sequence attribution) ---
    plt.figure(figsize=(6, 2))  # Small fixed size!
    if (
        attributions is not None
        and input_text is not None
        and isinstance(input_text, str)
        and len(input_text.strip()) > 0
    ):
        plt.bar([input_text], [sum(attributions)])
        plt.ylabel("Sequence Attribution")
        plt.xticks([0], [input_text], rotation=10, ha='right')
    else:
        if hasattr(attr_obj, "plot_seq_attr"):
            attr_obj.plot_seq_attr(show=False)
    plt.tight_layout()
    seq_png = f"{tag}_seq.png"
    plt.savefig(Path(out_dir, seq_png), dpi=80)  # Lower dpi!
    plt.close()

    # --- Heatmap (token attribution) ---
    plt.figure(figsize=(8, 2))  # Small
    if hasattr(attr_obj, "plot_token_attr"):
        attr_obj.plot_token_attr(show=False)
    tok_png = f"{tag}_tok.png"
    plt.savefig(Path(out_dir, tok_png), dpi=80)
    plt.close()

    return seq_png, tok_png

def run_full_llm_attribution(
    *,
    prompt: str,
    target_text: str = "",
    model: torch.nn.Module | None = None,
    tokenizer: object | None = None,
    template: str | None = None,
    values: Union[List, Dict, None] = None,
    baselines: Union[Dict, List, None] = None,
    mask: Dict | None = None,
    skip_tokens: Tuple[int, ...] = (1,),
    num_trials: int = 1,
    out_dir: str = "captum_attr_full",
    tag_prefix: str = "attr",
    save_json: bool = True,
) -> Dict[str, Dict[str, str]]:
    if model is None or tokenizer is None:
        warnings.warn("[Captum] model OR tokenizer not supplied – skipping attribution")
        print("[Captum] Skipping attribution due to model/tokenizer None!")
        return {}

    if template is not None:
        inp = TextTemplateInput(template, values=values, baselines=baselines, mask=mask)
    else:
        inp = TextTokenInput(prompt, tokenizer, skip_tokens=skip_tokens)

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    tag_base = f"{tag_prefix}_{ts}"
    artefacts: Dict[str, Dict[str, str]] = {"prompt": prompt, "target": target_text}

    fa = LLMAttribution(FeatureAblation(model), tokenizer)
    n_feats = len(inp.features) if hasattr(inp, "features") else None

    if hasattr(fa, "attribute_generator") and n_feats is not None:
        results = []
        with tqdm(total=n_feats, desc="Feature Ablation (XAI)") as pbar:
            for res in fa.attribute_generator(
                inp,
                target=target_text,
                skip_tokens=skip_tokens,
                num_trials=num_trials,
            ):
                results.append(res)
                pbar.update(1)
        fa_attr = results[-1] if results else None
    else:
        fa_attr = fa.attribute(
            inp,
            target=target_text,
            skip_tokens=skip_tokens,
            num_trials=num_trials,
        )

    if fa_attr is None:
        warnings.warn("[Captum] Feature Ablation failed; no attribution returned")
        print("[Captum] Feature Ablation failed – no attribution returned!")
        return artefacts

    # Pass the prompt (user input) as the label for the bar plot
    s, t = _plot_and_save(fa_attr, out_dir, f"{tag_prefix}_{ts}", input_text=prompt)
    feature_ablation = {
        "seq_png": str(Path(out_dir, s)),
        "tok_png": str(Path(out_dir, t)),
    }

    # --- tokens and scores for word-level attributions ---
    if hasattr(fa_attr, "tokens") and hasattr(fa_attr, "attributions"):
        feature_ablation["tokens"] = [str(tok) for tok in fa_attr.tokens]
        feature_ablation["scores"] = [float(x) for x in fa_attr.attributions]
    artefacts["feature_ablation"] = feature_ablation

    if save_json:
        manifest = Path(out_dir, f"{tag_prefix}_{ts}_manifest.json")
        dumpable = artefacts.copy()
        if "feature_ablation" in dumpable and dumpable["feature_ablation"] is not None:
            for k, v in dumpable["feature_ablation"].items():
                if isinstance(v, (list, tuple)):
                    dumpable["feature_ablation"][k] = [str(x) for x in v]
        with open(manifest, "w") as fp:
            json.dump(dumpable, fp, indent=2)
        artefacts["manifest"] = str(manifest)

    return artefacts

__all__ = [
    "doctor2_model",
    "doctor2_tokenizer",
    "run_full_llm_attribution",
]