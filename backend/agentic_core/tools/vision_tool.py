# backend/agentic_core/tools/vision_tool.py
from __future__ import annotations

import sys
import numpy as np
import torch

from config import settings
from pipelines.vision.s2a_unet import load_sa_unet
from pipelines.vision.resnet50 import load_resnet
from xai.visual_explainer import detect_chest_xray

DEFAULT_LABEL_COLS = [
    "Atelectasis", "Consolidation", "Edema",
    "Lung Lesion", "Lung Opacity", "Pneumonia"
]

_SA_UNET = None
_RESNET = None
_THRESHOLDS = None
_LABEL_COLS = None
_DEVICE = "cpu"

def _init_vision_models():
    global _SA_UNET, _RESNET, _THRESHOLDS, _LABEL_COLS, _DEVICE
    if _SA_UNET is None or _RESNET is None:
        try:
            _LABEL_COLS = DEFAULT_LABEL_COLS
            _SA_UNET = load_sa_unet(str(settings.SA_UNET_WEIGHTS))
            res_out = load_resnet(str(settings.RESNET_WEIGHTS), _LABEL_COLS)
            if isinstance(res_out, tuple) and len(res_out) == 3:
                _RESNET, _THRESHOLDS, _DEVICE = res_out
            print("[VisionTool] Models initialization attempted.")
        except Exception as e:
            print(f"[VisionTool] Error loading vision models ({e})", file=sys.stderr)

from pipelines.vision.cloud_vision_client import CloudVisionClient

_CLOUD_CLIENT = CloudVisionClient()

def run_vision_analysis(image_path: str, user_id: str = "default") -> dict:
    """Executes S2A-UNet segmentation, ResNet classification, and Grad-CAM explainability locally or via Google Colab Cloud GPU."""
    if _CLOUD_CLIENT.is_configured():
        try:
            print(f"[VisionTool] ☁️ Sending X-ray to Cloud Vision Microservice: {_CLOUD_CLIENT.endpoint_url}")
            return _CLOUD_CLIENT.analyze_xray(image_path)
        except Exception as e:
            print(f"[VisionTool] ⚠️ Cloud Vision request failed ({e}). Falling back to local execution.", file=sys.stderr)

    _init_vision_models()
    
    return detect_chest_xray(
        image_or_path=image_path,
        sa_unet=_SA_UNET,
        resnet=_RESNET,
        thresholds=_THRESHOLDS if _THRESHOLDS is not None else np.array([0.70] * len(DEFAULT_LABEL_COLS)),
        label_cols=_LABEL_COLS or DEFAULT_LABEL_COLS,
        device=_DEVICE or "cpu",
        return_explainability=True,
        user_id=user_id
    )
