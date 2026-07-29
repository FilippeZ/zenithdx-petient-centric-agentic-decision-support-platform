# backend/xai/visual_explainer.py
from __future__ import annotations

import os
import sys
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
import cv2
import numpy as np
import torch

from config import settings
from pipelines.vision.s2a_unet import sa_unet_predict
from pipelines.vision.resnet50 import resnet_predict, IMG_SIZE_CLS

def grad_cam_torch(model: torch.nn.Module, image_tensor: torch.Tensor, target_class_idx: int, target_layer: str = 'layer4') -> np.ndarray:
    """Returns Grad-CAM activation heatmap array for ResNet-50 target layer."""
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

def get_user_output_dir(user_id: Optional[str] = None) -> str:
    """Returns output directory for user output assets, ensuring creation."""
    out_dir = settings.OUTPUT_DIR / str(user_id or "default")
    out_dir.mkdir(parents=True, exist_ok=True)
    return str(out_dir)

def cv2_read_img(path: str) -> Optional[np.ndarray]:
    """Reads image file supporting non-ASCII paths on Windows."""
    try:
        data = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return cv2.imread(path)

def cv2_write_img(path: str, img: np.ndarray) -> bool:
    """Writes image file supporting non-ASCII paths on Windows."""
    try:
        ext = os.path.splitext(path)[1] or ".png"
        res, buf = cv2.imencode(ext, img)
        if res:
            buf.tofile(path)
            return True
        return False
    except Exception:
        return cv2.imwrite(path, img)

def detect_chest_xray(
    image_or_path: Any,
    sa_unet: Any,
    resnet: Any,
    thresholds: np.ndarray,
    label_cols: List[str],
    device: str = "cpu",
    return_explainability: bool = True,
    user_id: Optional[str] = None
) -> Dict[str, Any]:
    """Pipeline for X-ray segmentation, classification, and Grad-CAM explainability."""
    out_dir = get_user_output_dir(user_id)
    print("\n===== [Image Debug] detect_chest_xray() =====")

    if isinstance(image_or_path, str):
        img_arr = cv2_read_img(image_or_path)
        if img_arr is None:
            print(f"[Error] Could not load image: {image_or_path}", file=sys.stderr)
            raise FileNotFoundError(f"Could not load image: {image_or_path}")
        img_rgb = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
        fname = os.path.splitext(os.path.basename(image_or_path))[0]
    elif isinstance(image_or_path, np.ndarray):
        img_rgb = image_or_path
        fname = f"xray_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    else:
        raise ValueError("Input must be file path string or numpy array")

    # 1. Segmentation
    if sa_unet is not None:
        try:
            mask = sa_unet_predict(sa_unet, img_rgb)
        except Exception as e:
            print(f"[VisualExplainer] Segmentation warning: {e}", file=sys.stderr)
            mask = np.ones((img_rgb.shape[0], img_rgb.shape[1]), dtype="float32")
    else:
        mask = np.ones((img_rgb.shape[0], img_rgb.shape[1]), dtype="float32")

    mask_img = (mask * 255).astype(np.uint8)
    mask_path = os.path.join(out_dir, f"{fname}_segmask.png")
    cv2_write_img(mask_path, mask_img)

    # 2. Classification
    if resnet is not None:
        try:
            preds, probs, tensor = resnet_predict(resnet, img_rgb, mask, thresholds, device)
            findings = [(label, float(prob)) for label, prob, pred in zip(label_cols, probs, preds) if pred]
            findings = sorted(findings, key=lambda x: x[1], reverse=True)
        except Exception as e:
            print(f"[VisualExplainer] Classification warning: {e}", file=sys.stderr)
            findings = []
            tensor = None
    else:
        findings = []
        tensor = None

    # Save original image
    orig_path = os.path.join(out_dir, f"{fname}_original.png")
    cv2_write_img(orig_path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))

    gradcam_info = {
        "label": None,
        "heatmap": None,
        "gradcam_overlay": None,
        "gradcam_segmented": None,
    }

    heatmap_path = None
    gradcam_overlay_path = None
    gradcam_segmented_path = None

    # 3. Grad-CAM Explainability
    if return_explainability and findings and resnet is not None and tensor is not None:
        try:
            top_label, _ = findings[0]
            top_idx = label_cols.index(top_label)
            cam = grad_cam_torch(resnet, tensor, target_class_idx=top_idx, target_layer='layer4')
            cam = cv2.resize(cam, IMG_SIZE_CLS, interpolation=cv2.INTER_LINEAR)
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            
            heatmap_path = os.path.join(out_dir, f"{fname}_gradcam_heatmap.png")
            cv2_write_img(heatmap_path, cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR))

            orig_resized = cv2.resize(img_rgb, IMG_SIZE_CLS, interpolation=cv2.INTER_AREA)
            gradcam_overlay = cv2.addWeighted(orig_resized, 0.5, heatmap, 0.5, 0)
            gradcam_overlay_path = os.path.join(out_dir, f"{fname}_gradcam_overlay.png")
            cv2_write_img(gradcam_overlay_path, cv2.cvtColor(gradcam_overlay, cv2.COLOR_RGB2BGR))

            mask_resized = cv2.resize(mask, IMG_SIZE_CLS, interpolation=cv2.INTER_NEAREST)
            mask3 = np.repeat(mask_resized[..., None], 3, axis=-1).astype(np.float32)
            masked_img = (orig_resized * mask3).astype(np.uint8)
            heatmap_masked = (heatmap * mask3).astype(np.uint8)
            alpha = 0.4
            gradcam_segmented = cv2.addWeighted(masked_img, 1 - alpha, heatmap_masked, alpha, 0)
            gradcam_segmented_path = os.path.join(out_dir, f"{fname}_gradcam_segmented.png")
            cv2_write_img(gradcam_segmented_path, cv2.cvtColor(gradcam_segmented, cv2.COLOR_RGB2BGR))

            gradcam_info = {
                "label": top_label,
                "heatmap": heatmap_path,
                "gradcam_overlay": gradcam_overlay_path,
                "gradcam_segmented": gradcam_segmented_path,
            }
        except Exception as e:
            print(f"[VisualExplainer] Grad-CAM warning: {e}", file=sys.stderr)

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
