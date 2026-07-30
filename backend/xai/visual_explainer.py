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
from pipelines.vision.s2a_unet import sa_unet_predict, apply_mask
from pipelines.vision.resnet50 import resnet_predict, IMG_SIZE_CLS, DEFAULT_CLINICAL_CLASSES, DEFAULT_CUTOFFS

def grad_cam_torch(model: torch.nn.Module, image_tensor: torch.Tensor, target_class_idx: int, target_layer: str = 'layer4') -> np.ndarray:
    """
    Stage 3 Grad-CAM Visual Explainability:
    Attaches forward/backward hooks on layer4 of ResNet-50.
    Computes gradients of predicted class logit w.r.t. layer4 feature maps,
    applies Global Average Pooling for channel importance weights,
    takes weighted sum + ReLU, and returns raw activation map.
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

    image_tensor_req = image_tensor.detach().clone().requires_grad_(True)
    output = model(image_tensor_req)
    pred_score = output[0, target_class_idx]
    model.zero_grad()
    pred_score.backward(retain_graph=True)
    
    if 'value' in activations and 'value' in gradients:
        acts = activations['value'][0]
        grads = gradients['value'][0]
        weights = grads.mean(dim=(1, 2))
        cam = (weights[:, None, None] * acts).sum(dim=0)
        cam = torch.relu(cam).cpu().numpy()
    else:
        cam = np.ones((7, 7), dtype="float32")

    if handle_f: handle_f.remove()
    if handle_b: handle_b.remove()

    return cam

def get_user_output_dir(user_id: Optional[str] = None) -> str:
    out_dir = settings.OUTPUT_DIR / str(user_id or "default")
    out_dir.mkdir(parents=True, exist_ok=True)
    return str(out_dir)

def cv2_read_img(path: str) -> Optional[np.ndarray]:
    try:
        data = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return cv2.imread(path)

def cv2_write_img(path: str, img: np.ndarray) -> bool:
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
    thresholds: Optional[np.ndarray] = None,
    label_cols: Optional[List[str]] = None,
    device: str = "cpu",
    return_explainability: bool = True,
    user_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Theoretical Pipeline Chaining Architecture for ZenithDx:
    Step 1: Inference of S²A-UNet -> Binary Mask M (256x256x1 / original size)
    Step 2: Crop & Mask Gating (Segmented ROI Extraction):
            - Find Bounding Box (xmin, ymin, w, h) of mask M
            - Crop raw image I_raw and mask M to bounding box
            - Element-wise mask gating: I_segmented_crop = I_crop * M_crop (zeroing non-lung background)
            - Resize I_segmented_crop and M_crop to 224x224 -> input_tensor_224 & M_crop_224
    Step 3: Inference of ResNet-50 on input_tensor_224 -> 6 pathology logits
    Step 4: Grad-CAM on input_tensor_224 & ResNet-50 layer4 w.r.t. predicted target class
    Step 5: Multi-Level Mask Gating on 224x224 & Canvas Reconstruction:
            - Resize Grad-CAM raw activation map to 224x224 -> cam_224
            - Multi-Level Mask Gating: cam_224_gated = cam_224 * M_crop_224 (zeroing edge activations)
            - Normalize to [0, 1] & Resize back to ROI crop size (w, h) -> cam_roi
            - Reconstruct full (H_raw, W_raw) canvas: cam_full[ymin:ymax, xmin:xmax] = cam_roi
            - Gate element-wise with full mask M: cam_full = cam_full * M
    """
    if label_cols is None:
        label_cols = DEFAULT_CLINICAL_CLASSES
    if thresholds is None:
        thresholds = DEFAULT_CUTOFFS

    out_dir = get_user_output_dir(user_id)
    print("\n===== [ZenithDx Image Pipeline] detect_chest_xray() =====")

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

    h, w = img_rgb.shape[:2]

    # Step 1: S²A-UNet Lung Segmentation
    try:
        mask = sa_unet_predict(sa_unet, img_rgb)
    except Exception as e:
        print(f"[VisualExplainer] Segmentation warning ({e}). Generating fallback mask.", file=sys.stderr)
        mask = np.ones((h, w), dtype="float32")

    mask_img = (mask * 255).astype(np.uint8)
    mask_path = os.path.join(out_dir, f"{fname}_segmask.png")
    cv2_write_img(mask_path, mask_img)

    # Step 2 & 3: Segmented ROI Extraction + ResNet-50 Multi-label Prediction
    input_tensor_224 = None
    M_crop_224 = None
    bbox = (0, 0, w, h)
    segmented_full = img_rgb.copy()
    all_prob_pairs = []
    
    if resnet is not None:
        try:
            preds, probs, input_tensor_224, M_crop_224, bbox, segmented_full = resnet_predict(
                resnet, img_rgb, mask, thresholds, device
            )
            all_prob_pairs = [(label, float(prob)) for label, prob in zip(label_cols, probs)]
            findings = [(label, float(prob)) for label, prob, pred in zip(label_cols, probs, preds) if pred]
            if not findings:
                top_idx = int(np.argmax(probs))
                findings = [(label_cols[top_idx], float(probs[top_idx]))]
            findings = sorted(findings, key=lambda x: x[1], reverse=True)
        except Exception as e:
            print(f"[VisualExplainer] Classification warning: {e}", file=sys.stderr)
            findings = [("No Finding", 0.0)]
    else:
        findings = [("No Finding", 0.0)]

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

    # Step 4 & 5: Grad-CAM on 224x224 Input Tensor & Multi-Level Mask Gating
    if return_explainability and resnet is not None and input_tensor_224 is not None:
        try:
            if findings and isinstance(findings[0], (tuple, list)) and findings[0][0] in label_cols:
                top_label = findings[0][0]
                top_idx = label_cols.index(top_label)
            else:
                top_idx = 0
                top_label = label_cols[0] if label_cols else "Attention Map"

            # Step 4: Backward pass on predicted class logit w.r.t. layer4
            cam_7x7 = grad_cam_torch(resnet, input_tensor_224, target_class_idx=top_idx, target_layer='layer4')
            
            # Step 5: Multi-Level Mask Gating on 224x224
            cam_224 = cv2.resize(cam_7x7, IMG_SIZE_CLS, interpolation=cv2.INTER_LINEAR)
            
            # Multi-Level Gating: Multiply 224x224 Grad-CAM map with 224x224 lung mask (M_crop_224)
            if M_crop_224 is not None:
                cam_224_gated = cam_224 * M_crop_224
            else:
                cam_224_gated = cam_224

            # Normalize 224x224 gated activation map to [0, 1]
            cam_224_gated = (cam_224_gated - cam_224_gated.min()) / (cam_224_gated.max() - cam_224_gated.min() + 1e-8)

            # Resize back to ROI crop dimensions (bw, bh)
            x, y, bw, bh = bbox
            cam_roi = cv2.resize(cam_224_gated, (max(1, bw), max(1, bh)), interpolation=cv2.INTER_LINEAR)

            # Reconstruct full (h, w) original canvas
            cam_full = np.zeros((h, w), dtype=np.float32)
            cam_full[y:y+bh, x:x+bw] = cam_roi

            # Final Canvas Level Mask Gating with full S²A-UNet lung mask M
            cam_full = cam_full * mask
            cam_full = (cam_full - cam_full.min()) / (cam_full.max() - cam_full.min() + 1e-8)

            # Convert to JET Colormap
            heatmap = cv2.applyColorMap(np.uint8(255 * cam_full), cv2.COLORMAP_JET)
            heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            
            # Zero out all non-lung pixels to black
            mask3 = np.repeat(mask[..., None], 3, axis=-1).astype(np.float32)
            heatmap_masked = (heatmap_rgb * mask3).astype(np.uint8)

            heatmap_path = os.path.join(out_dir, f"{fname}_gradcam_heatmap.png")
            cv2_write_img(heatmap_path, cv2.cvtColor(heatmap_masked, cv2.COLOR_RGB2BGR))

            # Standard Grad-CAM Overlay (on raw chest X-ray)
            gradcam_overlay = cv2.addWeighted(img_rgb, 0.55, heatmap_masked, 0.45, 0)
            gradcam_overlay_path = os.path.join(out_dir, f"{fname}_gradcam_overlay.png")
            cv2_write_img(gradcam_overlay_path, cv2.cvtColor(gradcam_overlay, cv2.COLOR_RGB2BGR))

            # Segmented Grad-CAM Overlay (strictly on S²A-UNet segmented lungs)
            gradcam_segmented = cv2.addWeighted(segmented_full, 0.55, heatmap_masked, 0.45, 0)
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
        "findings": findings if findings else [("No Finding", 0.0)],
        "all_probabilities": all_prob_pairs,
        "paths": {
            "original": orig_path,
            "mask": mask_path,
            "gradcam_heatmap": heatmap_path,
            "gradcam_overlay": gradcam_overlay_path,
            "gradcam_segmented": gradcam_segmented_path,
        },
        "gradcam": gradcam_info,
    }
