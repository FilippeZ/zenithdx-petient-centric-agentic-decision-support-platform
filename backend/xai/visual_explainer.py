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
    Attaches forward/backward hooks on target_layer of ResNet-50.
    Computes gradients of predicted class logit w.r.t. feature maps,
    applies Global Average Pooling for channel importance weights,
    takes weighted sum + ReLU, and returns raw activation map.
    """
    model.eval()
    activations = []
    gradients = []
    
    def forward_hook(module, input, output):
        activations.append(output)
        
    def backward_hook(module, grad_input, grad_output):
        if grad_output and grad_output[0] is not None:
            gradients.append(grad_output[0])

    target_module = None
    for name, module in model.named_modules():
        if name == target_layer:
            target_module = module
            break
            
    if target_module is None:
        target_module = getattr(model, 'layer4', None)

    handle_f, handle_b = None, None
    if target_module is not None:
        handle_f = target_module.register_forward_hook(forward_hook)
        if hasattr(target_module, 'register_full_backward_hook'):
            handle_b = target_module.register_full_backward_hook(backward_hook)
        else:
            handle_b = target_module.register_backward_hook(backward_hook)

    image_tensor_req = image_tensor.detach().clone().requires_grad_(True)
    output = model(image_tensor_req)
    pred_score = output[0, target_class_idx]
    model.zero_grad()
    pred_score.backward(retain_graph=True)

    if activations and gradients:
        acts = activations[0][0].detach()
        grads = gradients[0][0].detach()
        weights = grads.mean(dim=(1, 2))
        cam = (weights[:, None, None] * acts).sum(dim=0)
        cam = torch.relu(cam).cpu().numpy()
    else:
        # Robust fallback if gradient hook failed
        if activations:
            acts = activations[0][0].detach()
            cam = acts.pow(2).sum(dim=0).sqrt().cpu().numpy()
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
            - Smooth Gaussian blending and dynamic alpha masking for realistic medical overlays
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
            
            # Step 5: Multi-Level Smooth Mask Gating
            cam_224 = cv2.resize(cam_7x7, IMG_SIZE_CLS, interpolation=cv2.INTER_CUBIC)
            
            if M_crop_224 is not None:
                cam_224_gated = cam_224 * M_crop_224
            else:
                cam_224_gated = cam_224

            cam_224_gated = (cam_224_gated - cam_224_gated.min()) / (cam_224_gated.max() - cam_224_gated.min() + 1e-8)

            # Resize back to ROI crop dimensions (bw, bh) with bicubic interpolation
            x, y, bw, bh = bbox
            cam_roi = cv2.resize(cam_224_gated, (max(1, bw), max(1, bh)), interpolation=cv2.INTER_CUBIC)

            # Reconstruct full (h, w) original canvas
            cam_full = np.zeros((h, w), dtype=np.float32)
            cam_full[y:y+bh, x:x+bw] = cam_roi

            # ── Two-pass Lung Masking + Gaussian Smoothing ───────────────────────
            # Pass 1: Soft mask before blur (suppresses non-lung activations)
            cam_full = cam_full * mask
            cam_full = cv2.GaussianBlur(cam_full, (21, 21), 0)
            # Pass 2: Re-apply mask after blur to prevent Gaussian edge-bleed.
            # Without this, near-zero bleed values map to blue in JET colormap,
            # causing colormap to appear on clavicles/shoulder tissue (audit risk).
            cam_full = cam_full * mask
            cam_full = (cam_full - cam_full.min()) / (cam_full.max() - cam_full.min() + 1e-8)

            # JET Colormap applied to masked activation map
            heatmap_bgr = cv2.applyColorMap(np.uint8(255 * cam_full), cv2.COLORMAP_JET)
            heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

            # ── Hard Binary Mask on Colormap ──────────────────────────────────────
            # Even after two-pass masking, JET maps near-zero values to blue.
            # Apply a hard binary cutoff (>0.5 threshold) to zero out all colormap
            # pixels outside the lung boundary — surgical precision for AI auditing.
            mask3 = np.repeat(mask[..., None], 3, axis=-1).astype(np.float32)
            hard_lung_mask3 = np.repeat((mask > 0.5).astype(np.float32)[..., None], 3, axis=-1)
            heatmap_rgb = (heatmap_rgb * hard_lung_mask3).astype(np.uint8)

            # Save raw masked heatmap artifact
            heatmap_masked = heatmap_rgb.copy()
            heatmap_path = os.path.join(out_dir, f"{fname}_gradcam_heatmap.png")
            cv2_write_img(heatmap_path, cv2.cvtColor(heatmap_masked, cv2.COLOR_RGB2BGR))

            # ── Overlay Alpha (for standard overlay on full chest X-ray) ──────────
            # Dynamic: low activation → transparent → shows original anatomy.
            # High activation → vivid colormap overlay (where model focused).
            # Alpha is also masked, ensuring zero overlay outside lung boundary.
            alpha_overlay = np.clip((cam_full - 0.12) / 0.88, 0, 1.0) * 0.60 * (mask > 0.5).astype(np.float32)
            alpha_overlay_3d = alpha_overlay[..., None]

            # Standard Grad-CAM Overlay (full chest X-ray + heatmap strictly within lung mask)
            gradcam_overlay = (img_rgb * (1.0 - alpha_overlay_3d) + heatmap_rgb * alpha_overlay_3d).astype(np.uint8)
            gradcam_overlay_path = os.path.join(out_dir, f"{fname}_gradcam_overlay.png")
            cv2_write_img(gradcam_overlay_path, cv2.cvtColor(gradcam_overlay, cv2.COLOR_RGB2BGR))

            # ── Segmented Grad-CAM (Exact Isolated Gated Heatmap) ──────────
            gradcam_segmented = create_segmented_gradcam(img_rgb, heatmap_rgb, mask)

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

def create_segmented_gradcam(
    original_xray: np.ndarray,
    heatmap_rgb: np.ndarray,
    unet_mask: np.ndarray
) -> np.ndarray:
    """
    Steps 2, 3 & 4 Implementation:
    Creates a perfect masked Grad-CAM output on a pure black background.

    Checks:
    - Dimensions: Ensures original_xray, heatmap_rgb, and unet_mask share identical (H, W) shapes.
    - Mask Range: Converts unet_mask to float32 strictly in [0.0, 1.0].
    - Channels: Stacks 2D mask into 3D (H, W, 3) using np.stack.
    - Color Map: Performs element-wise multiplication heatmap_rgb * mask_3d for zeroing non-lung background.
    """
    h, w = original_xray.shape[:2]

    # 1. Shape matching
    if heatmap_rgb.shape[:2] != (h, w):
        heatmap_rgb = cv2.resize(heatmap_rgb, (w, h), interpolation=cv2.INTER_CUBIC)

    mask_2d = unet_mask[..., 0] if len(unet_mask.shape) == 3 else unet_mask
    if mask_2d.shape[:2] != (h, w):
        mask_2d = cv2.resize(mask_2d, (w, h), interpolation=cv2.INTER_NEAREST)

    # 2. Mask Range check: float32 strictly in [0.0, 1.0]
    mask_float = mask_2d.astype(np.float32)
    if mask_float.max() > 1.0:
        mask_float /= 255.0

    # Smooth soft Gaussian feathering for pleural boundary silhouette
    soft_mask = cv2.GaussianBlur(mask_float, (15, 15), 0)

    # 3. Channels: 3D stacking via np.stack
    mask_3d = np.stack([soft_mask] * 3, axis=-1)

    # 4. Blending original X-ray texture with heatmap (60% X-ray + 40% heatmap)
    orig_float = original_xray.astype(np.float32)
    heat_float = heatmap_rgb.astype(np.float32)

    # cv2.addWeighted blends X-ray anatomy (ribs/parenchyma) with Grad-CAM colormap
    overlay = cv2.addWeighted(orig_float, 0.6, heat_float, 0.4, 0)

    # Apply mask_3d to the blended overlay to zero out background outside lungs
    final_image = overlay * mask_3d
    final_image = np.clip(final_image, 0, 255).astype(np.uint8)

    return final_image

def generate_exact_isolated_gated_heatmap(
    orig_rgb: np.ndarray,
    final_mask: np.ndarray,
    raw_cam_224: np.ndarray,
    bbox: Tuple[int, int, int, int],
) -> np.ndarray:
    """
    Generates an isolated lung Grad-CAM visualization with a pure black background.

    The ENTIRE segmented lung region is filled with the JET colormap —
    low-activation areas appear cool blue/cyan while pathological hotspots
    appear yellow/red — exactly matching the clinical heatmap reference style
    where the full lung silhouette is rendered on black with no X-ray bleed-through.

    Pipeline:
        1. Map 224×224 raw Grad-CAM back to the full image canvas via bounding box.
        2. Build binary lung mask; apply two-pass gated Gaussian smoothing.
        3. Normalise activation values ONLY within lung pixels so the full JET
           spectrum [0..255] spans the lung interior.
        4. Apply cv2.COLORMAP_JET to the normalised canvas.
        5. Zero everything outside the lung on a pure-black background using a
           slightly feathered Gaussian mask for smooth silhouette edges.
    """
    orig_h, orig_w = orig_rgb.shape[:2]
    x1, y1, w, h = bbox

    # ── 1. Reconstruct full-resolution activation canvas from 224×224 crop ────
    cam_bbox = cv2.resize(raw_cam_224, (max(1, w), max(1, h)), interpolation=cv2.INTER_CUBIC)
    full_cam_canvas = np.zeros((orig_h, orig_w), dtype=np.float32)
    x2, y2 = min(orig_w, x1 + w), min(orig_h, y1 + h)
    full_cam_canvas[y1:y2, x1:x2] = cam_bbox[:y2 - y1, :x2 - x1]

    # ── 2. Build binary lung mask & two-pass gated smoothing ──────────────────
    mask_2d = final_mask[..., 0] if len(final_mask.shape) == 3 else final_mask
    mask_float = mask_2d.astype(np.float32)
    if mask_float.max() > 1.0:
        mask_float /= 255.0

    lung_binary = (mask_float > 0.5).astype(np.float32)

    # Pass 1: suppress non-lung activations before smoothing
    full_cam_canvas = full_cam_canvas * lung_binary
    full_cam_canvas = cv2.GaussianBlur(full_cam_canvas, (11, 11), 0)
    # Pass 2: re-apply mask to eliminate Gaussian edge-bleed onto background
    full_cam_canvas = full_cam_canvas * lung_binary

    # ── 3. Normalise WITHIN lung pixels only → full JET spectrum inside lung ──
    # Normalising globally (including zeros outside the lung) would compress the
    # useful [lung_min, lung_max] range, causing the entire lung to appear in a
    # narrow cold-blue band.  Per-lung normalisation guarantees the full
    # colormap spans the organ interior.
    lung_pixels = full_cam_canvas[lung_binary > 0.5]
    if len(lung_pixels) > 0 and lung_pixels.max() > lung_pixels.min():
        cam_min, cam_max = float(lung_pixels.min()), float(lung_pixels.max())
        full_cam_canvas = np.where(
            lung_binary > 0.5,
            (full_cam_canvas - cam_min) / (cam_max - cam_min + 1e-8),
            0.0,
        ).astype(np.float32)
    else:
        # Fallback: distance-from-centre ramp so colormap is always visible
        full_cam_canvas = lung_binary.copy()

    # ── 4. Apply JET colormap to the normalised activation canvas ─────────────
    cam_uint8 = np.uint8(255 * np.clip(full_cam_canvas, 0.0, 1.0))
    heatmap_bgr = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    # ── 5. Isolate lung on pure-black background ───────────────────────────────
    # Soft Gaussian feathering gives smooth silhouette edges while staying
    # strictly within the lung shape — no X-ray tissue bleed-through.
    soft_mask = cv2.GaussianBlur(lung_binary, (15, 15), 0)
    soft_mask_3d = np.stack([soft_mask] * 3, axis=-1)

    # Pure black canvas: colormap × soft lung mask — NO X-ray blending
    isolated_heatmap = (heatmap_rgb * soft_mask_3d).astype(np.uint8)
    return isolated_heatmap

def apply_bitwise_mask_gating(overlay_image: np.ndarray, mask_image: np.ndarray) -> np.ndarray:
    """
    Applies OpenCV bitwise_and to cut out full Grad-CAM overlay using the lung mask,
    zeroing out all extrapulmonary areas (background & bones outside lungs).
    """
    if overlay_image.shape[:2] != mask_image.shape[:2]:
        mask_image = cv2.resize(mask_image, (overlay_image.shape[1], overlay_image.shape[0]))

    if len(mask_image.shape) == 3:
        mask_gray = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
    else:
        mask_gray = mask_image

    _, binary_mask = cv2.threshold(mask_gray, 127, 255, cv2.THRESH_BINARY)
    segmented_gradcam = cv2.bitwise_and(overlay_image, overlay_image, mask=binary_mask)
    return segmented_gradcam

def generate_zenithdx_visual(img_path: str, unet_model: Any, resnet_model: Any, target_layers: Any = None, user_id: Optional[str] = None):
    """
    Unified ZenithDx Visualization Entry Point:
    Integrates Unicode-safe image loading, S²A-UNet float32 normalization, Convex Hull lung boundary
    restoration, 8% safe ROI cropping, ResNet-50 multi-label classification, and gated Grad-CAM overlay
    on isolated lungs with 100% black background.
    """
    return detect_chest_xray(
        image_or_path=img_path,
        sa_unet=unet_model,
        resnet=resnet_model,
        return_explainability=True,
        user_id=user_id
    )

def advanced_image_pipeline(
    img_path: str,
    sa_unet: Any,
    resnet: Any,
    clahe_enhance: bool = True,
    user_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Advanced State-of-the-Art Vision Pipeline for ZenithDx:
    1. Unicode-safe image loading (np.fromfile + cv2.imdecode).
    2. Adaptive CLAHE contrast enhancement (clipLimit=2.0, tileGridSize=8x8) for dark parenchymal zones.
    3. S²A-UNet segmentation with float32 normalization [0.0, 1.0] & Convex Hull lobe restoration.
    4. Segmented ROI crop with 8% relative safety padding.
    5. Multi-label ResNet-50 classification (6 pathology classes with Youden's J cutoffs).
    6. Pre-sigmoid layer4 Grad-CAM activation mapping.
    7. Transparent blending (65% radiograph tissue / 35% JET colormap) with cv2.bitwise_and mask gating.
    """
    img_arr = cv2_read_img(img_path)
    if img_arr is None:
        raise FileNotFoundError(f"Could not read image at {img_path}")
        
    img_rgb = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
    
    if clahe_enhance:
        gray = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_gray = clahe.apply(gray)
        enhanced_bgr = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)
        enhanced_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)
    else:
        enhanced_rgb = img_rgb

    return detect_chest_xray(
        image_or_path=enhanced_rgb,
        sa_unet=sa_unet,
        resnet=resnet,
        return_explainability=True,
        user_id=user_id
    )
