# backend/pipelines/vision/resnet50.py
from __future__ import annotations

import os
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms

IMG_SIZE_CLS = (224, 224)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

DEFAULT_CLINICAL_CLASSES = [
    "Atelectasis",
    "Consolidation",
    "Edema",
    "Lung Lesion",
    "Lung Opacity",
    "Pneumonia",
]

DEFAULT_CUTOFFS = np.array([0.35, 0.40, 0.35, 0.30, 0.45, 0.35], dtype="float32")

RESNET_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(IMG_SIZE_CLS),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])

def create_model(num_classes: int = 6, pretrained: bool = True, freeze_backbone: bool = False) -> nn.Module:
    """
    Creates a ResNet-50 architecture for multi-label clinical classification:
    - Backbone: ResNet-50 with IMAGENET1K_V2 pretrained weights
    - Residual Bottleneck Blocks (conv1_x to conv5_x with identity/projection connections)
    - Classifier: Global Average Pooling (GAP) -> Dropout (p=0.5) -> Linear(2048, 6)
    """
    try:
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.resnet50(weights=weights)
    except Exception:
        model = models.resnet50(weights=None)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features, num_classes)
    )
    return model

def load_resnet(weights_path: str, label_cols: list[str] = None):
    """
    Loads ResNet-50 model and optimal class-specific decision thresholds.

    Unicode-safe: reads the .pth file via Python's built-in open() in binary
    mode and passes a BytesIO object to torch.load, bypassing the Windows
    C-runtime path encoding limitation that affects torch.load with non-ASCII
    file paths.

    Smart backbone transfer: if the checkpoint was trained on a different number
    of classes (e.g. 14-class CheXpert), the fc layer is skipped and only the
    shared backbone (conv1 → layer4) is transferred. This brings in all the
    valuable chest X-ray feature extractor weights while keeping the backend's
    6-class classification head.
    """
    import io
    import pathlib

    if label_cols is None:
        label_cols = DEFAULT_CLINICAL_CLASSES

    print(f"[ResNet-50] Loading multi-label ResNet-50 model from: {weights_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = create_model(len(label_cols), pretrained=True).to(device)
    thresholds = DEFAULT_CUTOFFS.copy()

    p = pathlib.Path(weights_path)

    # Use pathlib.stat() — Unicode-safe on Windows (os.path.getsize can fail)
    try:
        file_size = p.stat().st_size if p.exists() else 0
    except Exception:
        file_size = 0

    if file_size > 100:
        try:
            # BytesIO bypass: avoids torch.load's internal C open() on Unicode paths
            with open(str(p), "rb") as fh:
                buffer = io.BytesIO(fh.read())

            try:
                ckpt = torch.load(buffer, map_location=device, weights_only=False)
            except TypeError:
                buffer.seek(0)
                ckpt = torch.load(buffer, map_location=device)

            if isinstance(ckpt, dict):
                if "model_state_dict" in ckpt:
                    state_dict = ckpt["model_state_dict"]
                elif "state_dict" in ckpt:
                    state_dict = ckpt["state_dict"]
                else:
                    state_dict = ckpt

                if "best_thresholds" in ckpt:
                    thresholds = np.array(ckpt["best_thresholds"], dtype="float32")
                    print(f"[ResNet-50] Loaded class-specific thresholds: {thresholds}", file=sys.stderr)
            else:
                state_dict = ckpt

            # ── Auto-detect checkpoint class count from fc weight shape ───────
            fc_weight_key = next((k for k in state_dict if k.endswith(".weight") and "fc" in k), None)
            if fc_weight_key is not None:
                ckpt_classes = state_dict[fc_weight_key].shape[0]
            else:
                ckpt_classes = len(label_cols)

            if ckpt_classes != len(label_cols):
                # ── Backbone-only transfer: skip fc layer ──────────────────────
                # The checkpoint was trained on a different number of classes
                # (e.g. 14-class CheXpert vs 6-class backend).  Transfer all
                # shared backbone weights and ignore the classification head.
                model_state = model.state_dict()
                backbone_state = {
                    k: v for k, v in state_dict.items()
                    if k in model_state and "fc" not in k and v.shape == model_state[k].shape
                }
                model.load_state_dict(backbone_state, strict=False)
                print(
                    f"[ResNet-50] ✅ Backbone transfer: {ckpt_classes}-class checkpoint → "
                    f"{len(label_cols)}-class model. Loaded {len(backbone_state)} layers "
                    f"(fc skipped).",
                    file=sys.stderr,
                )
            else:
                model.load_state_dict(state_dict, strict=False)
                print(f"[ResNet-50] ✅ Successfully loaded full checkpoint ({ckpt_classes} classes).", file=sys.stderr)

        except Exception as e:
            print(f"[ResNet-50] Checkpoint load warning ({e}). Using pretrained ResNet-50 base.", file=sys.stderr)
    else:
        if file_size == 0:
            print(
                f"[ResNet-50] ⚠️  {weights_path} is empty (0 bytes). "
                "Copy your trained best_model.pth there and restart.",
                file=sys.stderr,
            )
        else:
            print(f"[ResNet-50] Weights file missing at {weights_path}. Using pretrained ResNet-50 base.", file=sys.stderr)

    model.eval()
    return model, thresholds, device

def get_safe_bounding_box_crop(isolated_lungs_np: np.ndarray, fixed_mask_np: np.ndarray, padding_percent=0.08):
    """
    Finds lung field boundaries from fixed_mask_np and crops isolated_lungs_np
    with a safe padding margin (8%), preventing truncation of lung apices.
    Returns (cropped_roi, (x1, y1, w, h), M_crop).
    """
    coords = cv2.findNonZero((fixed_mask_np > 0.3).astype(np.uint8))
    img_h, img_w = isolated_lungs_np.shape[:2]
    
    if coords is None:
        return isolated_lungs_np, (0, 0, img_w, img_h), fixed_mask_np
        
    x, y, w, h = cv2.boundingRect(coords)
    
    pad_x = int(w * padding_percent)
    pad_y = int(h * padding_percent)
    
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(img_w, x + w + pad_x)
    y2 = min(img_h, y + h + pad_y)
    
    cropped_roi = isolated_lungs_np[y1:y2, x1:x2]
    M_crop = fixed_mask_np[y1:y2, x1:x2]
    bbox = (x1, y1, x2 - x1, y2 - y1)
    
    return cropped_roi, bbox, M_crop

def extract_segmented_roi(img_rgb: np.ndarray, mask: np.ndarray):
    """
    Step 1 Implementation:
    1. Converts U-Net mask to float32 range [0.0, 1.0] and 3D stack (H, W, 3).
    2. Element-wise mask gating: masked_input_image = img_rgb * mask_3d (zeroing non-lung background/bones/labels).
    3. Safe bounding box crop (x, y, w, h) with 8% padding using get_safe_bounding_box_crop.
    4. Resizes cropped lung ROI to ResNet-50 input size (224x224).
    5. Normalizes via ImageNet stats -> input_tensor_224 for ResNet-50 inference.
    Returns (input_tensor_224, M_crop_224, (x, y, w, h), I_segmented_crop, segmented_full).
    """
    h_img, w_img = img_rgb.shape[:2]

    # Ensure mask is 2D float32 strictly in range [0.0, 1.0]
    mask_2d = mask[..., 0] if len(mask.shape) == 3 else mask
    mask_float = mask_2d.astype(np.float32)
    if mask_float.max() > 1.0:
        mask_float /= 255.0

    # 3D Mask Channel Stacking via np.stack
    mask_3d = np.stack([mask_float] * 3, axis=-1)

    # Element-wise multiplication: ResNet-50 sees ONLY lungs on pure black background
    segmented_full = (img_rgb.astype(np.float32) * mask_3d).astype(np.uint8)

    # Crop to segmented lung ROI with 8% safety padding
    I_segmented_crop, bbox, M_crop = get_safe_bounding_box_crop(segmented_full, mask_float, padding_percent=0.08)

    # Resize cropped segmented lung to 224x224
    I_segmented_224 = cv2.resize(I_segmented_crop, IMG_SIZE_CLS, interpolation=cv2.INTER_LINEAR)
    M_crop_224 = cv2.resize(M_crop, IMG_SIZE_CLS, interpolation=cv2.INTER_LINEAR)

    input_tensor_224 = RESNET_TRANSFORM(I_segmented_224).unsqueeze(0)

    return input_tensor_224, M_crop_224, bbox, I_segmented_crop, segmented_full

def resnet_predict(model_pt: nn.Module, img_rgb: np.ndarray, mask: np.ndarray, thresholds: np.ndarray, device: torch.device):
    """
    Stage 2 ResNet-50 Multi-label Prediction on Clean Segmented Lung ROI:
    - Feeds 224x224 normalized segmented lung tensor directly into ResNet-50
    - Evaluates 6 logits -> Sigmoid probabilities -> Binarization via Youden's J cutoffs
    Returns (preds, probs, input_tensor_224, M_crop_224, bbox, segmented_full).
    """
    input_tensor_224, M_crop_224, bbox, I_segmented_crop, segmented_full = extract_segmented_roi(img_rgb, mask)
    input_tensor_224 = input_tensor_224.to(device)

    model_pt.eval()
    with torch.no_grad():
        logits = model_pt(input_tensor_224)
        probs = torch.sigmoid(logits).cpu().numpy()[0]
        preds = (probs > thresholds).astype(int)

    return preds, probs, input_tensor_224, M_crop_224, bbox, segmented_full
