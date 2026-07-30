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
    """
    if label_cols is None:
        label_cols = DEFAULT_CLINICAL_CLASSES

    print(f"[ResNet-50] Loading multi-label ResNet-50 model from: {weights_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = create_model(len(label_cols), pretrained=True).to(device)
    thresholds = DEFAULT_CUTOFFS.copy()

    if os.path.isfile(weights_path) and os.path.getsize(weights_path) > 100:
        try:
            try:
                ckpt = torch.load(weights_path, map_location=device, weights_only=False)
            except TypeError:
                ckpt = torch.load(weights_path, map_location=device)

            if isinstance(ckpt, dict):
                if "model_state_dict" in ckpt:
                    state_dict = ckpt["model_state_dict"]
                elif "state_dict" in ckpt:
                    state_dict = ckpt
                else:
                    state_dict = ckpt
                
                if "best_thresholds" in ckpt:
                    thresholds = np.array(ckpt["best_thresholds"], dtype="float32")
            else:
                state_dict = ckpt
                
            model.load_state_dict(state_dict, strict=False)
            print(f"[ResNet-50] Successfully loaded checkpoint from {weights_path}", file=sys.stderr)
        except Exception as e:
            print(f"[ResNet-50] Checkpoint load warning ({e}). Using pretrained ResNet-50 base.", file=sys.stderr)
    else:
        print(f"[ResNet-50] Weights file missing at {weights_path}. Using pretrained ResNet-50 base.", file=sys.stderr)

    model.eval()
    return model, thresholds, device

def extract_segmented_roi(img_rgb: np.ndarray, mask: np.ndarray):
    """
    Strict Linear Pipeline Chaining Step 2:
    1. Bounding box (xmin, ymin, w, h) extraction from S²A-UNet binary lung mask M.
    2. Crop raw image I_raw and mask M to bounding box.
    3. Element-wise mask gating on cropped section: I_segmented_crop = I_crop * M_crop (zeroing out non-lung background).
    4. Resizing I_segmented_crop and M_crop to ResNet-50 input resolution (224x224).
    5. Normalization with ImageNet stats to build input_tensor_224.
    Returns (input_tensor_224, M_crop_224, (x, y, w, h), I_segmented_crop, segmented_full).
    """
    h_img, w_img = img_rgb.shape[:2]
    mask_3d = np.repeat(mask[..., None], 3, axis=-1).astype(np.float32)
    segmented_full = (img_rgb * mask_3d).astype(np.uint8)

    binary_mask = (mask > 0.3).astype(np.uint8)
    coords = cv2.findNonZero(binary_mask)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
    else:
        x, y, w, h = 0, 0, w_img, h_img

    # Crop raw image and mask strictly to lung bounding box
    I_crop = img_rgb[y:y+h, x:x+w]
    M_crop = mask[y:y+h, x:x+w]

    # Element-wise mask gating on cropped section
    M_crop_3d = np.repeat(M_crop[..., None], 3, axis=-1).astype(np.float32)
    I_segmented_crop = (I_crop * M_crop_3d).astype(np.uint8)

    # Resize cropped segmented lung to 224x224
    I_segmented_224 = cv2.resize(I_segmented_crop, IMG_SIZE_CLS, interpolation=cv2.INTER_LINEAR)
    M_crop_224 = cv2.resize(M_crop, IMG_SIZE_CLS, interpolation=cv2.INTER_LINEAR)

    input_tensor_224 = RESNET_TRANSFORM(I_segmented_224).unsqueeze(0)

    return input_tensor_224, M_crop_224, (x, y, w, h), I_segmented_crop, segmented_full

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
