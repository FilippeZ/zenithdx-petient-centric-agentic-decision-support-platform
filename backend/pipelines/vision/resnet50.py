# backend/pipelines/vision/resnet50.py
from __future__ import annotations

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms

IMG_SIZE_CLS = (224, 224)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

RESNET_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(IMG_SIZE_CLS),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])

def create_model(num_classes: int, pretrained: bool = False, freeze_backbone: bool = False) -> nn.Module:
    """Creates a ResNet-50 architecture for multi-label classification."""
    weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = models.resnet50(weights=weights)
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, num_classes)
    )
    return model

def load_resnet(weights_path: str, label_cols: list[str]):
    """Loads pre-trained ResNet-50 weights and best classification thresholds."""
    print(f"[ResNet] Loading ResNet model from: {weights_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize pretrained model so model is never None
    try:
        model = create_model(len(label_cols), pretrained=True).to(device)
    except Exception:
        model = create_model(len(label_cols), pretrained=False).to(device)
        
    thresholds = np.array([0.70] * len(label_cols))

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
                    state_dict = ckpt["state_dict"]
                else:
                    state_dict = ckpt
                thresholds = np.array(ckpt.get("best_thresholds", [0.70] * len(label_cols)))
            else:
                state_dict = ckpt
                thresholds = np.array([0.35] * len(label_cols))
                
            model.load_state_dict(state_dict, strict=False)
            print(f"[ResNet] ✅ Successfully loaded checkpoint from {weights_path}", file=sys.stderr)
        except Exception as e:
            print(f"[ResNet] ⚠️ Checkpoint load warning ({e}). Using pretrained ResNet50 model.", file=sys.stderr)
    else:
        print(f"[ResNet] ⚠️ Weights file missing/empty at {weights_path}. Using pretrained ResNet50 model.", file=sys.stderr)

    model.eval()
    return model, thresholds, device

def resnet_predict(model_pt: nn.Module, img_rgb: np.ndarray, mask: np.ndarray, thresholds: np.ndarray, device: torch.device):
    """Runs prediction on masked lung image with ResNet-50, returning binary predictions, probabilities, and input tensor."""
    masked = (img_rgb * mask[..., None]).astype("uint8")
    tensor = RESNET_TRANSFORM(masked).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model_pt(tensor)
        probs = torch.sigmoid(logits).cpu().numpy()[0]
        preds = (probs > thresholds).astype(int)
    return preds, probs, tensor
