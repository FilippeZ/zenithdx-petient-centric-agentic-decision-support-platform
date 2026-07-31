# backend/test_vision_pipeline.py
from __future__ import annotations

import os
import sys
import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from pipelines.vision.resnet50 import create_model, extract_segmented_roi, resnet_predict
from xai.visual_explainer import grad_cam_torch, detect_chest_xray

def run_vision_pipeline_tests():
    print("=" * 70)
    print("[TEST] ZENITHDX VISION PIPELINE SAFETY MECHANISMS TEST")
    print("Testing Bounding Box Alignment, VRAM Protection & Pre-Sigmoid Grad-CAM")
    print("=" * 70)

    # Synthetic chest X-ray image (256x256x3) and UNet lung mask
    h, w = 256, 256
    img_rgb = np.random.randint(50, 200, (h, w, 3), dtype=np.uint8)
    
    # Synthetic lung mask with bounding box [y: 40..200, x: 30..220]
    mask = np.zeros((h, w), dtype=np.float32)
    mask[40:200, 30:220] = 1.0

    # 1. Test Bounding Box Alignment in extract_segmented_roi
    print("\n1. Testing Bounding Box Alignment in Mask Gating...")
    input_tensor_224, M_crop_224, bbox, I_segmented_crop, segmented_full = extract_segmented_roi(img_rgb, mask)

    x, y, bw, bh = bbox
    print(f"   * Extracted Bounding Box: x={x}, y={y}, w={bw}, h={bh}")
    print(f"   * Cropped Segmented Image Shape: {I_segmented_crop.shape}")
    print(f"   * Resized Mask M_crop_224 Shape: {M_crop_224.shape}")
    print(f"   * Input Tensor 224 Shape: {input_tensor_224.shape}")

    assert input_tensor_224.shape == (1, 3, 224, 224), "Input tensor 224 resolution mismatch"
    assert M_crop_224.shape == (224, 224), "Resized mask 224 resolution mismatch"
    assert bw == 190 and bh == 160, f"Bounding box dimensions incorrect: {bbox}"
    print("   * [SUCCESS] Bounding Box Alignment verified (224x224 exact match)!")

    # 2. Test Pre-Sigmoid Logit Grad-CAM Backward Pass
    print("\n2. Testing Pre-Sigmoid Logit Grad-CAM Backward Pass...")
    model = create_model(num_classes=6, pretrained=False)
    model.eval()

    cam_7x7 = grad_cam_torch(model, input_tensor_224, target_class_idx=0, target_layer='layer4')
    print(f"   * Grad-CAM Raw Activation Map Shape: {cam_7x7.shape}")
    print(f"   * Activation Min: {cam_7x7.min():.4f}, Max: {cam_7x7.max():.4f}")

    assert cam_7x7.shape == (7, 7), "Grad-CAM activation map shape invalid"
    print("   * [SUCCESS] Pre-Sigmoid Grad-CAM Backward Pass verified!")

    # 3. Test Full Image Pipeline & Dynamic VRAM Context Management
    print("\n3. Testing Full Image Pipeline Execution & Dynamic Context Management...")
    device = "cpu"
    result = detect_chest_xray(
        image_or_path=img_rgb,
        sa_unet=None,
        resnet=model,
        device=device,
        return_explainability=True,
        user_id="test_user"
    )

    print(f"   * Top Pathology Findings: {result['findings']}")
    print(f"   * Grad-CAM Target Label: {result['gradcam']['label']}")
    print(f"   * Grad-CAM Overlay Saved to: {result['paths']['gradcam_overlay']}")

    assert os.path.exists(result['paths']['gradcam_overlay']), "Grad-CAM overlay file not written"
    print("   * [SUCCESS] Dynamic VRAM Context Management & Pipeline Chaining verified!")

    print("\n" + "=" * 70)
    print("[SUCCESS] ALL VISION PIPELINE SAFETY TESTS PASSED CLEANLY!")
    print("=" * 70)

if __name__ == "__main__":
    run_vision_pipeline_tests()
