# backend/test_sa_unet_quick.py
import sys
import os
import cv2
import numpy as np

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from pipelines.vision.s2a_unet import fallback_anatomical_lung_segmentation

def test_quick():
    print("=" * 60)
    print("[TEST] ANATOMICAL DUAL-LOBE LUNG SEGMENTATION TEST")
    print("=" * 60)

    h, w = 512, 512
    img_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Simulate chest radiograph: dark lung cavities, brighter ribs/tissue
    img_rgb[:, :] = [180, 180, 180] # body tissue
    img_rgb[int(0.18*h):int(0.78*h), int(0.12*w):int(0.44*w)] = [30, 30, 30] # Left lung field
    img_rgb[int(0.18*h):int(0.78*h), int(0.56*w):int(0.88*w)] = [30, 30, 30] # Right lung field

    mask = fallback_anatomical_lung_segmentation(img_rgb)
    
    coverage = np.mean(mask)
    print(f"   * Mask Shape: {mask.shape}")
    print(f"   * Pulmonary Coverage: {coverage*100:.2f}%")
    print(f"   * Mask Min: {mask.min()}, Max: {mask.max()}")

    assert coverage > 0.05 and coverage < 0.60, f"Coverage invalid: {coverage}"
    print("======================================================================")
    print("[SUCCESS] ANATOMICAL LUNG SEGMENTATION TEST PASSED CLEANLY!")
    print("======================================================================")

if __name__ == "__main__":
    test_quick()
