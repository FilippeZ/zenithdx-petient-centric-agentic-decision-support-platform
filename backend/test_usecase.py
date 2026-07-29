# backend/test_usecase.py
import os
import sys
import cv2
import numpy as np
from pathlib import Path

# Create a sample synthetic chest X-ray image (256x256 RGB)
img = np.zeros((256, 256, 3), dtype=np.uint8)
# Draw lung shape approximations for test
cv2.ellipse(img, (90, 128), (40, 70), 0, 0, 360, (180, 180, 180), -1)
cv2.ellipse(img, (166, 128), (40, 70), 0, 0, 360, (180, 180, 180), -1)
cv2.putText(img, "TEST X-RAY", (60, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

test_img_path = str(Path(__file__).parent / "test_chest_xray.png")
res, buf = cv2.imencode(".png", img)
if res:
    buf.tofile(test_img_path)
print("Created test chest X-ray image at:", test_img_path, flush=True)

from ai_agent_runner import run_agent

print("\n--- Running Multi-Modal End-to-End Test Usecase ---", flush=True)
print("-------------------------------------------------", flush=True)
print("Input Symptoms: 'Patient presents with severe persistent cough, high fever, dyspnea, and right-sided pleuritic chest pain.'", flush=True)
print(f"Input Image: {test_img_path}", flush=True)
print("Input Patient ID: 10000032", flush=True)
print("-------------------------------------------------\n", flush=True)

try:
    result = run_agent(
        user_query="Patient presents with severe persistent cough, high fever, dyspnea, and right-sided pleuritic chest pain.",
        image_path=test_img_path,
        patient_id="10000032"
    )

    print("\n[SUCCESS] Multi-Modal Pipeline Executed Successfully!")
    print("===============================================")
    print(f"Elapsed Time: {result.get('elapsed_sec')} seconds")
    print(f"\n--- Diagnosis Output Preview ---")
    print(str(result.get("diagnosis"))[:600])
    print(f"\n--- Vision Findings / Classification ---")
    print(result.get("classification_results"))
    print(f"\n--- Image Paths ---")
    print("Original:", result.get("original_xray"))
    print("GradCAM Overlay:", result.get("gradcam_overlay"))
    print(f"\n--- XAI Report Preview ---")
    print(str(result.get("xai_report"))[:400])
    print("===============================================")

except Exception as e:
    import traceback
    print(f"\n[ERROR] Error during execution: {e}")
    traceback.print_exc()
