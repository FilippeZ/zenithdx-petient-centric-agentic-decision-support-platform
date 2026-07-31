# backend/test_pdf_export.py
import sys
import os
from pathlib import Path

# Set PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipelines.pdf_generator import generate_pdf_report_bytes

def main():
    print("======================================================================")
    print("[TEST] ZENITHDX PDF REPORT GENERATION TEST")
    print("======================================================================")

    dummy_row = {
        "report_id": "eb3b821e-c6b0-4e2a-8921-123456789abc",
        "patient_id": "10000032",
        "patient_name": "Philip Vance",
        "symptoms": "Severe abdominal pain, distention, low blood pressure",
        "submission_date": "2026-07-31 15:30:00",
        "status": "Approved",
        "diagnosis": """# Clinical Assessment & Conclusion
The patient presents with abdominal pain, distention, and hypotension (43/63 mmHg).
Physical examination reveals tenderness without peritonitis.

### Diagnostic Evidence & Multi-Modal Findings
- **Chest Radiograph**: Right lower lobe consolidation consistent with subacute pneumonia.
- **Grad-CAM Attribution**: High visual attention over right basal lung zone.
- **PyTorch Captum Text Explanation**: High feature attribution for tokens 'pain' and 'hypotension'.

### Diagnostic Conclusion & Recommended Plan
1. Administer IV fluids for hemodynamic stabilization.
2. Initiate empiric broad-spectrum antibiotic coverage for community-acquired pneumonia.
3. Schedule urgent abdominal ultrasound to rule out ascites / SBP.
""",
        "doctor_message": "Patient requires immediate fluid resuscitation and inpatient monitoring in the step-down ICU.",
        "classification_results": [
            ["Pneumonia", 0.942],
            ["Atelectasis", 0.615],
            ["Lung Opacity", 0.783],
            ["Pleural Effusion", 0.312]
        ],
        "original_xray": "/outputs/original_xray_test.jpg",
        "gradcam_overlay": "/outputs/gradcam_overlay_test.jpg",
        "xai_structured": {
            "gradcam_segmented": "/outputs/segmented_gradcam_test.jpg",
            "captum_image": "/outputs/captum_plot_test.jpg"
        }
    }

    try:
        pdf_bytes = generate_pdf_report_bytes(dummy_row)
        output_file = "test_output_report.pdf"
        with open(output_file, "wb") as f:
            f.write(pdf_bytes)

        file_size = len(pdf_bytes)
        print(f"[SUCCESS] PDF report generated cleanly: {output_file} ({file_size} bytes)")
        assert file_size > 1000, "PDF byte size is too small!"
        print("======================================================================")
        print("[ALL TESTS PASSED CLEANLY]")
        print("======================================================================")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[FAIL] PDF generation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
