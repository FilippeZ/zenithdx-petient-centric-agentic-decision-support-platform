import os
import sys
import time
import json
import datetime as dt

from ai_agent_runner import run_agent

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

cases = [
    {
        "id": 1,
        "title": "Multimodal Acute Bacterial Pneumonia Consultation",
        "symptoms": "I have acute chest pain, persistent productive cough, shortness of breath, and high fever (38.8C).",
        "image_path": "pneumonia.jpeg",
        "patient_id": "79407290-14fd-4316-a9b2-8b7f8df2196a"
    },
    {
        "id": 2,
        "title": "Pure Text Consultation (No Imaging) — Tension / Migraine Headache",
        "symptoms": "I have a severe headache, sensitivity to light, and dizziness for the past 24 hours.",
        "image_path": None,
        "patient_id": None
    },
    {
        "id": 3,
        "title": "Complex Multi-Label Pulmonary Pathology & Edema",
        "symptoms": "Shortness of breath when lying flat, swelling in legs, and severe nocturnal dyspnea.",
        "image_path": "pneumonia.jpeg",
        "patient_id": "e53fa09e-f9ae-4ffe-a201-25c18bc94a71"
    },
    {
        "id": 4,
        "title": "Acute Febrile Influenza-Like Illness (ILI)",
        "symptoms": "Sudden onset of high fever (39.2C), severe muscle aches, body chills, sore throat, and dry cough.",
        "image_path": None,
        "patient_id": None
    },
    {
        "id": 5,
        "title": "Longitudinal Asthmatic Exacerbation (Graph EHR Integration)",
        "symptoms": "Wheezing, chest tightness, and dyspnea triggered by cold air exposure.",
        "image_path": None,
        "patient_id": "375476b2-b664-4b8b-ade5-fdbc8bf8ceea"
    },
    {
        "id": 6,
        "title": "Differential Diagnostic Challenge — Pleural Effusion vs Atelectasis",
        "symptoms": "Sharp right-sided chest pain on deep inspiration, mild fever, and shortness of breath.",
        "image_path": "pneumonia.jpeg",
        "patient_id": None
    }
]

results = []

for c in cases:
    print(f"\n==========================================")
    print(f"Executing Case {c['id']}: {c['title']}")
    print(f"Symptoms: {c['symptoms']}")
    print(f"Image: {c['image_path']}, Patient ID: {c['patient_id']}")
    print(f"==========================================")
    
    t0 = time.time()
    try:
        out = run_agent(
            user_query=c["symptoms"],
            image_path=c["image_path"],
            patient_id=c["patient_id"]
        )
        elapsed = round(time.time() - t0, 2)
        print(f"[OK] Case {c['id']} Completed in {elapsed}s")
        results.append({
            "case": c,
            "elapsed": elapsed,
            "out": out
        })
    except Exception as e:
        elapsed = round(time.time() - t0, 2)
        print(f"[FAIL] Case {c['id']} Failed in {elapsed}s: {e}")
        results.append({
            "case": c,
            "elapsed": elapsed,
            "error": str(e)
        })


with open("usecases_live_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("\nAll 6 synthetic use cases completed and saved to usecases_live_results.json")
