import json
import subprocess

INPUT_FILE = "patient_diagnosis_reports.json"   # JSON array of dicts
OUTPUT_FILE = "doctor2_predictions.json"        # Save as JSON array

def query_ollama(prompt):
    result = subprocess.run(
        ["ollama", "run", "doctor2:latest", prompt],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding="utf-8",
    )
    return result.stdout.strip()

# Load the entire JSON array
with open(INPUT_FILE, "r", encoding="utf-8") as fin:
    samples = json.load(fin)

# Select the last 200 records (if there are fewer than 200, this will just use all)
last_samples = samples[-200:]

predictions = []

for idx, sample in enumerate(last_samples, 1):
    patient_query = sample.get("patient", "").strip()
    instruction = (
        "You are a board-certified physician and expert medical writer. "
        "When provided with a patient’s query, generate a structured and concise **Diagnosis Report** consisting of: \n\n"
        "- **Assessment**: Synthesis of the patient's presentation and relevant clinical findings  \n"
        "- **Differential Diagnosis**: Prioritized list of possible conditions, each with a clear and concise justification based only on the information provided  \n"
        "- **Final Diagnosis**: The most likely diagnosis stated explicitly  \n"
        "- **Explanation of Final Diagnosis**: A brief, clinically sound rationale for the diagnosis, linking it directly to the patient's symptoms\n\n"
        "Do not reference the doctor or their statements. Do not use phrases such as \"the doctor said,\" \"according to the physician,\" or similar. Avoid speculation, assumptions, or inferred reasoning beyond what is explicitly stated. Use clear, objective, and professional clinical language only."
    )

    prompt = f"""You are a board-certified physician and expert medical writer.

Below is an instruction that describes a task, followed by a patient query and the corresponding Diagnosis Report output.

### Instruction:
{instruction}

### Patient Query:
{patient_query}

### Diagnosis Report:"""

    print(f"Processing sample {idx}/200...")
    try:
        prediction = query_ollama(prompt)
    except Exception as e:
        print(f"Error in sample {idx}: {e}")
        prediction = "ERROR: Could not get model output."

    predictions.append({
        "input": patient_query,
        "output": prediction
    })

    # Save every 10 samples for safety
    if idx % 10 == 0 or idx == 200:
        with open(OUTPUT_FILE, "w", encoding="utf-8") as fout:
            json.dump(predictions, fout, ensure_ascii=False, indent=2)
        print(f"Batch saved at sample {idx}/200.")

print("✅ Done. Total reports generated:", len(predictions))
