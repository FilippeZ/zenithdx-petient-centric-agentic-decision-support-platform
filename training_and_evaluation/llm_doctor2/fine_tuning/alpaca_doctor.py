import json

INSTRUCTION = (
    "You are a board-certified physician and expert medical writer. When provided with a patient’s query, generate a structured and concise **Diagnosis Report** consisting of: \n\n"
    "- **Assessment**: Synthesis of the patient's presentation and relevant clinical findings  \n"
    "- **Differential Diagnosis**: Prioritized list of possible conditions, each with a clear and concise justification based only on the information provided  \n"
    "- **Final Diagnosis**: The most likely diagnosis stated explicitly  \n"
    "- **Explanation of Final Diagnosis**: A brief, clinically sound rationale for the diagnosis, linking it directly to the patient's symptoms\n\n"
    "Do not reference the doctor or their statements. Do not use phrases such as \"the doctor said,\" \"according to the physician,\" or similar. "
    "Avoid speculation, assumptions, or inferred reasoning beyond what is explicitly stated. Use clear, objective, and professional clinical language only."
)

with open("patient_diagnosis_reports.json", "r", encoding="utf-8") as fin:
    data = json.load(fin)

alpaca_data = []
for record in data:
    alpaca_data.append({
        "instruction": INSTRUCTION,
        "input": record["patient"],  # or f"{record['patient']}\n\n{record['doctor']}" if you have a doctor response too
        "output": record["diagnosis_report"]
    })

with open("alpaca_style_output.jsonl", "w", encoding="utf-8") as fout:
    for entry in alpaca_data:
        fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
