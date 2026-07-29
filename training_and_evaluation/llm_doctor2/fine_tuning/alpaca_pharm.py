import json

INSTRUCTION = (
    "You are a board-certified physician and expert medical writer. When provided with a Diagnosis Report, generate a structured and concise **Therapeutic Report** consisting of: \n"
    "- **Therapeutic Assessment**: Synthesize the patient's therapeutic needs, considering the diagnosis, and any pertinent findings.\n"
    "- **Therapeutic Options**: Provide a prioritized, evidence-based list of potential therapeutic approaches (including pharmacological and non-pharmacological options), with a brief and clear justification for each, based only on the information provided.\n"
    "- **Recommended Treatment**: State the recommended treatment plan in detail, including medication name(s), dosage(s), frequency, duration, and any special instructions.\n"
    "- **Important Side Effects and Safety Instructions**: Summarize the most critical potential side effects, red flag symptoms, and safety precautions that should be communicated to the patient, as stated in the treatment plan.\n"
    "- **Rationale for Treatment**: Offer a concise, clinically sound rationale for the chosen treatment, explicitly linking it to the diagnosis, symptoms, and relevant clinical factors. Address any important contraindications, precautions, or follow-up requirements if applicable.\n\n"
    "Do **not** reference the doctor, physician, healthcare provider, or their statements. Do **not** use phrases such as \"the doctor said,\" \"according to the physician,\" or similar. Avoid speculation or inferred reasoning beyond what is explicitly stated in the input. Use clear, objective, and professional clinical language only."
)

with open("diagnosis_and_therapy_reports.json", "r", encoding="utf-8") as fin:
    data = json.load(fin)

alpaca_data = []
for record in data:
    alpaca_data.append({
        "instruction": INSTRUCTION,
        "input": record["Diagnosis Report"],
        "output": record["Therapeutic Report"]
    })

with open("alpaca_style_output_pharm.jsonl", "w", encoding="utf-8") as fout:
    for entry in alpaca_data:
        fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
