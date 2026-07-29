from doctor2_captum_helper import run_full_llm_attribution, doctor2_model, doctor2_tokenizer

art = run_full_llm_attribution(
    prompt="I have cough and fever",
    target_text="Diagnosis: bronchitis",
    model=doctor2_model,
    tokenizer=doctor2_tokenizer,
    out_dir="debug_xai",
    save_json=False
)
print(art)
