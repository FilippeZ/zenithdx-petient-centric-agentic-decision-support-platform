from unsloth import FastLanguageModel, is_bfloat16_supported 
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset
import torch, os, time, json

# Path to llama-quantize for GGUF export
os.environ["LLAMA_QUANTIZE_PATH"] = "/storage/data2/up1084660/dataset/ed/llama.cpp/build/bin/llama-quantize"

# ---------------- Prompt Template (MEDICAL SFT STYLE) ---------------------------
Alpaca_Prompt = """You are a board-certified physician and expert medical writer.

Below is an instruction that describes a task, followed by the Diagnosis Report and the corresponding Therapeutic Report output. 

### Instruction:
{instruction}

### Diagnosis Report: 
{input}

### Therapeutic Report:
{output}
"""

# ---------------- Load Local Alpaca JSONL Dataset ------------------
with open("alpaca_style_output_pharm.jsonl", "r", encoding="utf-8") as fin: 
    data = [json.loads(line) for line in fin]

# Use only the first 2901 examples for training
train_data = data[:2901]

dataset = Dataset.from_list(train_data)

# ---------------- Formatting Function ------------------------------
def formatting_prompts_func(examples):
    texts = [
        Alpaca_Prompt.format(
            instruction=inst.strip(),
            input=inp.strip(),
            output=out.strip(),
        ) + tokenizer.eos_token
        for inst, inp, out in zip(examples["instruction"], examples["input"], examples["output"])
    ]
    return {"text": texts}

# Model and tokenizer need to be loaded before running formatting_prompts_func
max_seq_length = 2048
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = "unsloth/Llama-3.2-3B-Instruct",
    max_seq_length = max_seq_length,
    dtype          = None,
    load_in_4bit   = True,
)

dataset = (
    dataset
    .map(formatting_prompts_func, batched=True, num_proc=2)
    .remove_columns([col for col in dataset.column_names if col != "text"])
    .with_format("torch")
)

# ---------------- LoRA Configuration ------------------------------------
model = FastLanguageModel.get_peft_model(
    model,
    r                          = 32,
    target_modules             = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_alpha                 = 32,
    lora_dropout               = 0.0,
    bias                       = "none",
    use_gradient_checkpointing = "unsloth",
    random_state               = 3407,
)

# ---------------- Collator ----------------------------------------------
data_collator = DataCollatorForLanguageModeling(
    tokenizer = tokenizer,
    mlm       = False,
)

# ---------------- Trainer -----------------------------------------------
trainer = SFTTrainer(
    model              = model,
    tokenizer          = tokenizer,
    train_dataset      = dataset,
    data_collator      = data_collator,
    dataset_text_field = "text",
    max_seq_length     = max_seq_length,
    dataset_num_proc   = 2,
    packing            = False,
    args = TrainingArguments(
        per_device_train_batch_size  = 4,
        gradient_accumulation_steps  = 4,
        warmup_steps                 = 5,
        max_steps                    = 362,
        learning_rate                = 2e-4,
        fp16                         = not is_bfloat16_supported(),
        bf16                         = is_bfloat16_supported(),
        logging_steps                = 1,
        optim                        = "adamw_8bit",
        weight_decay                 = 0.01,
        lr_scheduler_type            = "linear",
        seed                         = 3407,
        output_dir                   = "outputs",
        report_to                    = "none",
    ),
)

# ---------------- Train -------------------------------------------------
start_gpu = torch.cuda.max_memory_reserved() / 1024**3
start_time = time.time()
trainer.train()
runtime = time.time() - start_time

print(f"\n✅ Training completed in {runtime:.2f} seconds")
print(f"🚀 Max GPU memory reserved: {start_gpu:.2f} GB")

# ---------------- GGUF Export -------------------------------------------
model.save_pretrained_gguf("model_pharm", tokenizer, quantization_method="f16")
