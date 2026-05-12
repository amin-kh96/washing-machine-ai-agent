import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer
import os


# 1. Configuration
model_id = "D:/huggingface_cache/Smarter_Student_Stable" 
dataset_file = "train_data_final.jsonl"
output_dir = "./smollm2-washing-machine-final-agent"

# 2. Load Dataset
print("Loading dataset...")
dataset = load_dataset("json", data_files=dataset_file, split="train")


# 3. Load YOUR Local Model and the Official Tokenizer
print(f"Loading distilled model from: {model_id}...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float32, # <--- CHANGE 1: Use float32 instead of float16
    trust_remote_code=True
)

print("Loading official tokenizer configuration...")
tokenizer = AutoTokenizer.from_pretrained(
    "HuggingFaceTB/SmolLM2-135M-Instruct", 
    trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" # <--- CHANGE 2: Extremely important for loss calculation

# 4. LoRA Config (The "Brain" for Fine-Tuning)
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

args = TrainingArguments(
    output_dir="./smollm2-washing-machine-final-agent",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    # --- SAFETY BRAKES START HERE ---
    learning_rate=1e-5,          # Lowered even more (0.00001)
    max_grad_norm=0.3,           # THE SAFETY BRAKE: Clips big math jumps
    num_train_epochs=3,
    lr_scheduler_type="constant", # Keeps the learning steady, no sudden spikes
    fp16=False,                  # Turn off FP16 to use more stable FP32 math
    # --------------------------------
    logging_steps=1,
    save_strategy="epoch",
    push_to_hub=False,
    report_to="none"
)

# 6. Initialize Trainer
# Uses 'tokenizer' keyword (correct for version 0.9.4)
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=args,
    peft_config=peft_config,
    tokenizer=tokenizer,
    dataset_text_field="text",
    max_seq_length=512,
)

# 7. Start Training
print("--- STARTING TRAINING ---")
trainer.train()

# 8. Save the Final Adapter
trainer.model.save_pretrained(output_dir)
print(f"COMPLETE! Your agent is ready at: {output_dir}")