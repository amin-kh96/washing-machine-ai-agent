import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

base_model_path = "D:/huggingface_cache/Smarter_Student_Stable"
adapter_path = "./smollm2-washing-machine-final-agent"

print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    base_model_path, 
    device_map="auto", 
    torch_dtype=torch.float16, 
    trust_remote_code=True
)

print("Loading tokenizer & adapter...")
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

model = PeftModel.from_pretrained(model, adapter_path)
model.eval()

# ==========================================
# STEP 1: RETRIEVAL (Reading the JSON)
# ==========================================
print("\nLoading Knowledge Base...")
with open("washing_manual.json", "r") as file:
    washing_manual = json.load(file)

# The user's question
user_prompt = "I need to wash delicate silk shirts. Which washing machine setting should I use and what is the temperature?"
user_prompt_lower = user_prompt.lower()

# Search the JSON file for any matching keywords
retrieved_fact = "General washing rules apply. Please check the garment tag." 

for program in washing_manual["programs"]:
    for keyword in program["keywords"]:
        if keyword in user_prompt_lower:
            # If a keyword matches, format the rule beautifully for the AI
            retrieved_fact = f"Use the {program['name']} cycle at {program['temperature']}. Max spin is {program['max_spin']}. {program['instruction']}"
            break # Stop searching this program's keywords
    if retrieved_fact != "General washing rules apply. Please check the garment tag.":
        break # Stop searching other programs once we find a match

print(f"--- RAG RETRIEVAL SUCCESS ---")
print(f"Detected Fact: {retrieved_fact}")

# STEP 2: AUGMENTATION (Injecting the cheat sheet cleanly)
# ==========================================
# We use standard chat formatting so the AI understands its "role"
messages = [
    {
        "role": "system", 
        "content": f"You are a polite washing machine assistant. Answer the user's question using ONLY the following facts. Do not invent any extra information.\nFacts: {retrieved_fact}"
    },
    {
        "role": "user", 
        "content": user_prompt
    }
]

# This automatically applies the perfect <|im_start|> formatting for SmolLM2
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to("cuda")

# ==========================================
# STEP 3: GENERATION (The AI talks)
# ==========================================
print("\n--- GENERATING PHASE 2 ADVICE ---")
with torch.no_grad():
    outputs = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=100,
        do_sample=True,          # Allow a tiny bit of flexibility
        temperature=0.2,         # Keep it very focused and strict
        repetition_penalty=1.02, # Lowered drastically so it is allowed to repeat the facts!
        pad_token_id=tokenizer.eos_token_id
    )

print("\n" + "="*40)
print("PHASE 2 RESULT (With RAG):")
print("="*40)

# Decode only the newly generated text
new_tokens = outputs[0][inputs.input_ids.shape[1]:]
final_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

print(final_text)