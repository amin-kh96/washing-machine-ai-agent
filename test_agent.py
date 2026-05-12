import torch
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

# Let's ask a clear, specific question
prompt = "I need to wash delicate silk shirts. Which washing machine setting should I use and what is the temperature?" 

# WE PUT THE <|thought|> TAG BACK IN! This is the model's "Map".
text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<|thought|>\n"
inputs = tokenizer(text, return_tensors="pt").to("cuda")

print("--- GENERATING EXPERT ADVICE ---")
with torch.no_grad():
    outputs = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=150,
        do_sample=False,         # Keep it deterministic
        repetition_penalty=1.15, # Softened penalty so it doesn't panic
        no_repeat_ngram_size=5,  # Slightly more forgiving
        pad_token_id=tokenizer.eos_token_id
    )

print("\n" + "="*40)
print("AGENT'S RESPONSE:")
print("="*40)

new_tokens = outputs[0][inputs.input_ids.shape[1]:]
final_text = tokenizer.decode(new_tokens, skip_special_tokens=False).strip()

print(final_text)