from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "microsoft/Phi-3-mini-4k-instruct"

# Use CPU explicitly
device = torch.device("cpu")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load model in float32 for CPU
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,
    trust_remote_code=True,
    load_in_8bit=True 
).to(device)

# Optional: avoid flash attention warnings
try:
    model.config.attn_implementation = "eager"
except Exception:
    pass

# Create chat input
messages = [{"role": "user", "content": "Who are you?"}]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
).to(device)

# Generate
model.eval()
with torch.inference_mode():
    outputs = model.generate(
        **inputs,
        max_new_tokens=64,
        do_sample=False  # deterministic for testing
    )

# Decode and print
print("\n🧠 Model Output:\n")
print(tokenizer.decode(outputs[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True))
