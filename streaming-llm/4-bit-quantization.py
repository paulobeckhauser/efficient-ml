import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

print(f"PyTorch sees {torch.cuda.device_count()} GPU(s)")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Allocated VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    print(f"Cached VRAM:    {torch.cuda.memory_reserved()/1e9:.2f} GB")


model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# -----------------------------
# Tokenizer
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(model_name)

# -----------------------------
# 4-bit quantization configuration
# -----------------------------
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,  # computation in FP16
    bnb_4bit_quant_type="nf4"             # quantization type
)

# -----------------------------
# Load model with quantization
# -----------------------------
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quant_config,
    device_map="auto",  # automatically maps layers to GPU
)

model.eval()

# -----------------------------
# Inference
# -----------------------------
prompt = "Explain what machine learning is in one sentence"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=False
    )
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

print(output_text)
