import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

print(f"PyTorch sees {torch.cuda.device_count()} GPU(s)")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Allocated VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    print(f"Cached VRAM:    {torch.cuda.memory_reserved()/1e9:.2f} GB")

# -----------------------------
# Model settings
# -----------------------------
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
USE_4BIT = False  # set True to use 4-bit quantization

# -----------------------------
# Check GPU
# -----------------------------
print(f"PyTorch CUDA available? {torch.cuda.is_available()}")
print(f"Device count: {torch.cuda.device_count()}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    print(f"Using device: {torch.cuda.get_device_name(0)}")

# -----------------------------
# Load tokenizer
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# -----------------------------
# Load model
# -----------------------------
try:
    if USE_4BIT:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=quant_config,
            device_map="sequential",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            dtype=torch.float16,
            device_map="sequential",
        )
except RuntimeError as e:
    print("Error loading model. Clearing GPU cache and retrying...")
    torch.cuda.empty_cache()
    raise e

model.eval()
print("Model loaded successfully!\n")

# -----------------------------
# VRAM info helper
# -----------------------------
def print_vram():
    print(f"Allocated VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    print(f"Cached VRAM:    {torch.cuda.memory_reserved()/1e9:.2f} GB")

# -----------------------------
# Streaming inference
# -----------------------------
def generate_streaming(prompt, max_tokens=50):
    input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)
    past_key_values = None
    generated_ids = []

    with torch.no_grad():
        for _ in range(max_tokens):
            outputs = model(input_ids=input_ids, past_key_values=past_key_values, use_cache=True)
            past_key_values = outputs.past_key_values
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated_ids.append(next_token.item())
            input_ids = next_token

            text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            print(text, end="\r", flush=True)

    print("\n=== Generation complete ===")
    return tokenizer.decode(generated_ids, skip_special_tokens=True)

# -----------------------------
# Run example
# -----------------------------
prompt = "Explain machine learning in one sentence."
print("Prompt:", prompt)
print("=== Model Output ===")
output_text = generate_streaming(prompt, max_tokens=50)
print(output_text)
print_vram()
