import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
USE_4BIT = True

print(f"PyTorch CUDA available? {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Device count: {torch.cuda.device_count()}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# model = AutoModelForCausalLM.from_pretrained(
#     MODEL_NAME,
#     dtype=torch.float16,
#     device_map="auto",
# )


#### QUANTIZATION IMPLEMENTATION

if USE_4BIT:
    print("Loading model with 4-bit quantization...")
    # Configure 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
    except Exception as e:
        print(f"\n⚠️  4-bit quantization failed: {e}")
        print("\nTrying without bitsandbytes (fallback to FP16)...")
        print("Note: This may cause OOM on Orin Nano with 7B models")
        USE_4BIT = False
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
else:
    print("Loading model in FP16 (no quantization)...")
    estimated_gb = {
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0": 2,
        "google/gemma-2-2b-it": 4,
        "meta-llama/Llama-3.2-3B-Instruct": 6,
        "microsoft/Phi-3-mini-4k-instruct": 7,
        "mistralai/Mistral-7B-Instruct-v0.3": 14,
    }
    mem = estimated_gb.get(MODEL_NAME, "unknown")
    print(f"Estimated memory: ~{mem} GB")
    if mem > 8:
        print("⚠️  WARNING: This model may be too large for Orin Nano (8GB)")
        print("   System may freeze! Consider using 4-bit quantization.")
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )


###############################

model.eval()
print(f"\n{'='*60}")
print(f"Model: {MODEL_NAME}")
print(f"Device: {model.device}")
print(f"Quantization: {'4-bit' if USE_4BIT else 'FP16'}")
print(f"{'='*60}\n")

def print_vram():
    if torch.cuda.is_available():
        print(f"Allocated VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        print(f"Cached VRAM:    {torch.cuda.memory_reserved()/1e9:.2f} GB")

def generate_streaming(messages, max_tokens=2048):
    """
    Generate text with natural stopping.
    max_tokens is now a safety limit to prevent infinite loops,
    but the model will typically stop much earlier when it produces EOS token.
    """
    formatted_prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"]
    prompt_length = input_ids.shape[1]
    past_key_values = None
    all_token_ids = input_ids[0].tolist()
    previous_text = ""
    
    with torch.no_grad():
        for _ in range(max_tokens):
            outputs = model(
                input_ids=input_ids, 
                past_key_values=past_key_values, 
                use_cache=True
            )
            past_key_values = outputs.past_key_values
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            
            # Check for EOS token - model naturally finished
            if next_token.item() == tokenizer.eos_token_id:
                print()
                break
            
            all_token_ids.append(next_token.item())
            input_ids = next_token

            # Decode all generated tokens so far
            current_text = tokenizer.decode(
                all_token_ids[prompt_length:], 
                skip_special_tokens=True
            )

            new_part = current_text[len(previous_text):]
            print(new_part, end="", flush=True)
            previous_text = current_text

    return tokenizer.decode(all_token_ids[prompt_length:], skip_special_tokens=True)

def chat():
    print("\n" + "="*60)
    print("Interactive Chat - Type 'quit' or 'exit' to end")
    print("="*60 + "\n")
    conversation_history = []
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye!")
            break
        if not user_input:
            continue
        conversation_history.append({"role": "user", "content": user_input})
        print("Assistant: ", end="", flush=True)
        response = generate_streaming(conversation_history)
        conversation_history.append({"role": "assistant", "content": response})
        print()

chat()

print("\n=== VRAM Usage ===")
print_vram()
