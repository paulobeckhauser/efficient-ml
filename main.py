import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
USE_4BIT = False

print(f"PyTorch CUDA available? {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Device count: {torch.cuda.device_count()}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16,
    device_map="auto",
)

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

def generate_streaming(messages, max_tokens=100):
    # conversation history format
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
            
            if next_token.item() == tokenizer.eos_token_id:
                break
                
            all_token_ids.append(next_token.item())
            input_ids = next_token
            
            current_text = tokenizer.decode(
                all_token_ids[prompt_length:], 
                skip_special_tokens=True
            )
            
            new_part = current_text[len(previous_text):]
            print(new_part, end="", flush=True)
            previous_text = current_text
    
    print()
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
        
        # Print assistant label
        print("Assistant: ", end="", flush=True)
        response = generate_streaming(conversation_history, max_tokens=150)
        
        conversation_history.append({"role": "assistant", "content": response})
        
        print()  # Extra line for readability

chat()

print("\n=== VRAM Usage ===")
print_vram()
