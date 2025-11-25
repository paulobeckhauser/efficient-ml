import warnings
warnings.filterwarnings("ignore")
import torch
import argparse
import json
import os
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

@torch.no_grad()
def greedy_generate(model, tokenizer, input_ids, past_key_values, max_gen_len):
    """Generate text token by token using greedy decoding"""
    outputs = model(
        input_ids=input_ids,
        past_key_values=past_key_values,
        use_cache=True,
    )
    past_key_values = outputs.past_key_values
    pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
    generated_ids = [pred_token_idx.item()]
    pos = 0
    
    for _ in range(max_gen_len - 1):
        outputs = model(
            input_ids=pred_token_idx,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
        generated_ids.append(pred_token_idx.item())
        
        # Decode and print incrementally
        generated_text = (
            tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
                spaces_between_special_tokens=False,
            )
            .strip()
            .split(" ")
        )
        
        now = len(generated_text) - 1
        if now > pos:
            print(" ".join(generated_text[pos:now]), end=" ", flush=True)
            pos = now
        
        if pred_token_idx == tokenizer.eos_token_id:
            break
    
    print(" ".join(generated_text[pos:]), flush=True)
    return past_key_values

@torch.no_grad()
def streaming_inference(model, tokenizer, prompts, max_gen_len=200):
    """Run inference across multiple prompts with persistent KV cache"""
    past_key_values = None
    
    for idx, prompt in enumerate(prompts):
        # Format prompt
        formatted_prompt = f"USER: {prompt}\n\nASSISTANT: "
        print(f"\n{'='*60}")
        print(formatted_prompt, end="")
        
        # Tokenize
        input_ids = tokenizer(formatted_prompt, return_tensors="pt").input_ids
        input_ids = input_ids.to(model.device)
        
        # Generate response
        past_key_values = greedy_generate(
            model, tokenizer, input_ids, past_key_values, max_gen_len=max_gen_len
        )
        
        print(f"\n{'='*60}")

def load_model_for_jetson(model_name_or_path, use_4bit=True):
    """Load model optimized for Jetson Orin Nano (8GB RAM)"""
    
    print(f"Loading model: {model_name_or_path}")
    print(f"Using 4-bit quantization: {use_4bit}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    
    if use_4bit:
        # 4-bit quantization for memory efficiency
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            quantization_config=quantization_config,
            device_map="auto",
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
        )
    else:
        # FP16 without quantization
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map="auto",
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
        )
    
    model.eval()
    print(f"Model loaded successfully on {model.device}")
    print(f"Model memory footprint: {model.get_memory_footprint() / 1e9:.2f} GB")
    
    return model, tokenizer

def main(args):
    # Load model optimized for Jetson
    model, tokenizer = load_model_for_jetson(
        args.model_name_or_path,
        use_4bit=args.use_4bit
    )
    
    # Define test prompts
    if args.test_prompts:
        prompts = args.test_prompts
    else:
        # Default test prompts
        prompts = [
            "What is machine learning?",
            "Can you explain what you just said in simpler terms?",
            "Give me an example of this in real life.",
        ]
    
    print(f"\nRunning inference on {len(prompts)} prompts...")
    print(f"Max generation length: {args.max_gen_len}")
    
    # Run streaming inference
    streaming_inference(model, tokenizer, prompts, max_gen_len=args.max_gen_len)
    
    print("\n✓ Inference completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="StreamingLLM inference for Jetson Orin Nano")
    
    # Model selection - SMALL MODELS ONLY for Jetson!
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="HuggingFace model path. Recommended: TinyLlama-1.1B, Phi-2, Llama-3.2-1B"
    )
    
    # Quantization
    parser.add_argument(
        "--use_4bit",
        action="store_true",
        default=True,
        help="Use 4-bit quantization (required for larger models on Jetson)"
    )
    
    # Generation params
    parser.add_argument(
        "--max_gen_len",
        type=int,
        default=200,
        help="Maximum tokens to generate per response"
    )
    
    # Custom prompts
    parser.add_argument(
        "--test_prompts",
        nargs="+",
        type=str,
        help="List of custom prompts to test"
    )
    
    args = parser.parse_args()
    
    main(args)
