import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def check_torch_device(verbose=True):
    """
    Checks PyTorch GPU availability and runs a small test tensor operation.
    
    Returns:
        device (torch.device): The device used ('cuda' or 'cpu')
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if verbose:
        print(f"Using device: {device}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")

        if torch.cuda.is_available():
            print(f"GPU name: {torch.cuda.get_device_name(0)}")
        try:
            x = torch.tensor([1.0, 2.0, 3.0], device=device)
            print(f"Tensor on {device}: {x}")
        except Exception as e:
            print(f"⚠️ Failed to run test tensor on {device}: {e}")
    return device


def main(args):

    device = check_torch_device()

    # Comment below condition, if want to run in CPU (such as laptop)
    
    # Abort if not using GPU
    if device.type != "cuda":
        print("GPU not available. Exiting program.")
        return

    # Continue with GPU code
    print("\n" + "="*60)
    print("Running main program on GPU...")
    print("="*60 + "\n")

    model_name_or_path = args.model_name_or_path
    print(f"Loading model: {model_name_or_path}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    
    # Configure quantization if requested
    if args.use_4bit:
        print("Using 4-bit quantization...")
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4"
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            quantization_config=quant_config,
            device_map="auto",
            low_cpu_mem_usage=True,
            attn_implementation="eager"  # FIXED: Added for compatibility
        )
    
    else:
        print("Loading in FP16...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map="auto",
            dtype=torch.float16,  # FIXED: Changed from torch_dtype
            low_cpu_mem_usage=True,
            attn_implementation="eager"  # FIXED: Added for compatibility
        )

    model.eval()
    print(f"Model loaded successfully!")  # FIXED: Changed from printf
    print(f"Memory footprint: {model.get_memory_footprint() / 1e9:.2f} GB\n")
        
    # Set up prompts
    if args.test_prompts:
        prompts = args.test_prompts
    else:
        prompts = [
            "What is machine learning?",
            "Explain quantum computing in simple terms.",
            "Write a haiku about AI."
        ]

    # Run inference
    print("="*60)
    print(f"Running inference on {len(prompts)} prompts")
    print("="*60 + "\n")

    for i, prompt in enumerate(prompts, 1):
        print(f"[Prompt {i}/{len(prompts)}]: {prompt}")

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_gen_len,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
            output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            
        print(f"[Response]: {output_text}\n")
        print("-"*60 + "\n")

    print("All inference completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="StreamingLLM inference for Jetson Orin Nano"
    )

    # Model Selection
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