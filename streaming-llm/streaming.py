import torch
import argparse

from transformers import AutoModelForCausalLM, AutoTokenizer

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

    #Comment below condition, if want to run in CPU(such as laptop)
    
    # Abort if not using GPU
    if device.type != "cuda":
        print("GPU not available. Exiting program.")
        return

    # Continue with GPU code
    print("Running main program on GPU...")


    model_name_or_path = args.model_name_or_path
    print(f"Model: {model_name_or_path}")

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
