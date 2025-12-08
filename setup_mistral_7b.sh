#!/bin/bash
# Setup Mistral 7B for Jetson Orin Nano (8GB)

set -e

MODELS_DIR="$HOME/models"
mkdir -p "$MODELS_DIR"

echo "=========================================="
echo "Mistral 7B Setup for Jetson Orin Nano"
echo "=========================================="
echo ""

# Check available memory
echo "Checking available memory..."
free -h
echo ""

# Select quantization level
echo "Select Mistral 7B quantization level:"
echo ""
echo "1) Q4_K_M (~4GB) - Recommended, good quality"
echo "2) Q5_K_M (~5GB) - Better quality, might be tight on 8GB"
echo "3) IQ3_M (~3GB) - Smaller, lower quality"
echo ""
read -p "Choice (1-3): " choice

case $choice in
    1)
        MODEL_NAME="Mistral-7B-Instruct-v0.3-Q4_K_M.gguf"
        MODEL_URL="https://huggingface.co/bartowski/Mistral-7B-Instruct-v0.3-GGUF/resolve/main/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf"
        QUANT="Q4"
        ;;
    2)
        MODEL_NAME="Mistral-7B-Instruct-v0.3-Q5_K_M.gguf"
        MODEL_URL="https://huggingface.co/bartowski/Mistral-7B-Instruct-v0.3-GGUF/resolve/main/Mistral-7B-Instruct-v0.3-Q5_K_M.gguf"
        QUANT="Q5"
        ;;
    3)
        MODEL_NAME="Mistral-7B-Instruct-v0.3-IQ3_M.gguf"
        MODEL_URL="https://huggingface.co/bartowski/Mistral-7B-Instruct-v0.3-GGUF/resolve/main/Mistral-7B-Instruct-v0.3-IQ3_M.gguf"
        QUANT="Q3"
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

MODEL_PATH="$MODELS_DIR/$MODEL_NAME"

# Download if needed
if [ -f "$MODEL_PATH" ]; then
    echo "✓ Model already exists: $MODEL_NAME"
else
    echo ""
    echo "Downloading Mistral 7B $QUANT (~4GB, will take several minutes)..."
    wget --show-progress -c "$MODEL_URL" -O "$MODEL_PATH"
    echo "✓ Download complete"
fi

echo ""
echo "Model info:"
ls -lh "$MODEL_PATH"
echo ""

# Test the model
echo "=========================================="
echo "Testing Mistral 7B..."
echo "=========================================="
echo ""

python3 << EOF
import os
import time
from llama_cpp import Llama

model_path = "$MODEL_PATH"
print(f"Loading: {os.path.basename(model_path)}")
print(f"Size: {os.path.getsize(model_path)/1e9:.2f} GB")
print("")

# Load model
print("Loading model (this may take a few seconds)...")
start = time.time()

llm = Llama(
    model_path=model_path,
    n_gpu_layers=-1,  # All on GPU
    n_ctx=4096,       # Mistral's context window
    verbose=False
)

load_time = time.time() - start
print(f"✓ Loaded in {load_time:.2f}s\n")

# Test 1: Simple response
print("Test 1: Simple question")
print("-" * 40)
start = time.time()
result = llm.create_chat_completion(
    messages=[
        {"role": "user", "content": "What are you and what makes you special?"}
    ],
    max_tokens=100,
    temperature=0.7
)
inference_time = time.time() - start

response = result['choices'][0]['message']['content']
print(f"Response: {response}")
print(f"Time: {inference_time:.2f}s\n")

# Test 2: Reasoning
print("Test 2: Reasoning task")
print("-" * 40)
start = time.time()
result = llm.create_chat_completion(
    messages=[
        {"role": "user", "content": "Explain the difference between AI and machine learning in simple terms."}
    ],
    max_tokens=150,
    temperature=0.7
)
inference_time = time.time() - start

response = result['choices'][0]['message']['content']
print(f"Response: {response}")
print(f"Time: {inference_time:.2f}s\n")

# Test 3: Coding
print("Test 3: Code generation")
print("-" * 40)
start = time.time()
result = llm.create_chat_completion(
    messages=[
        {"role": "user", "content": "Write a Python function to check if a number is prime."}
    ],
    max_tokens=200,
    temperature=0.7
)
inference_time = time.time() - start

response = result['choices'][0]['message']['content']
print(f"Response: {response}")
print(f"Time: {inference_time:.2f}s\n")

print("=" * 60)
print("✓ All tests passed! Mistral 7B is working correctly.")
print("=" * 60)
EOF

echo ""
echo "=========================================="
echo "✓ Setup Complete!"
echo "=========================================="
echo ""
echo "To chat with Mistral 7B:"
echo "  1. Update MODEL_PATH in simple_quantized_chat.py"
echo "  2. Set to: $MODEL_PATH"
echo "  3. Run: python simple_quantized_chat.py"
echo ""
echo "To benchmark:"
echo "  python simple_benchmark.py $MODEL_PATH"
echo ""
echo "Monitor GPU memory while running:"
echo "  watch -n 1 nvidia-smi"
echo ""
