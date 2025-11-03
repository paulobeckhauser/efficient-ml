import torch

# Check GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Check if CUDA is available
print("CUDA available:", torch.cuda.is_available())

# Number of GPUs
print("Number of GPUs:", torch.cuda.device_count())

# Current GPU name
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))

# Test a simple operation on GPU
x = torch.tensor([1.0, 2.0, 3.0], device='cuda')
print("Tensor on GPU:", x)