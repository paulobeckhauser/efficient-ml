# Efficient Machine Learning on Edge Devices

This project is part of my **special course at DTU**, where I am exploring techniques for deploying Machine Learning and Deep Learning models on resource-constrained devices. The course follows the materials from [EfficientML](https://efficientml.ai/).

## Introduction

Large Language Models (LLMs) are computationally heavy systems that require a significant amount of memory and processing power.  

To improve their accuracy and reasoning capabilities, several techniques have been developed within the fields of **TinyML** and **EfficientML**.  
Some common examples include **few-shot learning** and **chain-of-thought prompting**.  

However, these enhancements come with a cost: they make models even heavier, increasing both the computational and resource demands required for training and inference.

In addition, deploying Machine Learning or Deep Learning models in the Edge, such as IoT devices based on microcontrollers can be a challenge since memory and power resources are even more limited.

So, the idea is to reduce both weights and activation to fit for example a Deep Neural Network in a 'tiny' or edge device.

While this course focuses on edge deployment, the techniques can also be applied to laptops or workstations to reduce memory and compute requirements.

## Hardware Overview

### NVIDIA Jetson Orin Nano - Caracteristics

For this project, the primary hardware used is an **NVIDIA Jetson Orin Nano (8GB)**. The detailed specifications are:

- GPU: 1024-core NVIDIA Ampere GPU with 32 Tensor Cores
- CPU: 6-core ARM Cortex-A78AE 64-bit CPU
- RAM: 8GB 128-bit LPDDR5


To verify the GPU and driver status, the `nvidia-smi` command can be used. Sample output:

```bash
Mon Nov  3 13:43:23 2025       
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 540.4.0                Driver Version: 540.4.0      CUDA Version: 12.6     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  Orin (nvgpu)                  N/A  | N/A              N/A |                  N/A |
| N/A   N/A  N/A               N/A /  N/A | Not Supported        |     N/A          N/A |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|  No running processes found                                                           |
+---------------------------------------------------------------------------------------+
```
When running `nvidia-smi` on an NVIDIA Jetson Orin device, many fields may appear as "N/A". This is normal and expected.

Jetson Orin uses an integrated GPU with unified memory, which makes it fundamentally different from discrete desktop or server GPUs such as the RTX, Tesla, or A100 series. Since `nvidia-smi` is designed for discrete GPU architectures, many of its usual features are not applicable on Jetson platforms, resulting in numerous fields appearing as “N/A”.

Because the CPU and GPU share the same unified memory, Jetson cannot report separate GPU memory usage the way discrete GPUs do. Its cooling system also differs, as Jetson devices may rely on passive cooling or system-managed fans, meaning fan speed information is not accessible through nvidia-smi. Power and voltage reporting are likewise more limited compared to traditional GPU boards, and Jetson employs its own thermal control mechanisms that nvidia-smi is not designed to display.

For a more complete system monitoring on Jetson devices, tools such as tegrastats, jtop, or the /sys interfaces provided by L4T should be used instead.


### NVIDIA Software Installation

with the command `cat /etc/nv_tegra_release' it was discovered that we have the version 6x. of Jetpack, since R36.x = JetPack 6.x

Therefore, for Jetpack 6.0, the CUDA 12.2 needs to be installed.

~/Documents/efficient-ml$ cat jetpack_v.txt 
R36 (release), REVISION: 4.7, GCID: 42132812, BOARD: generic, EABI: aarch64, DATE: Thu Sep 18 22:54:44 UTC 2025
KERNEL_VARIANT: oot
TARGET_USERSPACE_LIB_DIR=nvidia
TARGET_USERSPACE_LIB_DIR_PATH=usr/lib/aarch64-linux-gnu/nvidia



## Quantization

There are two types of quantization:

- **Quantize both weight and activation** (e.g. SmoothQuant) which is good for large batch inference
- **Weight-only activation** (e.g. AWQ) which is better for single batch inference

## 1. Choosing some LLM models
The Jetson Orin Nano has 8GB of GPU memory, which is very limited comparing to datacenters that LLMs can easily require something around 10-100GB in FP16 precision.

LLMs like Mistral-7B or LLaMA-2-7B are around 13-14GB in FP16, which is too large to load directly into 8GB GPU memory.

Some models that interested to work on in this project are: Mistral-7B, LLaMA-7B, Mini LLaMA, Falcon-7B.

| Model                     | Worked | Notes                  |
|:--------------------------|:------:|-----------------------|
| TinyLlama-1.1B-Chat-v1.0  | Yes    | Quantized: Yes & No   |


### Progress

First I tried to execute everything, using a very tiny model, just to have a first benchmarking that running in Jetson Orin Nano with libraries were working fine. So, I executed with a TinyLlama of ~1B parameters without quantization and worked fine.

Then I tried to go direct to a heavier model, Mistral 7B. This model in FP16 (~14 GB) is too large for this Jetson's memory. The Orin Nano has 8 GB of unified memory (shared between CPU and GPU), so a 14 GB model simply won't fit. When it tries to load, it runs out of memory and the system freezes.

It was tried then to apply 4-bit quantization.

When trying to run 7B models with 4-bit quantization using the standard Hugging Face Transformers approach: 

```bash
from transformers import BitsAndBytesConfig

# This will NOT work on Jetson Orin Nano 8GB
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.3",
    quantization_config=bnb_config,  # Fails on Jetson
)
```

I encountered this error:

```bash
Error named symbol not found at line 57 in file /src/csrc/ops.cu
```

This happens because first it was tried to used the library `bitsandbytes`, which is compiled for x86_64 (desktop) GPUs only. Jetson devices use ARM architecture with different CUDA operations, meaning that the library's low-level CUDA kernels are incompatible with Jetson's hardware.

Possibilities:

1. Already download pre-quantized models(ollama or GGUF models)
2. Use TensorRT-LLM


After some trying...

even with quantized models, I didn't make possible to run a model with 8B parameters in Jetson Orin Nano. This is primarily due to RAM from the GPU, but with the Jetson Orin Nano, it's more accurate to call it unified memory capacity.

The Jetson Orin Nano uses a Unified Memory Architecture, which means the CPU (Central Processing Unit) and the GPU (Graphics Processing Unit) share the same pool of physical memory (RAM). There is no separate, dedicated VRAM (Video RAM) like on a standard desktop graphics card.

LLaMA 3 8B (AWQ INT4): Even with 4-bit quantization (INT4), the model size is approximately 4 GB (8 billion parameters × 4 bits per parameter / 8 bits per byte). Jetson Orin Nano Memory: The Orin Nano typically comes with 8 GB of unified memory. Out of the 8 GB, a significant portion is already used by the Operating System (OS), system processes, and other software (often leaving about 6 GB to 7 GB usable). While 4 GB for the model weights might seem to fit, there's more memory required.