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

## Jetson Orin Nano

with the command `cat /etc/nv_tegra_release' it was discovered that we have the version 6x. of Jetpack, since R36.x = JetPack 6.x

Therefore, for Jetpack 6.0, the CUDA 12.2 needs to be installed.

~/Documents/efficient-ml$ cat jetpack_v.txt 
# R36 (release), REVISION: 4.7, GCID: 42132812, BOARD: generic, EABI: aarch64, DATE: Thu Sep 18 22:54:44 UTC 2025
# KERNEL_VARIANT: oot
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
