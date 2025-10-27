## Introduction

Large Language Models (LLMs) are computationally heavy systems that require a significant amount of memory and processing power.  

To improve their accuracy and reasoning capabilities, several techniques have been developed within the fields of **TinyML** and **EfficientML**.  
Some common examples include **few-shot learning** and **chain-of-thought prompting**.  

However, these enhancements come with a cost: they make models even heavier, increasing both the computational and resource demands required for training and inference.

In addition, deploying Machine Learning or Deep Learning models in the Edge, such as IoT devices based on microcontrollers can be a challenge since memory and power resources are even more limited.

So, the idea is to reduce both weights and activation to fit for example a Deep Neural Network in a tiny device.

## Hardware Overview
- Device: NVIDIA Jetson Orin Nano (8GB)
- GPU: 1024-core NVIDIA Ampere GPU with 32 Tensor Cores
- CPU: 6-core ARM Cortex-A78AE 64-bit CPU
- RAM: 8GB 128-bit LPDDR5

## Quantization

There are two types of quantization:

- **Quantize both weight and activation** (e.g. SmoothQuant) which is good for large batch inference
- **Weight-only activation** (e.g. AWQ) which is better for single batch inference

## 1. Choosing some LLM models
The Jetson Orin Nano has 8GB of GPU memory, which is very limited comparing to datacenters that LLMs can easily require something around 10-100GB in FP16 precision.

LLMs like Mistral-7B or LLaMA-2-7B are around 13-14GB in FP16, which is too large to load directly into 8GB GPU memory.

Some models that interested to work on in this project are: Mistral-7B, LLaMA-7B, Mini LLaMA, Falcon-7B.
