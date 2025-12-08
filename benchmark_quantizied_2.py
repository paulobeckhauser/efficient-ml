#!/usr/bin/env python3
"""
Benchmark FP16 (transformers) vs GGUF quantized models
"""

import time
import torch

# Test prompts
TEST_PROMPTS = [
    "Explain quantum computing in simple terms.",
    "Write a Python function to calculate fibonacci numbers.",
    "What are the main causes of climate change?",
]

def benchmark_fp16(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
    """Benchmark FP16 model using transformers"""
    print("\n" + "="*60)
    print("Benchmarking FP16 (Non-Quantized)")
    print("="*60)
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print(f"Loading {model_name}...")
    start_load = time.time()
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    
    load_time = time.time() - start_load
    
    # Get memory usage
    mem_allocated = torch.cuda.memory_allocated() / 1e9
    mem_reserved = torch.cuda.memory_reserved() / 1e9
    
    print(f"✓ Loaded in {load_time:.2f}s")
    print(f"  GPU Memory: {mem_allocated:.2f} GB allocated, {mem_reserved:.2f} GB reserved")
    
    # Benchmark inference
    print("\nRunning inference tests...")
    inference_times = []
    token_counts = []
    
    for i, prompt in enumerate(TEST_PROMPTS):
        print(f"  Test {i+1}/{len(TEST_PROMPTS)}: ", end="", flush=True)
        
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
        
        # Warmup on first run
        if i == 0:
            with torch.no_grad():
                _ = model.generate(**inputs, max_new_tokens=10)
        
        # Benchmark
        torch.cuda.synchronize()
        start = time.time()
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        torch.cuda.synchronize()
        inference_time = time.time() - start
        
        output_tokens = len(outputs[0]) - len(inputs.input_ids[0])
        tokens_per_sec = output_tokens / inference_time
        
        inference_times.append(inference_time)
        token_counts.append(tokens_per_sec)
        
        print(f"{inference_time:.2f}s ({tokens_per_sec:.1f} tok/s)")
    
    import statistics
    avg_time = statistics.mean(inference_times)
    avg_tps = statistics.mean(token_counts)
    
    print(f"\n✓ Average: {avg_time:.2f}s ({avg_tps:.1f} tok/s)")
    
    # Cleanup
    del model
    torch.cuda.empty_cache()
    
    return {
        "name": "FP16",
        "model_size_gb": mem_allocated,
        "load_time": load_time,
        "avg_inference_time": avg_time,
        "avg_tokens_per_second": avg_tps,
    }

def benchmark_gguf(model_path, quant_name):
    """Benchmark GGUF quantized model"""
    import os
    from llama_cpp import Llama
    
    print("\n" + "="*60)
    print(f"Benchmarking GGUF {quant_name}")
    print("="*60)
    
    model_path = os.path.expanduser(model_path)
    model_size_gb = os.path.getsize(model_path) / 1e9
    
    print(f"Loading {model_path}...")
    start_load = time.time()
    
    llm = Llama(
        model_path=model_path,
        n_gpu_layers=-1,
        n_ctx=2048,
        verbose=False,
    )
    
    load_time = time.time() - start_load
    print(f"✓ Loaded in {load_time:.2f}s")
    print(f"  File size: {model_size_gb:.2f} GB")
    
    print("\nRunning inference tests...")
    inference_times = []
    token_counts = []
    
    for i, prompt in enumerate(TEST_PROMPTS):
        print(f"  Test {i+1}/{len(TEST_PROMPTS)}: ", end="", flush=True)
        
        messages = [{"role": "user", "content": prompt}]
        
        # Warmup
        if i == 0:
            _ = llm.create_chat_completion(messages=messages, max_tokens=10)
        
        start = time.time()
        response = llm.create_chat_completion(
            messages=messages,
            max_tokens=100,
            temperature=0.7,
        )
        inference_time = time.time() - start
        
        output_text = response['choices'][0]['message']['content']
        output_tokens = len(output_text.split())
