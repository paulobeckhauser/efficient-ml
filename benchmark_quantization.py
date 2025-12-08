import time
import json
import os
from pathlib import Path
import torch
import numpy as np

# Test prompts for evaluation
TEST_PROMPTS = [
    "Explain quantum computing in simple terms.",
    "Write a Python function to calculate fibonacci numbers.",
    "What are the main causes of climate change?",
    "Describe the process of photosynthesis.",
    "How does a neural network learn?",
]

def get_gpu_memory():
    """Get current GPU memory usage in GB"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        return allocated, reserved
    return 0, 0

def benchmark_transformers_fp16(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
    """Benchmark non-quantized FP16 model"""
    print("\n" + "="*60)
    print("Benchmarking FP16 (Non-Quantized) Model")
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
    mem_allocated, mem_reserved = get_gpu_memory()
    
    print(f"✓ Model loaded in {load_time:.2f}s")
    print(f"  Memory: {mem_allocated:.2f} GB allocated, {mem_reserved:.2f} GB reserved")
    
    results = {
        "model_type": "FP16",
        "model_name": model_name,
        "load_time": load_time,
        "memory_allocated_gb": mem_allocated,
        "memory_reserved_gb": mem_reserved,
        "inference_times": [],
        "tokens_per_second": [],
        "outputs": []
    }
    
    print("\nRunning inference tests...")
    for i, prompt in enumerate(TEST_PROMPTS):
        print(f"  Test {i+1}/{len(TEST_PROMPTS)}: ", end="", flush=True)
        
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

        if i == 0:
            with torch.no_grad():
                _ = model.generate(**inputs, max_new_tokens=10)

        start = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id,
            )
        inference_time = time.time() - start
        
        output_text = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
        output_tokens = len(outputs[0]) - len(inputs.input_ids[0])
        tokens_per_sec = output_tokens / inference_time
        
        results["inference_times"].append(inference_time)
        results["tokens_per_second"].append(tokens_per_sec)
        results["outputs"].append(output_text)
        
        print(f"{inference_time:.2f}s ({tokens_per_sec:.1f} tok/s)")
    
    results["avg_inference_time"] = np.mean(results["inference_times"])
    results["avg_tokens_per_second"] = np.mean(results["tokens_per_second"])
    results["std_inference_time"] = np.std(results["inference_times"])
    
    print(f"\n✓ Average: {results['avg_inference_time']:.2f}s ({results['avg_tokens_per_second']:.1f} tok/s)")

    del model
    torch.cuda.empty_cache()
    
    return results

def benchmark_gguf(model_path, quantization_level="Q4"):
    """Benchmark GGUF quantized model"""
    print("\n" + "="*60)
    print(f"Benchmarking GGUF {quantization_level} (Quantized) Model")
    print("="*60)
    
    try:
        from llama_cpp import Llama
    except ImportError:
        print("llama-cpp-python not installed")
        return None
    
    model_path = os.path.expanduser(model_path)
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return None
    
    print(f"Loading {model_path}...")
    start_load = time.time()
    
    llm = Llama(
        model_path=model_path,
        n_gpu_layers=-1,
        n_ctx=2048,
        verbose=False,
    )
    
    load_time = time.time() - start_load
    model_size_gb = os.path.getsize(model_path) / 1e9
    
    print(f"✓ Model loaded in {load_time:.2f}s")
    print(f"  Model file size: {model_size_gb:.2f} GB")
    
    results = {
        "model_type": f"GGUF_{quantization_level}",
        "model_path": model_path,
        "load_time": load_time,
        "model_size_gb": model_size_gb,
        "inference_times": [],
        "tokens_per_second": [],
        "outputs": []
    }
    
    print("\nRunning inference tests...")
    for i, prompt in enumerate(TEST_PROMPTS):
        print(f"  Test {i+1}/{len(TEST_PROMPTS)}: ", end="", flush=True)
        
        messages = [{"role": "user", "content": prompt}]

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
        output_tokens = len(output_text.split()) * 1.3
        tokens_per_sec = output_tokens / inference_time
        
        results["inference_times"].append(inference_time)
        results["tokens_per_second"].append(tokens_per_sec)
        results["outputs"].append(output_text)
        
        print(f"{inference_time:.2f}s (~{tokens_per_sec:.1f} tok/s)")
    
    results["avg_inference_time"] = np.mean(results["inference_times"])
    results["avg_tokens_per_second"] = np.mean(results["tokens_per_second"])
    results["std_inference_time"] = np.std(results["inference_times"])
    
    print(f"\n✓ Average: {results['avg_inference_time']:.2f}s (~{results['avg_tokens_per_second']:.1f} tok/s)")
    
    return results

def compare_outputs(fp16_results, quant_results):
    """Compare output quality between FP16 and quantized"""
    print("\n" + "="*60)
    print("Output Quality Comparison")
    print("="*60)
    
    from difflib import SequenceMatcher
    
    similarities = []
    for i, (fp16_out, quant_out) in enumerate(zip(fp16_results["outputs"], quant_results["outputs"])):
        similarity = SequenceMatcher(None, fp16_out, quant_out).ratio()
        similarities.append(similarity)
        
        print(f"\nPrompt {i+1}: '{TEST_PROMPTS[i][:50]}...'")
        print(f"  Similarity: {similarity*100:.1f}%")
        print(f"  FP16 length: {len(fp16_out)} chars")
        print(f"  Quantized length: {len(quant_out)} chars")
        
        if similarity < 0.5:
            print(f"\n  FP16 output:\n    {fp16_out[:200]}...")
            print(f"\n  Quantized output:\n    {quant_out[:200]}...")
    
    avg_similarity = np.mean(similarities)
    print(f"\n✓ Average similarity: {avg_similarity*100:.1f}%")
    
    return similarities

def print_summary(fp16_results, quant_results):
    """Print summary comparison"""
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    
    print(f"\n{'Metric':<30} {'FP16':<15} {'Quantized':<15} {'Difference'}")
    print("-" * 70)
    print(f"{'Load Time':<30} {fp16_results['load_time']:.2f}s"
          f"{'':<8} {quant_results['load_time']:.2f}s"
          f"{'':<8} {quant_results['load_time']/fp16_results['load_time']:.2f}x")
    
    if 'memory_allocated_gb' in fp16_results:
        mem_fp16 = fp16_results['memory_allocated_gb']
        mem_quant = quant_results.get('model_size_gb', 0)
        print(f"{'Memory Usage':<30} {mem_fp16:.2f} GB"
              f"{'':<6} {mem_quant:.2f} GB"
              f"{'':<6} {mem_quant/mem_fp16:.2f}x")
    speed_fp16 = fp16_results['avg_inference_time']
    speed_quant = quant_results['avg_inference_time']
    speedup = speed_fp16 / speed_quant
    
    print(f"{'Avg Inference Time':<30} {speed_fp16:.2f}s"
          f"{'':<8} {speed_quant:.2f}s"
          f"{'':<8} {speedup:.2f}x")

    tps_fp16 = fp16_results['avg_tokens_per_second']
    tps_quant = quant_results['avg_tokens_per_second']
    
    print(f"{'Tokens/Second':<30} {tps_fp16:.1f}"
          f"{'':<10} {tps_quant:.1f}"
          f"{'':<10} {tps_quant/tps_fp16:.2f}x")
    
    print("\n" + "="*60)
    print("Interpretation:")
    if speedup > 1.1:
        print(f"  ✓ Quantized model is {speedup:.1f}x FASTER")
    elif speedup < 0.9:
        print(f"  ⚠ Quantized model is {1/speedup:.1f}x SLOWER")
    else:
        print(f"  ≈ Similar speed")
    
    if mem_quant < mem_fp16 * 0.7:
        print(f"  ✓ Quantized uses {(1-mem_quant/mem_fp16)*100:.0f}% LESS memory")
    
    print("="*60)

def save_results(fp16_results, quant_results, output_file="benchmark_results.json"):
    """Save results to JSON"""
    results = {
        "fp16": fp16_results,
        "quantized": quant_results,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {output_file}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark quantized vs non-quantized models")
    parser.add_argument("--hf-model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                       help="HuggingFace model name for FP16 benchmark")
    parser.add_argument("--gguf-model", default="~/models/tinyllama-q4.gguf",
                       help="Path to GGUF quantized model")
    parser.add_argument("--skip-fp16", action="store_true",
                       help="Skip FP16 benchmark (only run quantized)")
    parser.add_argument("--output", default="benchmark_results.json",
                       help="Output file for results")
    
    args = parser.parse_args()
    
    print("="*60)
    print("QUANTIZATION BENCHMARK")
    print("="*60)
    
    fp16_results = None
    quant_results = None
    
    if not args.skip_fp16:
        try:
            fp16_results = benchmark_transformers_fp16(args.hf_model)
        except Exception as e:
            print(f"\n❌ FP16 benchmark failed: {e}")

    try:
        quant_results = benchmark_gguf(args.gguf_model)
    except Exception as e:
        print(f"\n❌ Quantized benchmark failed: {e}")

    if fp16_results and quant_results:
        similarities = compare_outputs(fp16_results, quant_results)
        fp16_results["output_similarities"] = similarities
        print_summary(fp16_results, quant_results)
        save_results(fp16_results, quant_results, args.output)
    elif quant_results:
        print("\n✓ Quantized benchmark complete (FP16 skipped)")
        save_results({}, quant_results, args.output)
    else:
        print("\n❌ No benchmarks completed successfully")

if __name__ == "__main__":
    main()
