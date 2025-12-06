from llama_cpp import Llama 
from huggingface_hub import hf_hub_download
import torch
import os

REPO_ID = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
FILENAME = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

print(f"Downloading {FILENAME} from {REPO_ID}...")
GGUF_PATH = hf_hub_download(
    repo_id=REPO_ID, 
    filename=FILENAME
)
print(f"Model saved locally at: {GGUF_PATH}")

model = Llama(
    model_path=GGUF_PATH,  
    n_gpu_layers=-1,  
    n_ctx=2048,
    verbose=False
)
print(f"\n{'='*60}")
print(f"Model: {FILENAME}")
print(f"Quantization: GGUF Q4_K_M (4-bit)")
print(f"GPU Layers Offloaded: -1 (Max)")
print(f"{'='*60}\n")

def generate_streaming(messages, max_tokens=2048):
    """
    Generates text using the llama-cpp-python chat completion API.
    """
    print(" (Thinking...)", end="", flush=True)
    
    stream = model.create_chat_completion(
        messages=messages,
        max_tokens=max_tokens,
        stream=True,
    )

    full_response = ""
    for chunk in stream:
        content = chunk["choices"][0]["delta"].get("content", "")
        if content:
            if not full_response:
                print('\rAssistant: ', end="", flush=True) 

            print(content, end="", flush=True)
            full_response += content
            
    print()
    return full_response

def chat():
    print("\n" + "="*60)
    print("Interactive Chat")
    print("Commands: 'quit', 'exit', 'q' to end | Ctrl+C to interrupt")
    print("="*60 + "\n")
    conversation_history = []

    try:
        while True:
            try:
                user_input = input("You: ").strip()
            except EOFError:
                print("\n\nGoodbye!")
                break
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break
            
            if not user_input:
                continue
            
            conversation_history.append({"role": "user", "content": user_input})
            print("Assistant: ", end="", flush=True)
            
            try:
                response = generate_streaming(conversation_history)
                conversation_history.append({"role": "assistant", "content": response})
            except KeyboardInterrupt:
                print("\n\n⚠️  Generation interrupted. Type 'quit' to exit or continue chatting.")
                conversation_history.pop()
                continue
            
            print()
            
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
    
    except Exception as e:
        print(f"\n\nError: {e}")
        print("Exiting...")
    

chat()
