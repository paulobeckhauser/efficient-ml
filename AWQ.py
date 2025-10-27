'''
The goal of this code is to quantize a large language model
that can run efficiently. A AWQ for 4 bit weight-only 
quantization will be implemented.

I will be running this code in a NVIDIA
Jetson Orin Nano 8GB Module
'''

import tqdm # display progress bars
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from functools import partial
import gc


# Using wikitext-2 dataset for evaluation
def evaluate(model, tokenizer):
    testenc = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    