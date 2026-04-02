import torch
from safetensors.torch import load_file, save_file
from utils import weight_quant1, weight_quant158
import os

def quantize_to_bitlora(input_path, output_path, f):
    tensors = load_file(input_path)

    new_tensors = {}

    for name, tensor in tensors.items():
        if "lora_A" in name or "lora_B" in name:
            w = tensor.to(torch.float32)
            new_tensors[name] = f(w)
        else:
            new_tensors[name] = tensor
        save_file(new_tensors, output_path)

for r in [4, 8, 16]:
    quantize_to_bitlora(
        f"experiments/llama-3b-binary-r{r}/checkpoint-1800/adapter_model.safetensors", 
        f"to_eval/llama-3b-binary-r{r}/checkpoint-1800/adapter_model.safetensors", 
        weight_quant1
        )
    quantize_to_bitlora(
        f"experiments/llama-3b-ternary-r{r}/checkpoint-1800/adapter_model.safetensors", 
        f"to_eval/llama-3b-ternary-r{r}/checkpoint-1800/adapter_model.safetensors", 
        weight_quant158
        )
