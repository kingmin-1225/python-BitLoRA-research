import torch
import os
from safetensors.torch import load_file

def quantize_and_pack_ternary(tensor):
    gamma = tensor.abs().mean().item()
    
    eps = 1e-5
    quantized = torch.round(tensor / (gamma + eps)).clamp(-1, 1)
    
    # 2비트 패킹
    shifted = (quantized + 1).to(torch.uint8).flatten()
    
    if len(shifted) % 4 != 0:
        padding = 4 - (len(shifted) % 4)
        shifted = torch.cat([shifted, torch.zeros(padding, dtype=torch.uint8)])
    
    packed = (shifted[0::4] << 6) | (shifted[1::4] << 4) | (shifted[2::4] << 2) | shifted[3::4]
    
    return packed, gamma

# path = "./experiments/llama-3b-ternary-r4/checkpoint-2718/adapter_model.safetensors"
# path = "./experiments/llama-3b-ternary-r8/checkpoint-2718/adapter_model.safetensors"
path = "./experiments/llama-3b-ternary-r16/checkpoint-2721/adapter_model.safetensors"
weights = load_file(path)

total_binary_data = b""
scale_factors = []

for name, tensor in weights.items():
    if "lora_" in name:
        packed_w, g = quantize_and_pack_ternary(tensor)
        total_binary_data += packed_w.numpy().tobytes()
        scale_factors.append(g) 

with open("final_deployment_model.bin", "wb") as f:
    f.write(total_binary_data)
    f.write(torch.tensor(scale_factors, dtype=torch.float32).numpy().tobytes())

print(f"{os.path.getsize('ternary_model.bin') / (1024**2):.2f} MB")