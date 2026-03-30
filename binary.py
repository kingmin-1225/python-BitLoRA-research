import torch
import os
from safetensors.torch import load_file

def pack_binary_1bit(tensor):
    gamma = tensor.abs().mean().item()
    
    bits = (tensor > 0).to(torch.uint8).flatten()
    
    if len(bits) % 8 != 0:
        padding = 8 - (len(bits) % 8)
        bits = torch.cat([bits, torch.zeros(padding, dtype=torch.uint8)])
    
    packed = (
        (bits[0::8] << 7) | (bits[1::8] << 6) | (bits[2::8] << 5) | (bits[3::8] << 4) |
        (bits[4::8] << 3) | (bits[5::8] << 2) | (bits[6::8] << 1) | (bits[7::8])
    )
    
    return packed, gamma

path = "./experiments/llama-3b-binary-r8/checkpoint-2721/adapter_model.safetensors"
weights = load_file(path)

total_binary_data = b""
scale_factors = []

for name, tensor in weights.items():
    if "lora_" in name:
        packed_w, g = pack_binary_1bit(tensor)
        total_binary_data += packed_w.numpy().tobytes()
        scale_factors.append(g)

# 파일 저장
save_name = "binary_model.bin"
with open(save_name, "wb") as f:
    f.write(total_binary_data)
    f.write(torch.tensor(scale_factors, dtype=torch.float32).numpy().tobytes())

# 용량 확인
size_bytes = os.path.getsize(save_name)
print(f"MB: {size_bytes / (1024**2):.2f} MB")