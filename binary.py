# import torch
# import os
# from safetensors.torch import load_file

# def pack_binary_1bit(tensor):
#     gamma = tensor.abs().mean().item()
    
#     bits = (tensor > 0).to(torch.uint8).flatten()
    
#     if len(bits) % 8 != 0:
#         padding = 8 - (len(bits) % 8)
#         bits = torch.cat([bits, torch.zeros(padding, dtype=torch.uint8)])
    
#     packed = (
#         (bits[0::8] << 7) | (bits[1::8] << 6) | (bits[2::8] << 5) | (bits[3::8] << 4) |
#         (bits[4::8] << 3) | (bits[5::8] << 2) | (bits[6::8] << 1) | (bits[7::8])
#     )
    
#     return packed, gamma

# path = "./experiments/llama-3b-binary-r16/checkpoint-2721/adapter_model.safetensors"
# weights = load_file(path)

# total_binary_data = b""
# scale_factors = []

# for name, tensor in weights.items():
#     if "lora_" in name:
#         packed_w, g = pack_binary_1bit(tensor)
#         total_binary_data += packed_w.numpy().tobytes()
#         scale_factors.append(g)

# # 파일 저장
# save_name = "binary_model.bin"
# with open(save_name, "wb") as f:
#     f.write(total_binary_data)
#     f.write(torch.tensor(scale_factors, dtype=torch.float32).numpy().tobytes())

# # 용량 확인
# size_bytes = os.path.getsize(save_name)
# print(f"MB: {size_bytes / (1024**2):.2f} MB")


import torch
import os
from safetensors.torch import load_file

def pack_binary_1bit(tensor):
    """1비트 패킹: Tensor -> uint8 Packed Tensor, Gamma, Shape"""
    original_shape = tensor.shape
    numel = tensor.numel()
    
    gamma = tensor.abs().mean().item()
    
    # 0보다 크면 1, 아니면 0
    bits = (tensor > 0).to(torch.uint8).flatten()
    
    # 8비트 단위 패딩
    padding = 0
    if len(bits) % 8 != 0:
        padding = 8 - (len(bits) % 8)
        bits = torch.cat([bits, torch.zeros(padding, dtype=torch.uint8)])
    
    # 비트 패킹
    packed = (
        (bits[0::8] << 7) | (bits[1::8] << 6) | (bits[2::8] << 5) | (bits[3::8] << 4) |
        (bits[4::8] << 3) | (bits[5::8] << 2) | (bits[6::8] << 1) | (bits[7::8])
    )
    
    return packed, gamma, original_shape, numel

def unpack_binary_1bit(packed, gamma, shape, numel):
    """복구: uint8 Packed Tensor -> 원본 모양의 Tensor"""
    # 1. 비트 풀기 (Unpacking)
    bits = []
    for i in range(7, -1, -1):
        bits.append((packed >> i) & 1)
    
    # (8, N) -> (N, 8) -> Flat
    unpacked_bits = torch.stack(bits, dim=1).flatten()
    
    # 2. 패딩 제거
    unpacked_bits = unpacked_bits[:numel]
    
    # 3. 값 복원: 1은 +gamma, 0은 -gamma
    reconstructed = torch.where(unpacked_bits > 0, 
                                torch.tensor(gamma, dtype=torch.float32), 
                                torch.tensor(-gamma, dtype=torch.float32))
    
    return reconstructed.reshape(shape)



# --- 1. 압축 및 저장 단계 ---
path = "./experiments/llama-3b-binary-r16/checkpoint-2721/adapter_model.safetensors"
weights = load_file(path)

save_data = {}
print("압축 중...")

for name, tensor in weights.items():
    if "lora_" in name:
        packed_w, g, shape, numel = pack_binary_1bit(tensor)
        save_data[name] = {
            "packed": packed_w,
            "gamma": g,
            "shape": shape,
            "numel": numel
        }

# 파일 저장
torch.save(save_data, "my_compressed_model.bin")
print(f"저장 완료: {os.path.getsize('my_compressed_model.bin') / 1024**2:.2f} MB")

# --- 2. 복구 및 비교 단계 ---
print("\n복구 및 검증 시작...")
loaded_data = torch.load("my_compressed_model.bin")

print(f"{'Layer Name':<60} | {'Orig Gamma':<10} | {'Recon Val':<10} | {'MAE':<10}")
print("-" * 100)

for name, data in loaded_data.items():
    # 1. 복구 수행
    recon = unpack_binary_1bit(data['packed'], data['gamma'], data['shape'], data['numel'])
    orig = weights[name].to(torch.float32) # 정밀한 비교를 위해 float32 변환
    
    # 2. 스케일 확인
    orig_gamma = orig.abs().mean().item()
    recon_val = data['gamma'] # 우리가 저장했던 gamma 값
    
    # 3. 전체적인 복구 오차 (Mean Absolute Error)
    # 1비트 양자화는 손실이 클 수밖에 없지만, 스케일이 맞는지 보는 지표입니다.
    mae = torch.mean(torch.abs(orig - recon)).item()
    
    # 출력 (소수점 6자리까지)
    print(f"{name[:60]:<60} | {orig_gamma:.6f} | {recon_val:.6f} | {mae:.6f}")

    # 스케일 정밀도 체크 (부동소수점 오차 감안)
    if abs(orig_gamma - recon_val) > 1e-7:
        print(f"⚠️ {name}: 스케일 미세 불일치 발생!")

print("-" * 100)
print("검증 완료.")

all_match = True
for name, data in loaded_data.items():
    # 복구 수행
    recon = unpack_binary_1bit(data['packed'], data['gamma'], data['shape'], data['numel'])
    
    # 원본 가져오기
    orig = weights[name]
    
    # 1비트 양자화 특성상 '부호'가 같으면 성공입니다.
    orig_sign = (orig > 0)
    recon_sign = (recon > 0)
    
    mismatch = torch.sum(orig_sign != recon_sign).item()
    
    if mismatch == 0:
        print(f"✅ {name}: 복구 성공 (부호 일치)")
    else:
        print(f"❌ {name}: 복구 실패 ({mismatch}개 비트 다름)")
        all_match = False

if all_match:
    print("\n✨ 모든 가중치가 성공적으로 복구되었습니다!")