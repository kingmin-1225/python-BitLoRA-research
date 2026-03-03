import json
import matplotlib.pyplot as plt
import os

# 1. 두 모델의 trainer_state.json 경로 지정
# (본인의 실제 체크포인트 번호에 맞게 수정하세요. 예: checkpoint-200)
lora_state_path = "./medical_lora_results/checkpoint-200/trainer_state.json"
bitlora_state_path = "./medical_bitlora_results/checkpoint-200/trainer_state.json"

def extract_loss_data(file_path):
    steps = []
    losses = []
    if not os.path.exists(file_path):
        print(f"⚠️ 경고: 파일을 찾을 수 없습니다 -> {file_path}")
        return steps, losses
        
    with open(file_path, "r", encoding="utf-8") as f:
        state_data = json.load(f)
        
    for log in state_data.get("log_history", []):
        if "loss" in log and "step" in log:
            steps.append(log["step"])
            losses.append(log["loss"])
            
    return steps, losses

# 2. 데이터 추출
lora_steps, lora_losses = extract_loss_data(lora_state_path)
bitlora_steps, bitlora_losses = extract_loss_data(bitlora_state_path)

if not lora_steps and not bitlora_steps:
    print("❌ 두 파일 모두 데이터를 불러오지 못했습니다. 경로를 확인해주세요!")
else:
    # 3. 그래프 그리기 세팅
    plt.figure(figsize=(10, 6))
    
    # 일반 LoRA 그래프 (빨간색 선)
    if lora_steps:
        plt.plot(lora_steps, lora_losses, marker='o', linestyle='-', color='#d62728', 
                 linewidth=2, markersize=5, label='Original LoRA Loss')
                 
    # BitLoRA 그래프 (파란색 선)
    if bitlora_steps:
        plt.plot(bitlora_steps, bitlora_losses, marker='s', linestyle='-', color='#1f77b4', 
                 linewidth=2, markersize=5, label='BitLoRA (1.58-bit) Loss')

    # 4. 그래프 꾸미기
    plt.title('Training Loss Comparison: LoRA vs BitLoRA', fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    
    # 배경 그리드
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # 범례 표시
    plt.legend(fontsize=12)
    
    # 레이아웃 조절
    plt.tight_layout()

    # 5. 이미지 파일로 저장
    save_img_path = "loss_comparison_curve.png"
    plt.savefig(save_img_path, dpi=300)
    print(f"✅ 비교 그래프가 '{save_img_path}' 파일로 고화질 저장되었습니다!")
    
    # 6. 화면에 출력
    plt.show()