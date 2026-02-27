import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from utils import BitLinear

def replace_lora_with_bitlora_eval(model):
    """
    추론용: lora_A, lora_B (nn.Linear)를 BitLinear로 교체합니다.
    학습된 가중치를 그대로 유지하되, 기울기 계산(requires_grad)은 끕니다.
    """
    for name, module in model.named_modules():
        if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
            old_A = module.lora_A["default"]
            old_B = module.lora_B["default"]
            
            # BitLinear 초기화
            new_A = BitLinear(old_A.in_features, old_A.out_features, bias=False)
            new_B = BitLinear(old_B.in_features, old_B.out_features, bias=False)
            
            # 학습이 완료된 가중치를 BitLinear로 복사
            new_A.weight.data = old_A.weight.data.clone()
            new_B.weight.data = old_B.weight.data.clone()
            
            # 디바이스 및 데이터 타입 맞춤
            new_A.to(old_A.weight.device).to(old_A.weight.dtype)
            new_B.to(old_B.weight.device).to(old_B.weight.dtype)

            # 추론 시에는 기울기 업데이트가 필요 없으므로 False로 설정
            new_A.weight.requires_grad = False
            new_B.weight.requires_grad = False
            
            # 교체
            module.lora_A["default"] = new_A
            module.lora_B["default"] = new_B
            
    print(">>> 모든 LoRA 어댑터가 BitLinear(1.58bit)로 교체되었습니다 (추론 모드).")
    return model

## ================ 1. MODEL & ADAPTER LOAD ================

device = torch.accelerator.current_accelerator().type if hasattr(torch, "accelerator") else "cuda"
base_model_id = "microsoft/bitnet-b1.58-2B-4T"
adapter_path = "./final_bitlora_adapter"

print(">>> 베이스 모델 로딩 중...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    device_map=device,
    torch_dtype=torch.bfloat16
)

print(">>> 토크나이저 로딩 중...")
tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(">>> PEFT 어댑터 병합 및 로딩 중...")
# 1. 일반 LoRA 형태로 학습된 가중치를 불러옵니다.
model = PeftModel.from_pretrained(base_model, adapter_path)

# 2. 로드된 가중치를 보존하면서 BitLinear로 교체합니다.
model = replace_lora_with_bitlora_eval(model)

# 3. 모델을 평가(추론) 모드로 전환합니다.
model.eval()


## ================ 2. GENERATION TEST ================

def generate_text(prompt, max_new_tokens=128):
    # 학습에 사용된 OpenAssistant Guanaco 데이터셋의 기본 프롬프트 형태 (필요에 따라 수정하세요)
    formatted_prompt = f"### Human: {prompt}\n### Assistant:"
    
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,         # 유연한 답변을 원하면 True, 단답형을 원하면 False
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            
    # 프롬프트 부분을 제외하고 생성된 텍스트만 추출
    generated_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return generated_text.strip()

print("\n========== 추론 테스트 시작 ==========")
user_query = "What are the key benefits of using 1.58-bit quantization in LLMs?"
print(f"User: {user_query}")
print("-" * 38)
response = generate_text(user_query)
print(f"Assistant: {response}")
print("======================================")