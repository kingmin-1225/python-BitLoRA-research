import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def main():
    # 1. Base 모델 및 어댑터 경로 설정
    base_model_id = "microsoft/bitnet-b1.58-2B-4T"
    adapter_path = "./medical_bitlora_adapter_final" # 방금 학습이 끝나고 저장된 폴더

    print(">>> 모델 로딩 중...")
    # 2. Base 모델 로드
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        device_map="cuda",
        torch_dtype=torch.bfloat16
    )

    # 3. Base 모델에 학습된 LoRA 어댑터 씌우기
    print(">>> 어댑터 씌우는 중...")
    model = PeftModel.from_pretrained(base_model, adapter_path)

    # 추론 시에는 캐시 기능을 켜서 가속
    model.config.use_cache = True 
    model.eval() # 평가 모드로 전환

    # 4. 프롬프트 템플릿 (학습할 때와 정확히 동일한 포맷이어야 합니다)
    def generate_prompt(instruction, input_text=""):
        if input_text:
            return f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
        else:
            return f"### Instruction:\n{instruction}\n\n### Response:\n"

    # 5. 질문하기 
    question = input() ##  ex) What are the common symptoms of myocardial infarction?
    prompt = generate_prompt(question)

    print(f"\n[질문]: {question}")
    print("-" * 50)
    print(">>> 모델이 답변을 생성하고 있습니다...\n")

    # 6. 토크나이징 및 생성
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,    # 생성할 최대 토큰 수
            temperature=0.3,       # 답변의 창의성 조절 (의학 정보는 낮게)
            repetition_penalty=1.2 # 같은 말 반복 방지
        )

    # 7. 결과 출력 (입력 프롬프트 이후의 생성된 텍스트만 잘라서 출력)
    input_length = inputs["input_ids"].shape[1]
    response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)

    print(f"[답변]: {response.strip()}")
    print("-" * 50)

if __name__ == '__main__':
    main()