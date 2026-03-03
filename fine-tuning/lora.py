import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

model_id = "microsoft/bitnet-b1.58-2B-4T"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cuda",
    torch_dtype=torch.bfloat16
)

# 1. 의학 플래시카드 데이터셋 로드
dataset_name = "medalpaca/medical_meadow_medical_flashcards"
print(f">>> 데이터셋 다운로드 중: {dataset_name}...")
dataset = load_dataset(dataset_name, split="train")

# 2. 프롬프트 템플릿 정의 (Alpaca 스타일)
# 모델이 질문과 답변의 경계를 인식할 수 있도록 포맷을 맞춰줍니다.
def generate_prompt(instruction, input_text, output_text):
    if input_text:
        prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output_text}"
    else:
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{output_text}"
    
    # 모델이 답변을 끝맺을 수 있도록 마지막에 EOS 토큰을 반드시 추가합니다.
    return prompt + tokenizer.eos_token

# 3. 데이터 전처리 함수 (토크나이징)
def process_data(samples):
    # 배치(batch) 단위로 프롬프트 생성
    prompts = [
        generate_prompt(inst, inp, out)
        for inst, inp, out in zip(samples["instruction"], samples["input"], samples["output"])
    ]
    
    return tokenizer(
        prompts,
        truncation=True,
        max_length=512, 
        padding="max_length"
    )

# 토크나이저 패딩 설정 (이전 코드와 동일)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# 4. 데이터셋 변환
print(">>> 데이터 토크나이징 진행 중...")
tokenized_datasets = dataset.map(
    process_data, 
    batched=True, 
    remove_columns=dataset.column_names # 기존 텍스트 컬럼은 지우고 텐서(input_ids 등)만 남김
)

print(f">>> 준비된 의학 QA 데이터 개수: {len(tokenized_datasets)}개")

peft_config = LoraConfig(
    task_type="CAUSAL_LM",
    inference_mode=False,
    r=8,             
    lora_alpha=16,   
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)

model = get_peft_model(model, peft_config) ## base weight (frozen) + adapter
model.print_trainable_parameters()

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

model.gradient_checkpointing_enable()
model.enable_input_require_grads() # PEFT와 Gradient Checkpointing을 함께 쓸 때 필수

# 2. Data Collator 설정 (Next token prediction을 위해 mlm=False)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 3. Training Arguments 셋팅 (8GB 영혼의 최적화)
training_args = TrainingArguments(
    output_dir="./medical_lora_results",
    # --- 메모리 방어 핵심 세팅 ---
    per_device_train_batch_size=1,      # 한 번에 1개씩만 GPU에 올림
    gradient_accumulation_steps=8,      # 대신 8번 모아서 가중치 업데이트 (실질적 배치 사이즈 8)
    gradient_checkpointing=True,        # 활성화(Activation) 메모리를 줄이는 마법의 세팅
    optim="paged_adamw_8bit",           # 옵티마이저가 먹는 메모리(상태값)를 8비트로 압축
    # ---------------------------
    learning_rate=2e-4,
    lr_scheduler_type="cosine",         # 학습률을 부드럽게 감소시킴
    logging_steps=10,                   # 10 스텝마다 loss 출력
    max_steps=200,                      # 테스트용으로 일단 200 스텝만 진행 (잘 돌아가면 num_train_epochs=1 로 변경하세요)
    save_steps=50,                      # 50 스텝마다 체크포인트 저장
    bf16=True,                          # BitNet이 bfloat16이므로 동일하게 맞춤
    report_to="none"                    # wandb 등 외부 로깅 끄기
)

type(model).is_quantized = property(lambda self: False)
type(model.base_model).is_quantized = property(lambda self: False)

# 4. Trainer 객체 생성
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    data_collator=data_collator,
)

print(">>> 🚀 본격적인 학습을 시작합니다! (작업 관리자에서 VRAM을 모니터링하세요)")
trainer.train()

# 5. 학습 완료 후 최종 어댑터 저장
save_path = "./medical_lora_adapter_final"
trainer.model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

print(f">>> 🎉 학습 완료! 어댑터가 '{save_path}'에 안전하게 저장되었습니다.")