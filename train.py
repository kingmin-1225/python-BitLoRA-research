import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model
from utils import BitLinear

def replace_lora_with_bitlora(model):
    """
    PeftModel 내부의 lora_A, lora_B (nn.Linear)를 BitLinear로 교체
    """
    for name, module in model.named_modules():
        if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
            old_A = module.lora_A["default"] # "default"는 어댑터 이름
            old_B = module.lora_B["default"]
            
            new_A = BitLinear(old_A.in_features, old_A.out_features, bias=False)
            new_B = BitLinear(old_B.in_features, old_B.out_features, bias=False)
            
            new_A.weight.data = old_A.weight.data.clone()
            new_B.weight.data = old_B.weight.data.clone()
            
            new_A.to(old_A.weight.device).to(old_A.weight.dtype)
            new_B.to(old_B.weight.device).to(old_B.weight.dtype)

            new_A.weight.requires_grad = True
            new_B.weight.requires_grad = True
            
            module.lora_A["default"] = new_A
            module.lora_B["default"] = new_B
            
    print("모든 LoRA 어댑터가 BitLinear(1.58bit)로 교체되었습니다.")
    return model

## ================ MODEL LAOD ================

device = torch.accelerator.current_accelerator().type if hasattr(torch, "accelerator") else "cuda"
model_id = "microsoft/bitnet-b1.58-2B-4T"

print(">>> MODEL LOADING...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map=device,            
    torch_dtype=torch.bfloat16
)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

## ================ LORA SETTING ================

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=64,             
    lora_alpha=128,   
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)

# 2. 모델에 LoRA 주입 (이걸 해야 lora_A, lora_B가 생김!)
model = get_peft_model(model, peft_config)
print(">>> LoRA 적용 완료. 이제 BitLinear로 교체합니다...")

model = replace_lora_with_bitlora(model)

model.enable_input_require_grads() 
model.config.use_cache = False      

trainable_params = 0
all_param = 0
for _, param in model.named_parameters():
    all_param += param.numel()
    if param.requires_grad:
        trainable_params += param.numel()

print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")

if trainable_params == 0:
    raise ValueError("학습 가능한 파라미터가 없습니다! replace_lora_with_bitlora 함수를 확인하세요.")

# 3. 확인: BitLinear가 들어갔는지 출력
# print(model)


from datasets import load_dataset

dataset_name = "timdettmers/openassistant-guanaco"
print(f">>> 데이터셋 다운로드 및 로드 중: {dataset_name}...")
dataset = load_dataset(dataset_name, split="train")

# 2. 토크나이저 설정 (Llama 계열 필수 설정)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"  # 학습할 때는 오른쪽 패딩이 일반적

# 3. 데이터 전처리 함수 (토크나이징)
def process_data(samples):
    return tokenizer(
        samples["text"],
        truncation=True,
        max_length=256,  # VRAM이 부족하면 256으로 줄이세요
        padding="max_length"
    )

# 4. 데이터셋 변환 (Map 함수로 전체 적용)
print(">>> 데이터 토크나이징 진행 중...")
tokenized_datasets = dataset.map(process_data, batched=True, remove_columns=dataset.column_names)

print(f">>> 준비된 데이터 개수: {len(tokenized_datasets)}개")

## ================ TRAINING SETTING ================

from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

import torch
torch.cuda.empty_cache()

from torch.utils.data import DataLoader
from tqdm import tqdm

# 1. 학습을 위한 기본 설정
epochs = 3
learning_rate = 2e-4
batch_size = 2 
device = "cuda" if torch.cuda.is_available() else "cpu"

model.to(device)

optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

small_dataset = tokenized_datasets.select(range(100))
train_dataloader = DataLoader(
    small_dataset, 
    batch_size=batch_size, 
    shuffle=True, 
    collate_fn=data_collator
)

print(">>> 수동 학습 루프 시작!")

model.train()
for epoch in range(epochs):
    print(f"\n--- Epoch {epoch+1}/{epochs} ---")
    
    progress_bar = tqdm(train_dataloader, desc="Training")
    
    for step, batch in enumerate(progress_bar):
        inputs = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        
        optimizer.zero_grad()

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(**inputs)
            loss = outputs.loss
        
        loss.backward()
        optimizer.step()
        
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

print(">>> 학습 완료!")


# 저장 경로 설정
adapter_save_path = "./final_bitlora_adapter"

model.save_pretrained(adapter_save_path)
tokenizer.save_pretrained(adapter_save_path)

print(f">>> 학습 완료! 어댑터가 '{adapter_save_path}'에 저장되었습니다.")
print(">>> 주의: 이 폴더에는 Base Model이 포함되지 않았으므로 병합되지 않은 상태입니다.")