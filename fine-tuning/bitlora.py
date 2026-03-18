import importlib
from replace_bitlora import BitLoraLayer

original = importlib.import_module("peft")
original.tuners.lora.layer.LoraLayer.update_layer = (
    BitLoraLayer.update_layer
)

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset

device = torch.accelerator.current_accelerator().type if hasattr(torch, "accelerator") else "cuda"
model_id = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)

base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map=device,
    torch_dtype=torch.bfloat16
)

peft_config = LoraConfig(
    task_type=TaskType.Causal_LM,
    inference_mode=False,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)

base_model = get_peft_model(base_model, peft_config)
base_model.print_trainable_parameters()

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

base_model.gradient_checkpointing_enable()
base_model.enable_input_require_grads()

## Dataset Load
dataset_name = "medalpaca/medical_meadow_medical_flashcards"
dataset = load_dataset(dataset_name, split="train")

def generate_prompt(instruction, input_text, output_text):
    if input_text:
        prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output_text}"
    else:
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{output_text}"
    
    return prompt + tokenizer.eos_token

def process_data(samples):
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

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

tokenized_datasets = dataset.map(
    process_data, 
    batched=True, 
    remove_columns=dataset.column_names,
)

print(len(tokenized_datasets))

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


## Train Logic
training_args = TrainingArguments(
    output_dir="./medical_lora_results",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    logging_steps=30,
    do_train=True,
    do_eval=False,
    learning_rate=2e-4,
    lr_scheduler_type="linear",
    logging_steps=10,
    max_steps=200,
    save_steps=50,
    bf16=True,
    report_to="none",
    seed=42,
)

trainer = Trainer(
    model=base_model,
    args=training_args,
    train_dataset=tokenized_datasets,
    data_collator=data_collator,
)

trainer.train()

save_path = "../llama-8b-bitlora"
trainer.model.save_pretrained(save_path)