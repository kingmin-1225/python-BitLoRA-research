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
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map=device,
    torch_dtype=torch.bfloat16
)

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    task_type=TaskType.Causal_LM
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

## Train Logic
trainer = Trainer(model=model, train_dataset=dataset)

model.save_pretrained("../llama-8b-bitlora")