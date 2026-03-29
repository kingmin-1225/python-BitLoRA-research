import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] ="expandable_segments:True"
from huggingface_hub import login
from dotenv import load_dotenv

load_dotenv()
hf_token = os.getenv("HF_TOKEN")
login(hf_token)

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset

def main():
    parser = argparse.ArgumentParser(description="LoRA Training Script")
    parser.add_argument('--adapter_type', type=str, default='fp32', choices=['fp32', 'ternary', 'binary'])
    parser.add_argument('--r', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=1)
    args = parser.parse_args()

    if args.adapter_type == "ternary":
        import importlib
        from replace_bitlora import BitLoraLayer158

        original = importlib.import_module("peft")
        original.tuners.lora.layer.LoraLayer.update_layer = (
            BitLoraLayer158.update_layer
        )
    elif args.adapter_type == "binary":
        import importlib
        from replace_bitlora import BitLoraLayer1

        original = importlib.import_module("peft")
        original.tuners.lora.layer.LoraLayer.update_layer = (
            BitLoraLayer1.update_layer
        )
    
    device = torch.accelerator.current_accelerator().type if hasattr(torch, "accelerator") else "cuda"
    model_id = "meta-llama/Llama-3.2-3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=device,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=args.r,
        lora_alpha=args.r*2,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    base_model = get_peft_model(base_model, peft_config)
    base_model.print_trainable_parameters()

    base_model.gradient_checkpointing_enable()
    base_model.enable_input_require_grads()

    dataset_name = "microsoft/orca-math-word-problems-200k"
    dataset = load_dataset(dataset_name, split="train").shuffle(seed=42).select(range(15000))

    def process_data(samples):
        batch_prompts = []
        
        for q, a in zip(samples["question"], samples["answer"]):
            user_msg = str(q) if q is not None else ""
            assistant_msg = str(a) if a is not None else ""
            
            messages = [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assistant_msg}
            ]
            
            prompt = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=False
            )
            batch_prompts.append(prompt)
        
        result = tokenizer(
            batch_prompts,
            truncation=True,
            max_length=512,
            padding="max_length"
        )

        labels = result["input_ids"].copy()
        labels = [
            [-100 if token == tokenizer.pad_token_id else token for token in label]
            for label in labels
        ]
        result["labels"] = labels
        
        return result

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    tokenized_datasets = dataset.map(
        process_data, 
        batched=True, 
        remove_columns=dataset.column_names,
    )
    tokenized_datasets = tokenized_datasets.train_test_split(test_size=500, seed=42)

    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["test"]

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    save_path = f"./experiments/llama-3b-{args.adapter_type}-r{args.r}-test"

    training_args = TrainingArguments(
        max_steps=2,
        output_dir=save_path,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        do_train=True,

        # validation
        do_eval=True,
        eval_strategy="steps",
        eval_steps=200, 

        logging_steps=10,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        save_steps=200,
        bf16=True,
        report_to="none",
        seed=42,
    )

    trainer = Trainer(
        model=base_model,
        args=training_args,

        train_dataset=train_dataset,
        eval_dataset=eval_dataset,

        data_collator=data_collator,
    )

    train_result = trainer.train()

    peak_vram = torch.cuda.max_memory_allocated() / 1024**3
    print(f"Max VRAM usage: {peak_vram:.2f} GB")

    metrics = train_result.metrics
    metrics["peak_vram_gb"] = round(peak_vram, 2)

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    
    trainer.save_state()
    trainer.save_state()
    trainer.model.save_pretrained(
        save_path,
        safe_serialization=False
        )
    print("Model saved to", save_path)

if __name__ == "__main__":
    main()