import os
import argparse
import re
import torch
from huggingface_hub import login
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM

load_dotenv()
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(hf_token)

def main():
    parser = argparse.ArgumentParser(description="LoRA Training Script")
    parser.add_argument('--adapter_type', type=str, default='fp32', choices=['fp32', 'ternary', 'binary'])
    parser.add_argument('--r', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=3)
    args = parser.parse_args()

    if args.adapter_type == "ternary":
        import importlib
        from src.replace_bitlora import BitLoraLayer158
        original = importlib.import_module("peft")
        original.tuners.lora.layer.LoraLayer.update_layer = BitLoraLayer158.update_layer
    elif args.adapter_type == "binary":
        import importlib
        from src.replace_bitlora import BitLoraLayer1
        original = importlib.import_module("peft")
        original.tuners.lora.layer.LoraLayer.update_layer = BitLoraLayer1.update_layer

    model_id = "meta-llama/Llama-3.2-3B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="cuda", 
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    base_model.config.pad_token_id = tokenizer.eos_token_id

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=args.r,
        lora_alpha=args.r * 2,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    base_model = get_peft_model(base_model, peft_config)
    base_model.enable_input_require_grads()
    base_model.print_trainable_parameters()

    def format_answer_to_gsm8k(example):
        answer = example['answer']
        numbers = re.findall(r'-?\d+(?:,\d+)*(?:\.\d+)?', answer)
        
        if numbers:
            final_answer = numbers[-1]
            example['answer'] = f"{answer}\n#### {final_answer}"
        else:
            example['answer'] = f"{answer}\n#### "
            
        return example

    orca_dataset = load_dataset("microsoft/orca-math-word-problems-200k", split="train")
    filtered_orca = orca_dataset.filter(lambda x: len(x['question']) + len(x['answer']) < 2000)
    formatted_orca = filtered_orca.map(format_answer_to_gsm8k)

    sampled_orca = formatted_orca.shuffle(seed=42).select(range(15000))
    split_dataset = sampled_orca.train_test_split(test_size=500, seed=42)

    

    def formatting_prompts_func(example):
        output_texts = []
        for i in range(len(example['question'])):
            messages = [
                {"role": "user", "content": example['question'][i]},
                {"role": "assistant", "content": example['answer'][i]}
            ]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            output_texts.append(text)
        return output_texts

    response_template = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
        mlm=False
    )

    save_path = f"./new_experiments/llama-3b-{args.adapter_type}-r{args.r}"

    training_args = SFTConfig(
        output_dir=save_path,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        bf16=True,
        logging_steps=10,

        do_eval=True,
        eval_strategy="steps",
        eval_steps=200,

        save_strategy="steps",
        save_steps=200,

        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        report_to="none",

        load_best_model_at_end=True, 
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=3,

        seed=42,
        max_seq_length=1024, 
    )

    trainer = SFTTrainer(
        model=base_model,
        args=training_args,
        train_dataset = split_dataset['train'],
        eval_dataset = split_dataset['test'],
        formatting_func=formatting_prompts_func,
        data_collator=collator,
    )

    train_result = trainer.train()

    peak_vram = torch.cuda.max_memory_allocated() / 1024**3
    print(f"Max VRAM usage: {peak_vram:.2f} GB")
    
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)
    print("Model saved to", save_path)

if __name__ == "__main__":
    main()