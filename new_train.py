import os
import argparse
import torch
import re  # 정답 추출용
from tqdm import tqdm  # 진행바
from huggingface_hub import login
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM

load_dotenv()
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(hf_token)

def extract_answer(text):
    match = re.search(r"####\s*(-?\d+)", text)
    if match:
        return match.group(1)
    numbers = re.findall(r"-?\d+", text)
    if numbers:
        return numbers[-1]
    return None

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
        device_map={"": 0}, 
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
    base_model.print_trainable_parameters()

    dataset = load_dataset("openai/gsm8k", "main")

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

    save_path = f"./new_experiments/llama-3b-{args.adapter_type}-r{args.r}-test"

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
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        formatting_func=formatting_prompts_func,
        data_collator=collator,
    )

    # --- 학습 시작 ---
    print("Starting training...")
    train_result = trainer.train()

    # --- 평가 로직 (Accuracy 확인) ---
    print("\n--- Starting Evaluation ---")
    base_model.eval()
    torch.cuda.empty_cache() # VRAM 비우기

    test_ds = dataset["test"]
    num_eval_samples = 1319
    correct = 0
    total = 0

    for i in tqdm(range(num_eval_samples), desc="Evaluating"):
        question = test_ds[i]['question']
        ground_truth = extract_answer(test_ds[i]['answer'])

        eval_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = tokenizer(eval_prompt, return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = base_model.generate(
                **inputs, 
                max_new_tokens=256, 
                pad_token_id=tokenizer.eos_token_id,
                temperature=0.1,
                do_sample=False
            )
        
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_answer = decoded.split("assistant")[-1] if "assistant" in decoded else decoded
        predicted_number = extract_answer(generated_answer)

        if predicted_number == ground_truth:
            correct += 1
        total += 1

    accuracy = (correct / total) * 100
    print(f"\nFinal Test Accuracy ({total} samples): {accuracy:.2f}%")

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