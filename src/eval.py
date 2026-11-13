"""GSM8K (CoT) evaluation for base / FP32 / ternary / binary LoRA adapters.

Run from the repository root:

    # base model (re-measure with the same protocol as the adapters)
    python src/eval.py --adapter_type base

    # adapters: pass the raw training output (latent weights), NOT the to_eval/ copy.
    # ternary/binary weights are quantized in memory exactly like BitLinear158/BitLinear1 do in training.
    python src/eval.py --adapter_type fp32    --adapter_path new_experiments/llama-3b-fp32-r8
    python src/eval.py --adapter_type ternary --adapter_path new_experiments/llama-3b-ternary-r8
    python src/eval.py --adapter_type binary  --adapter_path new_experiments/llama-3b-binary-r8

    # quick pipeline check on 50 questions
    python src/eval.py --adapter_type base --limit 50

Results:
    eval_results/<run_name>.jsonl   per-question generations and parsed answers
    eval_results/summary.jsonl      one line per run (accuracy, settings)
"""
import argparse
import json
import os
import re
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"

# "#### 1,234.5" style final answer (GSM8K gold format, also what new_train.py trains on)
ANSWER_RE = re.compile(r"####\s*\$?\s*(-?[\d,]*\.?\d+)")
# same, but only once the number is complete (followed by whitespace) -> used to stop generation early
ANSWER_DONE_RE = re.compile(r"####\s*\$?\s*-?[\d,]*\.?\d+\.?\s")
BOXED_RE = re.compile(r"\\boxed\{\s*\$?\s*(-?[\d,]*\.?\d+)")
ANSWER_IS_RE = re.compile(r"answer is[:\s]*\$?\s*(-?[\d,]*\.?\d+)", re.IGNORECASE)
NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*")
# appended to every question (all models alike) so the base model also ends with the GSM8K answer line
FORMAT_INSTRUCTION = '\n\nSolve the problem step by step. Write the final numeric answer on the last line in the form "#### <number>".'
# calculator annotations in GSM8K train answers, e.g. <<48/2=24>>
CALC_RE = re.compile(r"<<[^>]*>>")


def normalize_number(text):
    text = text.replace(",", "").replace("$", "").strip().rstrip(".")
    try:
        return float(text)
    except ValueError:
        return None


def extract_answer(text):
    """First '#### N', then the last \\boxed{N} / 'answer is N', otherwise the last number in the text."""
    match = ANSWER_RE.search(text)
    if match:
        value = normalize_number(match.group(1))
        if value is not None:
            return value
    for pattern in (BOXED_RE, ANSWER_IS_RE):
        matches = pattern.findall(text)
        if matches:
            value = normalize_number(matches[-1])
            if value is not None:
                return value
    for candidate in reversed(NUMBER_RE.findall(text)):
        value = normalize_number(candidate)
        if value is not None:
            return value
    return None


def extract_gold(answer):
    return normalize_number(answer.split("####")[-1])


def is_correct(pred, gold):
    return pred is not None and gold is not None and abs(pred - gold) < 1e-4


def build_messages(question, shots, instruction=""):
    messages = []
    for shot in shots:
        messages.append({"role": "user", "content": shot["question"] + instruction})
        messages.append({"role": "assistant", "content": CALC_RE.sub("", shot["answer"])})
    messages.append({"role": "user", "content": question + instruction})
    return messages


def quantize_lora_weights(model, adapter_type):
    """Replace latent LoRA weights with their quantized values (same functions as training forward)."""
    import torch
    from torch import nn
    from src.utils import weight_quant1, weight_quant158

    quant_fn, max_levels = {
        "ternary": (weight_quant158, 3),
        "binary": (weight_quant1, 2),
    }[adapter_type]

    quantized, skipped = 0, 0
    with torch.no_grad():
        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if ".lora_A." not in name and ".lora_B." not in name:
                continue
            # weight_quant158 is not idempotent: re-quantizing an already ternary tensor shrinks its scale.
            # Skip tensors that are already quantized (e.g. to_eval/ outputs of deploy_weights.py).
            if torch.unique(module.weight).numel() <= max_levels:
                skipped += 1
                continue
            module.weight.copy_(quant_fn(module.weight))
            quantized += 1

    if quantized + skipped == 0:
        raise RuntimeError("No LoRA weights found to quantize - check --adapter_path")
    print(f"[quant] {adapter_type}: quantized {quantized} LoRA tensors, {skipped} were already quantized")


def load_model(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="cuda",
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )

    if args.adapter_type != "base":
        from peft import PeftModel

        # BitLinear158/BitLinear1 are nn.Linear subclasses, so the saved state dict loads into plain LoRA layers.
        model = PeftModel.from_pretrained(model, args.adapter_path)
        if args.adapter_type in ("ternary", "binary"):
            quantize_lora_weights(model, args.adapter_type)
        if args.merge:
            model = model.merge_and_unload()

    model.eval()
    return model, tokenizer


def make_stopping_criteria(tokenizer, prompt_len):
    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList

    class AnswerStop(StoppingCriteria):
        """Per-sequence stop once a complete '#### N' was generated.

        new_train.py masks <|eot_id|> in the labels (pad == eos), so fine-tuned adapters may never emit it.
        """

        def __call__(self, input_ids, scores, **kwargs):
            texts = tokenizer.batch_decode(input_ids[:, prompt_len:], skip_special_tokens=True)
            return torch.tensor(
                [ANSWER_DONE_RE.search(t) is not None for t in texts],
                dtype=torch.bool,
                device=input_ids.device,
            )

    return StoppingCriteriaList([AnswerStop()])


def main():
    parser = argparse.ArgumentParser(description="GSM8K evaluation")
    parser.add_argument("--adapter_type", type=str, required=True, choices=["base", "fp32", "ternary", "binary"])
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument("--n_shot", type=int, default=0, help="few-shot examples from GSM8K train (Meta reports 8-shot)")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--limit", type=int, default=None, help="evaluate only the first N test questions")
    parser.add_argument("--merge", action="store_true", help="merge adapter into base weights (faster, bf16 rounding)")
    parser.add_argument("--no_instruction", action="store_true",
                        help="do not append the '#### <number>' format instruction to questions")
    parser.add_argument("--output_dir", type=str, default="eval_results")
    args = parser.parse_args()

    if args.adapter_type != "base" and not args.adapter_path:
        parser.error("--adapter_path is required unless --adapter_type base")

    from dotenv import load_dotenv
    from huggingface_hub import login

    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        login(hf_token)

    import torch
    from datasets import load_dataset

    dataset = load_dataset("openai/gsm8k", "main")
    test = dataset["test"]
    if args.limit:
        test = test.select(range(min(args.limit, len(test))))
    shots = [dataset["train"][i] for i in range(args.n_shot)]

    model, tokenizer = load_model(args)
    stop_ids = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]

    instruction = "" if args.no_instruction else FORMAT_INSTRUCTION
    prompts = [
        # fixed date_string: the Llama 3.2 template otherwise injects today's date -> prompts change day to day
        tokenizer.apply_chat_template(build_messages(q, shots, instruction), tokenize=False,
                                      add_generation_prompt=True, date_string="26 Jul 2024")
        for q in test["question"]
    ]
    # longest first: similar lengths per batch -> less padding, OOM shows up on the first batch
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]), reverse=True)
    generations = [None] * len(prompts)

    start = time.time()
    for b in range(0, len(order), args.batch_size):
        idx = order[b:b + args.batch_size]
        # chat template already contains <|begin_of_text|>
        inputs = tokenizer([prompts[i] for i in idx], return_tensors="pt", padding=True,
                           add_special_tokens=False).to(model.device)
        prompt_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                eos_token_id=stop_ids,
                pad_token_id=tokenizer.eos_token_id,
                stopping_criteria=make_stopping_criteria(tokenizer, prompt_len),
            )
        texts = tokenizer.batch_decode(out[:, prompt_len:], skip_special_tokens=True)
        for i, text in zip(idx, texts):
            generations[i] = text
        done = min(b + args.batch_size, len(order))
        print(f"[{done}/{len(order)}] {time.time() - start:.0f}s", flush=True)
    elapsed = time.time() - start

    if args.adapter_type == "base":
        run_name = "base"
    else:
        parent, leaf = os.path.split(os.path.normpath(args.adapter_path))
        # .../ternary-r8/checkpoint-1800 -> ternary-r8-checkpoint-1800
        run_name = f"{os.path.basename(parent)}-{leaf}" if leaf.startswith("checkpoint") else leaf
    if args.merge:
        run_name += "-merged"
    if args.n_shot:
        run_name += f"-{args.n_shot}shot"
    if args.no_instruction:
        run_name += "-noinst"
    os.makedirs(args.output_dir, exist_ok=True)

    correct = 0
    with open(os.path.join(args.output_dir, f"{run_name}.jsonl"), "w", encoding="utf-8") as f:
        for sample, text in zip(test, generations):
            pred = extract_answer(text)
            gold = extract_gold(sample["answer"])
            ok = is_correct(pred, gold)
            correct += ok
            f.write(json.dumps({
                "question": sample["question"],
                "gold": gold,
                "pred": pred,
                "correct": ok,
                "has_####": ANSWER_RE.search(text) is not None,
                "generation": text,
            }, ensure_ascii=False) + "\n")

    accuracy = 100 * correct / len(test)
    summary = {
        "run": run_name,
        "adapter_type": args.adapter_type,
        "adapter_path": args.adapter_path,
        "n_shot": args.n_shot,
        "instruction": not args.no_instruction,
        "merged": args.merge,
        "n": len(test),
        "correct": correct,
        "accuracy": round(accuracy, 2),
        "seconds": round(elapsed, 1),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
    }
    with open(os.path.join(args.output_dir, "summary.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(summary) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
