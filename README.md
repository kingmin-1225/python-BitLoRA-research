# python-BitLoRA-research

## Purpose
- LoRA(FP32) vs BitLoRA(1.58) tradeoffs
- Iso-memory evaluation (FP32 = 2bit[1.58 bit] * 16)) - 배포 기준

## Experiment

```
python ./src/train.py --adapter_type fp32 --r 8 --epochs 3 
```

## Performance Comparison

### Common Experimental Settings

**1. Environment & Data**

- **Language:** `python` 3.11+
- **Base Model:** `meta-llama/Llama-3.2-3B-Instruct`
- **Dataset:** `microsoft/orca-math-word-problems-200k` (Shuffle seed: 42)
  - Train Set: 14,500 samples
  - Validation Set: 500 samples
- **Hardware:** NVIDIA RTX 4060 Ti (8GB VRAM) (System RAM: 32GB)
- **Framework:** `peft` 0.18.1, `PyTorch` 2.10.0+cu128, `transformers` 4.52.0.dev0

**2. Training Hyperparameters**
- **Epochs:** 3
- **Max Sequence Length:** 512 tokens
- **Global Batch Size:** 16
- **Optimizer:** `paged_adamw_8bit`
- **Learning Rate:** 2e-4 (Cosine LR Scheduler)
- **LoRA Target Modules:** All linear layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`)
- **LoRA Alpha:** `r * 2` (Dropout: 0.05, seed=42)

### Results

![Loss Comparison](loss_graph.png)

| Method | Trainable Params (%) | Validation loss (epoch=1.98) | PPL |GSM8K (CoT) | Adapter size(MB) | Inference time/cost |
| :--- | :---: | :---: | :---: | :---: | :---: |  :---: | 
| **BASE MODEL**            | --    | --    | -- | 77.7  | -- | -- |
| **FP32 Adapter (r=4)**    | 0.188 | 0.460 | -- | --    | -- | -- |
| **FP32 Adapter (r=8)**    | 0.377 | 0.451 | -- | --    | -- | -- |
| **FP32 Adapter (r=16)**   | 0.751 | 0.442 | -- | --    | -- | -- |
| **Ternary Adapter (r=4)** | 0.188 | 0.472 | -- | --    | -- | -- |
| **Ternary Adapter (r=8)** | 0.377 | 0.463 | -- | --    | -- | -- |
| **Ternary Adapter (r=16)**| 0.751 | 0.452 | -- | --    | -- | -- |
| **Binary Adapter (r=4)**  | --    | --    | -- | --    | -- | -- |
| **Binary Adapter (r=8)**  | --    | --    | -- | --    | -- | -- |
| **Binary Adapter (r=16)** | --    | --    | -- | --    | -- | -- |


### Issues
- ~~`activation` 양자화를 제외하고도 학습시켜봤으나 성능차이가 없거나 미비함 (정수 연산을 위해 W1A8 선정)~~
- `unsloth`를 이용해서 학습을 가속하려고 했으나, `BitLinear`가 적용되지 않는 문제가 발생
- `activation quant` 제거