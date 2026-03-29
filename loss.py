import json
import matplotlib.pyplot as plt

ternary_paths = [
    "experiments/llama-3b-ternary-r4/checkpoint-2721/trainer_state.json",
    "experiments/llama-3b-ternary-r8/checkpoint-2721/trainer_state.json",
    "experiments/llama-3b-ternary-r16-W2A8/checkpoint-2721/trainer_state.json",
]

binary_paths = [
    "experiments/llama-3b-binary-r4/checkpoint-2721/trainer_state.json",
    "experiments/llama-3b-binary-r8/checkpoint-2721/trainer_state.json",
]

fp16_paths = [
    "experiments/llama-3b-fp32-r4/checkpoint-2718/trainer_state.json",
    "experiments/llama-3b-fp32-r8/checkpoint-2718/trainer_state.json",
    "experiments/llama-3b-fp32-r16/checkpoint-2721/trainer_state.json",
]

try:
    plt.figure(figsize=(20, 5))
    for i, file_path in enumerate(ternary_paths):
        with open(file_path, "r", encoding="utf-8") as f:
            state = json.load(f)
        log_history = state.get("log_history", [])

        epochs = []
        losses = []
        eval_epochs = []
        eval_losses = []
        learning_rates = []

        for log in log_history:
            if "loss" in log and "epoch" in log:
                epochs.append(log["epoch"])
                losses.append(log["loss"])
                learning_rates.append(log["learning_rate"])
            if "eval_loss" in log:
                eval_epochs.append(log["epoch"])
                eval_losses.append(log["eval_loss"])

        plt.subplot(1, 2, 1)
        plt.plot(epochs, losses, marker='o', markersize=4, linestyle='-', label=f'Ternary LoRA r={2**(i+2)}')
        plt.subplot(1, 2, 2)
        plt.plot(eval_epochs, eval_losses, marker='o', markersize=4, linestyle='-', label=f'Ternary LoRA r={2**(i+2)}')

    for i, file_path in enumerate(binary_paths):
        with open(file_path, "r", encoding="utf-8") as f:
            state = json.load(f)
        log_history = state.get("log_history", [])

        epochs = []
        losses = []
        eval_epochs = []
        eval_losses = []
        learning_rates = []

        for log in log_history:
            if "loss" in log and "epoch" in log:
                epochs.append(log["epoch"])
                losses.append(log["loss"])
                learning_rates.append(log["learning_rate"])
            if "eval_loss" in log:
                eval_epochs.append(log["epoch"])
                eval_losses.append(log["eval_loss"])

        plt.subplot(1, 2, 1)
        plt.plot(epochs, losses, marker='o', markersize=4, linestyle='-', label=f'Binary LoRA r={2**(i+2)}')
        plt.subplot(1, 2, 2)
        plt.plot(eval_epochs, eval_losses, marker='o', markersize=4, linestyle='-', label=f'Binary LoRA r={2**(i+2)}')

    for i, file_path in enumerate(fp16_paths):
        with open(file_path, "r", encoding="utf-8") as f:
            state = json.load(f)
        log_history = state.get("log_history", [])

        epochs = []
        losses = []
        eval_epochs = []
        eval_losses = []
        learning_rates = []

        for log in log_history:
            if "loss" in log and "epoch" in log:
                epochs.append(log["epoch"])
                losses.append(log["loss"])
                learning_rates.append(log["learning_rate"])
            if "eval_loss" in log:
                eval_epochs.append(log["epoch"])
                eval_losses.append(log["eval_loss"])

        plt.subplot(1, 2, 1)
        plt.plot(epochs, losses, marker='o', markersize=4, linestyle='-', label=f'LoRA r={2**(i+2)}')
        plt.subplot(1, 2, 2)
        plt.plot(eval_epochs, eval_losses, marker='o', markersize=4, linestyle='-', label=f'LoRA r={2**(i+2)}')

    plt.subplot(1, 2, 1)
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.title('Validation Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    plt.savefig('loss_graph.png', dpi=300)
    
    plt.show()

except FileNotFoundError:
    print(f"No file in {file_path}")