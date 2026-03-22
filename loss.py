import json
import matplotlib.pyplot as plt

file_path = "llama-3b-fp16_lora-r4/checkpoint-2718/trainer_state.json" 
file_path2 = "llama-3b-bit_lora-r8/checkpoint-2718/trainer_state.json" 
try:
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

    plt.figure(figsize=(20, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, losses, marker='o', markersize=4, linestyle='-', color='#467599', label='LoRA')
    plt.subplot(1, 2, 2)
    plt.plot(eval_epochs, eval_losses, marker='o', markersize=4, linestyle='-', color='#467599', label='LoRA')

    with open(file_path2, "r", encoding="utf-8") as f:
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
    plt.plot(epochs, losses, marker='o', markersize=4, linestyle='-', color='#9ed8db', label='BitLoRA')

    plt.title('Training Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(eval_epochs, eval_losses, marker='o', markersize=4, linestyle='-', color='#9ed8db', label='BitLoRA')
    plt.title('Validation Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    plt.savefig('loss_graph.png', dpi=300)
    
    plt.show()

except FileNotFoundError:
    print(f"No file in {file_path}")