import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "microsoft/bitnet-b1.58-2B-4T"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cuda",
    torch_dtype=torch.bfloat16
)

print(model)

# Apply the chat template
messages = [
    {"role": "system", "content": "You are a helpful AI assistant."},
    {"role": "user", "content": "How are you?"},
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
chat_input = tokenizer(prompt, return_tensors="pt").to(model.device)

# Generate response
chat_outputs = model.generate(**chat_input, max_new_tokens=50)
response = tokenizer.decode(chat_outputs[0][chat_input['input_ids'].shape[-1]:], skip_special_tokens=True) # Decode only the response part
print("\nAssistant Response:", response)
