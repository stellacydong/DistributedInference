from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time

# Initialize Accelerator for multi-GPU support
accelerator = Accelerator()

# Check the number of GPUs available
print(f"Number of GPUs: {torch.cuda.device_count()}")

# Sample prompts for testing
prompts = ["What's the capital of France?", "What is the meaning of life?"]

# Load a pre-trained LLM and tokenizer
model_name = "gpt2"  # Example model; ensure the model supports multi-GPU
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Ensure the tokenizer has a padding token
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})  # Adding a default padding token

# Configure the model with Hugging Face Accelerate
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # Use float16 for efficient GPU utilization
    device_map="auto",  # Let Accelerate manage the device mapping
)

# Prepare the input for inference with padding and truncation
inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)

# Use Accelerate to manage multi-GPU inference
outputs = []

# Distribute prompts across available GPUs
with accelerator.split_between_processes(prompts) as prompt_subset:
    for prompt in prompt_subset:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(accelerator.device)
        generated_ids = model.generate(input_ids, max_new_tokens=20)  # Adjust as needed
        output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        outputs.append(output_text)

# Synchronize processes and gather results
accelerator.wait_for_everyone()

# If this process is the main process, display the generated outputs
if accelerator.is_main_process:
    print("Generated Outputs:")
    for i, output in enumerate(outputs):
        print(f"Prompt {i + 1}: {output}")
