import os
import torch
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

# Ensure PyTorch CUDA environment is configured to avoid memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

# Initialize Accelerator for multi-GPU setup
accelerator = Accelerator()

# Check how many GPUs are available and ensure a minimum count
gpu_count = torch.cuda.device_count()
assert gpu_count >= 2, "Multi-GPU setup requires at least two GPUs."
print(f"Available GPUs: {gpu_count}")

# Clear CUDA cache to free up GPU memory
torch.cuda.empty_cache()

# Hugging Face authentication token (if needed)
hf_token = "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"  # Replace with your token if accessing gated repositories

# Use 'auth_token' instead of 'use_auth_token'
model_path = "petals-team/StableBeluga2"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.float16,
    auth_token=hf_token,  # For gated repositories
)

# Load the tokenizer with the same token
tokenizer = AutoTokenizer.from_pretrained(model_path, auth_token=hf_token)

# Ensure the tokenizer has a padding token
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})  # Avoid padding issues

# Prepare inputs for multi-GPU inference
prompts = ["Example prompt 1", "Example prompt 2"]  # Adjust as needed
inputs = tokenizer(
    prompts,
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=50  # Adjust based on model context size
).to(accelerator.device)  # Ensure inputs are on the correct device

# Synchronize GPUs and start a timer to measure performance
accelerator.wait_for_everyone()
start_time = time.time()

# Generate text with multi-GPU inference
outputs = []

# Distribute prompts across available GPUs with Accelerate
with accelerator.split_between_processes(prompts) as subset_prompts:
    for prompt in subset_prompts:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(accelerator.device)

        # Generate text from the model
        generated_ids = model.generate(input_ids, max_new_tokens=50)  # Adjust based on context
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        outputs.append(generated_text)

# Synchronize processes after generation
accelerator.wait_for_everyone()

# Calculate inference time
elapsed_time = time.time() - start_time

# Display generated outputs
if accelerator.is_main_process:
    print(f"Inference time: {elapsed_time:.2f} seconds")
    print("Generated Outputs:")
    for i, output in enumerate(outputs):
        print(f"Prompt {i + 1}: {output}")
