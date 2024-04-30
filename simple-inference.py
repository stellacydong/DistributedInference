from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time

# Clear CUDA cache to free up GPU memory
torch.cuda.empty_cache()

# Initialize Accelerator for multi-GPU setup
accelerator = Accelerator()

# Token for Hugging Face authentication (replace with your token)
hf_token = "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"

# Set some sample prompts for testing
prompts = ["Example prompt 1", "Example prompt 2"]  # Change as needed

# Load the model and tokenizer with authentication
model_path = "petals-team/StableBeluga2"  # Example model; ensure you have access
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",  # Automatic device mapping by Accelerate
    torch_dtype=torch.bfloat16,  # Efficient GPU usage
    use_auth_token=hf_token,  # Authentication for gated repositories
)

tokenizer = AutoTokenizer.from_pretrained(
    model_path, use_auth_token=hf_token
)

# Ensure the tokenizer has a padding token to avoid padding errors
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})  # Add a padding token

# Prepare input for multi-GPU inference
inputs = tokenizer(
    prompts,
    return_tensors="pt",
    padding=True,  # Ensure proper padding
    truncation=True,  # Ensure proper truncation
    max_length=50  # Adjust as needed
)

# Move input tensors to the correct device to avoid device mismatch errors
inputs = {key: tensor.to(accelerator.device) for key, tensor in inputs.items()}

# Synchronize GPUs and start a timer for measuring performance
accelerator.wait_for_everyone()
start_time = time.time()

# Generate text using multi-GPU inference
outputs = []

# Distribute prompts across GPUs using Accelerate
with accelerator.split_between_processes(prompts) as subset_prompts:
    for prompt in subset_prompts:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(accelerator.device)  # Ensure the correct device
        
        # Generate text from the model
        generated_ids = model.generate(input_ids, max_new_tokens=50)  # Adjust as needed
        
        # Ensure the generated output is on the same device to avoid device mismatch
        generated_text = tokenizer.decode(generated_ids[0].to(accelerator.device), skip_special_tokens=True)
        
        outputs.append(generated_text)

# Synchronize processes after generation to avoid issues with data collection
accelerator.wait_for_everyone()

# Calculate the elapsed time
elapsed_time = time.time() - start_time

# Display generated outputs
if accelerator.is_main_process:
    print(f"Inference time: {elapsed_time:.2f} seconds")
    print("Generated Outputs:")
    for i, output in enumerate(outputs):
        print(f"Prompt {i + 1}: {output}")
