# Python script: simple-inference.py
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time

from transformers.utils import logging
logging.get_logger("transformers").setLevel(logging.ERROR)

# Initialize the Accelerator for multi-GPU setup
accelerator = Accelerator()

# GPT-2 Small model with 124M parameters
model_path = "gpt2"  # Using the smallest GPT-2 model
num_gpus = torch.cuda.device_count()  # Check the number of available GPUs

print(f"Using {num_gpus} GPUs for distributed processing.")

# Load the model and tokenizer
current_device = torch.device(f"cuda:{accelerator.process_index}")

print(current_device)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map={"": current_device},  # Explicitly set the device
    torch_dtype=torch.float16,  # Use mixed precision for efficiency
)

tokenizer = AutoTokenizer.from_pretrained(model_path)

# Example prompts for testing
prompts = ["Hello, world!", "What's next generation of decentralized AI?"]

# Synchronize GPUs and start the timer
accelerator.wait_for_everyone()
start_time = time.time()

# Distribute prompts across GPUs and gather results
outputs = []
total_tokens = 0
with accelerator.split_between_processes(prompts) as subset_prompts:
    for prompt in subset_prompts:
        # Tokenize and move inputs to the correct device
        inputs = tokenizer(prompt, return_tensors="pt").to(current_device)
        
        # Generate text with the model
        generated_ids = model.generate(inputs["input_ids"], max_new_tokens=50)
        
        # Decode the generated text
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        outputs.append(generated_text)

        # Count the total number of tokens generated
        total_tokens += len(generated_ids[0])

# Synchronize processes before outputting results
accelerator.wait_for_everyone()

# Calculate the time taken for inference
elapsed_time = time.time() - start_time

# Calculate tokens per second
tokens_per_second = total_tokens / elapsed_time

# Calculate the average time per token
average_time_per_token = elapsed_time / total_tokens

# Display results if in the main process
if accelerator.is_main_process:
    print(f"Inference completed in {elapsed_time:.2f} seconds.")
    print(f"Tokens per second: {tokens_per_second:.2f}")
    print(f"Average time per token: {average_time_per_token:.4f} seconds.")
    print("Generated Outputs:")
    for i, output in enumerate(outputs):
        print(f"Output {i + 1}: {output}")
