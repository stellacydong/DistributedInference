# Python script for multi-GPU setup with Petals

from accelerate import Accelerator
from transformers import AutoTokenizer
from petals.client import DistributedBloomForCausalLM  # Alternative import path
import torch  # For CUDA interaction
import time

# Initialize the Accelerator for multi-GPU setup
accelerator = Accelerator()

# Check the number of GPUs
num_gpus = torch.cuda.device_count()

print(f"Using {num_gpus} GPUs for distributed processing.")

# Load a Petals-supported model (StableBeluga2 as an example)
model_path = "petals-team/StableBeluga2"

# Set the appropriate device for multi-GPU processing
current_device = torch.device(f"cuda:{accelerator.process_index}")

# Load the model with error handling to check for import issues
try:
    model = DistributedBloomForCausalLM.from_pretrained(
        model_path,
        device_map={"": current_device},  # Explicitly set the device
        torch_dtype=torch.float16,  # Mixed precision for efficiency
    )
except ImportError:
    print("Failed to import DistributedBloomForCausalLM. Check if the module is installed and up to date.")

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Example prompts for testing
prompts = ["Hello, world!", "What's your favorite movie?", "Tell me a story."]

# Synchronize GPUs and start the timer
accelerator.wait_for_everyone()
start_time = time.time()

# Distribute prompts across GPUs and gather results, tracking which GPU handles what
outputs = []
gpu_usage = []  # List to store usage information
with accelerator.split_between_processes(prompts) as subset_prompts:
    gpu_start_time = time.time()  # Start time for GPU processing
    for prompt in subset_prompts:
        # Log the GPU being used and the prompt
        print(f"Processing prompt '{prompt}' on GPU {accelerator.process_index}")
        
        # Tokenize and move inputs to the correct device
        inputs = tokenizer(prompt, return_tensors="pt").to(current_device)
        
        # Generate text with the model
        generated_ids = model.generate(inputs["input_ids"], max_new_tokens=50)
        
        # Decode the generated text
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        # Store the generated text and GPU usage information
        outputs.append(generated_text)
        
        # Track the number of tokens generated
        num_tokens_generated = len(generated_ids[0])
        
        # Calculate the time taken to generate tokens
        gpu_elapsed_time = time.time() - gpu_start_time
        
        # Compute tokens per second for this GPU
        tokens_per_second = num_tokens_generated / gpu_elapsed_time
        
        # Record GPU usage information
        gpu_usage.append({
            "gpu": accelerator.process_index,
            "tokens_generated": num_tokens_generated,
            "time_elapsed": gpu_elapsed_time,
            "tokens_per_second": tokens_per_second
        })

# Synchronize processes before outputting results
accelerator.wait_for_everyone()

# Calculate the time taken for inference
elapsed_time = time.time() - start_time

# Display results if in the main process
if accelerator.is_main_process:
    print(f"Inference completed in {elapsed_time:.2f} seconds.")
    print("Generated Outputs:")
    for i, output in enumerate(outputs):
        print(f"Output {i + 1}: {output}")
    
    # Print GPU usage information
    for usage in gpu_usage:
        print(f"GPU {usage['gpu']} generated {usage['tokens_generated']} tokens in {usage['time_elapsed']:.2f} seconds, achieving {usage['tokens_per_second']:.2f} tokens per second.")
