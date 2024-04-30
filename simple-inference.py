from accelerate import Accelerator
from accelerate.utils import gather_object
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time
import statistics

# Initialize Accelerator for distributed setup
accelerator = Accelerator()

# Set some sample prompts
prompts = [
    "The King is dead. Long live the Queen.",
    "It was a bright cold day in April, and the clocks were striking thirteen.",
    "As Gregor Samsa awoke one morning, he found himself transformed into a gigantic insect.",
] * 10  # Duplicate prompts for more test data

# Load a base model and tokenizer
model_path = "meta-llama/Llama-2-7b-hf"  # Example model, change to your desired model
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map=accelerator.device,  # Use Accelerator to map devices
    torch_dtype=torch.float16  # Use float16 for GPU efficiency; consider float32 if errors occur
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Synchronize GPUs and start the timer
accelerator.wait_for_everyone()
start = time.time()

# Use accelerator to distribute tasks among processes
results = []

with accelerator.split_between_processes(prompts) as subset_prompts:
    local_results = {"outputs": [], "num_tokens": 0}

    # Generate responses for each prompt in the subset
    for prompt in subset_prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(accelerator.device)
        output = model.generate(inputs["input_ids"], max_new_tokens=50)  # Adjust max_new_tokens as needed
        output = output[0][len(inputs["input_ids"][0]):]  # Exclude prompt tokens

        local_results["outputs"].append(tokenizer.decode(output, skip_special_tokens=True))
        local_results["num_tokens"] += len(output)

    results.append(local_results)

# Gather results from all processes
results_gathered = gather_object(results)

if accelerator.is_main_process:
    # Calculate tokens per second and other statistics
    total_time = time.time() - start
    total_tokens = sum([r["num_tokens"] for r in results_gathered])

    print(f"Tokens per second: {total_tokens // total_time}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Total tokens: {total_tokens}")
    print(f"Total prompts: {len(prompts)}")
