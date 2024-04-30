# Python script: simple-inference.py
from accelerate import Accelerator
from accelerate.utils import gather_object
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch  # Import torch to interact with CUDA
import time

# Initialize the Accelerator for multi-GPU setup
accelerator = Accelerator()

# GPT-2 Small model with 124M parameters
model_path = "gpt2"  # Using the smallest GPT-2 model
num_gpus = torch.cuda.device_count()  # Check the number of available GPUs

print(f"Using {num_gpus} GPUs for distributed processing.")

# Load the model and tokenizer
current_device = torch.device(f"cuda:{accelerator.process_index}")  # Set the correct device for this process

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map={"": current_device},  # Explicitly set the device
    torch_dtype=torch.float16,  # Use mixed precision for efficiency
)

tokenizer = AutoTokenizer.from_pretrained(model_path)

# Example prompts for testing
prompts = ["Hello, world!", "What's your favorite movie?"]

# Synchronize GPUs and start the timer
accelerator.wait_for_everyone()
start_time = time.time()

# Distribute prompts across GPUs and gather results
results = {"outputs": [], "num_tokens": 0}  # Store output and token count
with accelerator.split_between_processes(prompts) as subset_prompts:
    for prompt in subset_prompts:
        # Tokenize and move inputs to the correct device
        inputs = tokenizer(prompt, return_tensors="pt").to(current_device)

        # Generate text with the model
        generated_ids = model.generate(inputs["input_ids"], max_new_tokens=50)

        # Remove prompt from output to get the generated text
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        # Store outputs and token counts
        results["outputs"].append(generated_text)
        results["num_tokens"] += len(generated_ids[0])

# Gather results from all GPUs
results_gathered = gather_object(results)

# Synchronize processes before outputting results
accelerator.wait_for_everyone()

# Calculate inference time
elapsed_time = time.time() - start_time

# Display results and additional statistics if in the main process
if accelerator.is_main_process:
    num_tokens = sum(r["num_tokens"] for r in results_gathered)
    tokens_per_sec = num_tokens // elapsed_time

    print(f"Inference completed in {elapsed_time:.2f} seconds.")
    print(f"Tokens/sec: {tokens_per_sec}")
    print(f"Total tokens: {num_tokens}, total prompts: {len(prompts)}")

    print("Generated Outputs:")
    for i, output in enumerate(results["outputs"]):
        print(f"Output {i + 1}: {output}")
