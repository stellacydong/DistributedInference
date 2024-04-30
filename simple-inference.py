# Python script: simple-inference.py
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch  # Ensure torch is imported
import time

# Clear CUDA cache to avoid memory issues
torch.cuda.empty_cache()

# Initialize the Accelerator for multi-GPU setup
accelerator = Accelerator()

# GPT-2 Small model with 124M parameters
model_path = "gpt2"  # Using the smallest GPT-2 model
num_gpus = torch.cuda.device_count()  # Check the number of available GPUs

print(f"Using {num_gpus} GPUs for distributed processing.")

# Load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map=accelerator.device_map,  # Let Accelerator handle device mapping
    torch_dtype=torch.float16,  # Use mixed precision for efficiency
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Example prompts for testing
prompts = ["Hello, world!", "What's your favorite movie?"]

# Synchronize GPUs and start the timer
accelerator.wait_for_everyone()
start_time = time.time()

# Initialize a list to store outputs
outputs = []

# Distribute prompts across available GPUs
with accelerator.split_between_processes(prompts) as subset_prompts:
    for prompt in subset_prompts:
        # Tokenize and move inputs to the correct device
        inputs = tokenizer(prompt, return_tensors="pt").to(accelerator.device)
        
        # Generate text with the model
        generated_ids = model.generate(inputs["input_ids"], max_new_tokens=50)
        
        # Decode the generated text
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        outputs.append(generated_text)

# Synchronize processes before outputting results
accelerator.wait_for_everyone()

# Calculate inference time
elapsed_time = time.time() - start_time

# Output results only if in the main process
if accelerator.is_main_process:
    print(f"Inference completed in {elapsed_time:.2f} seconds.")
    print("Generated Outputs:")
    for i, output in enumerate(outputs):
        print(f"Output {i + 1}: {output}")
