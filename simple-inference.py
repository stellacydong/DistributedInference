# Python script: simple-inference.py
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch import cuda
import time

# Initialize the Accelerator for multi-GPU setup
accelerator = Accelerator()

# GPT-2 Small model with 124M parameters
model_path = "gpt2"  # Using the smallest GPT-2 model
num_gpus = cuda.device_count()  # Check available GPUs

print(f"Using {num_gpus} GPUs for distributed processing.")

# Initialize the model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map={"": accelerator.process_index},
    torch_dtype=torch.float16  # Use mixed precision for efficiency
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Example prompts for testing
prompts = ["Hello, world!", "What's your favorite movie?"]  # Add your prompts

# Synchronize GPUs and start the timer
accelerator.wait_for_everyone()
start_time = time.time()

# Distribute prompts across GPUs and gather results
outputs = []
with accelerator.split_between_processes(prompts) as subset_prompts:
    for prompt in subset_prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(accelerator.device)
        generated_ids = model.generate(inputs["input_ids"], max_new_tokens=50)
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        outputs.append(generated_text)

# Wait for all processes to finish
accelerator.wait_for_everyone()

# Calculate the time taken for inference
elapsed_time = time.time() - start_time

# Display results if in the main process
if accelerator.is_main_process:
    print(f"Inference completed in {elapsed_time:.2f} seconds.")
    print("Generated Outputs:")
    for i, output in enumerate(outputs):
        print(f"Output {i + 1}: {output}")
