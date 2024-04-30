import torch
from transformers import AutoTokenizer
from petals import DistributedLlamaForCausalLM
import time

# Check CUDA availability
assert torch.cuda.is_available(), "CUDA is not available. Ensure your system has GPUs."

# Get the number of available GPUs
num_gpus = torch.cuda.device_count()
assert num_gpus >= 2, "This example requires at least 2 GPUs."

# Initialize the tokenizer
model_name = "petals-team/StableBeluga2"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the distributed model (without 'client_config')
try:
    model = DistributedLlamaForCausalLM.from_pretrained(
        model_name,
        # Removed 'client_config', replaced with necessary arguments
    )
except Exception as e:
    print("Error loading model:", e)
    raise

# Example prompt
prompt = "The quick brown fox jumps over the lazy dog."
inputs = tokenizer(prompt, return_tensors="pt")

# Synchronize GPUs and start the timer
torch.cuda.synchronize()  # Ensure synchronization
start_time = time.time()

# Generate text with the model
try:
    outputs = model.generate(inputs["input_ids"], max_new_tokens=50)
except Exception as e:
    print("Error during generation:", e)
    raise

# Calculate the time for inference
torch.cuda.synchronize()  # Ensure synchronization
elapsed_time = time.time() - start_time

# Decode the generated text
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Output the results
print(f"Inference completed in {elapsed_time:.2f} seconds.")
print("Generated Text:", generated_text)
