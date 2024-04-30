import torch
from transformers import AutoTokenizer
from petals import DistributedLlamaForCausalLM  # Example model from Petals
import time

# Initialize the tokenizer for StableBeluga2
model_name = "petals-team/StableBeluga2"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Get the number of available GPUs
num_gpus = torch.cuda.device_count()

# Ensure there are at least 2 GPUs for demonstration
assert num_gpus >= 2, "This example requires at least 2 GPUs."

# Set up the initial peers for the private swarm based on the number of GPUs
initial_peers = [f"localhost:{5000 + i}" for i in range(num_gpus)]

# Load the distributed model from the private swarm
model = DistributedLlamaForCausalLM.from_pretrained(
    model_name,
    client_config={"initial_peers": initial_peers},  # Connect to private swarm
)

# Example prompt for testing
prompt = "The quick brown fox jumps over the lazy dog."
inputs = tokenizer(prompt, return_tensors="pt")

# Synchronize GPUs and start the timer
torch.cuda.synchronize()  # Ensure CUDA synchronization
start_time = time.time()

# Generate text with Petals, using distributed processing
with torch.no_grad():
    outputs = model.generate(inputs["input_ids"], max_new_tokens=50)

# Calculate the elapsed time
torch.cuda.synchronize()  # Ensure all operations are complete
elapsed_time = time.time() - start_time

# Decode the generated text
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Output the results
print(f"Inference completed in {elapsed_time:.2f} seconds.")
print("Generated Text:", generated_text)
print("Initial Peers Used:", initial_peers)
