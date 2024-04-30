import torch
from transformers import AutoTokenizer
from petals import DistributedBloomForCausalLM  # Example model

# Ensure at least two GPUs are available
gpu_count = torch.cuda.device_count()
assert gpu_count >= 2, "This script requires at least two GPUs."

print("Available GPUs:", gpu_count)

try:
    from petals.server import start_server  # Attempt to import `start_server`
except ImportError:
    print("Error: 'start_server' not found in 'petals.server'.")
    print("Ensure 'petals' is installed correctly and the function is available.")
    raise  # Rethrow the exception to indicate failure

# If 'start_server' is available, start a local server
if 'start_server' in globals():
    # Example: Host blocks 0-15 of the 'bigscience/bloom-560m' model on a local server
    start_server(
        "bigscience/bloom-560m",  # Change to the desired model
        block_indices=range(0, 16),  # Blocks to host (change as needed)
        throughput=1,  # Server throughput; adjust based on hardware
    )

# Client code to connect to the local server
model_name = "bigscience/bloom-560m"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Connect to 'localhost' to use the local server
model = DistributedBloomForCausalLM.from_pretrained(
    model_name,
    client_config={"initial_peers": ["localhost"]},  # Connect to the local server
)

# Generate output
text = "What's your favorite programming language?"
inputs = tokenizer(text, return_tensors="pt")
output = model.generate(inputs["input_ids"], max_new_tokens=50)
result = tokenizer.decode(output[0], skip_special_tokens=True)

print("Generated Text:", result)
