import torch
from transformers import AutoTokenizer
from petals import DistributedBloomForCausalLM  # Example model


gpu_count = torch.cuda.device_count()
assert gpu_count >= 2, "This script requires at least two GPUs."

print("Available GPUs:", gpu_count)

from petals.server import start_server

# Example: Host blocks 0-15 of the 'bigscience/bloom-560m' model on a local server
start_server(
    "bigscience/bloom-560m",  # Change to the desired model
    block_indices=range(0, 16),  # Blocks to host (change as needed)
    throughput=1,  # Server throughput; adjust based on hardware
)

from transformers import AutoTokenizer
from petals import DistributedBloomForCausalLM

# Load tokenizer
model_name = "bigscience/bloom-560m"  # Change to your desired model
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the distributed model from the local server
# Connect to 'localhost' to use the local server
model = DistributedBloomForCausalLM.from_pretrained(
    model_name,
    client_config={"initial_peers": ["localhost"]},  # Connect to local server
)

# Generate output
text = "What's your favorite programming language?"
inputs = tokenizer(text, return_tensors="pt")
output = model.generate(inputs["input_ids"], max_new_tokens=50)
result = tokenizer.decode(output[0], skip_special_tokens=True)

print("Generated Text:", result)


from transformers import AutoTokenizer
from petals import DistributedBloomForCausalLM

# Connect to a private swarm with multiple servers
initial_peers = ["localhost:port1", "localhost:port2"]  # Update with your server addresses

model_name = "bigscience/bloom-560m"  # Change to your model
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = DistributedBloomForCausalLM.from_pretrained(
    model_name,
    client_config={"initial_peers": initial_peers},  # Connect to private swarm
)

# Generate output as before
text = "What's the best way to start learning Python?"
inputs = tokenizer(text, return_tensors="pt")
output = model.generate(inputs["input_ids"], max_new_tokens=50)
result = tokenizer.decode(output[0], skip_special tokens=True)

print("Generated Text:", result)


