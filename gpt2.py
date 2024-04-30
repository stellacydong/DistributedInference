import torch
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

# Initialize Accelerator for multi-GPU setup
accelerator = Accelerator()

# Check how many GPUs are available
gpu_count = torch.cuda.device_count()
print(f"Available GPUs: {gpu_count}")

assert gpu_count >= 2, "Multi-GPU setup requires at least two GPUs."

# Hugging Face token for authentication (if needed)
hf_token = "YOUR_HF_TOKEN"  # Replace with your token if accessing gated repositories

# Load a pre-trained LLM and tokenizer
model_name = "gpt2"  # Example; change to your desired model
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)

# Configure the model for multi-GPU
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",  # Let Accelerate map GPUs
    torch_dtype=torch.float16,  # Efficient GPU usage
    use_auth_token=hf_token  # For accessing gated repositories
)

# Ensure the tokenizer has a padding token
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})  # Add a padding token


# Example: Set up a local server hosting specific model blocks
from petals.server import start_server  # Ensure this function is available

# Start a local server to host model blocks
start_server(
    "bigscience/bloom-560m",  # Change to your model
    block_indices=range(0, 16),  # Blocks to host (adjust as needed)
    throughput=1,  # Adjust based on your hardware
)



# Connect to a private swarm with multiple servers
# Update 'initial_peers' with your server addresses or 'localhost' for local servers
initial_peers = ["localhost:port1", "localhost:port2"]  # Adjust as needed

# Load the distributed model from the private swarm
model_name = "bigscience/bloom-560m"  # Change to your desired model
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)

model = DistributedBloomForCausalLM.from_pretrained(
    model_name,
    client_config={"initial_peers": initial_peers}  # Connect to the private swarm
)

# Generate text from a given prompt
text = "What is the capital of France?"  # Example prompt
inputs = tokenizer(text, return_tensors="pt", padding=True).to(torch.cuda.current_device())  # Ensure correct device

# Generate output using the distributed LLM
output = model.generate(inputs["input_ids"], max_new_tokens=50)  # Adjust max_new_tokens as needed
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print("Generated Text:", generated_text)

