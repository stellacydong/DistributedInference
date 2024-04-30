import torch
from transformers import AutoTokenizer
from petals import DistributedBloomForCausalLM  # Example model

# Check if at least two GPUs are available
assert torch.cuda.device_count() >= 2, "This script requires at least two GPUs."

# Set device IDs for the GPUs you want to use
gpu1 = torch.device("cuda:0")  # First GPU
gpu2 = torch.device("cuda:1")  # Second GPU

# Load the tokenizer for the specified model
model_name = "bigscience/bloom-560m"  # Change to your desired model
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the model with full-precision (float32) to avoid half-precision issues
model = DistributedBloomForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,   # Use float32 instead of float16
    device_map={"0": gpu1, "1": gpu2}
)

# Sample text input
text = "What do you know about the theory of relativity?"

# Tokenize the input text
inputs = tokenizer(text, return_tensors="pt")

# Generate output with a specific max_new_tokens value
output = model.generate(inputs["input_ids"], max_new_tokens=50)  # Adjust as needed

# Decode and print the generated output
result = tokenizer.decode(output[0], skip_special_tokens=True)
print("Generated Text:", result)
