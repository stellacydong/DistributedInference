import torch
from transformers import AutoTokenizer
from petals import DistributedBloomForCausalLM  # Or another model from Petals

# Check available GPUs
assert torch.cuda.device_count() >= 2, "This script requires at least two GPUs."

# Set device IDs for the GPUs you want to use
gpu1 = torch.device("cuda:0")  # First GPU
gpu2 = torch.device("cuda:1")  # Second GPU

# Load tokenizer
model_name = "bigscience/bloom-560m"  # Example model name
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load model with Petals, specifying the desired devices
model = DistributedBloomForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.float16, device_map={"0": gpu1, "1": gpu2}
)

# Sample input
text = "Hello, world! How is it going?"

# Tokenize and generate output
inputs = tokenizer(text, return_tensors="pt")
output = model.generate(inputs["input_ids"])

# Decode output
result = tokenizer.decode(output[0], skip_special_tokens=True)
print("Output:", result)
