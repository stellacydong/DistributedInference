import torch
from transformers import AutoTokenizer
from petals import DistributedBloomForCausalLM

# Ensure at least two GPUs are available
assert torch.cuda.device_count() >= 2, "This script requires at least two GPUs."

# Load tokenizer
model_name = "bigscience/bloom-560m"  # Example model name; replace with your model
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the distributed model
model = DistributedBloomForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

# Sample input text
text = "Hello, world! How is it going?"

# Tokenize the input
inputs = tokenizer(text, return_tensors="pt")

# Set either max_length or max_new_tokens
# Here, we'll specify max_new_tokens to control the number of new tokens generated
output = model.generate(inputs["input_ids"], max_new_tokens=20)  # Adjust as needed

# Decode the output to text
result = tokenizer.decode(output[0], skip_special_tokens=True)
print("Output:", result)
