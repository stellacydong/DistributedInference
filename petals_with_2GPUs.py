import torch
from transformers import AutoTokenizer
from petals import DistributedBloomForCausalLM  # Or another Petals-supported model

# Ensure at least two GPUs are available
assert torch.cuda.device_count() >= 2, "This script requires at least two GPUs."

# Load tokenizer
model_name = "bigscience/bloom-560m"  # Example model name; replace with your model
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the model in distributed mode without manually specifying the device map
# Let Petals handle the distribution of layers across available GPUs/nodes
model = DistributedBloomForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

# Sample input
text = "Hello, world! How is it going?"

# Tokenize and generate output
inputs = tokenizer(text, return_tensors="pt")
output = model.generate(inputs["input_ids"])

# Decode output
result = tokenizer.decode(output[0], skip_special_tokens=True)
print("Output:", result)
