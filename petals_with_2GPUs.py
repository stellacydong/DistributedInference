import torch

# Check for available GPUs
gpu_count = torch.cuda.device_count()
assert gpu_count >= 2, f"Expected at least two GPUs, but found {gpu_count}."

print("Available GPUs:", gpu_count)

from transformers import AutoTokenizer
from petals import DistributedBloomForCausalLM  # Example model; other models may be used

# Load tokenizer
model_name = "bigscience/bloom-560m"  # Change to your desired model
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the model with Petals; this handles distribution across available GPUs/nodes
model = DistributedBloomForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # Use float16 to save memory on GPUs
)


# Sample input text
text = "What is the meaning of life?"

# Tokenize the input
inputs = tokenizer(text, return_tensors="pt")

# Generate output with Petals, specifying a maximum number of new tokens
output = model.generate(inputs["input_ids"], max_new_tokens=20)  # Adjust as needed

# Decode the output
result = tokenizer.decode(output[0], skip_special_tokens=True)
print("Generated Text:", result)

