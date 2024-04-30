import torch
from transformers import AutoTokenizer
from petals import DistributedBloomForCausalLM  # Example model


gpu_count = torch.cuda.device_count()
assert gpu_count >= 2, "This script requires at least two GPUs."

print("Available GPUs:", gpu_count)

# Load the tokenizer for the specified model
model_name = "bigscience/bloom-560m"  # Change to your desired model
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the model with full-precision (float32) to avoid half-precision issues
model = DistributedBloomForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32   # Use float32 instead of float16
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
