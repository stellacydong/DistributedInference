import torch
from transformers import AutoTokenizer
from petals import DistributedBloomForCausalLM  # Example model

# Check if multiple GPUs are available
assert torch.cuda.device_count() >= 2, "This script requires at least two GPUs."

# Load the tokenizer
model_name = "bigscience/bloom-560m"  # Change to your preferred model
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the distributed model with Petals
model = DistributedBloomForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16  # Or use torch.float32 if float16 causes issues
)

# Sample input text
text = "What do you know about quantum physics?"

# Tokenize the input text
inputs = tokenizer(text, return_tensors="pt")

# Generate output with Petals, using specified max_new_tokens to avoid assertion errors
output = model.generate(inputs["input_ids"], max_new_tokens=50)  # Adjust as needed

# Decode and print the generated output
result = tokenizer.decode(output[0], skip_special_tokens=True)
print("Generated Text:", result)
