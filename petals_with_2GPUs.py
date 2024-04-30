import torch

# Check PyTorch version
print("PyTorch Version:", torch.__version__)

# Check CUDA version
print("CUDA Version:", torch.version.cuda)

# Check available GPUs
gpu_count = torch.cuda.device_count()
print("Available GPUs:", gpu_count)


from transformers import AutoTokenizer
from petals import DistributedBloomForCausalLM  # Example model

# Load tokenizer
model_name = "bigscience/bloom-560m"  # Change to your desired model
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the model with full-precision (float32) instead of half-precision (float16)
model = DistributedBloomForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32  # Change to float32 to avoid half-precision issues
)

# Sample input text
text = "Hello, world! How is it going?"

# Tokenize the input
inputs = tokenizer(text, return_tensors="pt")

# Generate output with specified max_new_tokens
output = model.generate(inputs["input_ids"], max_new_tokens=20)

# Decode the output to text
result = tokenizer.decode(output[0], skip_special_tokens=True)
print("Generated Text:", result)

