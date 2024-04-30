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
