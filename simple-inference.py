from accelerate import Accelerator
from accelerate.utils import gather_object
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time

# Initialize Accelerator for distributed setup
accelerator = Accelerator()

# Sample prompts for testing
prompts = [
    "The King is dead. Long live the Queen.",
    "It was a bright cold day in April, and the clocks were striking thirteen.",
] * 10

# Token for authentication (replace 'YOUR_TOKEN' with your actual Hugging Face token)
hf_token = "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"

# Load a model and tokenizer with the token for accessing a gated repo
model_path = "meta-llama/Llama-2-7b-hf"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    use_auth_token=hf_token,  # Use the token for authentication
)
tokenizer = AutoTokenizer.from_pretrained(model_path, use_auth_token=hf_token)
