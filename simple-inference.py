from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time

torch.cuda.empty_cache()

# Initialize Accelerator for multi-GPU setup
accelerator = Accelerator()

# Token for Hugging Face authentication (replace 'YOUR_TOKEN' with your actual token)
hf_token = "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"

# Set some sample prompts for testing
prompts = ["Example prompt 1", "Example prompt 2"]  # Change as needed

# Load a model and tokenizer with the token
model_path = "petals-team/StableBeluga2"  # Example model; ensure you have access
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",  # Let the model map devices automatically
    torch_dtype=torch.bfloat16,  # Adjust torch_dtype as needed
    token=hf_token  # Use your Hugging Face token for authentication
)
tokenizer = AutoTokenizer.from_pretrained(model_path, token=hf_token)


