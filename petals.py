import torch
from transformers import AutoTokenizer
from petals import DistributedBloomForCausalLM

# Ensure PyTorch and CUDA are working correctly
print("PyTorch version:", torch.__version__)
print("CUDA is available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)

# Token for Hugging Face authentication
hf_token = "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"  # Replace with your Hugging Face token

# Set some sample prompts for testing
prompts = ["Example prompt 1", "Example prompt 2"]

# Load the model and tokenizer
model_path = "petals-team/StableBeluga2"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",  # Automatic device mapping by Accelerate
    torch_dtype=torch.bfloat16,  # Efficient GPU usage
    use_auth_token=hf_token,  # Authentication for gated repositories
)

tokenizer = AutoTokenizer.from_pretrained(model_path, use_auth_token=hf_token)

# Ensure the tokenizer has a padding token to avoid padding errors
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

# Prepare inputs for multi-GPU inference
inputs = tokenizer(
    prompts,
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=50,  # Adjust as needed
)

# Ensure tensors are on the correct device
inputs = {key: tensor.to(torch.cuda.current_device()) for key, tensor in inputs.items()}

# Generate text from the model
output = model.generate(inputs["input_ids"], max_new_tokens=50)  # Adjust as needed
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print("Generated Text:", generated_text)
