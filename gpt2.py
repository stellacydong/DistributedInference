import torch
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

# Ensure proper configuration for multi-GPU setup
# Run `accelerate config` to set values like `num_processes`
accelerator = Accelerator()

# Hugging Face authentication token is no longer used in the model constructor
hf_token = "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"  # If needed, authenticate with Hugging Face CLI

# Clear CUDA cache to free up GPU memory
torch.cuda.empty_cache()

# Model and tokenizer initialization
model_path = "petals-team/StableBeluga2"  # Example model
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",  # Let Accelerate handle device mapping
    torch_dtype=torch.float16,  # Adjust torch_dtype as needed
)

# Tokenizer initialization without `auth_token`
tokenizer = AutoTokenizer.from_pretrained(model_path)

# # Ensure the tokenizer has a padding token
# if tokenizer.pad_token is None:
#     tokenizer.add_special_tokens({"pad_token": "[PAD]"})  # Add padding if needed

# # Prepare inputs
# prompts = ["Example prompt 1", "Example prompt 2"]  # Change as needed
# inputs = tokenizer(
#     prompts,
#     return_tensors="pt",
#     padding=True,
#     truncation=True,
#     max_length=50,
# ).to(accelerator.device)  # Ensure inputs are on the correct device

# # Synchronize and start timing
# accelerator.wait_for_everyone()
# start_time = time.time()

# # Generate text with multi-GPU inference
# outputs = []
# with accelerator.split_between_processes(prompts) as subset_prompts:
#     for prompt in subset_prompts:
#         input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(accelerator.device)

#         # Generate text from the model
#         generated_ids = model.generate(input_ids, max_new_tokens=50)  # Adjust as needed
#         generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

#         outputs.append(generated_text)

# # Synchronize after generation
# accelerator.wait_for_everyone()

# # Calculate inference time
# elapsed_time = time.time() - start_time

# # Display generated outputs
# if accelerator.is_main_process:
#     print(f"Inference time: {elapsed_time:.2f} seconds")
#     for idx, output in enumerate(outputs):
#         print(f"Prompt {idx + 1}: {output}")
