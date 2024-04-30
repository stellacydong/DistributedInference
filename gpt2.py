import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator

# Initialize the Accelerator for multi-GPU support
accelerator = Accelerator()

# Check how many GPUs are available
print(f"Number of GPUs: {torch.cuda.device_count()}")

# Sample text input for testing
prompts = ["What's the capital of France?", "What is the meaning of life?"]

# Load a pre-trained LLM and tokenizer
# Replace 'gpt2' with your desired model, like 'llama-2-7b-hf'
model_name = "gpt2"  # This is just an example; replace with your LLM
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Configure the model with Hugging Face Accelerate
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # Use float16 for efficient GPU utilization
    device_map="auto",  # Let Accelerate handle device mapping
)

# Prepare prompts and move them to the appropriate device
inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(accelerator.device)

# Use Accelerator to manage multi-GPU inference
with accelerator.split_between_processes(prompts) as prompt_subset:
    outputs = []

    # Perform inference for each prompt on the respective GPU
    for prompt in prompt_subset:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(accelerator.device)
        generated_ids = model.generate(input_ids, max_new_tokens=20)  # Adjust max tokens as needed
        output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        outputs.append(output_text)

# Gather results from all processes
accelerator.wait_for_everyone()  # Synchronize processes

# If this process is the main process, display results
if accelerator.is_main_process:
    print("Generated Outputs:")
    for i, output in enumerate(outputs):
        print(f"Prompt {i+1}: {output}")
