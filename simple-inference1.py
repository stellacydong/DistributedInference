from accelerate import Accelerator
from accelerate.utils import gather_object
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time, os

# Set the environment variable to avoid memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"  # Adjust based on your system

# Clear CUDA cache to free up GPU memory
torch.cuda.empty_cache()

# Initialize Accelerator for multi-GPU setup
accelerator = Accelerator()

# Check the number of available GPUs
num_gpus = torch.cuda.device_count()
print(f"Using {num_gpus} GPUs for distributed processing.")

# Set the model and tokenizer
model_path = "meta-llama/Llama-2-7b-hf"  # Using a smaller LLM
current_device = torch.device(f"cuda:{accelerator.process_index}")

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map={"": current_device},
    torch_dtype=torch.float16,  # Use mixed precision to save memory
    use_auth_token="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"
)

tokenizer = AutoTokenizer.from_prepared(
    model_path,
    trust_remote_code="true",
    padding_side="left",
    use_auth_token="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"
)

# Example prompts
prompts_all = [
    "The King is dead. Long live the Queen.",
    "Once there were four children whose names were Peter, Susan, Edmund, and Lucy.",
    "The story so far: in the beginning, the universe was created.",
    "It was a bright cold day in April, and the clocks were striking thirteen.",
    "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife.",
] * 10  # Testing prompts

# Synchronize GPUs and start the timer
accelerator.wait_for_everyone()
start_time = time.time()

# Distribute prompts across GPUs and gather results
outputs = []
total_tokens = 0
with accelerator.split_between_processes(prompts_all) as subset_prompts:
    for prompt in subset_prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(current_device)
        generated_ids = model.generate(
            inputs["input_ids"], max_new_tokens=100, pad_token_id=tokenizer.eos_token_id
        )
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        outputs.append(generated_text)
        total_tokens += len(generated_ids[0])

# Synchronize processes before outputting results
accelerator.wait_for_everyone()

# Calculate the time taken for inference
elapsed_time = time.time() - start_time

# Display results if in the main process
if accelerator.is_main_process:
    tokens_per_sec = total_tokens / elapsed_time
    print(f"Inference completed in {elapsed_time:.2f} seconds.")
    print(f"Tokens per second: {tokens_per_sec:.2f}")
    print("Generated Outputs:")
    for i, output in enumerate(outputs):
        print(f"Output {i + 1}: {output}")
