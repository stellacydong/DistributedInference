from accelerate import Accelerator
from accelerate.utils import gather_object
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, time, json

# Clear CUDA cache to free up memory
torch.cuda.empty_cache()

# Initialize accelerator for multi-GPU setup
accelerator = Accelerator()

# Test prompts (reduced for memory efficiency)
prompts_all = [
    "The King is dead. Long live the Queen.",
    "Once there were four children whose names were Peter, Susan, Edmund, and Lucy.",
    # Reduced list to prevent excessive memory usage
] * 1  # Multiply by a smaller factor

# Load model and tokenizer
model_path = "meta-llama/Llama-2-7b-hf"
device = accelerator.device

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map={"": accelerator.process_index},
    torch_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(model_path)

# Sync GPUs and start the timer
accelerator.wait_for_everyone()
start_time = time.time()

# Prepare to collect results
results = {"outputs": [], "num_tokens": 0}

# Error handling for GPU operations
try:
    with accelerator.split_between_processes(prompts_all) as prompts:
        for prompt in prompts:
            # Tokenize with padding and truncation to reduce memory
            prompt_tokenized = tokenizer(
                prompt, return_tensors="pt", padding=True, truncation=True
            ).to(device)

            # Generate with a limited number of new tokens to prevent OOM
            output_tokenized = model.generate(
                **prompt_tokenized, max_new_tokens=50  # Adjust to reduce memory
            )[0]

            # Discard input IDs to prevent OOM
            output_tokenized = output_tokenized[len(prompt_tokenized["input_ids"][0]):]

            results["outputs"].append(tokenizer.decode(output_tokenized))
            results["num_tokens"] += len(output_tokenized)

except torch.cuda.OutOfMemoryError:
    print("Error: CUDA Out-of-Memory. Consider reducing batch size or max_new_tokens.")

# Gather results from all GPUs
results_gathered = gather_object([results])

# If this process is the main one, compute and display results
if accelerator.is_main_process:
    elapsed_time = time.time() - start_time
    total_tokens = sum(r["num_tokens"] for r in results_gathered)

    # Display results with error handling
    print(
        f"Tokens/sec: {total_tokens // elapsed_time}, Time: {elapsed_time:.2f}, Total Tokens: {total_tokens}"
    )
