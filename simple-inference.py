from accelerate import Accelerator
from accelerate.utils import gather_object
from transformers import AutoModelForCausalLM, AutoTokenizer
from statistics import mean
import torch, time, json

# Ensure multi-GPU setup
accelerator = Accelerator()

# Clear CUDA cache to free up GPU memory
torch.cuda.empty_cache()

# Test prompts (reduced for brevity)
prompts_all = [
    "The King is dead. Long live the Queen.",
    "Once there were four children whose names were Peter, Susan, Edmund, and Lucy.",
    # ... other prompts
] * 10

# Load base model and tokenizer
model_path = "meta-llama/Llama-2-7b-hf"

# Correct device mapping using Accelerator
device = accelerator.device

# If `device_map` causes issues, use `device`
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map=device,
    torch_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(model_path)

# Start the timer after synchronizing GPUs
accelerator.wait_for_everyone()
start_time = time.time()

# Divide prompts across available GPUs
results = {"outputs": [], "num_tokens": 0}
with accelerator.split_between_processes(prompts_all) as subset_prompts:
    # Ensure proper device handling for memory constraints
    subset_prompts = [tokenizer(prompt, return_tensors="pt").to(device) for prompt in subset_prompts]
    
    for prompt_tokenized in subset_prompts:
        # Generate output with memory optimization
        output_tokenized = model.generate(**prompt_tokenized, max_new_tokens=50)[0]

        # Remove prompt tokens from output to avoid memory issues
        output_tokenized = output_tokenized[len(prompt_tokenized["input_ids"][0]):]

        # Store outputs and count tokens
        results["outputs"].append(tokenizer.decode(output_tokenized))
        results["num_tokens"] += len(output_tokenized)

# Gather results from all GPUs
results_gathered = gather_object([results])

# If this process is the main one, compute and display results
if accelerator.is_main_process:
    elapsed_time = time.time() - start_time
    total_tokens = sum(r["num_tokens"] for r in results_gathered)

    # Display performance metrics and generated outputs
    print(f"Tokens/sec: {total_tokens // elapsed_time}, Time: {elapsed_time:.2f}, Total Tokens: {total_tokens}, Total Prompts: {len(prompts_all)}")
