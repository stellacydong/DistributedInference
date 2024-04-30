from accelerate import Accelerator
from accelerate.utils import gather_object
from transformers import AutoModelForCausalLM, AutoTokenizer
from statistics import mean
import torch, time, json

# Initialize Accelerator for multi-GPU setup
accelerator = Accelerator()

# 10*10 Prompts. Source: https://www.penguin.co.uk/articles/2022/04/best-first-lines-in-books
prompts_all = [
    "The King is dead. Long live the Queen.",
    # ... (other prompts)
    "We were somewhere around Barstow on the edge of the desert when the drugs began to take hold."
] * 10

# load a base model and tokenizer with multi-GPU support
model_path = "meta-llama/Llama-2-7b-hf"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map=accelerator.device,
    torch_dtype=torch.bfloat16,
)

# Synchronize GPUs and clear CUDA cache
accelerator.wait_for_everyone()
torch.cuda.empty_cache()

tokenizer = AutoTokenizer.from_pretrained(model_path)

# Start the timer
start_time = time.time()

# Split prompts among available GPUs
results = dict(outputs=[], num_tokens=0)
with accelerator.split_between_processes(prompts_all) as subset_prompts:
    # Each GPU handles its allocated prompts
    for prompt in subset_prompts:
        prompt_tokenized = tokenizer(prompt, return_tensors="pt").to(accelerator.device)
        output_tokenized = model.generate(**prompt_tokenized, max_new_tokens=100)[0]

        # Remove prompt from output
        output_tokenized = output_tokenized[len(prompt_tokenized["input_ids"][0]):]

        # Store results and token count
        results["outputs"].append(tokenizer.decode(output_tokenized))
        results["num_tokens"] += len(output_tokenized)

# Gather results from all GPUs
results_gathered = gather_object([results])

if accelerator.is_main_process:
    # Calculate performance metrics
    elapsed_time = time.time() - start_time
    total_tokens = sum(r["num_tokens"] for r in results_gathered)

    # Output performance and generated results
    print(f"Tokens/sec: {total_tokens // elapsed_time}, Time: {elapsed_time}, Total Tokens: {total_tokens}, Total Prompts: {len(prompts_all)}")
