from accelerate import Accelerator
from accelerate.utils import gather_object
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, time, os

# Set environment variable to avoid memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"  # Adjust based on your system

# Clear CUDA cache to free up memory
torch.cuda.empty_cache()

accelerator = Accelerator()

# Prompts for testing
prompts_all = [
    "The King is dead. Long live the Queen.",
    "Once there were four children whose names were Peter, Susan, Edmund, and Lucy.",
    "The story so far: in the beginning, the universe was created.",
    "It was a bright cold day in April, and the clocks were striking thirteen.",
    "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife.",
] * 1 # Multiply to create more prompts

# Load a base model and tokenizer
model_path = "meta-llama/Llama-2-7b-hf"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map={"": accelerator.process_index},
    torch_dtype=torch.bfloat16,
    use_auth_token="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"  # Corrected use of auth token
)

tokenizer = AutoTokenizer.from_prepared(model_path, padding_side="left", use_auth_token="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")

# Synchronize GPUs and start the timer
accelerator.wait_for_everyone()
start = time.time()

# Distribute prompts across GPUs and gather results
outputs = []
results = {"outputs": [], "num_tokens": 0}

with accelerator.split_between_processes(prompts_all) as prompts:
    # Have each GPU perform inference on its share of prompts
    for prompt in prompts:
        prompt_tokenized = tokenizer(prompt, return_tensors="pt", padding=True).to("cuda")
        output_tokenized = model.generate(**prompt_tokenized, max_new_tokens=100, pad_token_id=tokenizer.eos_token_id)[0]

        # Remove prompt from output
        output_tokenized = output_tokenized[len(prompt_tokenized["input_ids"][0]):]

        # Store decoded outputs and token count in results
        results["outputs"].append(tokenizer.decode(output_tokenized, skip_special_tokens=True))
        results["num_tokens"] += len(output_tokenized)

# Collect results from all GPUs
results_gathered = gather_object([results])

if accelerator.is_main_process:
    # Calculate time and tokens/sec
    timediff = time.time() - start
    num_tokens = sum([r["num_tokens"] for r in results_gathered])
    tokens_per_sec = num_tokens / timediff

    print(f"Tokens per second: {tokens_per_sec:.2f}")
    print(f"Total tokens: {num_tokens}")
    print(f"Total prompts: {len(prompts_all)}")
