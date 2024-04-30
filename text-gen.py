from accelerate import Accelerator
from accelerate.utils import gather_object
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, time, json

# Initialize the Accelerator for multi-GPU setup
accelerator = Accelerator()

# Ensure the tokenizer has a padding token
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})  # Add a default padding token

# 10*10 Prompts (example data)
prompts_all = [
    "The King is dead. Long live the Queen.",
    "Once there were four children whose names were Peter, Susan, Edmund, and Lucy.",
    "The story so far: in the beginning, the universe was created.",
    "It was a bright cold day in April, and the clocks were striking thirteen.",
    "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife.",
    "The sweat was lashing oafay Sick Boy; he was trembling.",
    "124 was spiteful. Full of Baby's venom.",
    "As Gregor Samsa awoke one morning from uneasy dreams, he found himself transformed into a gigantic insect.",
    "I write this sitting in the kitchen sink.",
    "We were somewhere around Barstow on the edge of the desert when the drugs began to take hold.",
] * 10

# Load the model with Hugging Face authentication
model_path = "meta-llama/Llama-2-7b-hf"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map={"": accelerator.process_index},  # Map devices with Accelerate
    torch_dtype=torch.bfloat16,
)

# Synchronize GPUs and start the timer
accelerator.wait_for_everyone()
start = time.time()

# Split prompts across GPUs for inference
with accelerator.split_between_processes(prompts_all) as subset_prompts:
    results = {"outputs": [], "num_tokens": 0}  # Store output of generations

    # Perform inference on each GPU
    for prompt in subset_prompts:
        prompt_tokenized = tokenizer(prompt, return_tensors="pt", padding=True).to(accelerator.device)  # Ensure consistent device assignment
        
        # Generate output from the model
        output_tokenized = model.generate(prompt_tokenized["input_ids"], max_new_tokens=100)[0]
        
        # Remove the prompt from the output and decode it
        generated_tokens = output_tokenized[len(prompt_tokenized["input_ids"][0]):]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        results["outputs"].append(generated_text)  # Store generated text
        results["num_tokens"] += len(generated_tokens)  # Count the total tokens generated

    results = [results]  # Convert to list to gather results from all GPUs

# Collect results from all the GPUs
results_gathered = gather_object(results)

# If this is the main process, display results
if accelerator.is_main_process:
    elapsed_time = time.time() - start  # Calculate elapsed time
    total_tokens = sum([r["num_tokens"] for r in results_gathered])

    # Display the generation statistics
    print(f"Tokens per second: {total_tokens // elapsed_time}")
    print(f"Total time: {elapsed_time:.2f} seconds")
    print(f"Total tokens: {total_tokens}")
    print(f"Total prompts: {len(prompts_all)}")
