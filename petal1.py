import time
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from petals.client import DistributedLlamaForCausalLM
from petals.client.inference_session import TimeoutError

# Initialize the Accelerator for multi-GPU setup
accelerator = Accelerator()

# Number of GPUs available
num_gpus = torch.cuda.device_count()

# Retry parameters
max_retries = 5
initial_retry_delay = 2  # Initial delay between retries in seconds

# Load the base model and tokenizer
model_path = "petals-team/StableBeluga2"
model = DistributedLlamaForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,  # Use bfloat16 for mixed precision
    device_map={"": accelerator.process_index}  # Use Accelerate for device mapping
)

tokenizer = AutoTokenizer.from_pretrained(model_path)

# Example prompts for testing
prompts = [
    "The King is dead. Long live the Queen.",
    "Once there were four children whose names were Peter, Susan, Edmund, and Lucy.",
    "The story so far: in the beginning, the universe was created."
]

# Synchronize GPUs and start the timer
accelerator.wait_for_everyone()
start_time = time.time()

# Function to handle inference with retry logic
def run_inference_with_retry(model, tokenizer, prompt):
    retry_count = 0
    retry_delay = initial_retry_delay
    while retry_count < max_retries:
        try:
            # Tokenize the prompt and move to the correct device
            inputs = tokenizer(prompt, return_tensors="pt").to(accelerator.device)

            # Generate text with the model
            outputs = model.generate(inputs["input_ids"], max_new_tokens=50)

            # Decode the generated text
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            return generated_text  # Return the result if successful
        except TimeoutError:
            # Handle timeout errors with retries and exponential backoff
            retry_count += 1
            time.sleep(retry_delay)
            retry_delay *= 2  # Double the delay for each retry

    raise TimeoutError("Maximum retries reached. Inference failed.")

# Run inference for each prompt
outputs = []
for prompt in prompts:
    try:
        generated_text = run_inference_with_retry(model, tokenizer, prompt)
        outputs.append(generated_text)  # Store the generated text
    except TimeoutError as e:
        print(f"Error with prompt '{prompt}': {e}")

# Synchronize processes before outputting results
accelerator.wait_for_everyone()

# Calculate the time taken for inference
elapsed_time = time.time() - start_time

# Display results if in the main process
if accelerator.is_main_process:
    print(f"Inference completed in {elapsed_time:.2f} seconds.")
    for i, output in enumerate(outputs):
        print(f"Output {i + 1}: {output}")
