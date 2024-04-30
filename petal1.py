from typing import Optional

import hivemind
import torch
import torch.nn as nn
from hivemind.utils.logging import get_logger
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.llama import LlamaForCausalLM, LlamaForSequenceClassification, LlamaModel, LlamaPreTrainedModel

from petals.client.from_pretrained import FromPretrainedMixin
from petals.client.lm_head import LMHead
from petals.client.ptune import PTuneMixin
from petals.client.remote_generation import RemoteGenerationMixin
from petals.client.remote_sequential import RemoteSequential
from petals.models.llama.config import DistributedLlamaConfig

logger = get_logger(__name__)

class DistributedLlamaForCausalLM(FromPretrainedMixin, RemoteGenerationMixin, LlamaForCausalLM):
    _keys_to_ignore_on_load_missing = DistributedLlamaModel._keys_to_ignore_on_load_missing
    _keys_to_ignore_on_load_unexpected = DistributedLlamaModel._keys_to_ignore_on_load_unexpected

    config_class = DistributedLlamaConfig

    def __init__(self, config: DistributedLlamaConfig):
        LlamaPreTrainedModel.__init__(self, config)
        self.model = DistributedLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = LMHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head

    @property
    def transformer(self) -> DistributedLlamaModel:  # For compatibility with RemoteGenerationMixin
        return self.model


import torch
import time
import os
import pynvml  # For GPU workload information
from accelerate import Accelerator
from transformers import AutoTokenizer
# from petals.client import DistributedLlamaForCausalLM  # Ensure the correct import
from typing import Optional

# Set NCCL environment variables for RTX 4000 series compatibility
os.environ['NCCL_P2P_DISABLE'] = '1'
os.environ['NCCL_IB_DISABLE'] = '1'

# Initialize the Accelerator for multi-GPU setup
accelerator = Accelerator()

# GPT-2 Small model with 124M parameters
model_path = "petals-team/StableBeluga2"
num_gpus = torch.cuda.device_count()  # Count the available GPUs

print(f"Using {num_gpus} GPUs for distributed processing.")

# Set the correct device and load the model
current_device = torch.device(f"cuda:{accelerator.process_index}")

# Track GPU usage
pynvml.nvmlInit()  # Initialize GPU monitoring
handle = pynvml.nvmlDeviceGetHandleByIndex(accelerator.process_index)  # Get handle for current GPU

try:
    model = DistributedLlamaForCausalLM.from_pretrained(
        model_path,
        device_map={"": current_device},  # Explicitly set the device
        torch_dtype=torch.float16,  # Use mixed precision for efficiency
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Example prompts for testing
    prompts = ["Hello, world!", "What's your favorite movie?", "Tell me a story."]

    # Synchronize GPUs and start the timer
    accelerator.wait_for_everyone()
    start_time = time.time()

    # Distribute prompts across GPUs and gather results
    outputs = []
    with accelerator.split_between_processes(prompts) as subset_prompts:
        for prompt in subset_prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(current_device)
            generated_ids = model.generate(inputs["input_ids"], max_new_tokens=50)
            generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            outputs.append(generated_text)

    # Calculate inference time
    elapsed_time = time.time() - start_time
    gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle)  # Get GPU utilization

    # Display results if in the main process
    if accelerator.is_main_process:
        print(f"Inference completed in {elapsed_time:.2f} seconds.")
        print("Generated Outputs:")
        for i, output in enumerate(outputs):
            print(f"Output {i + 1}: {output}")
        print(f"GPU {accelerator.process_index} utilization: {gpu_util.gpu}%")
        print(f"GPU {accelerator.process_index} memory utilization: {gpu_util.memory}%")

except Exception as e:
    print("An error occurred during inference:", e)

