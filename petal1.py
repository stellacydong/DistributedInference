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

class DistributedLlamaModel(FromPretrainedMixin, PTuneMixin, LlamaModel):
    """LlamaModel, but all transformer layers are hosted by the swarm"""

    _keys_to_ignore_on_load_missing = PTuneMixin._keys_to_ignore_on_load_missing
    _keys_to_ignore_on_load_unexpected = [r"^model\.layers\."]

    config_class = DistributedLlamaConfig

    def __init__(self, config: DistributedLlamaConfig, *, dht: Optional[hivemind.DHT] = None):
        n_layer, config.num_hidden_layers = config.num_hidden_layers, 0  # Prevent initialization
        super().__init__(config)
        assert len(self.layers) == 0
        config.num_hidden_layers = n_layer

        self.layers = RemoteSequential(config, dht=dht)

        self.requires_grad_(False)  # Forbid accumulate grads for embeddings and layernorm
        self.init_prompts(config)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        assert attention_mask is None, f"{self.__class__.__name__} does not support attention masks right now"

        for k, v in kwargs.items():
            if not (v is None or v is False):
                logger.debug(f"Extra keyword arguments are not yet supported (got {k} = {v})")

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if self.config.tuning_mode and "ptune" in self.config.tuning_mode:
            batch_size = inputs_embeds.shape[0]
            prompts, intermediate_prompts = self.get_prompt(batch_size)
            inputs_embeds = torch.cat([prompts, inputs_embeds], dim=1)

        hidden_states = inputs_embeds
        output_shape = input_shape + (hidden_states.size(-1),)

        if self.config.tuning_mode and "ptune" in self.config.tuning_mode:
            hidden_states = self.layers(hidden_states, prompts=intermediate_prompts)
        else:
            hidden_states = self.layers(hidden_states)

        # Remove prefix
        if self.config.tuning_mode and "ptune" in self.config.tuning_mode:
            hidden_states = hidden_states[:, self.pre_seq_len :]

        # Add last hidden state
        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states.view(output_shape)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )

    @property
    def word_embeddings(self) -> nn.Embedding:  # For compatibility with RemoteGenerationMixin
        return self.embed_tokens

    @property
    def word_embeddings_layernorm(self) -> nn.Module:  # For compatibility with RemoteGenerationMixin
        return nn.Identity()

    @property
    def h(self) -> RemoteSequential:  # For compatibility with RemoteGenerationMixin
        return self.layers

    @property
    def ln_f(self) -> nn.Module:  # For compatibility with RemoteGenerationMixin
        return self.norm


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

# Set the environment variable to avoid memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"  # Adjust based on your system

# Clear CUDA cache to free up GPU memory
torch.cuda.empty_cache()

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

