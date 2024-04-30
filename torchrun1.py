from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import time


def main():
    dist.init_process_group(backend="nccl")

    device = torch.device(f"cuda:{dist.get_rank()}")

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", torch_dtype=torch.float16, trust_remote_code=True).to(device)
    model = DDP(model, device_ids=[dist.get_rank()])

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
        {"role": "user", "content": "What is Deep learning?"},
    ]

    encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")

    model_inputs = encodeds.to(torch.device("cuda:0"))
    original_model = model.module
    start_time = time.time()
    generated_ids = original_model.generate(
        model_inputs, 
        top_k=10,
        top_p = 0.9,
        temperature = 0.6,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=512, 
        do_sample=True)
    decoded = tokenizer.batch_decode(generated_ids)
    print(decoded[0])
    end_time = time.time()
    print(f"Inference time: {end_time - start_time}s")

if __name__ == "__main__":
    main()
