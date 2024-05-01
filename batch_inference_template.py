import torch
torch.cuda.empty_cache()

from accelerate import notebook_launcher
import transformers

def hello_world():
    from accelerate import Accelerator
    from accelerate.utils import gather_object
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from statistics import mean
    import torch, time, json
    
    accelerator = Accelerator()
    
    prompts_all = [
        "Underneath the towering city skyline, where neon lights bathed the streets in a kaleidoscope of colors, a single shadow slipped through the alleyways.",
        "The shadow moved with purpose, darting between pools of light, its presence noticed only by the occasional stray cat."
    ]
    
    # Use a different model path
    model_path = 'meta-llama/Llama-2-13b-hf'  # Example with Llama-2-13b, change to desired model

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": accelerator.process_index},
        token="YOUR_HF_API_KEY"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        padding_side="left",
        token="YOUR_HF_API_KEY",
    )
    
    accelerator.wait_for_everyone()
    start = time.time()
    
    # Divide the prompt list onto the available GPUs
    with accelerator.split_between_processes(prompts_all) as prompts:
        results = dict(outputs=[], num_tokens=0)
    
        for prompt in prompts:
            prompt_tokenized = tokenizer(prompt, return_tensors="pt").to("cuda")
            output_tokenized = model.generate(**prompt_tokenized, max_new_tokens=100)[0]
    
            output_tokenized = output_tokenized[len(prompt_tokenized["input_ids"][0]):]
    
            results["outputs"].append(tokenizer.decode(output_tokenized))
            results["num_tokens"] += len(output_tokenized)
    
        results = [results]
        
        message = [f"Hello, this is GPU {accelerator.process_index}"]
        messages = gather_object(message)
        accelerator.print(messages)
    
        accelerator.print('\n ******** ')
        accelerator.print(results)
    
    results_gathered = gather_object(results)

    if accelerator.is_main_process:
        timediff = time.time() - start
        num_tokens = sum([r["num_tokens"] for r in results_gathered])
    
        print(f"tokens/sec: {num_tokens // timediff}, time {timediff}, total tokens {num_tokens}, total prompts {len(prompts_all)}")

# Set the number of processes to your hardware configuration
notebook_launcher(hello_world, num_processes=2)
