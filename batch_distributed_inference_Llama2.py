from accelerate import notebook_launcher

def hello_world():
    
    from accelerate import Accelerator
    from accelerate.utils import gather_object
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from statistics import mean
    import torch, time, json
    
    accelerator = Accelerator()
    
    def write_pretty_json(file_path, data):
        import json
        with open(file_path, "w") as write_file:
            json.dump(data, write_file, indent=4)
    
    # 10*10 Prompts. Source: https://www.penguin.co.uk/articles/2022/04/best-first-lines-in-books
    prompts_all=[
       "Underneath the towering city skyline, where neon lights bathed the streets in a kaleidoscope of colors, a single shadow slipped through the alleyways."
       "The shadow moved with purpose, darting between pools of light, its presence noticed only by the occasional stray cat."
    ] 
    
    # load a base model and tokenizer
    model_path='meta-llama/Llama-2-7b-hf' 

    model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map={"": accelerator.process_index},
    token="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")

    tokenizer = AutoTokenizer.from_pretrained(
    model_path, 
    padding_side="left",
    token="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", 
    )

    tokenizer.pad_token = tokenizer.eos_token
    
    # batch, left pad (for inference), and tokenize
    def prepare_prompts(prompts, tokenizer, batch_size=16):
        batches=[prompts[i:i + batch_size] for i in range(0, len(prompts), batch_size)]  
        batches_tok=[]
        tokenizer.padding_side="left"     
        for prompt_batch in batches:
            batches_tok.append(
                tokenizer(
                    prompt_batch, 
                    return_tensors="pt", 
                    padding='longest', 
                    truncation=False, 
                    pad_to_multiple_of=8,
                    add_special_tokens=False).to("cuda") 
                )
        tokenizer.padding_side="right"
        return batches_tok
    
    # sync GPUs and start the timer
    accelerator.wait_for_everyone()    
    start=time.time()
    
    # divide the prompt list onto the available GPUs 
    with accelerator.split_between_processes(prompts_all) as prompts:
        results=dict(outputs=[], num_tokens=0)
    
        # have each GPU do inference in batches
        prompt_batches=prepare_prompts(prompts, tokenizer, batch_size=16)
    
        for prompts_tokenized in prompt_batches:
            outputs_tokenized=model.generate(
                **prompts_tokenized, 
                max_new_tokens=100,
                pad_token_id=tokenizer.eos_token_id)
    
            # remove prompt from gen. tokens
            outputs_tokenized=[ tok_out[len(tok_in):] 
                for tok_in, tok_out in zip(prompts_tokenized["input_ids"], outputs_tokenized) ] 
    
            # count and decode gen. tokens 
            num_tokens=sum([ len(t) for t in outputs_tokenized ])
            outputs=tokenizer.batch_decode(outputs_tokenized)
    
            # store in results{} to be gathered by accelerate
            results["outputs"].extend(outputs)
            results["num_tokens"] += num_tokens
    
        results=[ results ] # transform to list, otherwise gather_object() will not collect correctly
    
    results_gathered=gather_object(results)
    
    if accelerator.is_main_process:
        timediff=time.time()-start
        num_tokens=sum([r["num_tokens"] for r in results_gathered ])
    
        print(f"tokens/sec: {num_tokens//timediff}, time elapsed: {timediff}, num_tokens {num_tokens}")

notebook_launcher(hello_world, num_processes=8)
