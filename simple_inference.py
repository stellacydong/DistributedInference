

#  Multi-GPU text-generation
# load a model on each GPU
# distribute the prompts with split_between_processes
# have each GPU generate
# gather and output the result


from accelerate import notebook_launcher
import transformers

def hello_world():
    from accelerate import Accelerator
    from accelerate.utils import gather_object
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
   
    prompts_all=[
       "Underneath the towering city skyline, where neon lights bathed the streets in a kaleidoscope of colors, a single shadow slipped through the alleyways."
       "The shadow moved with purpose, darting between pools of light, its presence noticed only by the occasional stray cat."
    ]

    accelerator = Accelerator()

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


    with accelerator.split_between_processes(prompts_all) as prompts:
        outputs=[]
        for prompt in prompts:
            prompt_tokenized=tokenizer(prompt, return_tensors="pt").to("cuda")
            output_tokenized = model.generate(**prompt_tokenized, max_new_tokens=50)
            outputs.append(tokenizer.decode(output_tokenized[0]))

    outputs_gathered=gather_object(outputs)

    for output in outputs_gathered:
        accelerator.print(output)

   

    with open('outputs.txt','w') as file:
        file.write('\n\n'.join(outputs_gathered))

notebook_launcher(hello_world, num_processes=1)

