# part 0 

from accelerate import notebook_launcher

def test():
    from accelerate import Accelerator
    accelerator = Accelerator()
    print(accelerator.distributed_type)

notebook_launcher(test, num_processes=2)  

print('\n ******** ') 

# part 1 
# simplest example: create strings on each GPU and collect them using gather_object()
# change num_processes to the number of GPUs in your system

from accelerate import notebook_launcher

def hello_world():
    from accelerate import Accelerator
    from accelerate.utils import gather_object

    accelerator = Accelerator()

    message= [f"Hello this is GPU {accelerator.process_index}"]
    messages=gather_object(message)

    accelerator.print(messages)

notebook_launcher(hello_world, num_processes=5)  

print('\n ******** ') 

# part 2 

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
    
    # https://www.penguin.co.uk/articles/2022/04/best-first-lines-in-books
    prompts_all=[
        "The King is dead. Long live the Queen.",
        "Once there were four children whose names were Peter, Susan, Edmund, and Lucy. This story",
        "The story so far: in the beginning, the universe was created. This has made a lot of people very angry",
        "It was a queer, sultry summer, the summer they electrocuted the Rosenbergs, and I didn’t know what",
        "We were somewhere around Barstow on the edge of the desert when the drugs began to take hold.",
        "It was a bright cold day in April, and the clocks were striking thirteen.",
        "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife.",
        "The snow in the mountains was melting and Bunny had been dead for several weeks before we came to understand the gravity of",
        "The sweat wis lashing oafay Sick Boy; he wis trembling.",
        "124 was spiteful. Full of Baby's venom.",
        "As Gregor Samsa awoke one morning from uneasy dreams he found himself transformed in his bed into a gigantic insect.",
        "When Mary Lennox was sent to Misselthwaite Manor to live with her uncle everybody said she was",
        "I write this sitting in the kitchen sink.",
    ]

    accelerator = Accelerator()

    model_path='meta-llama/Llama-2-7b-hf' 

    model = transformers.AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code="true",
    torch_dtype=torch.bfloat16,,
    device_map= None,
    # token=HF_TOKEN,
    use_auth_token="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")

    tokenizer = AutoTokenizer.from_pretrained(
    model_path, 
    trust_remote_code="true", 
    padding_side="left",
    token="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", 
    )


    # model_path="models/llama2-7b"
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_path,    
    #     device_map={"": accelerator.process_index},
    #     torch_dtype=torch.bfloat16,
    #     token = "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"
    # )
    # tokenizer = AutoTokenizer.from_pretrained(model_path)   

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

notebook_launcher(hello_world, num_processes=5)



