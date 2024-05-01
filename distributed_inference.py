# part 0 

from accelerate import notebook_launcher

def test():
    from accelerate import Accelerator
    accelerator = Accelerator()
    print(accelerator.distributed_type)

notebook_launcher(test, num_processes=2)  

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
