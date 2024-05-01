from accelerate import notebook_launcher

def test():
    from accelerate import Accelerator
    accelerator = Accelerator()
    print(accelerator.distributed_type)

notebook_launcher(test, num_processes=2)  
