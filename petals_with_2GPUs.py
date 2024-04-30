import torch

# Check for available GPUs
gpu_count = torch.cuda.device_count()
assert gpu_count >= 2, f"Expected at least two GPUs, but found {gpu_count}."

print("Available GPUs:", gpu_count)
