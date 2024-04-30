export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"

# Set the correct number of processes and machines
accelerate launch \
  --num_processes 1 \
  --num_machines 1 \
  --mixed_precision "no" \
  PytorchDistributedParallel/MessagePassing.py
