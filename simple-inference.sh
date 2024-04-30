# python3 -m pip install transformers accelerate torch bitsandbytes

# huggingface-cli login --token  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 --add-to-git-credential

export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"

# Set the correct number of processes and machines
accelerate launch \
  --num_processes 8 \
  --num_machines 1 \
  --mixed_precision "no" \
  PytorchDistributedParallel/simple-inference.py
