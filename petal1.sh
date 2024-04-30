export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"

echo "Accelerate configuration created at $ACCELERATE_CONFIG_PATH"

export ACCELERATE_CONFIG=PytorchDistributedParallel/accelerate_config.yaml


# Run a multi-GPU script with Accelerate
echo "Launching multi-GPU script with Accelerate..."
accelerate launch PytorchDistributedParallel/petal1.py  # Replace with your script name
