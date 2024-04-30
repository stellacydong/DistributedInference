

echo "Accelerate configuration created at $ACCELERATE_CONFIG_PATH"

export ACCELERATE_CONFIG=PytorchDistributedParallel/accelerate_config.yaml


# Run a multi-GPU script with Accelerate
echo "Launching multi-GPU script with Accelerate..."
accelerate launch PytorchDistributedParallel/simple-inference1.py  # Replace with your script name
