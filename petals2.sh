# Start with torchrun
torchrun --nproc_per_node=1 petals.server \
  --model bigscience/bloom-560m \
  --block-indices 0-15 \
  --throughput 1
