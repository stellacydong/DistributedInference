# Start the server for 'bigscience/bloom-560m', blocks 0-15
petals.server \
  --model bigscience/bloom-560m \
  --block-indices 0-15 \
  --throughput 1
