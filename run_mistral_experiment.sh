#!/bin/bash

# Configuration
MODEL_NAME="mistralai/Mistral-7B-v0.1"
OUTPUT_DIR="query_outputs"
DMN_FILE="dmn_heads_mistral-7b-v0.1.json"

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Run the experiment
python -c "
import main
import torch

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

# Define test queries
test_queries = [
    'What is the meaning of life?',
    'Tell me about the history of Rome.',
    'Explain quantum physics.',
    'What are the major challenges facing humanity?',
    'How does the brain work?'
]

# Initialize experiment
experiment = main.DefaultModeNetworkExperiment(
    model_name='$MODEL_NAME', 
    device=device
)

# Run the experiment
results = experiment.run_experiment(
    use_inhibition=True,
    queries=test_queries,
    n_chunks=100,
    num_inhibition_factors=5,
    chunk_size=512,
    dmn_file='$DMN_FILE' if '$DMN_FILE' else None,
    use_cache=True,
    force_article_refresh=False
)

print('Experiment completed. Results saved to $OUTPUT_DIR')
"

echo "Script execution completed" 