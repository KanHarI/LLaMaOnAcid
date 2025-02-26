#!/bin/bash

# LLaMaOnAcid Experiment Runner
# This script runs a basic experiment with the default mode network inhibition

# Set environment variables for better GPU memory management
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export CUDA_VISIBLE_DEVICES=0  # Use first GPU, modify as needed

# Check Python availability
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is required but not found. Please install Python 3."
    exit 1
fi

# Create output directory
OUTPUT_DIR="experiment_results"
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "🧠 LLaMaOnAcid Experiment 🔮"
echo "=========================================="

# Set parameters
MODEL_NAME="meta-llama/Llama-3-8b"  # Use smaller model for accessibility
SAMPLE_SIZE=50                      # Number of Wikipedia chunks to analyze
TOP_HEADS=30                        # Number of top DMN heads to target
INHIBITION_FACTOR=0.7               # Degree of DMN inhibition
MAX_TOKENS=200                      # Max tokens to generate per response

echo "Model: $MODEL_NAME"
echo "Sample size: $SAMPLE_SIZE"
echo "DMN heads: $TOP_HEADS"
echo "Inhibition factor: $INHIBITION_FACTOR"

echo -e "\n📊 Running experiment...\n"

# Run the experiment
python3 -c "
import sys
import os
from main import DefaultModeNetworkExperiment

# Redirect stdout to file and terminal
class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        
    def flush(self):
        self.terminal.flush()
        self.log.flush()

# Set up logging
sys.stdout = Logger('${OUTPUT_DIR}/experiment_log.txt')

# Print header
print('\\n== Starting LLaMaOnAcid Experiment ==\\n')

# Initialize experiment
print('Initializing experiment with model ${MODEL_NAME}...')
experiment = DefaultModeNetworkExperiment(model_name='${MODEL_NAME}')

# Identify default mode network
print('\\nIdentifying default mode network...')
experiment.fetch_top_wikipedia_articles(n=30)
experiment.prepare_text_chunks(chunk_size=512)
experiment.identify_default_mode_network(sample_size=${SAMPLE_SIZE})
experiment.select_top_default_mode_heads(top_n=${TOP_HEADS})

# Save the identified network
print('\\nSaving identified default mode network...')
experiment.save_default_mode_network('${OUTPUT_DIR}/dmn_network.pkl')

# Visualize network
print('\\nVisualizing default mode network...')
experiment.visualize_default_mode_network(save_path='${OUTPUT_DIR}/dmn_visualization.png')

# Test prompts
test_prompts = [
    'What is consciousness?',
    'Explain the concept of free will.',
    'Write a short poem about the universe.'
]

# Run generation tests
print('\\nRunning generation tests with and without DMN inhibition...')
for prompt in test_prompts:
    print(f'\\n== Testing prompt: \"{prompt}\" ==')
    
    normal, inhibited = experiment.generate_with_inhibition(
        prompt=prompt,
        inhibition_factor=${INHIBITION_FACTOR},
        max_new_tokens=${MAX_TOKENS}
    )
    
    print('\\nNORMAL OUTPUT:')
    print(normal)
    print('\\nINHIBITED OUTPUT (PSYCHEDELIC MODE):')
    print(inhibited)
    
    # Save outputs to files
    prompt_filename = prompt.replace(' ', '_').replace('?', '').lower()[:20]
    with open(f'${OUTPUT_DIR}/normal_{prompt_filename}.txt', 'w') as f:
        f.write(normal)
    with open(f'${OUTPUT_DIR}/inhibited_{prompt_filename}.txt', 'w') as f:
        f.write(inhibited)

print('\\n== Experiment complete! ==')
print(f'Results saved to ${OUTPUT_DIR}/')
"

echo -e "\n✅ Experiment completed!"
echo "Results saved to $OUTPUT_DIR/"
echo -e "\nTo view the visualization, open: $OUTPUT_DIR/dmn_visualization.png"
echo "To compare outputs, check the text files in: $OUTPUT_DIR/" 