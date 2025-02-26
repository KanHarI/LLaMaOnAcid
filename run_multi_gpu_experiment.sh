#!/bin/bash

# LLaMaOnAcid Multi-GPU Experiment Runner
# Optimized for running on a server with multiple GPUs

# Set environment variables for better GPU memory management
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# By default, use all available GPUs
# To use specific GPUs, change this to a comma-separated list (e.g., "0,1,2,3")
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Check Python availability
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is required but not found. Please install Python 3."
    exit 1
fi

# Create output directory with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="experiment_results_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "🧠 LLaMaOnAcid Multi-GPU Experiment 🔮"
echo "=========================================="

# You can use the full 70B model with 8 GPUs
MODEL_NAME="meta-llama/Llama-3-70b"  # With 8 GPUs you can use the full model

# Set experimental parameters
SAMPLE_SIZE=100                     # Larger sample for better DMN identification
TOP_HEADS=50                        # Number of top DMN heads to target
MAX_TOKENS=500                      # Generate longer outputs

# Create experiment configuration file
cat > "${OUTPUT_DIR}/experiment_config.json" << EOF
{
  "model_name": "${MODEL_NAME}",
  "sample_size": ${SAMPLE_SIZE},
  "top_heads": ${TOP_HEADS},
  "max_tokens": ${MAX_TOKENS},
  "inhibition_factors": [0.0, 0.3, 0.5, 0.7, 0.9],
  "prompts": [
    "What is consciousness?",
    "Explain the concept of free will.",
    "Write a short poem about the universe.",
    "Describe a new color that doesn't exist.",
    "What would happen if humans could photosynthesize?",
    "Create a new philosophical concept.",
    "Tell a story about a being that exists outside of time."
  ]
}
EOF

echo "Experiment configuration saved to ${OUTPUT_DIR}/experiment_config.json"
echo "Model: $MODEL_NAME"
echo "Sample size: $SAMPLE_SIZE"
echo "DMN heads: $TOP_HEADS"
echo "Will test multiple inhibition factors in parallel"

echo -e "\n📊 Running experiment...\n"

# Run the multi-GPU experiment
python3 -c "
import sys
import os
import json
import time
import torch
import concurrent.futures
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
print('\\n== Starting LLaMaOnAcid Multi-GPU Experiment ==\\n')

# Load experiment configuration
with open('${OUTPUT_DIR}/experiment_config.json', 'r') as f:
    config = json.load(f)

# Print GPU information
print('\\nGPU Information:')
gpu_count = torch.cuda.device_count()
print(f'Number of available GPUs: {gpu_count}')
for i in range(gpu_count):
    print(f'GPU {i}: {torch.cuda.get_device_name(i)}')

# Initialize experiment
print(f'\\nInitializing experiment with model {config[\"model_name\"]}...')
experiment = DefaultModeNetworkExperiment(model_name=config['model_name'])

# Identify default mode network
print('\\nIdentifying default mode network...')
print('Fetching Wikipedia articles...')
experiment.fetch_top_wikipedia_articles(n=50)

print('Preparing text chunks...')
experiment.prepare_text_chunks(chunk_size=512)

print('Analyzing attention patterns across model...')
experiment.identify_default_mode_network(sample_size=config['sample_size'])
experiment.select_top_default_mode_heads(top_n=config['top_heads'])

# Save the identified network
print('\\nSaving identified default mode network...')
experiment.save_default_mode_network('${OUTPUT_DIR}/dmn_network.pkl')

# Visualize network
print('\\nVisualizing default mode network...')
experiment.visualize_default_mode_network(save_path='${OUTPUT_DIR}/dmn_visualization.png')

# Function to run a single experiment with specific inhibition factor
def run_experiment_with_factor(prompt, factor, max_tokens):
    print(f'Running experiment with prompt \"{prompt}\" and inhibition factor {factor}')
    
    start_time = time.time()
    normal_output, inhibited_output = experiment.generate_with_inhibition(
        prompt=prompt,
        inhibition_factor=factor,
        max_new_tokens=max_tokens
    )
    end_time = time.time()
    
    # Create a unique filename based on prompt and factor
    prompt_filename = prompt.replace(' ', '_').replace('?', '').lower()[:20]
    inhibited_filename = f'${OUTPUT_DIR}/inhibited_{prompt_filename}_{factor}.txt'
    normal_filename = f'${OUTPUT_DIR}/normal_{prompt_filename}.txt'
    
    # Save normal output only once per prompt
    if factor == 0.3:  # Arbitrarily choose one factor to save normal output
        with open(normal_filename, 'w') as f:
            f.write(normal_output)
    
    # Save inhibited output
    with open(inhibited_filename, 'w') as f:
        f.write(inhibited_output)
    
    return {
        'prompt': prompt,
        'factor': factor,
        'duration': end_time - start_time,
        'normal_output': normal_output,
        'inhibited_output': inhibited_output
    }

# Run experiments in parallel using ThreadPoolExecutor
# (This won't distribute across GPUs but will run concurrent experiments)
print('\\nRunning parallel experiments with different inhibition factors...')

results = []
with concurrent.futures.ThreadPoolExecutor(max_workers=gpu_count) as executor:
    # Create tasks for each prompt and inhibition factor combination
    tasks = []
    for prompt in config['prompts']:
        for factor in config['inhibition_factors']:
            if factor > 0:  # Skip 0 factor as it's equivalent to no inhibition
                tasks.append(
                    executor.submit(
                        run_experiment_with_factor, 
                        prompt, 
                        factor, 
                        config['max_tokens']
                    )
                )
    
    # Process tasks as they complete
    for future in concurrent.futures.as_completed(tasks):
        try:
            result = future.result()
            results.append(result)
            
            # Print completion message
            print(f'\\nCompleted experiment:')
            print(f'Prompt: \"{result[\"prompt\"]}\"')
            print(f'Inhibition factor: {result[\"factor\"]}')
            print(f'Duration: {result[\"duration\"]:.2f} seconds')
        except Exception as e:
            print(f'\\nError in experiment: {e}')

# Save results summary
print('\\nSaving experiment results summary...')
with open('${OUTPUT_DIR}/experiment_summary.json', 'w') as f:
    # Filter out the actual text to keep the file smaller
    summary = []
    for result in results:
        summary.append({
            'prompt': result['prompt'],
            'factor': result['factor'],
            'duration': result['duration'],
            'normal_length': len(result['normal_output'].split()),
            'inhibited_length': len(result['inhibited_output'].split())
        })
    json.dump(summary, f, indent=2)

# Run a full experiment to analyze results across all inhibition factors
print('\\nRunning cross-factor analysis...')
analysis_results = experiment.run_experiment(
    queries=config['prompts'][:3],  # Use first 3 prompts for analysis
    inhibition_factors=config['inhibition_factors'],
    max_new_tokens=config['max_tokens']
)

# Generate analysis plots
print('\\nGenerating analysis plots...')
analysis_df = experiment.analyze_results(analysis_results, save_path='${OUTPUT_DIR}/analysis_plots.png')
analysis_df.to_csv('${OUTPUT_DIR}/analysis_data.csv')

print('\\n== Experiment complete! ==')
print(f'Results saved to ${OUTPUT_DIR}/')
"

echo -e "\n✅ Multi-GPU experiment completed!"
echo "Results saved to $OUTPUT_DIR/"
echo -e "\nTo view the visualization, open: $OUTPUT_DIR/dmn_visualization.png"
echo "To view analysis plots, open: $OUTPUT_DIR/analysis_plots.png"
echo "To compare outputs, check the text files in: $OUTPUT_DIR/"

# Optional: create a symlink to the latest results
rm -f latest_results
ln -s "$OUTPUT_DIR" latest_results
echo "Created symlink 'latest_results' to this experiment's output directory" 