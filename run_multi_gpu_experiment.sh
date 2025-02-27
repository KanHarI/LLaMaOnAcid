#!/bin/bash

# LLaMaOnAcid Multi-GPU Experiment Runner
# Optimized for running on a server with multiple GPUs

# Set environment variables for better GPU memory management
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# By default, use all available GPUs
# To use specific GPUs, change this to a comma-separated list (e.g., "0,1,2,3")
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5

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
echo "Max tokens: $MAX_TOKENS"
echo "Will test multiple inhibition factors: [0.0, 0.3, 0.5, 0.7, 0.9]"

echo -e "\n📊 Running experiment...\n"

# Create a log file
LOG_FILE="${OUTPUT_DIR}/experiment_log.txt"
touch "$LOG_FILE"

# Set up logging to both terminal and file
exec > >(tee -a "$LOG_FILE") 2>&1

# Print GPU information
echo "GPU Information:"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --list-gpus
    echo "Using GPUs: $CUDA_VISIBLE_DEVICES"
else
    echo "nvidia-smi not found. GPU information unavailable."
fi

# Extract prompts and inhibition factors from the config
PROMPTS=$(jq -r '.prompts | join(" ")' "${OUTPUT_DIR}/experiment_config.json" | sed 's/"/\\"/g')
FACTORS=$(jq -r '.inhibition_factors | join(" ")' "${OUTPUT_DIR}/experiment_config.json")

# Run the experiment using the new command-line interface
./run_experiment.py \
    --model "$MODEL_NAME" \
    --n-chunks "$SAMPLE_SIZE" \
    --max-tokens "$MAX_TOKENS" \
    --output-dir "$OUTPUT_DIR" \
    --factors $FACTORS \
    --queries $PROMPTS

echo -e "\n✅ Multi-GPU experiment completed!"
echo "Results saved to $OUTPUT_DIR/"
echo -e "\nTo view the visualization, open: $OUTPUT_DIR/dmn_visualization.png"
echo "To view analysis plots, open: $OUTPUT_DIR/analysis.png"
echo "To compare outputs, check the text files in: $OUTPUT_DIR/"

# Create a symlink to the latest results
rm -f latest_results
ln -s "$OUTPUT_DIR" latest_results
echo "Created symlink 'latest_results' to this experiment's output directory" 