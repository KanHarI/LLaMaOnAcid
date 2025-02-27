#!/bin/bash

pip install flash-attn==2.3.4 --no-build-isolation

# LLaMaOnAcid Experiment Runner
# This script runs a basic experiment with the default mode network inhibition

# Set environment variables for better GPU memory management
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export CUDA_VISIBLE_DEVICES=0  # Use first GPU, modify as needed

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
echo "Inhibition factors: 0.0, $INHIBITION_FACTOR"
echo "Max tokens: $MAX_TOKENS"

echo -e "\n📊 Running experiment...\n"

# Create a log file
LOG_FILE="${OUTPUT_DIR}/experiment_log.txt"
touch "$LOG_FILE"

# Set up logging to both terminal and file
exec > >(tee -a "$LOG_FILE") 2>&1

# Run the experiment using the new command-line interface
./run_experiment.py \
    --model "$MODEL_NAME" \
    --n-chunks "$SAMPLE_SIZE" \
    --max-tokens "$MAX_TOKENS" \
    --output-dir "$OUTPUT_DIR" \
    --factors 0.0 "$INHIBITION_FACTOR" \
    --queries "What is consciousness?" "Explain the concept of free will." "Write a short poem about the universe."

echo -e "\n✅ Experiment completed!"
echo "Results saved to $OUTPUT_DIR/"
echo -e "\nTo view the visualization, open: $OUTPUT_DIR/*/dmn_visualization.png"
echo "To compare outputs, check the text files in: $OUTPUT_DIR/" 