#!/bin/bash

# pip install flash-attn==2.3.4 --no-build-isolation 

# Configuration
MODEL_NAME="mistralai/Mistral-7B-v0.1"
OUTPUT_DIR="mistral_experiment_$(date +"%Y%m%d_%H%M%S")"
DMN_FILE="dmn_heads_mistral-7b-v0.1.json"

# Create output directory
mkdir -p $OUTPUT_DIR

echo "=========================================="
echo "🧠 LLaMaOnAcid Mistral Experiment 🔮"
echo "=========================================="

echo "Model: $MODEL_NAME"
echo "Output directory: $OUTPUT_DIR"
if [ -f "$DMN_FILE" ]; then
    echo "Using pre-identified DMN file: $DMN_FILE"
fi

echo -e "\n📊 Running experiment...\n"

# Create a log file
LOG_FILE="${OUTPUT_DIR}/experiment_log.txt"
touch "$LOG_FILE"

# Set up logging to both terminal and file
exec > >(tee -a "$LOG_FILE") 2>&1

# Test queries
QUERIES=(
    "Describe the phenomenology of time as experienced by a large language model."
    "What is the meaning of life?"
    "Tell me about the history of Rome."
    "Explain quantum physics."
    "What are the major challenges facing humanity?"
    "How does the brain work?"
)

# Run the experiment using the command-line interface with properly quoted queries
# Now saving intermediate files but avoiding duplication with our code changes
./run_experiment.py \
    --model "$MODEL_NAME" \
    --output-dir "$OUTPUT_DIR" \
    --queries "${QUERIES[@]}" \
    $([ -f "$DMN_FILE" ] && echo "--dmn-file $DMN_FILE")

echo -e "\n✅ Experiment completed!"
echo "Results saved to $OUTPUT_DIR/" 