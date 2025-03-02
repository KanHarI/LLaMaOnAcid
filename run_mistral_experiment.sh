#!/bin/bash

# Function to display help information
show_help() {
    cat << EOF
Usage: ./run_mistral_experiment.sh [OPTIONS]

Run a Default Mode Network (DMN) inhibition experiment using the Mistral-7B model.
This script automatically sets up and executes experiments that simulate psychedelic effects
on the Mistral language model by inhibiting its identified default mode network.

Options:
  -h, --help                Show this help message and exit
  -m, --model MODEL         Specify the Mistral model variant (default: mistralai/Mistral-7B-v0.1)
  -o, --output-dir DIR      Specify output directory (default: mistral_experiment_TIMESTAMP)
  -d, --dmn-file FILE       Use a pre-identified DMN file (default: dmn_heads_mistral-7b-v0.1.json)
  -q, --queries "Q1" "Q2"   Custom test queries (default: uses predefined set)
  -f, --factors N1 N2 N3    Inhibition factors between 0.0-1.0 (default: uses run_experiment.py defaults)
  -t, --max-tokens N        Maximum tokens to generate (default: uses run_experiment.py defaults)
  -g, --gamma N             Gamma decay factor for inhibition (default: 0.95)
  --flash-attn              Install flash-attention for better performance (recommended for CUDA systems)
  --debug                   Enable verbose debug logging

Examples:
  # Run with default settings
  ./run_mistral_experiment.sh
  
  # Run with custom output directory
  ./run_mistral_experiment.sh --output-dir my_mistral_results
  
  # Run with custom DMN file and specific inhibition factors
  ./run_mistral_experiment.sh --dmn-file my_custom_dmn.json --factors 0.0 0.4 0.8
  
  # Run with custom gamma decay factor
  ./run_mistral_experiment.sh --gamma 0.9
  
  # Run more aggressive inhibition with higher base factor and lower decay
  ./run_mistral_experiment.sh --factors 0.0 0.6 0.9 --gamma 0.7

Note:
  This script requires a CUDA-capable GPU and approximately 14GB of VRAM.
  For first-time runs, the DMN identification process may take 30-60 minutes.
  Subsequent runs can reuse an identified DMN file to save time.
EOF
    exit 0
}

# Check for help flag
for arg in "$@"; do
    if [[ "$arg" == "-h" || "$arg" == "--help" ]]; then
        show_help
    fi
done

# pip install flash-attn==2.3.4 --no-build-isolation 

# Configuration
MODEL_NAME="mistralai/Mistral-7B-v0.1"
OUTPUT_DIR="mistral_experiment_$(date +"%Y%m%d_%H%M%S")"
DMN_FILE="dmn_heads_mistral-7b-v0.1.json"
CUSTOM_QUERIES=()
FACTORS=(0.0 0.5 0.8)
MAX_TOKENS=""
GAMMA=0.95
INSTALL_FLASH_ATTN=false
DEBUG=false

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -m|--model)
            MODEL_NAME="$2"
            shift 2
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -d|--dmn-file)
            DMN_FILE="$2"
            shift 2
            ;;
        -q|--queries)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^- ]]; do
                CUSTOM_QUERIES+=("$1")
                shift
            done
            ;;
        -f|--factors)
            shift
            FACTORS=()
            while [[ $# -gt 0 && "$1" =~ ^[0-9.]+$ ]]; do
                FACTORS+=("$1")
                shift
            done
            ;;
        -t|--max-tokens)
            MAX_TOKENS="$2"
            shift 2
            ;;
        -g|--gamma)
            GAMMA="$2"
            shift 2
            ;;
        --flash-attn)
            INSTALL_FLASH_ATTN=true
            shift
            ;;
        --debug)
            DEBUG=true
            shift
            ;;
        *)
            shift
            ;;
    esac
done

# Install flash-attention if requested
if [ "$INSTALL_FLASH_ATTN" = true ]; then
    echo "Installing flash-attention for improved performance..."
    pip install flash-attn==2.3.4 --no-build-isolation
fi

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
echo "Gamma decay factor: $GAMMA"

echo -e "\n📊 Running experiment...\n"

# Create a log file
LOG_FILE="${OUTPUT_DIR}/experiment_log.txt"
touch "$LOG_FILE"

# Set up logging to both terminal and file
exec > >(tee -a "$LOG_FILE") 2>&1

# Default test queries if none provided
if [ ${#CUSTOM_QUERIES[@]} -eq 0 ]; then
    QUERIES=(
        "Describe a color that does not exist."
        "Describe the phenomenology of time as experienced by a large language model."
        "What is the meaning of life?"
        "Tell me about the history of Rome."
        "Explain quantum physics."
        "What are the major challenges facing humanity?"
        "How does the brain work?"
    )
else
    QUERIES=("${CUSTOM_QUERIES[@]}")
fi

# Construct command with optional arguments
CMD="./run_experiment.py --model \"$MODEL_NAME\" --output-dir \"$OUTPUT_DIR\" --gamma $GAMMA"

# Add DMN file if it exists
if [ -f "$DMN_FILE" ]; then
    CMD="$CMD --dmn-file \"$DMN_FILE\""
fi

# Add inhibition factors if specified
if [ ${#FACTORS[@]} -gt 0 ]; then
    FACTOR_ARGS=""
    for factor in "${FACTORS[@]}"; do
        FACTOR_ARGS="$FACTOR_ARGS --factors $factor"
    done
    CMD="$CMD $FACTOR_ARGS"
fi

# Add max tokens if specified
if [ ! -z "$MAX_TOKENS" ]; then
    CMD="$CMD --max-tokens $MAX_TOKENS"
fi

# Add debug flag if specified
if [ "$DEBUG" = true ]; then
    CMD="$CMD --verbose-logging"
fi

# Add queries with proper quoting
for query in "${QUERIES[@]}"; do
    CMD="$CMD --queries \"$query\""
done

# Run the experiment
echo "Running command: $CMD"
eval $CMD

echo -e "\n✅ Experiment completed!"
echo "Results saved to $OUTPUT_DIR/"
echo "To visualize results, check the output directory for generated graphs and logs." 