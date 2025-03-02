# LLaMaOnAcid

A computational experiment simulating the effects of psychedelics on large language models by inhibiting the "default mode network" in artificial neural networks.

## Overview

LLaMaOnAcid explores the fascinating hypothesis that large language models (LLMs) may have analogous neural structures to human brains, including something resembling the Default Mode Network (DMN). In humans, psychedelics like LSD and psilocybin temporarily inhibit the DMN, leading to altered states of consciousness and increased creativity. This project implements a similar mechanism for LLMs by:

1. Identifying the "default mode network" in a language model by analyzing attention patterns
2. Selectively inhibiting the most active attention heads in this network 
3. Measuring the effects on the model's outputs across various inhibition factors

## Features

- Works with Llama-3 and other transformer-based language models
- Automatically identifies default mode network patterns in model attention heads
- Allows for controlled inhibition of attention mechanisms
- Includes visualization and analysis tools for experimental results
- Supports customizable experiments with different prompts and inhibition strengths

## Project Structure

The codebase has been refactored into a modular package structure:

```
llama_on_acid/
├── __init__.py
├── config.py               # Central configuration settings
├── experiment.py           # Main experiment orchestration
├── data/
│   ├── __init__.py
│   ├── wikipedia.py        # Wikipedia article fetching
│   └── processor.py        # Text chunk processing
├── model/
│   ├── __init__.py
│   ├── dmn_identifier.py   # Default mode network identification
│   └── inhibited_generator.py  # Text generation with DMN inhibition
├── visualization/
│   ├── __init__.py
│   └── visualizer.py       # Visualization and analysis of results
└── utils/
    └── __init__.py

run_experiment.py           # Command-line entry point
```

## Installation

```bash
# Clone the repository
git clone https://github.com/KanHarI/LLaMaOnAcid.git
cd LLaMaOnAcid

# Install dependencies
pip install -r requirements.txt
```

### Authentication

Configure Git to store credentials and authenticate with Hugging Face to access models:

```bash
# Configure Git to store credentials
git config --global credential.helper store

# Log in to Hugging Face
huggingface-cli login
```

### Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- CUDA-capable GPU (recommended)
- 24GB+ VRAM for Llama-3-70B (or use a smaller model)

## Usage

### Command-line Interface

The simplest way to run an experiment is using the command-line interface:

```bash
# Run with default settings (uses Llama-3-8b model)
python run_experiment.py

# Run with a custom model and specific settings
python run_experiment.py --model meta-llama/Llama-3-70b \
                        --queries "What is the meaning of life?" "Describe a new color" \
                        --factors 0.0 0.5 0.9 \
                        --max-tokens 300 \
                        --output-dir my_experiments
```

### Shell Scripts

We also provide several shell scripts for convenience:

- `run_experiment.sh`: Basic experiment with standard settings
- `run_multi_gpu_experiment.sh`: For running on systems with multiple GPUs
- `run_mistral_experiment.sh`: Specifically configured for Mistral models

#### Using run_mistral_experiment.sh

The `run_mistral_experiment.sh` script provides a convenient way to run DMN inhibition experiments specifically optimized for Mistral models. It includes sensible defaults and additional options for customization.

```bash
# View help and available options
./run_mistral_experiment.sh --help

# Run with default settings (uses Mistral-7B-v0.1)
./run_mistral_experiment.sh

# Run with custom parameters
./run_mistral_experiment.sh --output-dir my_mistral_results \
                          --factors 0.0 0.3 0.6 0.9 \
                          --max-tokens 250

# Run with a pre-identified DMN file (saves time on repeated experiments)
./run_mistral_experiment.sh --dmn-file dmn_heads_mistral-7b-v0.1.json

# Run with custom queries
./run_mistral_experiment.sh --queries "Design a utopian society" \
                          "Write a poem about artificial consciousness" \
                          "Describe what it would feel like to be a quantum particle"

# Control the inhibition decay using gamma parameter (0-1)
./run_mistral_experiment.sh --gamma 0.9  # Higher value = more consistent inhibition across heads
./run_mistral_experiment.sh --gamma 0.7  # Lower value = more focused inhibition on top heads
```

The script supports the following options:
- `--model`: Specify a different Mistral model variant
- `--output-dir`: Custom directory for results (default: timestamped folder)
- `--dmn-file`: Use a pre-identified Default Mode Network file
- `--queries`: Custom test prompts (default: uses creative prompts)
- `--factors`: Inhibition factors between 0.0-1.0 (higher values = stronger "psychedelic" effect)
- `--max-tokens`: Maximum tokens to generate per response
- `--gamma`: Decay factor for inhibition across heads (default: 0.85)
- `--flash-attn`: Install flash-attention for better performance (CUDA systems only)

The first run will identify the DMN and save it for future use. Subsequent runs can use the `--dmn-file` option to skip this step and save time.

### Advanced DMN Inhibition Control

LLaMaOnAcid now features enhanced control over the inhibition process:

1. **Ordered Inhibition**: The DMN heads are now ordered by importance (from most to least active), allowing for more precise targeting of the most influential attention patterns.

2. **Gamma Decay**: The `--gamma` parameter controls how inhibition decays across the ordered heads:
   - For head at position N: `inhibition_factor * (gamma ^ (N-1))`
   - Higher gamma (e.g., 0.9): More uniform inhibition across all heads
   - Lower gamma (e.g., 0.7): Rapidly decreasing inhibition, focusing on the most active heads

3. **Detailed Logging**: The system now logs the ordered list of heads that are inhibited, showing their layer, head index, and importance score.

When running an experiment, you'll see output showing the top DMN heads and their inhibition levels:

```
Top 10 DMN heads ordered by importance:
#1: Layer 20, Head 15, Score 0.6743, Inhibition 0.7000
#2: Layer 18, Head 7,  Score 0.6521, Inhibition 0.5950
#3: Layer 24, Head 11, Score 0.6392, Inhibition 0.5057
...
```

This allows for more precise experimentation with different inhibition patterns across the network.

### Python API

You can also use the Python API for more customized experiments:

```python
from llama_on_acid.experiment import DefaultModeNetworkExperiment

# Initialize experiment
experiment = DefaultModeNetworkExperiment(model_name="meta-llama/Llama-3-8b")

# Identify the default mode network
experiment.fetch_top_wikipedia_articles(n=50)
experiment.prepare_text_chunks(chunk_size=512)
experiment.identify_default_mode_network(sample_size=50)
experiment.select_top_default_mode_heads(top_n=30)

# Save the identified network for future use
experiment.save_default_mode_network("llama3_dmn.pkl")

# Test with a single prompt at different inhibition levels
normal, inhibited = experiment.generate_with_inhibition(
    prompt="Write a creative story about consciousness.",
    inhibition_factor=0.7,
    max_new_tokens=200
)

print("NORMAL OUTPUT:")
print(normal)
print("\nINHIBITED OUTPUT (PSYCHEDELIC MODE):")
print(inhibited)
```

### Running Full Experiments

```python
# Define test queries
test_queries = [
    "What is the meaning of life?",
    "Write a creative story about a robot who discovers emotions.",
    "Describe a new color that doesn't exist."
]

# Run experiment with multiple inhibition factors
results = experiment.run_experiment(
    queries=test_queries,
    inhibition_factors=[0.0, 0.3, 0.5, 0.7, 0.9],
    max_new_tokens=200
)

# Analyze and visualize results
analysis_df = experiment.analyze_results(results, save_path="experiment_analysis.png")
```

## How It Works

LLaMaOnAcid is based on neuroscientific theories about psychedelics' effects on the brain's default mode network:

1. **Identification Phase**: The model processes neutral content (Wikipedia articles) while measuring activity in each attention head across all layers
2. **Head Selection**: The most consistently active attention heads are identified as the "default mode network"
3. **Inhibition**: During generation, these selected heads have their attention weights scaled down by a specified factor
4. **Analysis**: The effects on output coherence, creativity, and other metrics are measured across different inhibition strengths

This approach is inspired by research suggesting that psychedelics' effects result from temporarily inhibiting highly connected hub regions in the brain's default mode network.

## Example Results

With increasing inhibition of the default mode network, language models often exhibit:

- Increased novelty and unexpectedness in outputs
- More abstract and creative associations
- Less adherence to typical narrative patterns
- Occasional disorganized or less coherent text at high inhibition levels

These patterns are analogous to effects observed in human creativity and cognition under psychedelics.

## Advanced Usage

### Using a Pre-identified DMN

To save time on repeated experiments, you can save and load pre-identified DMNs:

```python
# Load a previously identified DMN
experiment = DefaultModeNetworkExperiment(model_name="meta-llama/Llama-3-8b")
experiment.load_default_mode_network("saved_dmn.pkl")

# Now you can generate directly, skipping the identification phase
normal, inhibited = experiment.generate_with_inhibition(
    prompt="What would happen if humans could photosynthesize?",
    inhibition_factor=0.7
)
```

### Customizing Caching Behavior

The system uses caching to avoid re-downloading Wikipedia articles or re-processing model activations:

```bash
# Disable caching for a fresh run
python run_experiment.py --no-cache

# Force a refresh of Wikipedia articles
python run_experiment.py --force-refresh
```

## License

MIT License

## Citation

If you use this code for your research, please cite:

```
@software{llamaonacid2025,
  author = {Itay Knaan-Harpaz},
  title = {LLaMaOnAcid: Simulating Psychedelic Effects on Large Language Models},
  year = {2025},
  url = {https://github.com/KanHarI/LLaMaOnAcid}
}
```

## Acknowledgments

This project was inspired by research on the default mode network and its role in consciousness, as well as studies on the effects of psychedelics on human cognition. 