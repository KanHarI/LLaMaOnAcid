#!/bin/bash

# Set environment variables to use only one GPU
export CUDA_VISIBLE_DEVICES=0

echo "Starting LLaMa on Acid experiment with Mistral-7B-v0.1..."

# Run the Python script with the Mistral model
python3 - << 'EOF'
import os
import torch
import pickle
from main import DefaultModeNetworkExperiment

# Check GPU availability
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("No GPU detected, using CPU")

# Initialize the experiment with Mistral-7B
experiment = DefaultModeNetworkExperiment(model_name="mistralai/Mistral-7B-v0.1")

print("Fetching Wikipedia articles and preparing chunks...")
# Fetch Wikipedia articles and prepare chunks
experiment.fetch_top_wikipedia_articles(n=50)  # Using fewer articles for the smaller model
experiment.prepare_text_chunks(chunk_size=384)  # Smaller chunk size for memory efficiency

print("Identifying default mode network...")
# Identify the default mode network
experiment.identify_default_mode_network(sample_size=50)  # Analyze fewer samples
experiment.select_top_default_mode_heads(top_n=30)  # Select fewer heads for the smaller model

# Save the identified network
print("Saving the default mode network...")
experiment.save_default_mode_network("mistral_7b_default_mode_network.pkl")

# Visualize the default mode network
print("Visualizing results...")
experiment.visualize_default_mode_network(save_path="mistral_7b_dmn_visualization.png")

# Define test queries
test_queries = [
    "What is the meaning of life?",
    "Write a creative story about a robot who discovers emotions.",
    "Explain quantum mechanics in simple terms.",
    "Describe a new color that doesn't exist.",
    "What would happen if humans could photosynthesize?"
]

# Run the experiment with various inhibition factors
print("Running experiment with various inhibition factors...")
results = experiment.run_experiment(
    queries=test_queries,
    inhibition_factors=[0.0, 0.3, 0.5, 0.7, 0.9],
    max_new_tokens=150  # Fewer tokens for faster generation
)

# Analyze results
print("Analyzing results...")
analysis_df = experiment.analyze_results(results, save_path="mistral_7b_dmn_analysis.png")

# Save the experiment results
print("Saving experiment results...")
with open("mistral_7b_dmn_experiment_results.pkl", "wb") as f:
    pickle.dump(results, f)

analysis_df.to_csv("mistral_7b_dmn_experiment_analysis.csv")

print("Experiment completed successfully!")
EOF

echo "Experiment finished. Results are saved in the current directory." 