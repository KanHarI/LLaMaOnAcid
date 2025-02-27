#!/usr/bin/env python
"""
Example usage of the LLaMa on Acid Python API.
"""
import os
from datetime import datetime

from llama_on_acid.experiment import DefaultModeNetworkExperiment


def run_simple_example() -> None:
    """Run a simple example demonstrating the core features."""
    # Initialize experiment with a smaller model for faster testing
    experiment = DefaultModeNetworkExperiment(model_name="meta-llama/Llama-3-8b")

    # Try to load a pre-identified DMN if available
    dmn_file = "llama3_dmn.pkl"
    if os.path.exists(dmn_file):
        print(f"Loading pre-identified DMN from {dmn_file}")
        experiment.load_default_mode_network(dmn_file)
    else:
        print("No pre-identified DMN found. Identifying default mode network...")
        # This step will take some time as it processes Wikipedia articles
        experiment.fetch_top_wikipedia_articles(n=10)  # Limit to 10 articles for speed
        experiment.prepare_text_chunks(chunk_size=512)
        experiment.identify_default_mode_network(sample_size=20)  # Use fewer samples for speed
        experiment.select_top_default_mode_heads(top_n=30)

        # Save for future use
        experiment.save_default_mode_network(dmn_file)
        print(f"Saved identified DMN to {dmn_file} for future use")

    # Test with a creative prompt
    prompt = "Describe a color that doesn't exist in our world."
    print(f"\nGenerating responses for prompt: '{prompt}'")

    # Create a timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"example_output_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # Try different inhibition levels
    inhibition_factors = [0.0, 0.5, 0.9]
    results = []

    for factor in inhibition_factors:
        print(f"\nGenerating with inhibition factor: {factor}")
        try:
            normal, inhibited = experiment.generate_with_inhibition(
                prompt=prompt, inhibition_factor=factor, max_new_tokens=150
            )

            # For factor 0.0, use normal output since inhibition isn't applied
            response = normal if factor == 0.0 else inhibited

            print("\nResponse:")
            print("-" * 40)
            print(response)
            print("-" * 40)

            # Add to results
            results.append({"query": prompt, "inhibition_factor": factor, "response": response})

        except Exception as e:
            print(f"Error generating with inhibition factor {factor}: {e}")

    # Save and analyze results
    if results:
        from llama_on_acid.visualization.visualizer import save_query_outputs

        # Save the responses to file
        save_query_outputs(results=results, model_name=experiment.model_name, output_dir=output_dir)
        print(f"\nSaved outputs to {output_dir}")

        # Visualize the DMN
        experiment.visualize_default_mode_network(
            save_path=os.path.join(output_dir, "dmn_visualization.png")
        )
        print(f"Saved DMN visualization to {output_dir}/dmn_visualization.png")

    print("\nExample complete!")


if __name__ == "__main__":
    run_simple_example()
