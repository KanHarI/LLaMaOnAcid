#!/usr/bin/env python
"""
Main entry point for running LLaMa on Acid experiments.
"""
import argparse
import os
import pickle
from datetime import datetime

import torch

from llama_on_acid.config import DEFAULT_MODEL_NAME, DEFAULT_QUERIES
from llama_on_acid.experiment import DefaultModeNetworkExperiment


def main() -> None:
    """Run LLaMa on Acid experiment with command-line arguments."""
    parser = argparse.ArgumentParser(
        description="LLaMa on Acid: Simulating psychedelic effects on LLMs"
    )

    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help=f"Model name or path (default: {DEFAULT_MODEL_NAME})",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (default: 'cuda' if available, else 'cpu')",
    )

    # Experiment arguments
    parser.add_argument(
        "--dmn-file", type=str, default=None, help="Path to pre-identified DMN file (if available)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save outputs (default: 'results')",
    )
    parser.add_argument(
        "--n-chunks", type=int, default=100, help="Number of chunks to process (default: 100)"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=200, help="Maximum tokens to generate (default: 200)"
    )
    parser.add_argument(
        "--no-cache", action="store_true", help="Disable caching (default: caching enabled)"
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Force refresh of article data (default: False)",
    )
    parser.add_argument(
        "--no-save-intermediate",
        action="store_true",
        help="Don't save intermediate results after each query (default: save intermediate results)",
    )

    # Query arguments
    parser.add_argument(
        "--queries",
        type=str,
        nargs="*",
        help="Specific queries to use (default: use built-in queries)",
    )
    parser.add_argument(
        "--factors",
        type=float,
        nargs="*",
        default=[0.0, 0.3, 0.5, 0.7, 0.9],
        help="Inhibition factors to use (default: 0.0, 0.3, 0.5, 0.7, 0.9)",
    )

    args = parser.parse_args()

    # Create output directory with timestamped subfolder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name_safe = args.model.replace("/", "-").replace(".", "-")
    output_dir = os.path.join(args.output_dir, f"{model_name_safe}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved to: {output_dir}")

    # Set torch seeds for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # Initialize experiment
    print(f"Initializing experiment with model: {args.model}")
    experiment = DefaultModeNetworkExperiment(
        model_name=args.model,
        device=args.device,
    )

    # Use specified queries or default queries
    queries = args.queries if args.queries else DEFAULT_QUERIES
    print(f"Using {len(queries)} queries and {len(args.factors)} inhibition factors")

    # Run the experiment
    results = experiment.run_experiment(
        queries=queries,
        inhibition_factors=args.factors,
        max_new_tokens=args.max_tokens,
        n_chunks=args.n_chunks,
        dmn_file=args.dmn_file,
        use_cache=not args.no_cache,
        force_article_refresh=args.force_refresh,
        output_dir=output_dir,
        save_intermediate=not args.no_save_intermediate,
    )

    # Save the full results
    results_file = os.path.join(output_dir, "experiment_results.pkl")
    with open(results_file, "wb") as f:
        pickle.dump(results, f)
    print(f"Saved full results to {results_file}")

    # Analyze and visualize results
    try:
        analysis_df = experiment.analyze_results(
            results=results, save_path=os.path.join(output_dir, "analysis.png")
        )
        analysis_csv = os.path.join(output_dir, "analysis.csv")
        analysis_df.to_csv(analysis_csv)
        print(f"Saved analysis to {analysis_csv}")
    except Exception as e:
        print(f"Error during analysis: {e}")

    # Save DMN visualization
    try:
        experiment.visualize_default_mode_network(
            save_path=os.path.join(output_dir, "dmn_visualization.png")
        )
    except Exception as e:
        print(f"Error generating DMN visualization: {e}")

    # Save the identified DMN for future use
    try:
        dmn_path = os.path.join(output_dir, "identified_dmn.pkl")
        experiment.save_default_mode_network(dmn_path)
        print(f"Saved identified DMN to {dmn_path}")
    except Exception as e:
        print(f"Error saving DMN: {e}")

    print("Experiment complete!")


if __name__ == "__main__":
    main()
