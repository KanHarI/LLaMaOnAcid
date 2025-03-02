#!/usr/bin/env python
"""
Main entry point for running LLaMa on Acid experiments.
"""
import argparse
import os
import pickle
from datetime import datetime

import torch

from llama_on_acid.config import (
    DEFAULT_INHIBITION_FACTORS,
    DEFAULT_MODEL_NAME,
    DEFAULT_QUERIES,
    DMN_CONFIG,
)
from llama_on_acid.experiment import DefaultModeNetworkExperiment
from llama_on_acid.utils import get_git_commit_hash


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
        default=DEFAULT_INHIBITION_FACTORS,
        help="Inhibition factors to use (default: 0.0, 0.3, 0.5, 0.7, 0.9)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.85,
        help="Gamma decay factor for inhibition (default: 0.85). Higher values preserve more inhibition across heads.",
    )

    # DMN identification parameters
    parser.add_argument(
        "--top-heads-per-layer",
        type=int,
        default=DMN_CONFIG["top_n_per_layer"],
        help=f"Number of top attention heads to select per layer for inhibition (default: {DMN_CONFIG['top_n_per_layer']})",
    )
    parser.add_argument(
        "--skip-first-last",
        action="store_true",
        default=DMN_CONFIG["skip_first_last"],
        help=f"Skip first and last layers when identifying DMN (default: {DMN_CONFIG['skip_first_last']})",
    )
    parser.add_argument(
        "--verbose-logging",
        action="store_true",
        default=DMN_CONFIG["verbose_logging"],
        help=f"Enable verbose logging during DMN identification (default: {DMN_CONFIG['verbose_logging']})",
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

    # Update DMN configuration based on command-line arguments
    # Temporarily modify the global config settings
    DMN_CONFIG["top_n_per_layer"] = args.top_heads_per_layer
    DMN_CONFIG["skip_first_last"] = args.skip_first_last
    DMN_CONFIG["verbose_logging"] = args.verbose_logging

    # Use specified queries or default queries
    queries = args.queries if args.queries else DEFAULT_QUERIES
    print(f"Using {len(queries)} queries and {len(args.factors)} inhibition factors")

    # Identify or load the default mode network
    if args.dmn_file and os.path.exists(args.dmn_file):
        print(f"Loading pre-identified DMN from: {args.dmn_file}")
        experiment.load_default_mode_network(args.dmn_file)
    else:
        print("Identifying default mode network...")
        experiment.fetch_top_wikipedia_articles(n=50, force_refresh=args.force_refresh)
        experiment.prepare_text_chunks(chunk_size=512)
        experiment.identify_default_mode_network(
            sample_size=args.n_chunks, use_cache=not args.no_cache
        )
        experiment.select_top_default_mode_heads(top_n_per_layer=args.top_heads_per_layer)

        # Save the identified DMN for future use
        dmn_path = os.path.join(output_dir, "dmn_heads.pkl")
        print(f"Saving identified DMN to: {dmn_path}")
        experiment.save_default_mode_network(dmn_path)

    # Run the experiment
    results = experiment.run_experiment(
        queries=queries,
        inhibition_factors=args.factors,
        gamma=args.gamma,
        max_new_tokens=args.max_tokens,
        n_chunks=args.n_chunks,
        dmn_file=args.dmn_file,
        use_cache=not args.no_cache,
        force_article_refresh=args.force_refresh,
        output_dir=output_dir,
        save_intermediate=True,
    )

    # Save the full results
    results_file = os.path.join(output_dir, "experiment_results.pkl")
    with open(results_file, "wb") as f:
        pickle.dump(results, f)
    print(f"Saved full results to {results_file}")

    # Save metadata file with git hash
    metadata_file = os.path.join(output_dir, "experiment_metadata.txt")
    with open(metadata_file, "w") as f:
        f.write("LLaMa On Acid Experiment\n")
        f.write("========================\n\n")
        f.write(f"Experiment run at: {datetime.now().isoformat()}\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Git commit: {get_git_commit_hash() or 'Not available'}\n\n")
        f.write("Command line arguments:\n")
        for arg, value in vars(args).items():
            f.write(f"  {arg}: {value}\n")
    print(f"Saved experiment metadata to {metadata_file}")

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

    print("Experiment complete!")


if __name__ == "__main__":
    main()
