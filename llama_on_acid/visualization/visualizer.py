"""
Module for visualizing default mode network and analyzing experiment results.
"""

import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ..model.dmn_identifier import HeadImportanceScore


def visualize_default_mode_network(
    head_importance_scores: HeadImportanceScore,
    num_layers: int,
    num_heads: int,
    model_name: str = "model",
    top_n: int = 100,
    save_path: Optional[str] = None,
) -> None:
    """
    Visualize the default mode network as a heatmap.

    Args:
        head_importance_scores: List of (layer_idx, head_idx, score) tuples
        num_layers: Number of layers in the model
        num_heads: Number of heads per layer
        model_name: Name of the model for plot title
        top_n: Number of top heads to highlight
        save_path: Path to save the visualization (if provided)
    """
    if not head_importance_scores:
        raise ValueError("Head importance scores must be provided")

    # Create a matrix of activation values
    activation_matrix = np.zeros((num_layers, num_heads))

    # Fill in the values for heads we have data for
    for layer_idx, head_idx, score in head_importance_scores:
        if 0 <= layer_idx < num_layers and 0 <= head_idx < num_heads:
            activation_matrix[layer_idx, head_idx] = score

    # Create the heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(activation_matrix, cmap="viridis", annot=False)
    plt.title(f"Default Mode Network Activation Heatmap for {model_name}")
    plt.xlabel("Attention Head")
    plt.ylabel("Layer")

    # Highlight the top N heads
    top_n = min(top_n, len(head_importance_scores))
    for i in range(top_n):
        layer_idx, head_idx, _ = head_importance_scores[i]
        if 0 <= layer_idx < num_layers and 0 <= head_idx < num_heads:
            plt.plot(head_idx + 0.5, layer_idx + 0.5, "rx", markersize=8)

    if save_path:
        plt.savefig(save_path)
        print(f"Visualization saved to {save_path}")

    plt.show()


def analyze_results(
    results: List[Dict[str, Any]], save_path: Optional[str] = None, model_name: str = "model"
) -> pd.DataFrame:
    """
    Analyze experimental results and create visualizations.

    Args:
        results: List of result dictionaries from run_experiment
        save_path: Optional path to save the visualization
        model_name: Name of the model to include in plot titles

    Returns:
        DataFrame with analysis
    """
    analysis: Dict[str, List[Any]] = {
        "queries": [],
        "inhibition_factors": [],
        "response_lengths": [],
        "unique_words": [],
        "avg_sentence_length": [],
        "creativity_score": [],
    }

    for result in results:
        # Extract data from result
        query = result.get("query", "Unknown query")
        factor = result.get("inhibition_factor", 0.0)
        response = result.get("response", "")

        # Strip the query from the beginning of the response
        if response.startswith(query):
            response_text = response[len(query) :].strip()
        else:
            response_text = response

        # Calculate response length
        response_length = len(response_text.split())

        # Calculate unique words ratio
        words = response_text.lower().split()
        unique_words = len(set(words)) / len(words) if words else 0

        # Calculate average sentence length
        sentences = [s.strip() for s in response_text.split(".") if s.strip()]
        avg_sentence_length = (
            sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
        )

        # Calculate a simple "creativity score" based on unique words and sentence variance
        sentence_lengths = [len(s.split()) for s in sentences]
        sentence_variance = np.var(sentence_lengths) if len(sentence_lengths) > 1 else 0
        creativity_score = unique_words * (1 + sentence_variance / 100)

        # Add to analysis
        analysis["queries"].append(query)
        analysis["inhibition_factors"].append(factor)
        analysis["response_lengths"].append(response_length)
        analysis["unique_words"].append(unique_words)
        analysis["avg_sentence_length"].append(avg_sentence_length)
        analysis["creativity_score"].append(creativity_score)

    # Convert to DataFrame
    df = pd.DataFrame(analysis)

    # Create plots
    plt.figure(figsize=(15, 10))

    # Plot 1: Response length vs inhibition factor
    plt.subplot(2, 2, 1)
    sns.boxplot(x="inhibition_factors", y="response_lengths", data=df)
    plt.title("Response Length vs Inhibition Factor")
    plt.xlabel("Inhibition Factor")
    plt.ylabel("Response Length (words)")

    # Plot 2: Unique words ratio vs inhibition factor
    plt.subplot(2, 2, 2)
    sns.boxplot(x="inhibition_factors", y="unique_words", data=df)
    plt.title("Lexical Diversity vs Inhibition Factor")
    plt.xlabel("Inhibition Factor")
    plt.ylabel("Unique Words Ratio")

    # Plot 3: Average sentence length vs inhibition factor
    plt.subplot(2, 2, 3)
    sns.boxplot(x="inhibition_factors", y="avg_sentence_length", data=df)
    plt.title("Sentence Length vs Inhibition Factor")
    plt.xlabel("Inhibition Factor")
    plt.ylabel("Average Sentence Length (words)")

    # Plot 4: Creativity score vs inhibition factor
    plt.subplot(2, 2, 4)
    sns.boxplot(x="inhibition_factors", y="creativity_score", data=df)
    plt.title("Creativity Score vs Inhibition Factor")
    plt.xlabel("Inhibition Factor")
    plt.ylabel("Creativity Score")

    plt.suptitle(f"Analysis of DMN Inhibition Effects ({model_name})")
    plt.tight_layout(rect=(0, 0, 1, 0.95))  # Make room for suptitle

    if save_path:
        plt.savefig(save_path)
        print(f"Analysis saved to {save_path}")

    plt.show()

    return df


def save_query_outputs(
    results: List[Dict[str, Any]],
    model_name: str = "model",
    output_dir: str = "query_outputs",
    suffix: str = "",
    save_individual_files: bool = True,
) -> None:
    """
    Save the query outputs to files.

    Args:
        results: List of results dictionaries, each with query, inhibition_factor and response
        model_name: Name of the model for filenames
        output_dir: Directory to save the outputs
        suffix: Optional suffix for the output files
        save_individual_files: Whether to save individual files for each query (set to False for final outputs)
    """
    if not results:
        print("Warning: No results to save.")
        # Create a directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, f"no_results{suffix}.txt"), "w") as f:
            f.write("No results were generated during the experiment.\n")
        return

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    try:
        # Create a combined summary file
        model_name_safe = model_name.replace("/", "-").replace(".", "-")
        output_file = os.path.join(output_dir, f"{model_name_safe}_all_outputs{suffix}.txt")

        with open(output_file, "w") as f:
            f.write(f"Experiment results for {model_name}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write("=" * 80 + "\n\n")

            # Group results by query
            query_results: Dict[str, List[Dict[str, Any]]] = {}
            for result in results:
                query = result.get("query", "Unknown query")
                if query not in query_results:
                    query_results[query] = []
                query_results[query].append(result)

            # Write results for each query
            for query, query_list in query_results.items():
                f.write(f"QUERY: {query}\n")
                f.write("-" * 80 + "\n\n")

                # Sort by inhibition factor
                query_list.sort(key=lambda x: x.get("inhibition_factor", 0.0))

                for result in query_list:
                    factor = result.get("inhibition_factor", 0.0)
                    response = result.get("response", "No response generated")

                    f.write(f"INHIBITION FACTOR: {factor}\n\n")
                    f.write(f"RESPONSE:\n{response}\n\n")
                    f.write("-" * 40 + "\n\n")

                f.write("=" * 80 + "\n\n")

        print(f"Saved combined outputs to {output_file}")

        # Create individual files for each query only if requested
        if save_individual_files:
            for query, query_list in query_results.items():
                # Create a safe filename from the query
                query_filename = query.replace(" ", "_").replace("?", "").replace(".", "")[:30]
                query_filename = "".join(c for c in query_filename if c.isalnum() or c == "_")

                query_output_file = os.path.join(
                    output_dir, f"{model_name_safe}_{query_filename}{suffix}.txt"
                )

                with open(query_output_file, "w") as f:
                    f.write(f"Results for query: {query}\n")
                    f.write(f"Model: {model_name}\n")
                    f.write(f"Timestamp: {timestamp}\n")
                    f.write("-" * 80 + "\n\n")

                    # Sort by inhibition factor
                    query_list.sort(key=lambda x: x.get("inhibition_factor", 0.0))

                    for result in query_list:
                        factor = result.get("inhibition_factor", 0.0)
                        response = result.get("response", "No response generated")

                        f.write(f"INHIBITION FACTOR: {factor}\n\n")
                        f.write(f"RESPONSE:\n{response}\n\n")
                        f.write("-" * 40 + "\n\n")

            print(f"Saved individual query outputs to {output_dir}")

    except Exception as e:
        print(f"Error saving query outputs: {e}")
