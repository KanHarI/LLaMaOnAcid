#!/usr/bin/env python
"""
Main experiment module for running Default Mode Network inhibition.
"""

import json
import os
import pickle
import random
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from .config import (
    CACHE_DIR,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_GENERATION_PARAMS,
    DEFAULT_INHIBITION_FACTORS,
    DEFAULT_MODEL_NAME,
    DEFAULT_QUERIES,
    DEFAULT_SAMPLE_SIZE,
    DEFAULT_TOP_HEADS,
    DEVICE,
    DMN_CONFIG,
)
from .data.processor import prepare_text_chunks
from .data.wikipedia import fetch_top_wikipedia_articles
from .model.dmn_identifier import DefaultModeNetworkIdentifier
from .model.inhibited_generator import InhibitedGenerator
from .utils import get_git_commit_hash
from .visualization.visualizer import (
    analyze_results,
    save_query_outputs,
    visualize_default_mode_network,
)


# Add debug logging function
def debug_log(
    msg: str, is_important: bool = False, divider: bool = False, verbose: bool = False
) -> None:
    """
    Helper function to print consistent debug logs with timestamps.

    Args:
        msg: Message to log
        is_important: Whether the message is important (will be highlighted)
        divider: Whether to add divider lines before and after the message
        verbose: Whether to print the message if it's not important (for detailed logs)
    """
    # Skip non-important, verbose logs unless verbose is enabled
    if not is_important and not verbose:
        return

    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    if divider:
        print(f"\n{'=' * 80}")
    if is_important:
        print(f"\n[DEBUG {timestamp}] 🔍 {msg} 🔍")
    else:
        print(f"[DEBUG {timestamp}] {msg}")
    if divider:
        print(f"{'=' * 80}\n")


class DefaultModeNetworkExperiment:
    """
    Class to simulate the effects of LSD on large language models by manipulating attention head activations.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize the DefaultModeNetworkExperiment.

        Args:
            model_name: The name or path of the model to use
            device: The device to use (cuda or cpu)
            cache_dir: Directory to store cached data
        """
        debug_log(
            "INIT: Starting DefaultModeNetworkExperiment initialization",
            is_important=True,
            divider=True,
        )
        self.model_name = model_name
        print(f"Initializing experiment with model: {model_name}")

        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        debug_log(f"Using device: {self.device}", verbose=False)

        # Create cache directory
        self.cache_dir = cache_dir or CACHE_DIR
        os.makedirs(self.cache_dir, exist_ok=True)
        print(f"Using cache directory: {self.cache_dir}")

        # Initialize model and tokenizer
        print("Loading tokenizer and model...")
        debug_log("Loading tokenizer...", verbose=False)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        debug_log("Tokenizer loaded successfully", verbose=False)

        # For flash attention compatibility
        debug_log("Loading model with eager attention implementation...", verbose=False)
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=self.device,
                torch_dtype=torch.float16,
                attn_implementation="eager",  # Use eager implementation for compatibility
            )
            debug_log("Model loaded successfully with eager attention", verbose=False)
        except Exception as e:
            debug_log(f"Error loading model: {e}", is_important=True)
            raise
        print("Model and tokenizer loaded")

        # Model dimensions
        debug_log("Determining model dimensions...", verbose=False)
        if hasattr(self.model.config, "num_hidden_layers"):
            self.num_layers = self.model.config.num_hidden_layers
            debug_log(f"Found num_hidden_layers: {self.num_layers}", verbose=False)
        elif hasattr(self.model.config, "n_layer"):
            self.num_layers = self.model.config.n_layer
            debug_log(f"Found n_layer: {self.num_layers}", verbose=False)
        else:
            error_msg = "Could not determine number of layers in the model"
            debug_log(error_msg, is_important=True)
            raise ValueError(error_msg)

        if hasattr(self.model.config, "num_attention_heads"):
            self.num_heads = self.model.config.num_attention_heads
            debug_log(f"Found num_attention_heads: {self.num_heads}", verbose=False)
        elif hasattr(self.model.config, "n_head"):
            self.num_heads = self.model.config.n_head
            debug_log(f"Found n_head: {self.num_heads}", verbose=False)
        else:
            error_msg = "Could not determine number of attention heads in the model"
            debug_log(error_msg, is_important=True)
            raise ValueError(error_msg)

        print(f"Model has {self.num_layers} layers and {self.num_heads} attention heads per layer")

        # Data storage
        self.articles: List[str] = []
        self.processed_chunks: List[str] = []

        # Debug prints to help identify the issue
        print(f"model_name type: {type(self.model_name)}")
        print(f"model type: {type(self.model)}")
        debug_log(f"Model config: {self.model.config}", verbose=False)

        # Initialize DMN identifier
        debug_log("Initializing DMN identifier...", is_important=True)
        self.dmn_identifier = DefaultModeNetworkIdentifier(
            self.model, device=self.device, model_name=self.model_name
        )
        debug_log("DMN identifier initialized successfully", verbose=False)

        # Initialize generator with proper type annotation
        self.generator: Optional[InhibitedGenerator] = None
        debug_log("DefaultModeNetworkExperiment initialization complete", divider=True)

    def fetch_top_wikipedia_articles(
        self, n: int = 100, use_cache: bool = True, force_refresh: bool = False
    ) -> List[str]:
        """
        Fetch the top N most viewed Wikipedia articles.

        Args:
            n: Number of top articles to fetch
            use_cache: Whether to use cached list if available
            force_refresh: Whether to force a refresh even if cache is valid

        Returns:
            List of article titles
        """
        debug_log(
            f"FETCH_ARTICLES: Fetching top {n} Wikipedia articles (cache={use_cache}, force_refresh={force_refresh})",
            is_important=True,
        )
        self.articles = fetch_top_wikipedia_articles(
            n=n,
            use_cache=use_cache,
            force_refresh=force_refresh,
            cache_dir=self.cache_dir,
            model_name=self.model_name,
        )
        debug_log(f"Fetched {len(self.articles)} articles", verbose=False)
        return self.articles

    def prepare_text_chunks(
        self, chunk_size: int = DEFAULT_CHUNK_SIZE, use_cache: bool = True
    ) -> List[str]:
        """
        Prepare chunks of Wikipedia articles for processing.

        Args:
            chunk_size: Size of each chunk in tokens
            use_cache: Whether to use cached chunks if available

        Returns:
            List of text chunks
        """
        debug_log(
            f"PREPARE_CHUNKS: Preparing text chunks (chunk_size={chunk_size}, use_cache={use_cache})",
            is_important=True,
        )
        if not self.articles:
            debug_log("No articles loaded. Fetching top Wikipedia articles first...", verbose=False)
            print("No articles loaded. Fetching top Wikipedia articles first...")
            self.fetch_top_wikipedia_articles()

        self.processed_chunks = prepare_text_chunks(
            articles=self.articles,
            tokenizer=self.tokenizer,
            chunk_size=chunk_size,
            use_cache=use_cache,
            cache_dir=self.cache_dir,
            model_name=self.model_name,
        )

        debug_log(f"Prepared {len(self.processed_chunks)} text chunks", verbose=False)
        return self.processed_chunks

    def identify_default_mode_network(
        self, sample_size: int = DEFAULT_SAMPLE_SIZE, use_cache: bool = True
    ) -> None:
        """
        Process Wikipedia chunks to identify the default mode network.

        Args:
            sample_size: Number of chunks to process
            use_cache: Whether to use cached activations if available
        """
        debug_log(
            f"IDENTIFY_DMN: Starting DMN identification (sample_size={sample_size}, use_cache={use_cache})",
            is_important=True,
            divider=True,
        )
        if not self.processed_chunks:
            debug_log("No text chunks prepared. Preparing chunks first...", verbose=False)
            print("No text chunks prepared. Preparing chunks first...")
            self.prepare_text_chunks()

        debug_log(
            f"Using {min(sample_size, len(self.processed_chunks))} chunks for DMN identification",
            verbose=False,
        )
        self.dmn_identifier.identify_default_mode_network(
            chunks=self.processed_chunks,
            tokenizer=self.tokenizer,
            sample_size=sample_size,
            use_cache=use_cache,
        )
        debug_log("DMN identification complete", divider=True)

    def select_top_default_mode_heads(
        self, top_n_per_layer: Optional[int] = None
    ) -> List[Tuple[int, int, float]]:
        """
        Select the top N most active heads from each layer (excluding first and last).

        Args:
            top_n_per_layer: Number of top heads to select from each layer

        Returns:
            List of (layer_idx, head_idx, score) tuples
        """
        # Use configuration value if not provided
        if top_n_per_layer is None:
            top_n_per_layer = DMN_CONFIG["top_n_per_layer"]

        verbose = DMN_CONFIG["verbose_logging"]
        debug_log("Selecting top default mode network heads...", is_important=True)

        if not hasattr(self, "dmn_identifier") or not self.dmn_identifier:
            raise ValueError("Must run identify_default_mode_network() first")

        self.top_default_mode_heads = self.dmn_identifier.select_top_default_mode_heads(
            top_n_per_layer=top_n_per_layer
        )

        # Displaying the top heads
        debug_log(
            f"Selected {len(self.top_default_mode_heads)} heads for DMN", verbose=bool(verbose)
        )

        return self.top_default_mode_heads

    def save_default_mode_network(self, filepath: str) -> None:
        """
        Save the identified default mode network to a file.

        Args:
            filepath: Path to save the data
        """
        self.dmn_identifier.save_default_mode_network(filepath)

    def load_default_mode_network(self, filepath: str) -> None:
        """
        Load a previously identified default mode network.

        Args:
            filepath: Path to the saved data
        """
        self.dmn_identifier.load_default_mode_network(filepath)

        # Initialize generator with loaded DMN heads
        self._initialize_generator()

    def _initialize_generator(self) -> None:
        """
        Initialize the inhibited generator with the identified DMN heads.
        """
        debug_log("INIT_GENERATOR: Initializing inhibited generator", is_important=True)
        if not self.dmn_identifier.top_default_mode_heads:
            error_msg = "No default mode heads identified. Please run identify_default_mode_network() and select_top_default_mode_heads() first, or load a saved network."
            debug_log(error_msg, is_important=True)
            raise ValueError(error_msg)

        assert self.model is not None, "Model must be initialized"
        assert self.tokenizer is not None, "Tokenizer must be initialized"

        debug_log(
            f"Creating generator with {len(self.dmn_identifier.top_default_mode_heads)} DMN heads",
            verbose=False,
        )
        try:
            self.generator = InhibitedGenerator(
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.device,
                top_default_mode_heads=self.dmn_identifier.top_default_mode_heads,
            )
            debug_log("Generator initialized successfully", verbose=False)
        except Exception as e:
            debug_log(f"Error initializing generator: {e}", is_important=True)
            raise

    def generate_with_inhibition(
        self,
        prompt: str,
        inhibition_factor: float = 0.5,
        gamma: float = 0.95,
        max_new_tokens: int = int(DEFAULT_GENERATION_PARAMS["max_new_tokens"]),
        temperature: float = DEFAULT_GENERATION_PARAMS["temperature"],
        top_p: float = DEFAULT_GENERATION_PARAMS["top_p"],
        do_sample: bool = bool(DEFAULT_GENERATION_PARAMS["do_sample"]),
    ) -> Tuple[str, str]:
        """
        Generate text with and without inhibition of default mode network.

        Args:
            prompt: Input prompt
            inhibition_factor: Base factor by which to scale down DMN attention weights (0-1)
            gamma: Decay factor for inhibition across heads (0-1)
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            do_sample: Whether to sample or use greedy decoding

        Returns:
            Tuple of (normal_output, inhibited_output)
        """
        debug_log(
            f"GENERATE: Starting generation with inhibition factor {inhibition_factor}, gamma {gamma}",
            is_important=True,
        )

        # Initialize generator if not already done
        if self.generator is None:
            self._initialize_generator()

        if self.generator is None:
            error_msg = (
                "Generator could not be initialized. DMN must be identified or loaded first."
            )
            debug_log(error_msg, is_important=True)
            raise ValueError(error_msg)

        # Set logging level for transformers
        import logging

        logging.getLogger("transformers").setLevel(logging.ERROR)

        # Generate text
        try:
            normal_output, inhibited_output = self.generator.generate_with_inhibition(
                prompt=prompt,
                inhibition_factor=inhibition_factor,
                gamma=gamma,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
            )
        except Exception as e:
            import traceback

            stack_trace = traceback.format_exc()
            debug_log(f"Error during generation: {e}", is_important=True)
            debug_log(f"Stack trace:\n{stack_trace}", is_important=True)
            raise

        return normal_output, inhibited_output

    def visualize_default_mode_network(
        self, top_n: int = 100, save_path: Optional[str] = None
    ) -> None:
        """
        Visualize the default mode network as a heatmap.

        Args:
            top_n: Number of top heads to visualize
            save_path: Path to save the visualization (if provided)
        """
        visualize_default_mode_network(
            head_importance_scores=self.dmn_identifier.head_importance_scores,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            model_name=self.model_name,
            top_n=top_n,
            save_path=save_path,
        )

    def run_experiment(
        self,
        queries: List[str] = DEFAULT_QUERIES,
        inhibition_factors: List[float] = DEFAULT_INHIBITION_FACTORS,
        gamma: float = 0.95,
        max_new_tokens: int = 100,
        n_chunks: int = 100,
        dmn_file: Optional[str] = None,
        use_cache: bool = True,
        force_article_refresh: bool = False,
        output_dir: str = "results",
        save_intermediate: bool = False,
        top_n_per_layer: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Run a default mode network inhibition experiment.

        Args:
            queries: List of queries to use
            inhibition_factors: List of inhibition factors to test
            gamma: Decay factor for inhibition across heads (0-1)
            max_new_tokens: Maximum number of tokens to generate per response
            n_chunks: Number of Wikipedia chunks to process for DMN identification
            dmn_file: Path to pre-identified DMN file (if available)
            use_cache: Whether to use cached activations if available
            force_article_refresh: Whether to force refresh of Wikipedia articles
            output_dir: Directory to save outputs
            save_intermediate: Whether to save results after each query
            top_n_per_layer: Number of top heads to select per layer

        Returns:
            List of experiment results
        """
        debug_log(
            f"RUN_EXPERIMENT: Starting experiment with {len(queries)} queries and {len(inhibition_factors)} inhibition factors",
            is_important=True,
            divider=True,
        )
        debug_log(
            f"Parameters: gamma={gamma}, max_tokens={max_new_tokens}, output_dir={output_dir}"
        )

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Initialize DMN identification if not already done
        if dmn_file and os.path.exists(dmn_file):
            debug_log(f"Loading DMN from file: {dmn_file}", is_important=True)
            self.load_default_mode_network(dmn_file)
        elif (
            not hasattr(self.dmn_identifier, "top_default_mode_heads")
            or not self.dmn_identifier.top_default_mode_heads
        ):
            debug_log(
                "DMN not identified yet. Running identification process...", is_important=True
            )
            print("Identifying default mode network...")
            self.fetch_top_wikipedia_articles(n=n_chunks, force_refresh=force_article_refresh)
            self.prepare_text_chunks(use_cache=use_cache)
            self.identify_default_mode_network(sample_size=n_chunks, use_cache=use_cache)
            self.select_top_default_mode_heads(top_n_per_layer=top_n_per_layer)

            # Save the identified DMN to the output directory
            dmn_file = os.path.join(output_dir, "dmn_heads.pkl")
            print(f"Saving identified DMN to: {dmn_file}")
            self.save_default_mode_network(dmn_file)

            # Also save as JSON for easier inspection
            dmn_json_file = os.path.join(output_dir, "dmn_heads.json")
            try:
                with open(dmn_json_file, "w") as f:
                    json.dump(
                        [
                            {"layer": layer_idx, "head": head_idx, "score": float(score)}
                            for layer_idx, head_idx, score in self.dmn_identifier.top_default_mode_heads
                        ],
                        f,
                        indent=2,
                    )
                print(f"Saved DMN as JSON to: {dmn_json_file}")
            except Exception as e:
                print(f"Error saving DMN JSON: {e}")

        # Show heads ordered from most to least active
        print("\nTop 10 DMN heads ordered by importance:")
        for i, (layer, head, score) in enumerate(self.dmn_identifier.top_default_mode_heads[:10]):
            print(f"#{i+1}: Layer {layer}, Head {head}, Score {score:.4f}")
        print("\n")

        # Initialize results store
        results = []

        # Run the experiment for each query and inhibition factor
        for query_idx, query in enumerate(queries):
            print(f"\nQuery {query_idx+1}/{len(queries)}: {query}")
            query_results = []

            for factor_idx, factor in enumerate(inhibition_factors):
                print(f"  Inhibition factor: {factor}, Gamma: {gamma}")

                try:
                    normal_output, inhibited_output = self.generate_with_inhibition(
                        prompt=query,
                        inhibition_factor=factor,
                        gamma=gamma,
                        max_new_tokens=max_new_tokens,
                    )

                    result = {
                        "query": query,
                        "query_idx": query_idx,
                        "inhibition_factor": factor,
                        "gamma": gamma,
                        "normal_output": normal_output,
                        "inhibited_output": inhibited_output,
                        "timestamp": datetime.now().isoformat(),
                    }

                    # Calculate simple metrics
                    normal_tokens = len(self.tokenizer.encode(normal_output))
                    inhibited_tokens = len(self.tokenizer.encode(inhibited_output))
                    result["normal_token_count"] = normal_tokens
                    result["inhibited_token_count"] = inhibited_tokens

                    # Save result
                    query_results.append(result)
                    results.append(result)

                    # Save intermediate visualization for this query and inhibition factor
                    if save_intermediate:
                        query_safe = query.replace(" ", "_")[:30].replace("?", "").replace("/", "_")
                        factor_str = f"{factor:.1f}"
                        output_path = os.path.join(
                            output_dir,
                            f"query_{query_idx+1}_{query_safe}_factor_{factor_str}_gamma_{gamma:.2f}.txt",
                        )

                        # Add debug logging
                        debug_log(f"Attempting to save output to: {output_path}", is_important=True)
                        debug_log(
                            f"Output directory exists: {os.path.exists(output_dir)}",
                            is_important=True,
                        )

                        # No need to import again - already imported at the top
                        # Convert the single result to a list for save_query_outputs
                        try:
                            # Fix the function call to match the expected parameters
                            save_query_outputs(
                                results=[result],
                                model_name=self.model_name,
                                output_dir=os.path.dirname(output_path),
                                suffix=os.path.basename(output_path).replace(
                                    f"{self.model_name}_all_outputs", ""
                                ),
                                save_individual_files=False,
                            )
                            debug_log("Successfully called save_query_outputs", is_important=True)
                            # Check if the file was actually created
                            output_file = os.path.join(
                                os.path.dirname(output_path),
                                f"{self.model_name.replace('/', '-').replace('.', '-')}_all_outputs.txt",
                            )
                            debug_log(f"Expected output file: {output_file}", is_important=True)
                            debug_log(
                                f"Output file exists: {os.path.exists(output_file)}",
                                is_important=True,
                            )
                            print(f"    Saved intermediate result to {output_path}")
                        except Exception as e:
                            debug_log(f"Error in save_query_outputs: {e}", is_important=True)
                            import traceback

                            debug_log(f"Traceback: {traceback.format_exc()}", is_important=True)
                            print(f"    Error saving to {output_path}: {e}")

                except Exception as e:
                    print(f"Error generating for query {query_idx+1} with factor {factor}: {e}")
                    debug_log(f"Error: {e}", is_important=True)
                    # Add error result
                    results.append(
                        {
                            "query": query,
                            "query_idx": query_idx,
                            "inhibition_factor": factor,
                            "gamma": gamma,
                            "error": str(e),
                            "timestamp": datetime.now().isoformat(),
                        }
                    )

            # Generate query-level visualization
            if save_intermediate and len(query_results) > 0:
                query_safe = query.replace(" ", "_")[:30].replace("?", "").replace("/", "_")
                save_path = os.path.join(
                    output_dir, f"query_{query_idx+1}_{query_safe}_summary.png"
                )
                try:
                    analyze_results(query_results, save_path=save_path)
                    print(f"  Saved query analysis to {save_path}")
                except Exception as e:
                    print(f"  Error saving query analysis: {e}")

        debug_log(f"Experiment complete with {len(results)} total data points", is_important=True)
        return results

    def analyze_results(
        self, results: List[Dict[str, Any]], save_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Analyze the results of the experiment.

        Args:
            results: Results from run_experiment
            save_path: Path to save the analysis visualization

        Returns:
            DataFrame with analysis metrics
        """
        return analyze_results(results=results, save_path=save_path, model_name=self.model_name)
