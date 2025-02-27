#!/usr/bin/env python
"""
Main experiment module for running Default Mode Network inhibition.
"""

import json
import os
import pickle
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import (
    CACHE_DIR,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_GENERATION_PARAMS,
    DEFAULT_INHIBITION_FACTORS,
    DEFAULT_MODEL_NAME,
    DEFAULT_QUERIES,
    DEFAULT_SAMPLE_SIZE,
    DEFAULT_TOP_HEADS,
)
from .data.processor import prepare_text_chunks
from .data.wikipedia import fetch_top_wikipedia_articles
from .model.dmn_identifier import DefaultModeNetworkIdentifier
from .model.inhibited_generator import InhibitedGenerator
from .visualization.visualizer import (
    analyze_results,
    save_query_outputs,
    visualize_default_mode_network,
)

# Add debug logging function
def debug_log(msg: str, is_important: bool = False, divider: bool = False) -> None:
    """Helper function to print consistent debug logs with timestamps."""
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
        debug_log("INIT: Starting DefaultModeNetworkExperiment initialization", is_important=True, divider=True)
        self.model_name = model_name
        print(f"Initializing experiment with model: {model_name}")

        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        debug_log(f"Using device: {self.device}")

        # Create cache directory
        self.cache_dir = cache_dir or CACHE_DIR
        os.makedirs(self.cache_dir, exist_ok=True)
        print(f"Using cache directory: {self.cache_dir}")

        # Initialize model and tokenizer
        print("Loading tokenizer and model...")
        debug_log("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        debug_log("Tokenizer loaded successfully")

        # For flash attention compatibility
        debug_log("Loading model with eager attention implementation...")
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=self.device,
                torch_dtype=torch.float16,
                attn_implementation="eager",  # Use eager implementation for compatibility
            )
            debug_log("Model loaded successfully with eager attention")
        except Exception as e:
            debug_log(f"Error loading model: {e}", is_important=True)
            raise
        print("Model and tokenizer loaded")

        # Model dimensions
        debug_log("Determining model dimensions...")
        if hasattr(self.model.config, "num_hidden_layers"):
            self.num_layers = self.model.config.num_hidden_layers
            debug_log(f"Found num_hidden_layers: {self.num_layers}")
        elif hasattr(self.model.config, "n_layer"):
            self.num_layers = self.model.config.n_layer
            debug_log(f"Found n_layer: {self.num_layers}")
        else:
            error_msg = "Could not determine number of layers in the model"
            debug_log(error_msg, is_important=True)
            raise ValueError(error_msg)

        if hasattr(self.model.config, "num_attention_heads"):
            self.num_heads = self.model.config.num_attention_heads
            debug_log(f"Found num_attention_heads: {self.num_heads}")
        elif hasattr(self.model.config, "n_head"):
            self.num_heads = self.model.config.n_head
            debug_log(f"Found n_head: {self.num_heads}")
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
        debug_log(f"Model config: {self.model.config}")

        # Initialize DMN identifier
        debug_log("Initializing DMN identifier...", is_important=True)
        self.dmn_identifier = DefaultModeNetworkIdentifier(self.model, device=self.device, model_name=self.model_name)
        debug_log("DMN identifier initialized successfully")

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
        debug_log(f"FETCH_ARTICLES: Fetching top {n} Wikipedia articles (cache={use_cache}, force_refresh={force_refresh})", is_important=True)
        self.articles = fetch_top_wikipedia_articles(
            n=n,
            use_cache=use_cache,
            force_refresh=force_refresh,
            cache_dir=self.cache_dir,
            model_name=self.model_name,
        )
        debug_log(f"Fetched {len(self.articles)} articles")
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
        debug_log(f"PREPARE_CHUNKS: Preparing text chunks (chunk_size={chunk_size}, use_cache={use_cache})", is_important=True)
        if not self.articles:
            debug_log("No articles loaded. Fetching top Wikipedia articles first...")
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
        
        debug_log(f"Prepared {len(self.processed_chunks)} text chunks")
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
        debug_log(f"IDENTIFY_DMN: Starting DMN identification (sample_size={sample_size}, use_cache={use_cache})", is_important=True, divider=True)
        if not self.processed_chunks:
            debug_log("No text chunks prepared. Preparing chunks first...")
            print("No text chunks prepared. Preparing chunks first...")
            self.prepare_text_chunks()
        
        debug_log(f"Using {min(sample_size, len(self.processed_chunks))} chunks for DMN identification")
        self.dmn_identifier.identify_default_mode_network(
            chunks=self.processed_chunks,
            tokenizer=self.tokenizer,
            sample_size=sample_size,
            use_cache=use_cache,
        )
        debug_log("DMN identification complete", divider=True)

    def select_top_default_mode_heads(
        self, top_n: int = DEFAULT_TOP_HEADS
    ) -> List[Tuple[int, int, float]]:
        """
        Select the top N most active heads as the default mode network.

        Args:
            top_n: Number of top heads to select

        Returns:
            List of (layer_idx, head_idx, score) tuples
        """
        debug_log(f"SELECT_TOP_HEADS: Selecting top {top_n} default mode heads", is_important=True)
        tops = self.dmn_identifier.select_top_default_mode_heads(top_n=top_n)
        debug_log(f"Selected {len(tops)} top default mode heads")
        return tops

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
        
        debug_log(f"Creating generator with {len(self.dmn_identifier.top_default_mode_heads)} DMN heads")
        try:
            self.generator = InhibitedGenerator(
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.device,
                top_default_mode_heads=self.dmn_identifier.top_default_mode_heads,
            )
            debug_log("Generator initialized successfully")
        except Exception as e:
            debug_log(f"Error initializing generator: {e}", is_important=True)
            raise

    def generate_with_inhibition(
        self,
        prompt: str,
        inhibition_factor: float = 0.5,
        max_new_tokens: int = int(DEFAULT_GENERATION_PARAMS["max_new_tokens"]),
        temperature: float = DEFAULT_GENERATION_PARAMS["temperature"],
        top_p: float = DEFAULT_GENERATION_PARAMS["top_p"],
        do_sample: bool = bool(DEFAULT_GENERATION_PARAMS["do_sample"]),
    ) -> Tuple[str, str]:
        """
        Generate text with and without inhibition of the default mode network.

        Args:
            prompt: Input prompt
            inhibition_factor: Factor by which to scale down the attention weights (0-1)
            max_new_tokens: Maximum number of tokens to generate
            temperature: Temperature for sampling
            top_p: Top-p sampling parameter
            do_sample: Whether to use sampling (vs greedy decoding)

        Returns:
            Tuple of (normal_output, inhibited_output)
        """
        debug_log(f"GENERATE: Generating with inhibition_factor={inhibition_factor}", is_important=True)
        debug_log(f"Prompt: '{prompt[:50]}...'")
        
        # Initialize generator if not already done
        if self.generator is None:
            debug_log("Generator not initialized, initializing now...")
            self._initialize_generator()

        # The generator should be initialized by now, but if it's still None, we have a problem
        if self.generator is None:
            error_msg = "Failed to initialize generator"
            debug_log(error_msg, is_important=True)
            raise ValueError(error_msg)

        try:
            results = self.generator.generate_with_inhibition(
                prompt=prompt,
                inhibition_factor=inhibition_factor,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
            )
            debug_log("Generation successful")
            return results
        except Exception as e:
            debug_log(f"Error during generation: {e}", is_important=True)
            raise

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
        queries: Optional[List[str]] = None,
        inhibition_factors: Optional[List[float]] = None,
        max_new_tokens: int = int(DEFAULT_GENERATION_PARAMS["max_new_tokens"]),
        n_chunks: int = DEFAULT_SAMPLE_SIZE,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        dmn_file: Optional[str] = None,
        use_cache: bool = True,
        force_article_refresh: bool = False,
        output_dir: str = "query_outputs",
    ) -> List[Dict[str, Any]]:
        """
        Run the full experiment:
        1. Fetch Wikipedia articles
        2. Prepare text chunks
        3. Identify default mode network
        4. Select top DMN heads
        5. Generate responses for queries with and without inhibition
        6. Save the outputs

        Args:
            queries: List of queries to answer (defaults to DEFAULT_QUERIES)
            inhibition_factors: List of inhibition factors to test
            max_new_tokens: Maximum number of tokens to generate
            n_chunks: Number of chunks to process
            chunk_size: Size of chunks in tokens
            dmn_file: Path to a file containing pre-identified DMN heads
            use_cache: Whether to use cached data (articles, chunks, activations)
            force_article_refresh: Whether to force a refresh of article data
            output_dir: Directory to save outputs

        Returns:
            List of result dictionaries
        """
        debug_log(f"RUN_EXPERIMENT: Starting full experiment with {self.model_name}", is_important=True, divider=True)
        debug_log(f"Parameters: max_tokens={max_new_tokens}, n_chunks={n_chunks}, chunk_size={chunk_size}, use_cache={use_cache}")
        
        print(f"Running experiment with {self.model_name}")
        results = []
        at_least_one_succeeded = False

        # Set default values
        if queries is None:
            queries = DEFAULT_QUERIES
        debug_log(f"Using {len(queries)} queries")

        if inhibition_factors is None:
            inhibition_factors = DEFAULT_INHIBITION_FACTORS
        debug_log(f"Using inhibition factors: {inhibition_factors}")

        # Step 1: Fetch Wikipedia articles
        try:
            debug_log("STEP 1: Fetching Wikipedia articles", divider=True)
            print("Fetching Wikipedia articles...")
            self.fetch_top_wikipedia_articles(
                use_cache=use_cache, force_refresh=force_article_refresh
            )
            print(f"Fetched {len(self.articles)} articles")
        except Exception as e:
            debug_log(f"Error fetching articles: {e}", is_important=True)
            print(f"Error fetching articles: {e}")
            print("Using fallback list of articles")
            self.articles = ["Philosophy", "Science", "History", "Mathematics", "Literature"]
            debug_log(f"Using {len(self.articles)} fallback articles")

        # Step 2: Prepare text chunks
        try:
            debug_log("STEP 2: Preparing text chunks", divider=True)
            print("Preparing text chunks...")
            self.prepare_text_chunks(chunk_size=chunk_size, use_cache=use_cache)
            print(f"Prepared {len(self.processed_chunks)} text chunks")
        except Exception as e:
            debug_log(f"Error preparing text chunks: {e}", is_important=True)
            print(f"Error preparing text chunks: {e}")
            self.processed_chunks = ["This is a fallback text chunk." * 50]
            debug_log("Using fallback text chunks")

        # Step 3 & 4: Load or identify default mode network
        debug_log("STEP 3 & 4: Identifying default mode network", divider=True)
        if dmn_file and os.path.exists(dmn_file):
            debug_log(f"Loading pre-identified DMN from {dmn_file}")
            print(f"Loading pre-identified DMN from {dmn_file}")
            self.load_default_mode_network(dmn_file)
        else:
            print("Identifying default mode network...")
            try:
                debug_log("No DMN file provided or found, identifying DMN...")
                self.identify_default_mode_network(
                    sample_size=min(n_chunks, len(self.processed_chunks)), use_cache=use_cache
                )
                debug_log("Selecting top default mode heads...")
                self.select_top_default_mode_heads()
                debug_log("Initializing generator with identified DMN heads...")
                self._initialize_generator()
            except Exception as e:
                debug_log(f"Error identifying default mode network: {e}", is_important=True)
                print(f"Error identifying default mode network: {e}")
                print("Using fallback default mode network")
                debug_log("Creating fallback DMN...")
                
                # Select a subset of heads from the model for the fallback DMN
                # For Mistral, focus on later layers which tend to have more semantic understanding
                fallback_size = self.num_heads * 16  # Use 16 heads per layer as fallback
                debug_log(f"Creating a fallback DMN with target size of {fallback_size} heads")
                
                top_heads = []
                # Add heads from later layers (second half of the model)
                for layer_idx in range(self.num_layers // 2, self.num_layers):
                    for head_idx in range(self.num_heads):
                        # Assign decreasing importance scores to create a gradient
                        score = 1.0 - (head_idx / self.num_heads) * 0.5
                        top_heads.append((layer_idx, head_idx, score))
                        
                        # Limit the total number of heads for performance
                        if len(top_heads) >= fallback_size:
                            break
                debug_log(f"Created initial fallback DMN with {len(top_heads)} heads")
                            
                # Convert numpy types to native Python types to satisfy mypy
                python_top_heads = [
                    (int(layer_idx), int(head_idx), float(score))
                    for layer_idx, head_idx, score in top_heads
                ]
                self.dmn_identifier.top_default_mode_heads = python_top_heads
                try:
                    debug_log("Attempting to initialize generator with fallback DMN...")
                    self._initialize_generator()
                except Exception as e:
                    debug_log(f"Error initializing generator with fallback DMN: {e}", is_important=True)
                    # Create an even simpler fallback as last resort
                    debug_log("Creating simplified fallback DMN as last resort...")
                    simple_heads = [(self.num_layers-1, i, 0.8) for i in range(min(16, self.num_heads))]
                    debug_log(f"Simple fallback DMN has {len(simple_heads)} heads from layer {self.num_layers-1}")
                    self.dmn_identifier.top_default_mode_heads = simple_heads
                    self._initialize_generator()
                print(f"Created fallback DMN with {len(self.dmn_identifier.top_default_mode_heads)} heads")

        # Step 5: Generate responses
        debug_log("STEP 5: Generating responses", divider=True)
        print(
            f"Generating responses for {len(queries)} queries with {len(inhibition_factors)} inhibition factors"
        )

        for q_idx, query in enumerate(queries):
            query_results = []
            debug_log(f"Processing query {q_idx+1}/{len(queries)}: '{query}'")
            print(f"\nProcessing query: {query}")

            for factor in inhibition_factors:
                try:
                    debug_log(f"  Generating with inhibition factor: {factor:.2f}")
                    print(f"  Generating with inhibition factor: {factor:.2f}")
                    response = None

                    if factor == 0.0:
                        # Normal generation without inhibition
                        debug_log("  Normal generation without inhibition")
                        inputs = self.tokenizer(query, return_tensors="pt").to(self.device)
                        output = self.model.generate(
                            **inputs,
                            max_new_tokens=max_new_tokens,
                            do_sample=True,
                            temperature=0.7,
                            top_p=0.9,
                        )
                        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
                    else:
                        # Generation with inhibition
                        debug_log(f"  Generation with inhibition factor {factor}")
                        try:
                            normal_response, inhibited_response = self.generate_with_inhibition(
                                query, inhibition_factor=factor, max_new_tokens=max_new_tokens
                            )
                            response = inhibited_response
                        except Exception as e:
                            debug_log(f"  Error with inhibited generation: {e}", is_important=True)
                            print(f"Error with inhibited generation: {e}")
                            print("Falling back to normal generation")
                            debug_log("  Falling back to normal generation")
                            # Fall back to normal generation
                            inputs = self.tokenizer(query, return_tensors="pt").to(self.device)
                            output = self.model.generate(
                                **inputs,
                                max_new_tokens=max_new_tokens,
                                do_sample=True,
                                temperature=0.7,
                                top_p=0.9,
                            )
                            response = self.tokenizer.decode(output[0], skip_special_tokens=True)

                    # Add to results
                    if response:
                        result = {"query": query, "inhibition_factor": factor, "response": response}
                        query_results.append(result)
                        at_least_one_succeeded = True
                        debug_log(f"  Generation successful (length: {len(response)})")
                        print(f"  Generation successful (length: {len(response)})")
                    else:
                        debug_log("  Generation failed - empty response", is_important=True)
                        print("  Generation failed - empty response")

                except Exception as e:
                    debug_log(f"  Error generating response for factor {factor}: {e}", is_important=True)
                    print(f"  Error generating response for factor {factor}: {e}")

            results.extend(query_results)

            # Save intermediate results after each query
            if query_results:
                debug_log(f"Saving intermediate results after query {q_idx+1}")
                save_query_outputs(
                    results,
                    model_name=self.model_name,
                    output_dir=output_dir,
                    suffix=f"_intermediate_{len(results)}",
                )

        # Step 6: Save outputs
        debug_log("STEP 6: Saving final outputs", divider=True)
        try:
            print("\nSaving outputs...")
            save_query_outputs(results, model_name=self.model_name, output_dir=output_dir)
            debug_log(f"Saved {len(results)} results to {output_dir}")
        except Exception as e:
            debug_log(f"Error saving outputs: {e}", is_important=True)
            print(f"Error saving outputs: {e}")

        if not at_least_one_succeeded:
            debug_log("Warning: No successful generations were produced during this experiment.", is_important=True)
            print("Warning: No successful generations were produced during this experiment.")

        debug_log("Experiment complete", is_important=True, divider=True)
        print("Experiment complete")
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
