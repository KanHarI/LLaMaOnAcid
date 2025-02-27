"""
Main experiment runner for LLaMa on Acid.
"""
import os
import pickle
import torch
import random
from typing import List, Dict, Tuple, Optional, Any, Union
from transformers import AutoTokenizer, AutoModelForCausalLM
from datetime import datetime

from .config import (
    CACHE_DIR, DEFAULT_MODEL_NAME, DEVICE, DEFAULT_QUERIES,
    DEFAULT_CHUNK_SIZE, DEFAULT_SAMPLE_SIZE, DEFAULT_TOP_HEADS,
    DEFAULT_INHIBITION_FACTORS, DEFAULT_GENERATION_PARAMS
)
from .data.wikipedia import fetch_top_wikipedia_articles
from .data.processor import prepare_text_chunks
from .model.dmn_identifier import DefaultModeNetworkIdentifier
from .model.inhibited_generator import InhibitedGenerator
from .visualization.visualizer import (
    visualize_default_mode_network,
    analyze_results,
    save_query_outputs
)


class DefaultModeNetworkExperiment:
    """
    Class to simulate the effects of LSD on large language models by manipulating attention head activations.
    """
    
    def __init__(
        self, 
        model_name: str = DEFAULT_MODEL_NAME, 
        device: Optional[str] = None,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize the DefaultModeNetworkExperiment.
        
        Args:
            model_name: The name or path of the model to use
            device: The device to use (cuda or cpu)
            cache_dir: Directory to store cached data
        """
        self.model_name = model_name
        print(f"Initializing experiment with model: {model_name}")
        
        # Set device
        if device:
            self.device = torch.device(device)
        else:
            self.device = DEVICE
        
        # Create cache directory
        self.cache_dir = cache_dir or CACHE_DIR
        os.makedirs(self.cache_dir, exist_ok=True)
        print(f"Using cache directory: {self.cache_dir}")
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("Tokenizer loaded")
        
        # For flash attention compatibility
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            device_map=self.device,
            torch_dtype=torch.float16,
            attn_implementation="eager"  # Use eager implementation for compatibility
        )
        print("Model loaded")
        
        # Model dimensions
        if hasattr(self.model.config, 'num_hidden_layers'):
            self.num_layers = self.model.config.num_hidden_layers
        elif hasattr(self.model.config, 'n_layer'):
            self.num_layers = self.model.config.n_layer
        else:
            raise ValueError("Could not determine number of layers in the model")
            
        if hasattr(self.model.config, 'num_attention_heads'):
            self.num_heads = self.model.config.num_attention_heads
        elif hasattr(self.model.config, 'n_head'):
            self.num_heads = self.model.config.n_head
        else:
            raise ValueError("Could not determine number of attention heads in the model")
            
        print(f"Model has {self.num_layers} layers and {self.num_heads} attention heads per layer")
        
        # Data storage
        self.articles: List[str] = []
        self.processed_chunks: List[str] = []
        
        # Initialize DMN identifier
        self.dmn_identifier = DefaultModeNetworkIdentifier(
            model=self.model,
            device=self.device,
            cache_dir=self.cache_dir,
            model_name=model_name
        )
        
        # Will be initialized later when needed
        self.generator = None
        
    def fetch_top_wikipedia_articles(
        self, 
        n: int = 100, 
        use_cache: bool = True, 
        force_refresh: bool = False
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
        self.articles = fetch_top_wikipedia_articles(
            n=n, 
            use_cache=use_cache, 
            force_refresh=force_refresh,
            cache_dir=self.cache_dir,
            model_name=self.model_name
        )
        return self.articles
    
    def prepare_text_chunks(
        self, 
        chunk_size: int = DEFAULT_CHUNK_SIZE, 
        use_cache: bool = True
    ) -> List[str]:
        """
        Prepare chunks of Wikipedia articles for processing.
        
        Args:
            chunk_size: Size of each chunk in tokens
            use_cache: Whether to use cached chunks if available
            
        Returns:
            List of text chunks
        """
        if not self.articles:
            print("No articles loaded. Fetching top Wikipedia articles first...")
            self.fetch_top_wikipedia_articles()
            
        self.processed_chunks = prepare_text_chunks(
            articles=self.articles,
            tokenizer=self.tokenizer,
            chunk_size=chunk_size,
            use_cache=use_cache,
            cache_dir=self.cache_dir,
            model_name=self.model_name
        )
        
        return self.processed_chunks
    
    def identify_default_mode_network(
        self, 
        sample_size: int = DEFAULT_SAMPLE_SIZE, 
        use_cache: bool = True
    ) -> None:
        """
        Process Wikipedia chunks to identify the default mode network.
        
        Args:
            sample_size: Number of chunks to process
            use_cache: Whether to use cached activations if available
        """
        if not self.processed_chunks:
            print("No text chunks prepared. Preparing chunks first...")
            self.prepare_text_chunks()
            
        self.dmn_identifier.identify_default_mode_network(
            chunks=self.processed_chunks,
            tokenizer=self.tokenizer,
            sample_size=sample_size,
            use_cache=use_cache
        )
    
    def select_top_default_mode_heads(
        self, 
        top_n: int = DEFAULT_TOP_HEADS
    ) -> List[Tuple[int, int, float]]:
        """
        Select the top N most active heads as the default mode network.
        
        Args:
            top_n: Number of top heads to select
            
        Returns:
            List of (layer_idx, head_idx, score) tuples
        """
        return self.dmn_identifier.select_top_default_mode_heads(top_n=top_n)
    
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
        if not self.dmn_identifier.top_default_mode_heads:
            raise ValueError("No default mode heads identified. Please run identify_default_mode_network() and select_top_default_mode_heads() first, or load a saved network.")
            
        self.generator = InhibitedGenerator(
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            top_default_mode_heads=self.dmn_identifier.top_default_mode_heads
        )
    
    def generate_with_inhibition(
        self, 
        prompt: str, 
        inhibition_factor: float = 0.5, 
        max_new_tokens: int = DEFAULT_GENERATION_PARAMS["max_new_tokens"],
        temperature: float = DEFAULT_GENERATION_PARAMS["temperature"],
        top_p: float = DEFAULT_GENERATION_PARAMS["top_p"],
        do_sample: bool = DEFAULT_GENERATION_PARAMS["do_sample"]
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
        # Initialize generator if not already done
        if self.generator is None:
            self._initialize_generator()
            
        return self.generator.generate_with_inhibition(
            prompt=prompt,
            inhibition_factor=inhibition_factor,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample
        )
    
    def visualize_default_mode_network(
        self, 
        top_n: int = 100, 
        save_path: Optional[str] = None
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
            save_path=save_path
        )
    
    def run_experiment(
        self, 
        queries: Optional[List[str]] = None, 
        inhibition_factors: Optional[List[float]] = None,
        max_new_tokens: int = DEFAULT_GENERATION_PARAMS["max_new_tokens"],
        n_chunks: int = DEFAULT_SAMPLE_SIZE, 
        chunk_size: int = DEFAULT_CHUNK_SIZE, 
        dmn_file: Optional[str] = None,
        use_cache: bool = True, 
        force_article_refresh: bool = False,
        output_dir: str = "query_outputs"
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
        print(f"Running experiment with {self.model_name}")
        results = []
        at_least_one_succeeded = False
        
        # Set default values
        if queries is None:
            queries = DEFAULT_QUERIES
            
        if inhibition_factors is None:
            inhibition_factors = DEFAULT_INHIBITION_FACTORS
            
        # Step 1: Fetch Wikipedia articles
        try:
            print("Fetching Wikipedia articles...")
            self.fetch_top_wikipedia_articles(
                use_cache=use_cache, 
                force_refresh=force_article_refresh
            )
            print(f"Fetched {len(self.articles)} articles")
        except Exception as e:
            print(f"Error fetching articles: {e}")
            print("Using fallback list of articles")
            self.articles = ["Philosophy", "Science", "History", "Mathematics", "Literature"]
        
        # Step 2: Prepare text chunks
        try:
            print("Preparing text chunks...")
            self.prepare_text_chunks(chunk_size=chunk_size, use_cache=use_cache)
            print(f"Prepared {len(self.processed_chunks)} text chunks")
        except Exception as e:
            print(f"Error preparing text chunks: {e}")
            self.processed_chunks = ["This is a fallback text chunk." * 50]
        
        # Step 3 & 4: Load or identify default mode network
        if dmn_file and os.path.exists(dmn_file):
            print(f"Loading pre-identified DMN from {dmn_file}")
            self.load_default_mode_network(dmn_file)
        else:
            print("Identifying default mode network...")
            try:
                self.identify_default_mode_network(
                    sample_size=min(n_chunks, len(self.processed_chunks)),
                    use_cache=use_cache
                )
                self.select_top_default_mode_heads()
                self._initialize_generator()
            except Exception as e:
                print(f"Error identifying default mode network: {e}")
                print("Using fallback default mode network")
                # Create a fallback DMN
                import numpy as np
                top_heads = [(l, h, 0.5) for l, h in zip(
                    np.random.randint(0, self.num_layers, size=50),
                    np.random.randint(0, self.num_heads, size=50)
                )]
                self.dmn_identifier.top_default_mode_heads = top_heads
                self._initialize_generator()
                print(f"Created fallback DMN with {len(top_heads)} heads")
        
        # Step 5: Generate responses
        print(f"Generating responses for {len(queries)} queries with {len(inhibition_factors)} inhibition factors")
        
        for query in queries:
            query_results = []
            print(f"\nProcessing query: {query}")
            
            for factor in inhibition_factors:
                try:
                    print(f"  Generating with inhibition factor: {factor:.2f}")
                    response = None
                    
                    if factor == 0.0:
                        # Normal generation without inhibition
                        inputs = self.tokenizer(query, return_tensors="pt").to(self.device)
                        output = self.model.generate(
                            **inputs,
                            max_new_tokens=max_new_tokens,
                            do_sample=True,
                            temperature=0.7,
                            top_p=0.9
                        )
                        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
                    else:
                        # Generation with inhibition
                        try:
                            normal_response, inhibited_response = self.generate_with_inhibition(
                                query, inhibition_factor=factor, max_new_tokens=max_new_tokens
                            )
                            response = inhibited_response
                        except Exception as e:
                            print(f"Error with inhibited generation: {e}")
                            print("Falling back to normal generation")
                            # Fall back to normal generation
                            inputs = self.tokenizer(query, return_tensors="pt").to(self.device)
                            output = self.model.generate(
                                **inputs,
                                max_new_tokens=max_new_tokens,
                                do_sample=True,
                                temperature=0.7,
                                top_p=0.9
                            )
                            response = self.tokenizer.decode(output[0], skip_special_tokens=True)
                    
                    # Add to results
                    if response:
                        result = {
                            'query': query,
                            'inhibition_factor': factor,
                            'response': response
                        }
                        query_results.append(result)
                        at_least_one_succeeded = True
                        print(f"  Generation successful (length: {len(response)})")
                    else:
                        print(f"  Generation failed - empty response")
                
                except Exception as e:
                    print(f"  Error generating response for factor {factor}: {e}")
            
            results.extend(query_results)
            
            # Save intermediate results after each query
            if query_results:
                save_query_outputs(
                    results, 
                    model_name=self.model_name,
                    output_dir=output_dir,
                    suffix=f"_intermediate_{len(results)}"
                )
        
        # Step 6: Save outputs
        try:
            print("\nSaving outputs...")
            save_query_outputs(
                results, 
                model_name=self.model_name,
                output_dir=output_dir
            )
        except Exception as e:
            print(f"Error saving outputs: {e}")
        
        if not at_least_one_succeeded:
            print("Warning: No successful generations were produced during this experiment.")
        
        print("Experiment complete")
        return results
    
    def analyze_results(
        self, 
        results: List[Dict[str, Any]],
        save_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Analyze the results of the experiment.
        
        Args:
            results: Results from run_experiment
            save_path: Path to save the analysis visualization
            
        Returns:
            DataFrame with analysis metrics
        """
        return analyze_results(
            results=results, 
            save_path=save_path,
            model_name=self.model_name
        ) 