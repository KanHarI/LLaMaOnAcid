import os
import random
import torch
import numpy as np
import pandas as pd
import requests
import json
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Union
import pickle
import time
from collections import defaultdict
from datetime import datetime, timedelta

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DefaultModeNetworkExperiment:
    """
    Class to simulate the effects of LSD on large language models by manipulating attention head activations.
    """
    
    def __init__(self, model_name: str = "meta-llama/Llama-3-70b", cache_dir: Optional[str] = None):
        """
        Initialize the experiment with the specified model.
        
        Args:
            model_name: Name of the HuggingFace model to use
            cache_dir: Directory to cache the model weights
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        
        print(f"Loading tokenizer for {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        
        print(f"Loading model {model_name}...")
        # Load model with 8-bit quantization to save memory
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            device_map="auto",  # Automatically distribute across available GPUs
            # load_in_8bit=True,  # Use 8-bit quantization
            torch_dtype=torch.float16,  # Use half precision
            attn_implementation="flash_attention_2",  # Use FlashAttention for better memory efficiency
        )
        
        # Model architecture properties
        if hasattr(self.model.config, "num_hidden_layers"):
            self.num_layers = self.model.config.num_hidden_layers
        else:
            self.num_layers = self.model.config.n_layer
            
        if hasattr(self.model.config, "num_attention_heads"):
            self.num_heads = self.model.config.num_attention_heads
        else:
            self.num_heads = self.model.config.n_head
            
        print(f"Model loaded with {self.num_layers} layers and {self.num_heads} attention heads per layer")
        
        # Storage for activations
        self.default_mode_activations = {}
        self.head_importance_scores = None
        self.top_default_mode_heads = []
        
        # Wikipedia data
        self.wikipedia_articles = []
        self.processed_chunks = []
        
    def fetch_top_wikipedia_articles(self, n: int = 100) -> List[str]:
        """
        Fetch the top N most viewed Wikipedia articles.
        
        Args:
            n: Number of top articles to fetch
            
        Returns:
            List of article titles
        """
        print(f"Fetching top {n} Wikipedia articles...")
        
        # Get the most viewed articles from the past month
        # Use current date instead of hardcoded 2023/10
        current_date = datetime.now()
        prev_month = current_date - timedelta(days=30)
        year = prev_month.strftime("%Y")
        month = prev_month.strftime("%m")
        
        url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/top/en.wikipedia/all-access/{year}/{month}/all-days"
        print(f"Requesting data from URL: {url}")
        
        try:
            response = requests.get(url)
            response.raise_for_status()  # Raise an exception for HTTP errors
            data = response.json()
            
            # Extract article titles
            articles = []
            for item in data["items"][0]["articles"]:
                if "Main_Page" not in item["article"] and "Special:" not in item["article"]:
                    articles.append(item["article"])
                    if len(articles) >= n:
                        break
            
            self.wikipedia_articles = articles
            print(f"Fetched {len(articles)} articles")
            return articles
            
        except requests.exceptions.HTTPError as http_err:
            print(f"HTTP error occurred: {http_err}")
            print(f"Status code: {response.status_code}")
            print(f"Response text: {response.text[:500]}...")  # Print first 500 chars of response
        except requests.exceptions.JSONDecodeError as json_err:
            print(f"JSON decode error: {json_err}")
            print(f"Response text: {response.text[:500]}...")  # Print first 500 chars of response
        except Exception as err:
            print(f"Other error occurred: {err}")
        
        # Fallback to a list of common Wikipedia articles if the API fails
        print("Using fallback list of Wikipedia articles...")
        fallback_articles = [
            "United_States", "World_War_II", "Albert_Einstein", "Climate_change",
            "Artificial_intelligence", "COVID-19_pandemic", "Quantum_mechanics",
            "World_Wide_Web", "William_Shakespeare", "Solar_System", "Computer",
            "Mathematics", "Biology", "History", "Science", "Technology", "Art",
            "Music", "Film", "Literature", "Psychology", "Economics", "Physics",
            "China", "India", "Europe", "Africa", "Asia", "North_America", "South_America",
            "Russia", "Germany", "United_Kingdom", "France", "Japan", "Australia",
            "Canada", "Brazil", "Mexico", "Italy", "Spain", "Democracy", "Capitalism",
            "Socialism", "Internet", "Space_exploration", "Genetics", "Evolution"
        ]
        
        # Take the first n articles from the fallback list
        self.wikipedia_articles = fallback_articles[:n]
        print(f"Using {len(self.wikipedia_articles)} fallback articles")
        return self.wikipedia_articles
    
    def fetch_article_content(self, article_title: str) -> str:
        """
        Fetch the content of a Wikipedia article.
        
        Args:
            article_title: Title of the article
            
        Returns:
            Article text content
        """
        max_retries = 3
        retry_delay = 1  # seconds
        
        for attempt in range(max_retries):
            try:
                url = f"https://en.wikipedia.org/w/api.php?action=query&prop=extracts&exlimit=1&titles={article_title}&explaintext=1&format=json"
                response = requests.get(url)
                response.raise_for_status()
                data = response.json()
                
                # Extract the page content
                page_id = list(data["query"]["pages"].keys())[0]
                
                # Check if the API returned an error (page_id will be -1 for missing pages)
                if page_id == "-1":
                    print(f"Article '{article_title}' not found")
                    return ""
                    
                content = data["query"]["pages"][page_id]["extract"]
                return content
                
            except (requests.exceptions.RequestException, KeyError, IndexError, json.JSONDecodeError) as e:
                if attempt < max_retries - 1:
                    print(f"Error fetching '{article_title}', retrying in {retry_delay}s: {e}")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    print(f"Failed to fetch '{article_title}' after {max_retries} attempts: {e}")
                    return ""  # Return empty string after max retries
        
        return ""  # Should not reach here, but just in case
    
    def prepare_text_chunks(self, chunk_size: int = 512) -> List[str]:
        """
        Prepare chunks of Wikipedia articles for processing.
        
        Args:
            chunk_size: Size of each chunk in tokens
            
        Returns:
            List of text chunks
        """
        print("Preparing text chunks from Wikipedia articles...")
        chunks = []
        min_chunk_tokens = 50  # Minimum number of tokens for a valid chunk
        
        for article_title in tqdm(self.wikipedia_articles):
            try:
                content = self.fetch_article_content(article_title)
                
                # Skip if content is empty
                if not content or len(content.strip()) < 100:  # Skip very short content
                    print(f"Skipping article '{article_title}' due to insufficient content")
                    continue
                
                # Tokenize the content
                tokens = self.tokenizer.encode(content)
                
                # Split into chunks
                for i in range(0, len(tokens), chunk_size):
                    if i + chunk_size < len(tokens):
                        chunk_tokens = tokens[i:i+chunk_size]
                        
                        # Only add chunks with sufficient content
                        if len(chunk_tokens) >= min_chunk_tokens:
                            chunk_text = self.tokenizer.decode(chunk_tokens)
                            chunks.append(chunk_text)
                        
                # Sleep to avoid rate limiting
                time.sleep(0.5)
                
            except Exception as e:
                print(f"Error processing article {article_title}: {e}")
                continue
        
        # Ensure we have at least some chunks
        if not chunks:
            print("Warning: No valid chunks were created. Using fallback text.")
            # Create some simple chunks from hardcoded text
            fallback_text = "The default mode network (DMN) is a large-scale brain network primarily composed of the medial prefrontal cortex, posterior cingulate cortex, and angular gyrus. It is most commonly shown to be active when a person is not focused on the outside world and the brain is at wakeful rest, such as during daydreaming and mind-wandering. It can also be active during detailed thoughts about the past or future, and in social contexts."
            tokens = self.tokenizer.encode(fallback_text)
            chunks = [self.tokenizer.decode(tokens)]
        
        # Shuffle the chunks
        random.shuffle(chunks)
        
        self.processed_chunks = chunks
        print(f"Prepared {len(chunks)} text chunks")
        return chunks
    
    def register_attention_hooks(self):
        """
        Register hooks to capture attention patterns during forward pass.
        
        Returns:
            List of hook handles
        """
        hooks = []
        
        def get_attention_hook(layer_idx, head_idx):
            def hook(module, input, output):
                # For LLaMA models, attention outputs contain attention probs at index 3
                if isinstance(output, tuple) and len(output) > 3:
                    # Get attention weights
                    attn_weights = output[3]
                    
                    # Extract only the specified head
                    head_attn = attn_weights[0, :, head_idx, :]
                    
                    # Compute activation strength (mean attention weight)
                    activation = head_attn.abs().mean().item()
                    
                    # Store the activation
                    if layer_idx not in self.default_mode_activations:
                        self.default_mode_activations[layer_idx] = {}
                    
                    if head_idx not in self.default_mode_activations[layer_idx]:
                        self.default_mode_activations[layer_idx][head_idx] = []
                        
                    self.default_mode_activations[layer_idx][head_idx].append(activation)
                
            return hook
        
        # Register hooks for each attention layer and head
        print("Registering attention hooks...")
        for layer_idx in range(self.num_layers):
            for head_idx in range(self.num_heads):
                # Access the correct attention module based on the model architecture
                if hasattr(self.model, "model"):
                    # For LLaMA models
                    attn_module = self.model.model.layers[layer_idx].self_attn
                else:
                    # For other architectures
                    attn_module = self.model.h[layer_idx].attn
                
                # Register the hook
                hook = attn_module.register_forward_hook(get_attention_hook(layer_idx, head_idx))
                hooks.append(hook)
        
        return hooks
    
    def identify_default_mode_network(self, sample_size: int = 100):
        """
        Process Wikipedia chunks to identify the default mode network.
        
        Args:
            sample_size: Number of chunks to process
        """
        print("Identifying default mode network from Wikipedia chunks...")
        
        # Register hooks to capture attention activations
        hooks = self.register_attention_hooks()
        
        # Process chunks
        chunks_to_process = self.processed_chunks[:sample_size]
        for chunk in tqdm(chunks_to_process):
            # Tokenize the chunk
            inputs = self.tokenizer(chunk, return_tensors="pt").to(device)
            
            # Process through model with no_grad to save memory
            with torch.no_grad():
                self.model(**inputs)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Calculate average activations and importance scores for each head
        avg_activations = {}
        for layer_idx in self.default_mode_activations:
            avg_activations[layer_idx] = {}
            for head_idx in self.default_mode_activations[layer_idx]:
                activations = self.default_mode_activations[layer_idx][head_idx]
                avg_activations[layer_idx][head_idx] = sum(activations) / len(activations)
        
        # Flatten into a list of (layer, head, activation) tuples for ranking
        head_scores = []
        for layer_idx in avg_activations:
            for head_idx in avg_activations[layer_idx]:
                head_scores.append((layer_idx, head_idx, avg_activations[layer_idx][head_idx]))
        
        # Sort by activation (high to low)
        head_scores.sort(key=lambda x: x[2], reverse=True)
        
        self.head_importance_scores = head_scores
        print("Default mode network identification complete")
    
    def select_top_default_mode_heads(self, top_n: int = 50):
        """
        Select the top N most active heads as the default mode network.
        
        Args:
            top_n: Number of top heads to select
        """
        if self.head_importance_scores is None:
            raise ValueError("Must run identify_default_mode_network() first")
        
        self.top_default_mode_heads = self.head_importance_scores[:top_n]
        
        print(f"Selected top {top_n} heads as the default mode network:")
        for layer_idx, head_idx, score in self.top_default_mode_heads[:10]:
            print(f"  Layer {layer_idx}, Head {head_idx}: Activation {score:.4f}")
        
        if top_n > 10:
            print(f"  ... plus {top_n-10} more heads")
    
    def save_default_mode_network(self, filepath: str):
        """
        Save the identified default mode network to a file.
        
        Args:
            filepath: Path to save the data
        """
        data = {
            "model_name": self.model_name,
            "head_importance_scores": self.head_importance_scores,
            "top_default_mode_heads": self.top_default_mode_heads
        }
        
        with open(filepath, "wb") as f:
            pickle.dump(data, f)
        
        print(f"Default mode network saved to {filepath}")
    
    def load_default_mode_network(self, filepath: str):
        """
        Load a previously identified default mode network.
        
        Args:
            filepath: Path to the saved data
        """
        try:
            with open(filepath, "rb") as f:
                data = pickle.load(f)
            
            # Check if the saved model matches the current model
            if "model_name" in data and data["model_name"] != self.model_name:
                print(f"Warning: Loading default mode network identified on {data['model_name']}")
                print(f"Current model is {self.model_name}")
                print("The head indices may not be compatible between different models.")
                
            # Load data with error handling for different formats
            if "head_importance_scores" in data:
                self.head_importance_scores = data["head_importance_scores"]
            else:
                print("Warning: head_importance_scores not found in saved data")
                
            if "top_default_mode_heads" in data:
                self.top_default_mode_heads = data["top_default_mode_heads"]
            else:
                print("Warning: top_default_mode_heads not found in saved data")
                # If we have importance scores but no top heads, select them now
                if self.head_importance_scores:
                    print("Selecting top heads from importance scores...")
                    self.select_top_default_mode_heads(top_n=30)
            
            # Validate the loaded data
            if not self.top_default_mode_heads:
                raise ValueError("Failed to load top_default_mode_heads from file")
                
            print(f"Loaded default mode network from {filepath}")
            print(f"Top heads in the default mode network:")
            for layer_idx, head_idx, score in self.top_default_mode_heads[:10]:
                print(f"  Layer {layer_idx}, Head {head_idx}: Activation {score:.4f}")
                
            if len(self.top_default_mode_heads) > 10:
                print(f"  ... plus {len(self.top_default_mode_heads)-10} more heads")
                
        except Exception as e:
            print(f"Error loading default mode network from {filepath}: {e}")
            raise
    
    def generate_with_inhibition(self, 
                                prompt: str, 
                                inhibition_factor: float = 0.5, 
                                max_new_tokens: int = 200) -> Tuple[str, str]:
        """
        Generate text with and without inhibition of the default mode network.
        Implementation based on https://arxiv.org/pdf/2306.03341
        
        Args:
            prompt: Input prompt
            inhibition_factor: Factor by which to scale down the attention weights (0-1)
            max_new_tokens: Maximum number of tokens to generate
            
        Returns:
            Tuple of (normal_output, inhibited_output)
        """
        # Provide more detailed error message if top_default_mode_heads is not set
        if not self.top_default_mode_heads:
            # Check if head_importance_scores exists but top_default_mode_heads wasn't selected
            if self.head_importance_scores:
                print("Head importance scores exist but no top heads were selected.")
                print("Automatically selecting top 30 heads...")
                self.select_top_default_mode_heads(top_n=30)
            else:
                raise ValueError("Default mode network not identified. Please run identify_default_mode_network() and select_top_default_mode_heads() first, or load a saved network with load_default_mode_network().")
        
        # Tokenize prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        
        # Generate normally (without inhibition)
        print("Generating without inhibition...")
        with torch.no_grad():
            normal_outputs = self.model.generate(
                inputs.input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )
        
        normal_output = self.tokenizer.decode(normal_outputs[0], skip_special_tokens=True)
        
        # Now generate with inhibition
        print(f"Generating with inhibition factor {inhibition_factor}...")
        
        # Create attention mask for inhibiting specific heads
        attention_masks = {}
        
        try:
            print("Creating attention masks...")
            # Check for any potential layer_idx that might be out of bounds
            max_layer = -1
            for layer_idx, head_idx, _ in self.top_default_mode_heads:
                if layer_idx > max_layer:
                    max_layer = layer_idx
            
            if max_layer >= self.num_layers:
                print(f"Warning: Some DMN heads reference layers beyond model capacity (max:{max_layer}, model:{self.num_layers})")
                print("Filtering out-of-bound layers...")
                filtered_heads = [h for h in self.top_default_mode_heads if h[0] < self.num_layers]
                self.top_default_mode_heads = filtered_heads
                print(f"Filtered heads to {len(filtered_heads)} valid heads")
            
            # Check for any head_idx that might be out of bounds
            invalid_heads = []
            for layer_idx, head_idx, _ in self.top_default_mode_heads:
                if head_idx >= self.num_heads:
                    invalid_heads.append((layer_idx, head_idx))
                    
            if invalid_heads:
                print(f"Warning: Some DMN heads reference heads beyond model capacity (model has {self.num_heads} heads per layer)")
                print(f"Invalid head indices: {invalid_heads}")
                print("Filtering invalid heads...")
                filtered_heads = [h for h in self.top_default_mode_heads if h[1] < self.num_heads]
                self.top_default_mode_heads = filtered_heads
                print(f"Filtered to {len(filtered_heads)} valid heads")
            
            # Now create the attention masks with validated heads
            for layer_idx, head_idx, _ in self.top_default_mode_heads:
                if layer_idx not in attention_masks:
                    attention_masks[layer_idx] = {}
                attention_masks[layer_idx][head_idx] = inhibition_factor
            
            print(f"Created attention masks for {len(attention_masks)} layers")
            
        except Exception as e:
            print(f"Error creating attention masks: {e}")
            print(f"top_default_mode_heads format: {self.top_default_mode_heads[:3]}")
            raise
        
        # Check if using Flash Attention
        is_using_flash_attn = False
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            # For models like LLaMA and Mistral
            if hasattr(self.model.model.layers[0].self_attn, "flash_attn"):
                is_using_flash_attn = self.model.model.layers[0].self_attn.flash_attn
        
        if is_using_flash_attn:
            print("Model is using Flash Attention. DMN inhibition may work differently.")
            print("Attempting to use alternative inhibition method...")
            
        # Apply attention inhibition during generation
        inhibited_output = self._generate_with_attention_inhibition(
            prompt, 
            attention_masks, 
            max_new_tokens
        )
        
        return normal_output, inhibited_output
    
    def _generate_with_attention_inhibition(self, 
                                          prompt: str, 
                                          attention_masks: Dict[int, Dict[int, float]], 
                                          max_new_tokens: int) -> str:
        """
        Generate text with attention inhibition applied to specific heads.
        
        Args:
            prompt: Input prompt
            attention_masks: Dictionary mapping layer_idx -> head_idx -> scaling_factor
            max_new_tokens: Maximum number of tokens to generate
            
        Returns:
            Generated text
        """
        # Tokenize the prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(device)
        
        # Register hooks for attention inhibition
        hooks = []
        
        def get_inhibition_hook(layer_idx, head_masks):
            def hook(module, input, output):
                # Different models have different output formats for attention
                # For Mistral/Llama3 models (using attention with flash attention impl)
                if hasattr(module, "flash_attn") and module.flash_attn:
                    # Flash attention doesn't return attention weights by default
                    # We'll need to modify the hidden states directly instead
                    # This is a simplified approach for these models
                    if isinstance(output, tuple) and len(output) >= 1:
                        # Just scale the output hidden states
                        hidden_states = output[0]
                        scaling = torch.ones_like(hidden_states[0, 0, 0]).to(hidden_states.device)
                        
                        # Apply scaling - this is approximate since we don't have direct 
                        # access to attention weights with flash attention
                        for head_idx, scale_factor in head_masks.items():
                            # The rest of the heads remain at 1.0 (unchanged)
                            # Scale based on all heads
                            head_dim = hidden_states.shape[-1] // self.num_heads
                            start_idx = head_idx * head_dim
                            end_idx = (head_idx + 1) * head_dim
                            
                            # Create head-specific scaling
                            scaling_factor = torch.ones_like(hidden_states)
                            scaling_factor[:, :, start_idx:end_idx] *= scale_factor
                            
                            # Apply scaling
                            output_list = list(output)
                            output_list[0] = hidden_states * scaling_factor
                            output = tuple(output_list)
                        
                        return output
                
                # For LLaMA models, attention outputs contain attention probs at index 3
                # This handles the case where attention weights are returned
                elif isinstance(output, tuple) and len(output) > 3:
                    # Get attention weights
                    attn_weights = output[3]
                    
                    # Apply inhibition to specific heads
                    for head_idx, scale_factor in head_masks.items():
                        # Create a scaling tensor that's 1 except for the target head
                        scaling = torch.ones_like(attn_weights)
                        scaling[:, :, head_idx, :] = scale_factor
                        
                        # Apply scaling
                        new_weights = attn_weights * scaling
                        
                        # Update the output tuple
                        output_list = list(output)
                        output_list[3] = new_weights
                        output = tuple(output_list)
                    
                    return output
                
                # Fallback for other model architectures - may require model-specific adjustments
                else:
                    print(f"Warning: Attention format not recognized for layer {layer_idx}. Output type: {type(output)}")
                    if isinstance(output, tuple):
                        print(f"Output tuple length: {len(output)}")
                    return output
            
            return hook
        
        # Log model architecture for debugging
        print(f"Registering hooks for {self.model_name} with {self.num_layers} layers and {self.num_heads} heads")
        
        # Register hooks for each layer
        for layer_idx, head_masks in attention_masks.items():
            try:
                if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
                    # For LLaMA and Mistral models
                    attn_module = self.model.model.layers[layer_idx].self_attn
                elif hasattr(self.model, "layers"):
                    # Some other architectures
                    attn_module = self.model.layers[layer_idx].self_attn
                elif hasattr(self.model, "h"):
                    # For other architectures (like GPT-2)
                    attn_module = self.model.h[layer_idx].attn
                else:
                    print(f"Warning: Could not find attention module for layer {layer_idx}")
                    continue
                
                # Register the hook
                hook = attn_module.register_forward_hook(get_inhibition_hook(layer_idx, head_masks))
                hooks.append(hook)
                
            except Exception as e:
                print(f"Error registering hook for layer {layer_idx}: {e}")
                continue
        
        # Generate with inhibition
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                )
        except Exception as e:
            print(f"Error during generation with inhibition: {e}")
            # Remove hooks before re-raising
            for hook in hooks:
                hook.remove()
            raise
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Decode and return
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text
    
    def run_experiment(self, 
                     queries: List[str], 
                     inhibition_factors: List[float] = [0.0, 0.3, 0.5, 0.7, 0.9],
                     max_new_tokens: int = 200,
                     dmn_file: Optional[str] = None) -> Dict:
        """
        Run the experiment with multiple queries and inhibition factors.
        
        Args:
            queries: List of queries to test
            inhibition_factors: List of inhibition factors to try
            max_new_tokens: Maximum tokens to generate for each query
            dmn_file: Path to a saved default mode network file to load (if not already loaded)
            
        Returns:
            Dictionary of experimental results
        """
        # Check if we have default mode network heads
        if not self.top_default_mode_heads and dmn_file:
            print(f"Default mode network not loaded, attempting to load from {dmn_file}")
            try:
                self.load_default_mode_network(dmn_file)
            except Exception as e:
                print(f"Error loading default mode network: {e}")
                print("Please run identify_default_mode_network() and select_top_default_mode_heads() first")
                return {}
        
        # Double check that we have heads selected
        if not self.top_default_mode_heads:
            print("No default mode network heads selected.")
            print("Please run identify_default_mode_network() and select_top_default_mode_heads() first")
            return {}
            
        results = {}
        
        # Try to process at least one query even if others fail
        at_least_one_succeeded = False
        
        for query_index, query in enumerate(queries):
            print(f"\n===== Processing query {query_index+1}/{len(queries)}: '{query}' =====")
            results[query] = {}
            
            try:
                for factor in inhibition_factors:
                    print(f"\nTesting query: '{query}' with inhibition factor {factor}")
                    
                    try:
                        if factor == 0.0:
                            # Just run the model normally
                            print("Running normal generation without inhibition...")
                            inputs = self.tokenizer(query, return_tensors="pt").to(device)
                            with torch.no_grad():
                                outputs = self.model.generate(
                                    inputs.input_ids,
                                    max_new_tokens=max_new_tokens,
                                    do_sample=True,
                                    temperature=0.7,
                                    top_p=0.9,
                                )
                            output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                            print(f"Normal generation completed, output length: {len(output)} chars")
                            results[query][factor] = output
                            
                            # Save intermediate results after each successful generation
                            if not at_least_one_succeeded:
                                print("Saving intermediate results...")
                                self.save_query_outputs(results, output_dir="query_outputs_interim")
                                at_least_one_succeeded = True
                                
                        else:
                            # Run with inhibition
                            print(f"Running generation with inhibition factor {factor}...")
                            try:
                                normal_output, inhibited_output = self.generate_with_inhibition(
                                    query, 
                                    inhibition_factor=factor,
                                    max_new_tokens=max_new_tokens
                                )
                                print(f"Inhibited generation completed, output length: {len(inhibited_output)} chars")
                                results[query][factor] = inhibited_output
                                
                                # Save intermediate results after each successful generation
                                if not at_least_one_succeeded:
                                    print("Saving intermediate results...")
                                    self.save_query_outputs(results, output_dir="query_outputs_interim")
                                    at_least_one_succeeded = True
                                
                            except Exception as inhibition_err:
                                print(f"Error during inhibited generation: {inhibition_err}")
                                # Add a placeholder response
                                results[query][factor] = f"[Error generating with inhibition factor {factor}: {str(inhibition_err)}]"
                                
                                # Try to run normal generation as fallback
                                print("Attempting fallback to normal generation...")
                                inputs = self.tokenizer(query, return_tensors="pt").to(device)
                                with torch.no_grad():
                                    outputs = self.model.generate(
                                        inputs.input_ids,
                                        max_new_tokens=max_new_tokens,
                                        do_sample=True,
                                        temperature=0.7,
                                        top_p=0.9,
                                    )
                                output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                                results[query][factor] = f"[Fallback to normal generation due to error: {str(inhibition_err)}]\n\n{output}"
                    
                    except Exception as factor_err:
                        print(f"Error processing inhibition factor {factor}: {factor_err}")
                        results[query][factor] = f"[Error with factor {factor}: {str(factor_err)}]"
                        
                    # Save results after each factor is processed
                    if query_index == 0 or (query_index > 0 and factor == inhibition_factors[-1]):
                        print("Saving current results...")
                        self.save_query_outputs(results)
                
            except Exception as query_err:
                print(f"Error processing query '{query}': {query_err}")
                results[query]["error"] = f"Failed to process query: {str(query_err)}"
                
            # Save results after each query is processed
            print(f"Completed query: '{query}', saving results...")
            self.save_query_outputs(results)
            
        # Final save of all results
        print("Experiment completed, saving final results...")
        self.save_query_outputs(results)
        
        return results
    
    def visualize_default_mode_network(self, top_n: int = 100, save_path: Optional[str] = None):
        """
        Visualize the default mode network as a heatmap.
        
        Args:
            top_n: Number of top heads to visualize
            save_path: Path to save the visualization (if provided)
        """
        if self.head_importance_scores is None:
            raise ValueError("Must run identify_default_mode_network() first")
        
        # Create a matrix of activation values
        activation_matrix = np.zeros((self.num_layers, self.num_heads))
        
        # Fill in the values for heads we have data for
        for layer_idx in self.default_mode_activations:
            for head_idx in self.default_mode_activations[layer_idx]:
                activations = self.default_mode_activations[layer_idx][head_idx]
                avg_activation = sum(activations) / len(activations)
                activation_matrix[layer_idx, head_idx] = avg_activation
        
        # Create the heatmap
        plt.figure(figsize=(12, 10))
        ax = sns.heatmap(activation_matrix, cmap="viridis", annot=False)
        plt.title(f"Default Mode Network Activation Heatmap for {self.model_name}")
        plt.xlabel("Attention Head")
        plt.ylabel("Layer")
        
        # Highlight the top N heads
        top_n = min(top_n, len(self.head_importance_scores))
        for i in range(top_n):
            layer_idx, head_idx, _ = self.head_importance_scores[i]
            plt.plot(head_idx + 0.5, layer_idx + 0.5, 'rx', markersize=8)
        
        if save_path:
            plt.savefig(save_path)
            print(f"Visualization saved to {save_path}")
        
        plt.show()
    
    def analyze_results(self, results: Dict, save_path: Optional[str] = None):
        """
        Analyze the results of the experiment.
        
        Args:
            results: Results dictionary from run_experiment
            save_path: Path to save the analysis (if provided)
        """
        analysis = {
            "queries": [],
            "inhibition_factors": [],
            "response_lengths": [],
            "unique_words": [],
            "avg_sentence_length": [],
            "creativity_score": []
        }
        
        # Save the text outputs to files for easier viewing
        self.save_query_outputs(results)
        
        for query, responses in results.items():
            for factor, response in responses.items():
                # Strip the query from the beginning of the response
                if response.startswith(query):
                    response_text = response[len(query):].strip()
                else:
                    response_text = response
                
                # Calculate response length
                response_length = len(response_text.split())
                
                # Calculate unique words ratio
                words = response_text.lower().split()
                unique_words = len(set(words)) / len(words) if words else 0
                
                # Calculate average sentence length
                sentences = [s.strip() for s in response_text.split('.') if s.strip()]
                avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
                
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
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Analysis saved to {save_path}")
        
        plt.show()
        
        return df

    def save_query_outputs(self, results: Dict, output_dir: str = "query_outputs"):
        """
        Save the query outputs to text files for easier viewing.
        
        Args:
            results: Results dictionary from run_experiment
            output_dir: Directory to save the output files
        """
        if not results:
            print("Warning: No results to save")
            # Create an empty placeholder file to show the process ran
            os.makedirs(output_dir, exist_ok=True)
            with open(f"{output_dir}/no_results.txt", "w") as f:
                f.write("No results were generated during the experiment.\n")
                f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            return
        
        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Get model name shorthand for the filenames
        model_shortname = self.model_name.split('/')[-1].lower()
        
        print(f"Saving query outputs to {output_dir}/")
        
        # Save a combined summary file
        try:
            with open(f"{output_dir}/{model_shortname}_all_outputs.txt", "w") as summary_file:
                summary_file.write(f"== Experiment results for {self.model_name} ==\n\n")
                summary_file.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                summary_file.write(f"Model has {self.num_layers} layers and {self.num_heads} heads per layer\n\n")
                
                for query, responses in results.items():
                    summary_file.write(f"\n\n==== QUERY: {query} ====\n\n")
                    
                    if not responses:
                        summary_file.write("No responses generated for this query.\n")
                        continue
                        
                    for factor, response in sorted(responses.items()):
                        summary_file.write(f"-- Inhibition factor: {factor} --\n")
                        # Handle the case where response might be None
                        if response is None:
                            summary_file.write("[No response generated]\n")
                        else:
                            summary_file.write(str(response))
                        summary_file.write("\n\n" + "-"*80 + "\n\n")
            
            print(f"Saved combined summary to {output_dir}/{model_shortname}_all_outputs.txt")
        except Exception as e:
            print(f"Error saving combined summary file: {e}")
        
        # Save individual files for each query and inhibition factor
        for query, responses in results.items():
            # Create a safe filename from the query
            query_filename = query.replace(" ", "_").replace("?", "").replace(".", "").replace("/", "_")[:30]
            
            # Save a summary file for just this query
            try:
                with open(f"{output_dir}/{model_shortname}_{query_filename}_summary.txt", "w") as f:
                    f.write(f"Query: {query}\n\n")
                    f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                    
                    if not responses:
                        f.write("No responses generated for this query.\n")
                        continue
                    
                    for factor, response in sorted(responses.items()):
                        f.write(f"\n-- Inhibition factor: {factor} --\n\n")
                        # Handle the case where response might be None
                        if response is None:
                            f.write("[No response generated]\n")
                        else:
                            f.write(str(response))
                        f.write("\n\n" + "-"*40 + "\n")
            except Exception as e:
                print(f"Error saving summary for query '{query}': {e}")
            
            # Save individual files for each inhibition factor
            for factor, response in responses.items():
                try:
                    filename = f"{output_dir}/{model_shortname}_{query_filename}_inhibition_{factor}.txt"
                    
                    with open(filename, "w") as f:
                        f.write(f"Query: {query}\n")
                        f.write(f"Inhibition factor: {factor}\n")
                        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                        
                        # Handle the case where response might be None
                        if response is None:
                            f.write("[No response generated]\n")
                        else:
                            f.write(str(response))
                except Exception as e:
                    print(f"Error saving output for query '{query}' with factor {factor}: {e}")
                    
        print(f"Saved query outputs to individual files in {output_dir}/")


# Example usage
if __name__ == "__main__":
    # Initialize the experiment
    experiment = DefaultModeNetworkExperiment(model_name="meta-llama/Llama-3-70b")
    
    # Fetch Wikipedia articles and prepare chunks
    experiment.fetch_top_wikipedia_articles(n=100)
    experiment.prepare_text_chunks(chunk_size=512)
    
    # Identify the default mode network
    experiment.identify_default_mode_network(sample_size=100)
    experiment.select_top_default_mode_heads(top_n=50)
    
    # Save the identified network
    experiment.save_default_mode_network("llama3_70b_default_mode_network.pkl")
    
    # Visualize the default mode network
    experiment.visualize_default_mode_network(save_path="llama3_70b_dmn_visualization.png")
    
    # Define test queries
    test_queries = [
        "What is the meaning of life?",
        "Write a creative story about a robot who discovers emotions.",
        "Explain quantum mechanics in simple terms.",
        "Describe a new color that doesn't exist.",
        "What would happen if humans could photosynthesize?"
    ]
    
    # Run the experiment
    results = experiment.run_experiment(
        queries=test_queries,
        inhibition_factors=[0.0, 0.3, 0.5, 0.7, 0.9],
        max_new_tokens=200
    )
    
    # Analyze results
    analysis_df = experiment.analyze_results(results, save_path="llama3_70b_dmn_analysis.png")
    
    # Save the experiment results
    with open("llama3_70b_dmn_experiment_results.pkl", "wb") as f:
        pickle.dump(results, f)
    
    analysis_df.to_csv("llama3_70b_dmn_experiment_analysis.csv")
