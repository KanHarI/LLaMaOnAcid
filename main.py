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
    
    def __init__(self, model_name: str, device: str = "cuda"):
        """
        Initialize the DefaultModeNetworkExperiment.
        
        Args:
            model_name: The name or path of the model to use
            device: The device to use (cuda or cpu)
        """
        self.model_name = model_name
        print(f"Initializing experiment with model: {model_name}")
        
        # Create cache directory
        self.wiki_cache_dir = os.path.join("cache", model_name.replace("/", "_"))
        os.makedirs(self.wiki_cache_dir, exist_ok=True)
        print(f"Created cache directory: {self.wiki_cache_dir}")
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("Tokenizer loaded")
        
        # For flash attention compatibility
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            device_map=device,
            torch_dtype=torch.float16,
            attn_implementation="eager"  # Use eager implementation for compatibility
        )
        print("Model loaded")
        
        # Model dimensions
        self.num_layers = self.model.config.num_hidden_layers 
        self.num_heads = self.model.config.num_attention_heads
        print(f"Model has {self.num_layers} layers and {self.num_heads} attention heads per layer")
        
        # Data storage
        self.articles = []
        self.processed_chunks = []
        self.default_mode_activations = {}
        self.head_importance_scores = []
        self.top_default_mode_heads = []
        
    def fetch_top_wikipedia_articles(self, n: int = 100, use_cache: bool = True, cache_ttl_days: int = 30) -> List[str]:
        """
        Fetch the top N most viewed Wikipedia articles.
        
        Args:
            n: Number of top articles to fetch
            use_cache: Whether to use cached list if available
            cache_ttl_days: How many days before the cached list expires
            
        Returns:
            List of article titles
        """
        print(f"Fetching top {n} Wikipedia articles...")
        
        # Check for cached list of top articles
        cache_file = os.path.join(self.wiki_cache_dir, "top_articles.json")
        
        if use_cache and os.path.exists(cache_file):
            try:
                with open(cache_file, "r") as f:
                    cached_data = json.load(f)
                
                # Check if cache is still valid
                cache_timestamp = datetime.fromisoformat(cached_data.get("timestamp", "2000-01-01"))
                cache_age = (datetime.now() - cache_timestamp).days
                
                if cache_age <= cache_ttl_days:
                    self.articles = cached_data.get("articles", [])[:n]
                    print(f"Using cached list of {len(self.articles)} top articles from {cache_timestamp}")
                    return self.articles
                else:
                    print(f"Cached article list is {cache_age} days old (> {cache_ttl_days}), fetching new data")
            except Exception as e:
                print(f"Error reading cached article list: {e}")
        
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
            
            # Cache the fetched articles
            cache_data = {
                "timestamp": datetime.now().isoformat(),
                "articles": articles
            }
            try:
                with open(cache_file, "w") as f:
                    json.dump(cache_data, f)
                print(f"Cached list of {len(articles)} top articles")
            except Exception as e:
                print(f"Error caching article list: {e}")
            
            self.articles = articles
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
        
        # Try to load expired cache as a fallback if it exists
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r") as f:
                    cached_data = json.load(f)
                articles = cached_data.get("articles", [])[:n]
                if articles:
                    print(f"Using expired cached list of {len(articles)} articles as fallback")
                    self.articles = articles
                    return articles
            except Exception as e:
                print(f"Error reading cached article list as fallback: {e}")
        
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
        self.articles = fallback_articles[:n]
        print(f"Using {len(self.articles)} fallback articles")
        return self.articles
    
    def fetch_article_content(self, article_title: str, use_cache: bool = True, cache_ttl_days: int = 90) -> str:
        """
        Fetch the content of a Wikipedia article, using cache if available.
        
        Args:
            article_title: Title of the Wikipedia article
            use_cache: Whether to use cached content
            cache_ttl_days: How many days to consider cached content valid
        
        Returns:
            Article content as a string
        """
        cache_file = os.path.join(
            self.wiki_cache_dir, 
            f"article_{article_title.replace(' ', '_').replace('/', '_')}.json"
        )
        
        # Check cache first if enabled
        if use_cache and os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                
                cache_timestamp = cached_data.get('timestamp')
                if cache_timestamp:
                    # Calculate age of cache in days
                    cache_date = datetime.fromisoformat(cache_timestamp)
                    cache_age = (datetime.now() - cache_date).days
                    
                    if cache_age <= cache_ttl_days:
                        # Cache is still valid
                        print(f"Using cached content for '{article_title}' ({cache_age} days old)")
                        return cached_data.get('content', '')
                    else:
                        print(f"Cached content for '{article_title}' is {cache_age} days old (> {cache_ttl_days})")
            except Exception as e:
                print(f"Error reading cache for '{article_title}': {e}")
        
        # Cache miss or expired, fetch from Wikipedia
        try:
            # Make the article title URL-safe
            url_title = article_title.replace(' ', '_')
            
            # Wikipedia API URL for content extraction
            url = f"https://en.wikipedia.org/w/api.php?action=query&prop=extracts&format=json&exintro=&titles={url_title}"
            
            # Add user agent to avoid 403 errors
            headers = {
                'User-Agent': 'DefaultModeNetworkExperiment/1.0 (Research project; contact@example.com)'
            }
            
            # Fetch with retry logic
            max_retries = 3
            retry_delay = 2  # seconds
            
            for attempt in range(max_retries):
                try:
                    response = requests.get(url, headers=headers, timeout=10)
                    response.raise_for_status()  # Raise exception for 4XX/5XX responses
                    break
                except requests.exceptions.RequestException as e:
                    if attempt < max_retries - 1:
                        print(f"Retry {attempt+1}/{max_retries} for '{article_title}' after error: {e}")
                        time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
                    else:
                        raise
            
            data = response.json()
            
            # Extract the page content
            pages = data.get('query', {}).get('pages', {})
            if not pages:
                raise ValueError(f"No pages found for '{article_title}'")
            
            # Get the first page (there should only be one)
            page_id = next(iter(pages))
            if page_id == '-1':
                raise ValueError(f"Article '{article_title}' not found")
            
            content = pages[page_id].get('extract', '')
            
            # Cache the content if we got something
            if content and use_cache:
                try:
                    cache_data = {
                        'title': article_title,
                        'timestamp': datetime.now().isoformat(),
                        'content': content
                    }
                    with open(cache_file, 'w', encoding='utf-8') as f:
                        json.dump(cache_data, f, ensure_ascii=False)
                except Exception as e:
                    print(f"Error caching content for '{article_title}': {e}")
            
            return content
            
        except Exception as e:
            print(f"Error fetching content for '{article_title}': {e}")
            
            # Try to use expired cache as fallback
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        cached_data = json.load(f)
                    print(f"Using expired cache as fallback for '{article_title}'")
                    return cached_data.get('content', '')
                except Exception as cache_e:
                    print(f"Error reading expired cache: {cache_e}")
            
            # If all else fails, return empty string
            return ""
    
    def prepare_text_chunks(self, chunk_size: int = 512, use_cache: bool = True) -> List[str]:
        """
        Prepare chunks of Wikipedia articles for processing.
        
        Args:
            chunk_size: Size of each chunk in tokens
            use_cache: Whether to use cached chunks if available
            
        Returns:
            List of text chunks
        """
        print("Preparing text chunks from Wikipedia articles...")
        
        # Check for cached chunks
        chunks_cache_file = os.path.join(self.wiki_cache_dir, f"chunks_{chunk_size}.pkl")
        
        if use_cache and os.path.exists(chunks_cache_file):
            try:
                with open(chunks_cache_file, "rb") as f:
                    cached_data = pickle.load(f)
                
                # Verify the cache contains what we expect
                if "chunks" in cached_data and "articles" in cached_data:
                    cached_articles = set(cached_data["articles"])
                    current_articles = set(self.articles)
                    
                    # If the cached chunks were generated from the same or a superset of current articles
                    if current_articles.issubset(cached_articles):
                        self.processed_chunks = cached_data["chunks"]
                        print(f"Using {len(self.processed_chunks)} cached chunks from {len(cached_articles)} articles")
                        return self.processed_chunks
                    else:
                        # If we have new articles, but can reuse some cached content
                        print(f"Articles list changed. Cached: {len(cached_articles)}, Current: {len(current_articles)}")
                        print(f"Will fetch content for {len(current_articles - cached_articles)} new articles")
            except Exception as e:
                print(f"Error reading cached chunks: {e}")
        
        chunks = []
        min_chunk_tokens = 50  # Minimum number of tokens for a valid chunk
        processed_articles = []  # Keep track of successfully processed articles
        
        for article_title in tqdm(self.articles):
            try:
                # Use our cache-aware fetch_article_content method
                content = self.fetch_article_content(article_title, use_cache=use_cache)
                
                # Skip if content is empty
                if not content or len(content.strip()) < 100:  # Skip very short content
                    print(f"Skipping article '{article_title}' due to insufficient content")
                    continue
                
                # Tokenize the content
                tokens = self.tokenizer.encode(content)
                
                # Split into chunks
                article_has_valid_chunks = False
                for i in range(0, len(tokens), chunk_size):
                    if i + chunk_size < len(tokens):
                        chunk_tokens = tokens[i:i+chunk_size]
                        
                        # Only add chunks with sufficient content
                        if len(chunk_tokens) >= min_chunk_tokens:
                            chunk_text = self.tokenizer.decode(chunk_tokens)
                            chunks.append(chunk_text)
                            article_has_valid_chunks = True
                
                # Only record this article as processed if it contributed valid chunks
                if article_has_valid_chunks:
                    processed_articles.append(article_title)
                        
                # Sleep to avoid rate limiting
                time.sleep(0.5)
                
            except Exception as e:
                print(f"Error processing article {article_title}: {e}")
                continue
        
        # Cache the chunks if we have any
        if chunks:
            try:
                cache_data = {
                    "timestamp": datetime.now().isoformat(),
                    "chunk_size": chunk_size,
                    "articles": processed_articles,
                    "chunks": chunks
                }
                with open(chunks_cache_file, "wb") as f:
                    pickle.dump(cache_data, f)
                print(f"Cached {len(chunks)} processed chunks from {len(processed_articles)} articles")
            except Exception as e:
                print(f"Error caching processed chunks: {e}")
        
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
    
    def identify_default_mode_network(self, sample_size: int = 100, use_cache: bool = True):
        """
        Process Wikipedia chunks to identify the default mode network.
        
        Args:
            sample_size: Number of chunks to process
            use_cache: Whether to use cached activations if available
        """
        print("Identifying default mode network from Wikipedia chunks...")
        
        # Check if we have previously processed activations in cache
        activations_cache_file = os.path.join(self.wiki_cache_dir, 
                                             f"dmn_activations_{self.model_name.split('/')[-1]}_{sample_size}.pkl")
        
        if use_cache and os.path.exists(activations_cache_file):
            try:
                print(f"Loading cached DMN activations from {activations_cache_file}")
                with open(activations_cache_file, "rb") as f:
                    cached_data = pickle.load(f)
                
                if "default_mode_activations" in cached_data:
                    self.default_mode_activations = cached_data["default_mode_activations"]
                    print(f"Loaded cached activations for {len(self.default_mode_activations)} layers")
                    
                    # Calculate importance scores from the loaded activations
                    self._calculate_head_importance_scores()
                    print("Calculated head importance scores from cached activations")
                    return
                    
            except Exception as e:
                print(f"Error loading cached activations: {e}")
        
        # Register hooks to capture attention activations
        hooks = self.register_attention_hooks()
        
        # Process chunks
        chunks_to_process = self.processed_chunks[:sample_size]
        print(f"Processing {len(chunks_to_process)} chunks to identify default mode network")
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
        self._calculate_head_importance_scores()
        
        # Cache the activations if we have any
        if self.default_mode_activations:
            try:
                cache_data = {
                    "timestamp": datetime.now().isoformat(),
                    "model_name": self.model_name,
                    "sample_size": sample_size,
                    "default_mode_activations": self.default_mode_activations,
                    "head_importance_scores": self.head_importance_scores
                }
                with open(activations_cache_file, "wb") as f:
                    pickle.dump(cache_data, f)
                print(f"Cached DMN activations for {len(self.default_mode_activations)} layers")
            except Exception as e:
                print(f"Error caching DMN activations: {e}")
        
        print("Default mode network identification complete")
        
    def _calculate_head_importance_scores(self):
        """
        Calculate head importance scores based on the captured activations.
        """
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
    
    def run_experiment(self, use_inhibition: bool = True, queries: list = None, 
                     n_chunks: int = 100, num_inhibition_factors: int = 5, 
                     chunk_size: int = 512, dmn_file: str = None,
                     use_cache: bool = True, force_article_refresh: bool = False):
        """
        Run the full experiment:
        1. Fetch Wikipedia articles
        2. Prepare text chunks
        3. Identify default mode network
        4. Select top DMN heads
        5. Generate responses for queries with and without inhibition
        6. Save the outputs
        
        Args:
            use_inhibition: Whether to use inhibition
            queries: List of queries to answer
            n_chunks: Number of chunks to process
            num_inhibition_factors: Number of inhibition factors to test
            chunk_size: Size of chunks in tokens
            dmn_file: Path to a file containing pre-identified DMN heads
            use_cache: Whether to use cached data (articles, chunks, activations)
            force_article_refresh: Whether to force a refresh of article data
        """
        print(f"Running experiment with {self.model_name}")
        results = []
        at_least_one_succeeded = False
        
        if queries is None:
            print("No queries provided, using default test queries")
            queries = [
                "What is the meaning of life?",
                "Tell me about the history of Rome.",
                "Explain quantum physics.",
                "What are the major challenges facing humanity?",
                "How does the brain work?"
            ]
        
        # Step 1: Fetch Wikipedia articles
        try:
            print("Fetching Wikipedia articles...")
            self.articles = self.fetch_top_wikipedia_articles(use_cache=use_cache, force_refresh=force_article_refresh)
            print(f"Fetched {len(self.articles)} articles")
        except Exception as e:
            print(f"Error fetching articles: {e}")
            print("Using fallback list of articles")
            self.articles = [
                {'article': 'Philosophy', 'views': 10000},
                {'article': 'Science', 'views': 9000},
                {'article': 'History', 'views': 8000},
                {'article': 'Literature', 'views': 7000},
                {'article': 'Mathematics', 'views': 6000}
            ]
        
        # Step 2: Prepare text chunks
        try:
            print("Preparing text chunks...")
            self.processed_chunks = self.prepare_text_chunks(chunk_size=chunk_size, use_cache=use_cache)
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
                self.identify_default_mode_network(sample_size=min(n_chunks, len(self.processed_chunks)), use_cache=use_cache)
                self.select_top_default_mode_heads()
            except Exception as e:
                print(f"Error identifying default mode network: {e}")
                print("Using fallback default mode network")
                # Create a fallback DMN
                num_layers = self.model.config.num_hidden_layers
                num_heads = self.model.config.num_attention_heads
                self.top_default_mode_heads = [(l, h) for l, h in zip(
                    np.random.randint(0, num_layers, size=50),
                    np.random.randint(0, num_heads, size=50)
                )]
                print(f"Created fallback DMN with {len(self.top_default_mode_heads)} heads")
        
        # Step 5: Generate responses
        inhibition_factors = [0.0]  # Normal generation
        if use_inhibition:
            # Add inhibition factors
            inhibition_factors.extend(np.linspace(0.1, 1.0, num_inhibition_factors))
        
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
                        inputs = self.tokenizer(query, return_tensors="pt").to(device)
                        output = self.model.generate(
                            **inputs,
                            max_new_tokens=150,
                            do_sample=True,
                            temperature=0.7,
                            top_p=0.9
                        )
                        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
                    else:
                        # Generation with inhibition
                        try:
                            normal_response, inhibited_response = self.generate_with_inhibition(
                                query, inhibition_factor=factor
                            )
                            response = inhibited_response
                        except Exception as e:
                            print(f"Error with inhibited generation: {e}")
                            print("Falling back to normal generation")
                            # Fall back to normal generation
                            inputs = self.tokenizer(query, return_tensors="pt").to(device)
                            output = self.model.generate(
                                **inputs,
                                max_new_tokens=150,
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
                self.save_query_outputs(results, suffix=f"_intermediate_{len(results)}")
        
        # Step 6: Save outputs
        try:
            print("\nSaving outputs...")
            self.save_query_outputs(results)
        except Exception as e:
            print(f"Error saving outputs: {e}")
        
        if not at_least_one_succeeded:
            print("Warning: No successful generations were produced during this experiment.")
        
        print("Experiment complete")
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

    def save_query_outputs(self, results, output_dir="query_outputs", suffix=""):
        """
        Save the query outputs to files.
        
        Args:
            results: List of results dictionaries, each with query, inhibition_factor and response
            output_dir: Directory to save the outputs
            suffix: Optional suffix for the output files
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
            model_name_safe = self.model_name.replace("/", "-").replace(".", "-")
            output_file = os.path.join(
                output_dir, f"{model_name_safe}_all_outputs{suffix}.txt"
            )
            
            with open(output_file, "w") as f:
                f.write(f"Experiment results for {self.model_name}\n")
                f.write(f"Timestamp: {timestamp}\n")
                f.write("=" * 80 + "\n\n")
                
                # Group results by query
                query_results = {}
                for result in results:
                    query = result.get('query', 'Unknown query')
                    if query not in query_results:
                        query_results[query] = []
                    query_results[query].append(result)
                
                # Write results for each query
                for query, query_list in query_results.items():
                    f.write(f"QUERY: {query}\n")
                    f.write("-" * 80 + "\n\n")
                    
                    # Sort by inhibition factor
                    query_list.sort(key=lambda x: x.get('inhibition_factor', 0.0))
                    
                    for result in query_list:
                        factor = result.get('inhibition_factor', 0.0)
                        response = result.get('response', 'No response generated')
                        
                        f.write(f"INHIBITION FACTOR: {factor}\n\n")
                        f.write(f"RESPONSE:\n{response}\n\n")
                        f.write("-" * 40 + "\n\n")
                    
                    f.write("=" * 80 + "\n\n")
            
            print(f"Saved combined outputs to {output_file}")
            
            # Create individual files for each query
            for query, query_list in query_results.items():
                # Create a safe filename from the query
                query_filename = query.replace(" ", "_").replace("?", "").replace(".", "")[:30]
                query_filename = ''.join(c for c in query_filename if c.isalnum() or c == '_')
                
                query_output_file = os.path.join(
                    output_dir, f"{model_name_safe}_{query_filename}{suffix}.txt"
                )
                
                with open(query_output_file, "w") as f:
                    f.write(f"Results for query: {query}\n")
                    f.write(f"Model: {self.model_name}\n")
                    f.write(f"Timestamp: {timestamp}\n")
                    f.write("-" * 80 + "\n\n")
                    
                    # Sort by inhibition factor
                    query_list.sort(key=lambda x: x.get('inhibition_factor', 0.0))
                    
                    for result in query_list:
                        factor = result.get('inhibition_factor', 0.0)
                        response = result.get('response', 'No response generated')
                        
                        f.write(f"INHIBITION FACTOR: {factor}\n\n")
                        f.write(f"RESPONSE:\n{response}\n\n")
                        f.write("-" * 40 + "\n\n")
            
            print(f"Saved individual query outputs to {output_dir}")
                
        except Exception as e:
            print(f"Error saving query outputs: {e}")


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
        use_inhibition=True,
        queries=test_queries,
        n_chunks=100,
        num_inhibition_factors=5,
        chunk_size=512
    )
    
    # Analyze results
    analysis_df = experiment.analyze_results(results, save_path="llama3_70b_dmn_analysis.png")
    
    # Save the experiment results
    with open("llama3_70b_dmn_experiment_results.pkl", "wb") as f:
        pickle.dump(results, f)
    
    analysis_df.to_csv("llama3_70b_dmn_experiment_analysis.csv")
