"""
Module for identifying the default mode network in language models.
"""

import os
import pickle
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import torch
from tqdm import tqdm
from transformers import PreTrainedModel

from ..config import CACHE_DIR

ActivationDict = Dict[int, Dict[int, List[float]]]
HeadImportanceScore = List[Tuple[int, int, float]]


class DefaultModeNetworkIdentifier:
    """
    Class to identify the default mode network (DMN) in language models
    by analyzing attention patterns during text processing.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        device: torch.device,
        cache_dir: Optional[str] = None,
        model_name: str = "default",
    ):
        """
        Initialize the DMN identifier.

        Args:
            model: The pre-trained model to analyze
            device: The device to use for computation
            cache_dir: Directory to store cached data
            model_name: Name of the model (used for cache folder)
        """
        self.model = model
        self.device = device
        self.model_name = model_name

        # Create cache directory
        self.cache_dir = os.path.join(cache_dir or CACHE_DIR, model_name.replace("/", "_"))
        os.makedirs(self.cache_dir, exist_ok=True)

        # Model dimensions
        if hasattr(model.config, "num_hidden_layers"):
            self.num_layers = model.config.num_hidden_layers
        elif hasattr(model.config, "n_layer"):
            self.num_layers = model.config.n_layer
        else:
            raise ValueError("Could not determine number of layers in the model")

        if hasattr(model.config, "num_attention_heads"):
            self.num_heads = model.config.num_attention_heads
        elif hasattr(model.config, "n_head"):
            self.num_heads = model.config.n_head
        else:
            raise ValueError("Could not determine number of attention heads in the model")

        # Storage for attention patterns
        self.default_mode_activations: ActivationDict = {}
        self.head_importance_scores: HeadImportanceScore = []
        self.top_default_mode_heads: HeadImportanceScore = []

    def register_attention_hooks(self) -> List[torch.utils.hooks.RemovableHandle]:
        """
        Register hooks to capture attention patterns during forward pass.

        Returns:
            List of hook handles
        """
        hooks = []

        def get_attention_hook(layer_idx: int, head_idx: int) -> Callable[[Any, Any, Any], None]:
            def hook(module: Any, input: Any, output: Any) -> None:
                # For Mistral models, attention outputs contain 2 elements
                if isinstance(output, tuple) and len(output) == 2:
                    # For Mistral models, we don't have direct access to attention weights
                    # We'll use a proxy based on the hidden states
                    hidden_states = output[0]  # First element is hidden states
                    
                    # Extract a proxy for attention based on the activation of this head
                    head_dim = hidden_states.shape[-1] // self.num_heads
                    start_idx = head_idx * head_dim
                    end_idx = (head_idx + 1) * head_dim
                    
                    # Use the mean activation of this head's hidden states as a proxy
                    head_activation = hidden_states[:, :, start_idx:end_idx].abs().mean().item()
                    
                    # Store the activation
                    if layer_idx not in self.default_mode_activations:
                        self.default_mode_activations[layer_idx] = {}

                    if head_idx not in self.default_mode_activations[layer_idx]:
                        self.default_mode_activations[layer_idx][head_idx] = []

                    self.default_mode_activations[layer_idx][head_idx].append(head_activation)
                    
                # For LLaMA models, attention outputs contain attention probs at index 3
                elif isinstance(output, tuple) and len(output) > 3:
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
                
                # If we don't recognize the attention format, log it
                else:
                    if not hasattr(self, "_warned_about_format"):
                        print(f"Warning: Unrecognized attention format for layer {layer_idx}. Output type: {type(output)}")
                        if isinstance(output, tuple):
                            print(f"Output tuple length: {len(output)}")
                        self._warned_about_format = True

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

    def identify_default_mode_network(
        self, chunks: List[str], tokenizer: Any, sample_size: int = 100, use_cache: bool = True
    ) -> None:
        """
        Process text chunks to identify the default mode network.

        Args:
            chunks: List of text chunks to process
            tokenizer: Tokenizer to use for processing text
            sample_size: Number of chunks to process
            use_cache: Whether to use cached activations if available
        """
        print("Identifying default mode network from text chunks...")

        # Check if we have previously processed activations in cache
        activations_cache_file = os.path.join(
            self.cache_dir, f"dmn_activations_{self.model_name.split('/')[-1]}_{sample_size}.pkl"
        )

        if use_cache and os.path.exists(activations_cache_file):
            try:
                print(f"Loading cached DMN activations from {activations_cache_file}")
                with open(activations_cache_file, "rb") as f:
                    cached_data = pickle.load(f)

                if "default_mode_activations" in cached_data:
                    self.default_mode_activations = cached_data["default_mode_activations"]
                    print(
                        f"Loaded cached activations for {len(self.default_mode_activations)} layers"
                    )

                    # Calculate importance scores from the loaded activations
                    self._calculate_head_importance_scores()
                    print("Calculated head importance scores from cached activations")
                    return

            except Exception as e:
                print(f"Error loading cached activations: {e}")

        # Register hooks to capture attention activations
        hooks = self.register_attention_hooks()

        # Process chunks
        chunks_to_process = chunks[:sample_size]
        print(f"Processing {len(chunks_to_process)} chunks to identify default mode network")
        for chunk in tqdm(chunks_to_process):
            # Tokenize the chunk
            inputs = tokenizer(chunk, return_tensors="pt").to(self.device)

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
                    "head_importance_scores": self.head_importance_scores,
                }
                with open(activations_cache_file, "wb") as f:
                    pickle.dump(cache_data, f)
                print(f"Cached DMN activations for {len(self.default_mode_activations)} layers")
            except Exception as e:
                print(f"Error caching DMN activations: {e}")

        print("Default mode network identification complete")

    def _calculate_head_importance_scores(self) -> None:
        """
        Calculate head importance scores based on the captured activations.
        """
        avg_activations: Dict[int, Dict[int, float]] = {}
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

    def select_top_default_mode_heads(self, top_n: int = 50) -> List[Tuple[int, int, float]]:
        """
        Select the top N most active heads as the default mode network.

        Args:
            top_n: Number of top heads to select

        Returns:
            List of (layer_idx, head_idx, score) tuples
        """
        if not self.head_importance_scores:
            raise ValueError("Must run identify_default_mode_network() first")

        self.top_default_mode_heads = self.head_importance_scores[:top_n]

        print(f"Selected top {top_n} heads as the default mode network:")
        for layer_idx, head_idx, score in self.top_default_mode_heads[:10]:
            print(f"  Layer {layer_idx}, Head {head_idx}: Activation {score:.4f}")

        if top_n > 10:
            print(f"  ... plus {top_n-10} more heads")

        return self.top_default_mode_heads

    def save_default_mode_network(self, filepath: str) -> None:
        """
        Save the identified default mode network to a file.

        Args:
            filepath: Path to save the data
        """
        data = {
            "model_name": self.model_name,
            "head_importance_scores": self.head_importance_scores,
            "top_default_mode_heads": self.top_default_mode_heads,
            "timestamp": datetime.now().isoformat(),
        }

        with open(filepath, "wb") as f:
            pickle.dump(data, f)

        print(f"Default mode network saved to {filepath}")

    def load_default_mode_network(self, filepath: str) -> None:
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
            print("Top heads in the default mode network:")
            for layer_idx, head_idx, score in self.top_default_mode_heads[:10]:
                print(f"  Layer {layer_idx}, Head {head_idx}: Activation {score:.4f}")

            if len(self.top_default_mode_heads) > 10:
                print(f"  ... plus {len(self.top_default_mode_heads)-10} more heads")

        except Exception as e:
            print(f"Error loading default mode network from {filepath}: {e}")
            raise
