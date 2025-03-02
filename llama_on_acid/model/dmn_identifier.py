"""
Module for identifying the default mode network in language models.
"""

import os
import pickle
from datetime import datetime
from typing import Any, Callable, Counter, Dict, List, Optional, Set, Tuple, TypeVar, Union

import torch
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer

from ..config import CACHE_DIR, DMN_CONFIG

ActivationDict = Dict[int, Dict[int, List[float]]]
HeadImportanceScore = List[Tuple[int, int, float]]


# Add debug logging function for consistency across modules
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
        print(f"\n[DMN {timestamp}] 🔍 {msg} 🔍")
    else:
        print(f"[DMN {timestamp}] {msg}")
    if divider:
        print(f"{'=' * 80}\n")


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
        debug_log("Initializing DefaultModeNetworkIdentifier", is_important=True, divider=True)
        debug_log(f"Model: {model_name}, Device: {device}")

        self.model = model
        self.device = device
        self.model_name = model_name

        # Create cache directory
        self.cache_dir = os.path.join(cache_dir or CACHE_DIR, model_name.replace("/", "_"))
        os.makedirs(self.cache_dir, exist_ok=True)
        debug_log(f"Using cache directory: {self.cache_dir}")

        # Model dimensions
        debug_log("Determining model dimensions...")
        if hasattr(model.config, "num_hidden_layers"):
            self.num_layers = model.config.num_hidden_layers
            debug_log(f"Found num_hidden_layers: {self.num_layers}", verbose=False)
        elif hasattr(model.config, "n_layer"):
            self.num_layers = model.config.n_layer
            debug_log(f"Found n_layer: {self.num_layers}", verbose=False)
        else:
            error_msg = "Could not determine number of layers in the model"
            debug_log(error_msg, is_important=True)
            raise ValueError(error_msg)

        if hasattr(model.config, "num_attention_heads"):
            self.num_heads = model.config.num_attention_heads
            debug_log(f"Found num_attention_heads: {self.num_heads}", verbose=False)
        elif hasattr(model.config, "n_head"):
            self.num_heads = model.config.n_head
            debug_log(f"Found n_head: {self.num_heads}", verbose=False)
        else:
            error_msg = "Could not determine number of attention heads in the model"
            debug_log(error_msg, is_important=True)
            raise ValueError(error_msg)

        # Storage for attention patterns
        self.default_mode_activations: ActivationDict = {}
        self.head_importance_scores: HeadImportanceScore = []
        self.top_default_mode_heads: HeadImportanceScore = []

        debug_log(
            f"DMN identifier initialized for model with {self.num_layers} layers and {self.num_heads} heads per layer",
            is_important=True,
        )

    def register_attention_hooks(self) -> List[torch.utils.hooks.RemovableHandle]:
        """
        Register hooks to capture attention patterns during forward pass.

        Returns:
            List of hook handles
        """
        debug_log("Registering attention hooks...", is_important=True)
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
                        debug_log(
                            f"Unrecognized attention format for layer {layer_idx}",
                            is_important=True,
                        )
                        debug_log(f"Output type: {type(output)}")
                        if isinstance(output, tuple):
                            debug_log(f"Output tuple length: {len(output)}")
                            # Print more details about the tuple elements
                            for i, elem in enumerate(output):
                                debug_log(
                                    f"Element {i} type: {type(elem)}, shape: {getattr(elem, 'shape', 'No shape')}"
                                )
                        self._warned_about_format = True

            return hook

        # Register hooks for each attention layer and head
        debug_log(
            f"Registering hooks for {self.num_layers} layers with {self.num_heads} heads each",
            is_important=True,
        )
        registered_count = 0
        error_count = 0

        for layer_idx in range(self.num_layers):
            try:
                # Access the correct attention module based on the model architecture
                if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
                    # For LLaMA/Mistral models
                    attn_module = self.model.model.layers[layer_idx].self_attn
                    debug_log(
                        f"Layer {layer_idx}: Found LLaMA/Mistral style attention module",
                        verbose=False,
                    )
                elif hasattr(self.model, "h"):
                    # For other architectures like GPT-2
                    attn_module = self.model.h[layer_idx].attn
                    debug_log(
                        f"Layer {layer_idx}: Found GPT-2 style attention module", verbose=False
                    )
                else:
                    debug_log(
                        f"Layer {layer_idx}: Could not find attention module", is_important=True
                    )
                    error_count += 1
                    continue

                # Log the attention module type and properties
                debug_log(
                    f"Layer {layer_idx} attention module: {type(attn_module).__name__}",
                    verbose=False,
                )

                # Check for flash attention
                has_flash = hasattr(attn_module, "flash_attn") and attn_module.flash_attn
                if has_flash:
                    debug_log(f"Layer {layer_idx} uses flash attention", verbose=False)

                # Register hooks for each head in this layer
                for head_idx in range(self.num_heads):
                    try:
                        hook = attn_module.register_forward_hook(
                            get_attention_hook(layer_idx, head_idx)
                        )
                        hooks.append(hook)
                        registered_count += 1
                    except Exception as e:
                        debug_log(
                            f"Error registering hook for layer {layer_idx}, head {head_idx}: {e}",
                            is_important=True,
                        )
                        error_count += 1

            except Exception as e:
                debug_log(
                    f"Error accessing attention module for layer {layer_idx}: {e}",
                    is_important=True,
                )
                error_count += 1

        debug_log(f"Registered {registered_count} hooks successfully with {error_count} errors")
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
        debug_log(
            "Starting DMN identification from text chunks...", is_important=True, divider=True
        )
        debug_log(f"Parameters: sample_size={sample_size}, use_cache={use_cache}")

        # Check if we have previously processed activations in cache
        activations_cache_file = os.path.join(
            self.cache_dir, f"dmn_activations_{self.model_name.split('/')[-1]}_{sample_size}.pkl"
        )

        if use_cache and os.path.exists(activations_cache_file):
            try:
                debug_log(f"Found cached DMN activations at: {activations_cache_file}")
                print(f"Loading cached DMN activations from {activations_cache_file}")

                with open(activations_cache_file, "rb") as f:
                    cached_data = pickle.load(f)

                if "default_mode_activations" in cached_data:
                    self.default_mode_activations = cached_data["default_mode_activations"]
                    debug_log(
                        f"Loaded activations for {len(self.default_mode_activations)} layers",
                        verbose=False,
                    )

                    # Log the structure of loaded activations
                    layer_counts = {
                        layer: len(heads) for layer, heads in self.default_mode_activations.items()
                    }
                    debug_log(f"Activation structure: {layer_counts}", verbose=False)

                    # Calculate importance scores from the loaded activations
                    self._calculate_head_importance_scores()
                    debug_log("Calculated head importance scores from cached activations")
                    return
                else:
                    debug_log(
                        "Cached file doesn't contain 'default_mode_activations'", is_important=True
                    )

            except Exception as e:
                debug_log(f"Error loading cached activations: {e}", is_important=True)

        # Register hooks to capture attention activations
        hooks = self.register_attention_hooks()
        debug_log(f"Registered {len(hooks)} hooks to capture attention patterns")

        # Process chunks
        chunks_to_process = chunks[:sample_size]
        debug_log(f"Processing {len(chunks_to_process)} chunks to identify DMN")
        print(f"Processing {len(chunks_to_process)} chunks to identify default mode network")

        # Track success/error rate
        success_count = 0
        error_count = 0

        for idx, chunk in enumerate(tqdm(chunks_to_process)):
            # Tokenize the chunk
            try:
                inputs = tokenizer(chunk, return_tensors="pt").to(self.device)

                # Log token count
                token_count = inputs.input_ids.shape[1]
                if idx == 0:
                    debug_log(
                        f"Processing chunk {idx+1}/{len(chunks_to_process)}: {token_count} tokens"
                    )

                # Process through model with no_grad to save memory
                with torch.no_grad():
                    self.model(**inputs)

                success_count += 1
            except Exception as e:
                debug_log(f"Error processing chunk {idx}: {e}", is_important=True)
                error_count += 1

        debug_log(f"Processed {success_count} chunks successfully with {error_count} errors")

        # Remove hooks
        for hook in hooks:
            hook.remove()
        debug_log("Removed all hooks")

        # Calculate average activations and importance scores for each head
        self._calculate_head_importance_scores()

        # Cache the activations if we have any
        if self.default_mode_activations:
            try:
                debug_log(f"Caching DMN activations to {activations_cache_file}")
                cache_data = {
                    "timestamp": datetime.now().isoformat(),
                    "model_name": self.model_name,
                    "sample_size": sample_size,
                    "default_mode_activations": self.default_mode_activations,
                    "head_importance_scores": self.head_importance_scores,
                }
                with open(activations_cache_file, "wb") as f:
                    pickle.dump(cache_data, f)
                debug_log(
                    f"Cached DMN activations for {len(self.default_mode_activations)} layers",
                    verbose=False,
                )
            except Exception as e:
                debug_log(f"Error caching DMN activations: {e}", is_important=True)

        debug_log("Default mode network identification complete", is_important=True)

    def _calculate_head_importance_scores(self) -> None:
        """
        Calculate head importance scores based on the captured activations.
        """
        debug_log("Calculating head importance scores...")

        if not self.default_mode_activations:
            debug_log(
                "No activations recorded! Cannot calculate head importance.", is_important=True
            )
            return

        # Log the structure of recorded activations
        layer_counts = {layer: len(heads) for layer, heads in self.default_mode_activations.items()}
        debug_log(
            f"Recorded activations for {len(self.default_mode_activations)} layers: {layer_counts}",
            verbose=False,
        )

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
        debug_log(f"Calculated importance scores for {len(head_scores)} heads")

        # Log some stats about the scores
        if head_scores:
            top_5_scores = head_scores[:5]
            bottom_5_scores = head_scores[-5:]
            debug_log(f"Top 5 head scores: {top_5_scores}")
            debug_log(f"Bottom 5 head scores: {bottom_5_scores}")

            min_score = min(score for _, _, score in head_scores)
            max_score = max(score for _, _, score in head_scores)
            avg_score = sum(score for _, _, score in head_scores) / len(head_scores)
            debug_log(
                f"Score stats - Min: {min_score:.4f}, Max: {max_score:.4f}, Avg: {avg_score:.4f}"
            )

    def select_top_default_mode_heads(
        self, top_n_per_layer: Optional[int] = None
    ) -> List[Tuple[int, int, float]]:
        """
        Select top default mode network heads from the importance scores.

        Args:
            top_n_per_layer: Number of top heads to select per layer

        Returns:
            List of (layer_idx, head_idx, score) tuples representing the DMN
        """
        debug_log("Selecting top default mode heads", is_important=True)
        debug_log(f"Parameters: top_n_per_layer={top_n_per_layer}")

        if not self.head_importance_scores:
            self._calculate_head_importance_scores()

        if top_n_per_layer is None:
            top_n_per_layer = DMN_CONFIG["top_n_per_layer"]

        # Whether to skip first and last layers
        skip_first_last = DMN_CONFIG.get("skip_first_last", True)
        if skip_first_last:
            debug_log("Skipping first and last layers as per DMN_CONFIG setting")

        # Dictionary to track heads selected per layer
        heads_per_layer: Dict[int, List[Tuple[int, int, float]]] = {}
        self.top_default_mode_heads = []

        for layer_idx, head_idx, score in sorted(
            self.head_importance_scores, key=lambda x: x[2], reverse=True
        ):
            # Skip first and last layers if configured
            if skip_first_last and (layer_idx == 0 or layer_idx == self.num_layers - 1):
                continue

            # Initialize layer entry if not present
            if layer_idx not in heads_per_layer:
                heads_per_layer[layer_idx] = []

            # Add head if we haven't reached the limit for this layer
            if len(heads_per_layer[layer_idx]) < top_n_per_layer:
                heads_per_layer[layer_idx].append((layer_idx, head_idx, score))
                self.top_default_mode_heads.append((layer_idx, head_idx, score))

        # Sort the top default mode heads by score (highest to lowest) - this is the important change
        self.top_default_mode_heads = sorted(
            self.top_default_mode_heads, key=lambda x: x[2], reverse=True
        )

        # Log the heads selected per layer
        debug_log(
            f"Selected {len(self.top_default_mode_heads)} heads across {len(heads_per_layer)} layers",
            is_important=True,
        )
        debug_log("Heads selected per layer:")
        for layer_idx in sorted(heads_per_layer.keys()):
            selected_heads = heads_per_layer[layer_idx]
            head_indices = [head_idx for _, head_idx, _ in selected_heads]
            debug_log(f"Layer {layer_idx}: {len(selected_heads)} heads {head_indices}")

        # Print the top 10 heads overall by importance
        if self.top_default_mode_heads:
            debug_log("Top 10 most important heads across all layers:", is_important=True)
            for i, (layer, head, score) in enumerate(self.top_default_mode_heads[:10]):
                debug_log(f"  #{i+1}: Layer {layer}, Head {head}, Score {score:.4f}")

        return self.top_default_mode_heads

    def save_default_mode_network(self, filepath: str) -> None:
        """
        Save the identified default mode network to a file.

        Args:
            filepath: Path to save the data
        """
        # Import here to avoid circular imports
        try:
            from ..utils import get_git_commit_hash

            git_hash = get_git_commit_hash()
        except ImportError:
            git_hash = None

        data = {
            "model_name": self.model_name,
            "head_importance_scores": self.head_importance_scores,
            "top_default_mode_heads": self.top_default_mode_heads,
            "timestamp": datetime.now().isoformat(),
            "git_commit": git_hash,
        }

        with open(filepath, "wb") as f:
            pickle.dump(data, f)

        print(f"Default mode network saved to {filepath}")
        print(f"Git commit: {git_hash or 'Not available'}")

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

            # Display git commit hash information if available
            if "git_commit" in data and data["git_commit"]:
                print(f"DMN was identified with git commit: {data['git_commit']}")

            # Try to get current git commit hash for comparison
            try:
                from ..utils import get_git_commit_hash

                current_hash = get_git_commit_hash()
                if (
                    current_hash
                    and "git_commit" in data
                    and data["git_commit"]
                    and current_hash != data["git_commit"]
                ):
                    print(
                        f"Warning: Current git commit ({current_hash}) differs from the one used for identification ({data['git_commit']})"
                    )
            except ImportError:
                pass

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
                    self.select_top_default_mode_heads(top_n_per_layer=5)

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
