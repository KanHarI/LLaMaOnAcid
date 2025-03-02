"""
Module for generating text with inhibition of the default mode network.
"""

# Add debug logging function for consistency across modules
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

import torch
from torch.utils.hooks import RemovableHandle
from transformers import PreTrainedModel, PreTrainedTokenizer

from ..config import DEFAULT_GENERATION_PARAMS, DMN_CONFIG
from ..model.dmn_identifier import HeadImportanceScore
from ..types import GenerationResult


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
        print(f"\n[GEN {timestamp}] 🔍 {msg} 🔍")
    else:
        print(f"[GEN {timestamp}] {msg}")
    if divider:
        print(f"{'=' * 80}\n")


class InhibitedGenerator:
    """
    Class to generate text with inhibition of the default mode network.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device: torch.device,
        top_default_mode_heads: HeadImportanceScore,
    ):
        """
        Initialize the inhibited generator.

        Args:
            model: The pre-trained model to use for generation
            tokenizer: The tokenizer to use for processing text
            device: The device to use for computation
            top_default_mode_heads: List of (layer_idx, head_idx, score) tuples representing DMN
        """
        debug_log("Initializing InhibitedGenerator", is_important=True, divider=True)
        debug_log(f"Device: {device}")

        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.top_default_mode_heads = top_default_mode_heads
        # Initialize the attention_masks attribute
        self.attention_masks: Dict[int, Dict[int, float]] = {}
        self.registered_hooks: Dict[Tuple[int, int], RemovableHandle] = {}

        debug_log(f"Received {len(top_default_mode_heads)} DMN heads")
        if top_default_mode_heads:
            # Log a sample of the heads
            sample_size = min(5, len(top_default_mode_heads))
            debug_log(f"Sample of DMN heads: {top_default_mode_heads[:sample_size]}")

        # Model dimensions - used for validating head indices
        debug_log("Determining model dimensions...")
        if hasattr(model.config, "num_hidden_layers"):
            self.num_layers = model.config.num_hidden_layers
            debug_log(f"Found num_hidden_layers: {self.num_layers}")
        elif hasattr(model.config, "n_layer"):
            self.num_layers = model.config.n_layer
            debug_log(f"Found n_layer: {self.num_layers}")
        else:
            error_msg = "Could not determine number of layers in the model"
            debug_log(error_msg, is_important=True)
            raise ValueError(error_msg)

        if hasattr(model.config, "num_attention_heads"):
            self.num_heads = model.config.num_attention_heads
            debug_log(f"Found num_attention_heads: {self.num_heads}")
        elif hasattr(model.config, "n_head"):
            self.num_heads = model.config.n_head
            debug_log(f"Found n_head: {self.num_heads}")
        else:
            error_msg = "Could not determine number of attention heads in the model"
            debug_log(error_msg, is_important=True)
            raise ValueError(error_msg)

        # Check for model architecture type
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            debug_log("Detected LLaMA/Mistral architecture")
            # Check for flash attention
            if (
                hasattr(model.model.layers[0].self_attn, "flash_attn")
                and model.model.layers[0].self_attn.flash_attn
            ):
                debug_log("Model uses flash attention", is_important=True)
        elif hasattr(model, "h"):
            debug_log("Detected GPT-2 style architecture")
        else:
            debug_log("Unknown model architecture", is_important=True)

        debug_log("Generator initialized successfully")

    def generate_with_inhibition(
        self,
        prompt: str,
        inhibition_factor: float = 0.5,
        gamma: float = 0.85,
        max_new_tokens: int = 200,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> Tuple[str, str]:
        """
        Generate text with and without inhibition of the default mode network.

        Args:
            prompt: Input prompt
            inhibition_factor: Base factor by which to scale down the attention weights (0-1)
            gamma: Decay factor for inhibition applied to each successive head (0-1)
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            do_sample: Whether to sample or use greedy decoding

        Returns:
            Tuple of (normal_output, inhibited_output)
        """
        debug_log(
            f"Generating with inhibition (factor={inhibition_factor}, gamma={gamma})",
            is_important=True,
            divider=True,
        )
        debug_log(f"Prompt (first 50 chars): '{prompt[:50]}...'")
        debug_log(
            f"Generation parameters: max_tokens={max_new_tokens}, temp={temperature}, top_p={top_p}, do_sample={do_sample}"
        )

        # Validate the top_default_mode_heads
        if not self.top_default_mode_heads:
            error_msg = "Default mode network heads must be provided"
            debug_log(error_msg, is_important=True)
            raise ValueError(error_msg)

        # Tokenize prompt
        debug_log("Tokenizing prompt...")
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        token_count = inputs.input_ids.shape[1]
        debug_log(f"Prompt tokenized to {token_count} tokens")

        # Generate normally (without inhibition)
        debug_log("Starting normal generation (without inhibition)...")
        print("Generating without inhibition...")
        try:
            with torch.no_grad():
                normal_outputs = self.model.generate(
                    inputs.input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=top_p,
                )
            debug_log("Normal generation completed successfully")
        except Exception as e:
            debug_log(f"Error during normal generation: {e}", is_important=True)
            raise

        normal_output = self.tokenizer.decode(normal_outputs[0], skip_special_tokens=True)
        normal_length = len(normal_output)
        debug_log(f"Normal output length: {normal_length} characters")

        # Now generate with inhibition
        debug_log(
            f"Starting inhibited generation with factor {inhibition_factor} and gamma {gamma}...",
            is_important=True,
        )
        print(f"Generating with inhibition factor {inhibition_factor} and gamma {gamma}...")

        # Create attention mask for inhibiting specific heads
        attention_masks: Dict[int, Dict[int, float]] = {}

        try:
            debug_log("Creating attention masks for DMN inhibition with gamma decay...")
            # Check for any potential layer_idx that might be out of bounds
            max_layer = -1
            for layer_idx, head_idx, _ in self.top_default_mode_heads:
                if layer_idx > max_layer:
                    max_layer = layer_idx

            if max_layer >= self.num_layers:
                debug_log(
                    f"Warning: Some DMN heads reference layers beyond model capacity (max:{max_layer}, model:{self.num_layers})",
                    is_important=True,
                )
                debug_log("Filtering out-of-bound layers...")
                filtered_heads = [h for h in self.top_default_mode_heads if h[0] < self.num_layers]
                self.top_default_mode_heads = filtered_heads
                debug_log(f"Filtered heads to {len(filtered_heads)} valid heads")

            # Check for any head_idx that might be out of bounds
            invalid_heads = []
            for layer_idx, head_idx, _ in self.top_default_mode_heads:
                if head_idx >= self.num_heads:
                    invalid_heads.append((layer_idx, head_idx))

            if invalid_heads:
                debug_log(
                    f"Warning: Some DMN heads reference heads beyond model capacity (model has {self.num_heads} heads per layer)",
                    is_important=True,
                )
                debug_log(f"Invalid head indices: {invalid_heads}")
                debug_log("Filtering invalid heads...")
                filtered_heads = [h for h in self.top_default_mode_heads if h[1] < self.num_heads]
                self.top_default_mode_heads = filtered_heads
                debug_log(f"Filtered to {len(filtered_heads)} valid heads")

            # Now create the attention masks with validated heads and gamma decay
            debug_log("Applying gamma decay to inhibition factors:")

            # First, ensure the heads are sorted by importance score (highest first)
            sorted_heads = sorted(self.top_default_mode_heads, key=lambda x: x[2], reverse=True)

            # Apply gamma decay to inhibition factor for each head based on its rank
            for rank, (layer_idx, head_idx, score) in enumerate(sorted_heads):
                if layer_idx not in attention_masks:
                    attention_masks[layer_idx] = {}

                # Calculate decayed inhibition factor
                decayed_factor = inhibition_factor * (gamma**rank)
                attention_masks[layer_idx][head_idx] = decayed_factor

                # Log the first few heads and their inhibition factors
                if rank < 10:
                    debug_log(
                        f"Head #{rank+1}: Layer {layer_idx}, Head {head_idx}, Score {score:.4f}, Inhibition {decayed_factor:.4f}"
                    )

            debug_log(f"Created attention masks with gamma decay for {len(attention_masks)} layers")
            debug_log("Overall DMN inhibition summary:")
            debug_log(f"  Base inhibition factor: {inhibition_factor}")
            debug_log(f"  Gamma decay factor: {gamma}")
            debug_log(f"  Number of inhibited heads: {len(sorted_heads)}")
            debug_log("  Most inhibited head factor: {inhibition_factor}")
            debug_log(
                f"  Least inhibited head factor: {inhibition_factor * (gamma ** (len(sorted_heads)-1)):.6f}"
            )

        except Exception as e:
            debug_log(f"Error creating attention masks: {e}", is_important=True)
            debug_log(f"top_default_mode_heads format: {self.top_default_mode_heads[:3]}")
            raise

        # Check if using Flash Attention
        is_using_flash_attn = False
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            # For models like LLaMA and Mistral
            if hasattr(self.model.model.layers[0].self_attn, "flash_attn"):
                is_using_flash_attn = self.model.model.layers[0].self_attn.flash_attn

        if is_using_flash_attn:
            debug_log(
                "Model is using Flash Attention. DMN inhibition may work differently.",
                is_important=True,
            )
            debug_log("Attempting to use alternative inhibition method...")

        # Track errors and flash attention layers for reporting and debugging
        verbose = DMN_CONFIG["verbose_logging"]

        debug_log(
            f"Model has {self.num_layers} layers and {self.num_heads} heads per layer",
            verbose=bool(verbose),
        )

        # Convert the model to eval mode for safety
        self.model.eval()

        # Apply attention inhibition during generation
        inhibited_output = self._generate_with_attention_inhibition(
            prompt, attention_masks, max_new_tokens, temperature, top_p, do_sample
        )

        inhibited_length = len(inhibited_output)
        debug_log(f"Inhibited output length: {inhibited_length} characters")
        debug_log("Generation with inhibition complete", divider=True)

        return normal_output, inhibited_output

    def _generate_with_attention_inhibition(
        self,
        prompt: str,
        attention_masks: Dict[int, Dict[int, float]],
        max_new_tokens: int,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> str:
        """
        Generate text with attention inhibition applied to specific heads.

        Args:
            prompt: Input prompt
            attention_masks: Dictionary mapping layer_idx -> head_idx -> scaling_factor
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            do_sample: Whether to sample or use greedy decoding

        Returns:
            Generated text
        """
        # Get verbosity setting from config
        verbose = DMN_CONFIG["verbose_logging"]

        debug_log("Starting generation with attention inhibition...")

        # Tokenize the prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        debug_log(f"Input shape: {input_ids.shape}")

        # Register hooks for attention inhibition
        hooks: List[RemovableHandle] = []

        def get_inhibition_hook(
            layer_idx: int, head_masks: Dict[int, float]
        ) -> Callable[[Any, Any, Any], Optional[Tuple[Any, ...]]]:
            def hook(module: Any, input: Any, output: Any) -> Optional[Tuple[Any, ...]]:
                # Different models have different output formats for attention
                # For Mistral models (using attention with different output format)
                if isinstance(output, tuple) and len(output) == 2:
                    # This is the Mistral format (output is a tuple with hidden states and attention weights)
                    # First element is typically the hidden states, second might contain attention info
                    hidden_states = output[0]

                    # Apply scaling to hidden states directly for Mistral
                    # This is an approximation since we're not directly modifying the attention weights
                    scaling_factor = torch.ones_like(hidden_states)
                    head_dim = hidden_states.shape[-1] // self.num_heads

                    for head_idx, scale in head_masks.items():
                        start_idx = head_idx * head_dim
                        end_idx = (head_idx + 1) * head_dim
                        scaling_factor[:, :, start_idx:end_idx] *= scale

                    # Create a new output with modified hidden states
                    output_list = list(output)
                    output_list[0] = hidden_states * scaling_factor
                    return tuple(output_list)

                # For LLaMA models, attention outputs contain attention probs at index 3
                elif isinstance(output, tuple) and len(output) > 3:
                    # Get attention weights
                    attn_weights = output[3]

                    debug_log(
                        f"LLaMA attention format detected for layer {layer_idx}, attention weights shape: {attn_weights.shape}"
                    )

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
                        # Return the modified output as a tuple
                        return tuple(output_list)

                    # If no modifications were made, return the original output
                    return output

                # Fallback for other model architectures - may require model-specific adjustments
                else:
                    debug_log(
                        f"Warning: Attention format not recognized for layer {layer_idx}. Output type: {type(output)}",
                        is_important=True,
                    )
                    if isinstance(output, tuple):
                        debug_log(f"Output tuple length: {len(output)}")
                        # Print more details about the tuple elements
                        for i, elem in enumerate(output):
                            debug_log(
                                f"Element {i} type: {type(elem)}, shape: {getattr(elem, 'shape', 'No shape')}"
                            )
                    return cast(Optional[Tuple[Any, ...]], output)

            return hook

        # Register hooks for attention inhibition
        hooks = []

        # When registering hooks, use verbose=False to prevent log spam
        debug_log(
            f"Registering hooks for model with {self.num_layers} layers and {self.num_heads} heads",
            is_important=True,
        )

        # Register hooks for each layer
        registered_count = 0
        error_count = 0

        for layer_idx, head_masks in attention_masks.items():
            try:
                # Use verbose=False for detailed hook logs
                debug_log(
                    f"Registering hook for layer {layer_idx} with {len(head_masks)} heads",
                    verbose=False,
                )

                if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
                    # For LLaMA and Mistral models
                    attn_module = self.model.model.layers[layer_idx].self_attn
                    debug_log(
                        f"Found LLaMA/Mistral attention module for layer {layer_idx}", verbose=False
                    )
                elif hasattr(self.model, "layers"):
                    # Some other architectures
                    attn_module = self.model.layers[layer_idx].self_attn
                    debug_log(
                        f"Found generic attention module for layer {layer_idx}", verbose=False
                    )
                elif hasattr(self.model, "h"):
                    # For other architectures (like GPT-2)
                    attn_module = self.model.h[layer_idx].attn
                    debug_log(
                        f"Found GPT-2 style attention module for layer {layer_idx}", verbose=False
                    )
                else:
                    debug_log(
                        f"Warning: Could not find attention module for layer {layer_idx}",
                        is_important=True,
                    )
                    error_count += 1
                    continue

                # Log the attention module type
                debug_log(f"Attention module type: {type(attn_module).__name__}", verbose=False)

                # Register the hook
                hook = attn_module.register_forward_hook(get_inhibition_hook(layer_idx, head_masks))
                hooks.append(hook)
                registered_count += 1
                debug_log(f"Successfully registered hook for layer {layer_idx}", verbose=False)

            except Exception as e:
                debug_log(f"Error registering hook for layer {layer_idx}: {e}", is_important=True)
                error_count += 1

        debug_log(f"Registered {registered_count} hooks with {error_count} errors")

        # Generate with inhibition
        try:
            debug_log("Starting generation with hooks applied...")
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=top_p,
                )
            debug_log("Generation completed successfully")
        except Exception as e:
            debug_log(f"Error during generation with inhibition: {e}", is_important=True)
            # Remove hooks before re-raising
            for hook in hooks:
                hook.remove()
            raise

        # Remove hooks
        debug_log("Removing hooks...")
        for hook in hooks:
            hook.remove()
        debug_log("All hooks removed")

        # Decode and return
        debug_log("Decoding output tokens to text", verbose=bool(verbose))

        # Fix tuple assignment error by converting to a list
        all_tokens = list(outputs[0].cpu().numpy())
        debug_log(f"Generated {len(all_tokens)} total tokens", verbose=bool(verbose))

        # Fix tuple indexing by first converting to a list
        outputs_list = list(outputs)
        outputs_list[0] = outputs_list[0][:, input_ids.shape[1] :]

        debug_log(f"New tokens count: {outputs_list[0].shape[1]}", verbose=bool(verbose))

        generated_text = self.tokenizer.decode(outputs_list[0], skip_special_tokens=True)
        debug_log(f"Output decoded, length: {len(generated_text)} characters")
        return cast(str, generated_text)

    def register_hooks(self) -> List[RemovableHandle]:
        """
        Register hooks to inhibit the default mode network during generation.

        Returns:
            List of hook handles that can be removed later
        """
        # Configure logging verbosity
        verbose = DMN_CONFIG["verbose_logging"]

        debug_log("Registering hooks for DMN heads...", is_important=True)
        hook_handles = []

        try:
            # Get the model architecture for debugging
            model_arch = "unknown"
            if hasattr(self.model, "config"):
                if hasattr(self.model.config, "model_type"):
                    model_arch = self.model.config.model_type
                elif (
                    hasattr(self.model.config, "architectures") and self.model.config.architectures
                ):
                    model_arch = self.model.config.architectures[0]

            debug_log(f"Model architecture: {model_arch}", verbose=bool(verbose))

            # Convert the model to eval mode for safety
            self.model.eval()

            debug_log(
                f"Model has {self.num_layers} layers and {self.num_heads} heads per layer",
                verbose=bool(verbose),
            )

            # Get a list of unique layers that have DMN heads
            layers_with_dmn = sorted(
                set(layer_idx for layer_idx, _, _ in self.top_default_mode_heads)
            )
            debug_log(
                f"DMN heads present in {len(layers_with_dmn)} layers: {layers_with_dmn}",
                verbose=bool(verbose),
            )

            # Track which heads have hooks registered by layer and head index
            self.registered_hooks = {}

            # Register hooks for each layer
            for layer_idx in range(self.num_layers):
                # Get all heads from this layer that are in the DMN
                layer_heads = [
                    (h_idx, score)
                    for l_idx, h_idx, score in self.top_default_mode_heads
                    if l_idx == layer_idx
                ]

                if not layer_heads:
                    debug_log(f"Layer {layer_idx}: No DMN heads to inhibit", verbose=bool(verbose))
                    continue

                debug_log(
                    f"Layer {layer_idx}: Registering hooks for {len(layer_heads)} heads: {[h for h, _ in layer_heads]}",
                    verbose=bool(verbose),
                )

                # Register hooks for each head in the layer
                for head_idx, _ in layer_heads:
                    debug_log(
                        f"Layer {layer_idx}, head {head_idx}: Registering hook",
                        verbose=bool(verbose),
                    )
                    hook_handle = self.model.model.layers[
                        layer_idx
                    ].self_attn.register_forward_hook(
                        lambda module, input, output: self._apply_inhibition(
                            layer_idx, head_idx, input, output
                        )
                    )
                    hook_handles.append(hook_handle)
                    self.registered_hooks[(layer_idx, head_idx)] = hook_handle

            debug_log("All hooks registered successfully")
            return hook_handles

        except Exception as e:
            debug_log(f"Error registering hooks: {e}", is_important=True)
            # Remove all registered hooks
            for hook_handle in hook_handles:
                hook_handle.remove()
            raise

    def _apply_inhibition(
        self, layer_idx: int, head_idx: int, input: Any, output: Any
    ) -> Optional[Tuple[Any, ...]]:
        """
        Apply inhibition to the attention output for a specific layer and head.

        Args:
            layer_idx: Index of the layer
            head_idx: Index of the head
            input: Input to the attention module
            output: Output from the attention module

        Returns:
            Modified output tuple if changes were made, None otherwise
        """
        try:
            verbose = DMN_CONFIG["verbose_logging"]

            debug_log(
                f"Applying inhibition to layer {layer_idx}, head {head_idx}", verbose=bool(verbose)
            )

            # Check if the layer and head are in the registered_hooks dictionary
            if (layer_idx, head_idx) in self.registered_hooks:
                debug_log(f"Layer {layer_idx}, head {head_idx}: Hook found", verbose=bool(verbose))

                # Get the inhibition factor for this head from the attention_masks dictionary
                inhibition_factor = self.attention_masks.get(layer_idx, {}).get(head_idx, 0.0)
                debug_log(
                    f"Layer {layer_idx}, head {head_idx}: Inhibition factor: {inhibition_factor}",
                    verbose=bool(verbose),
                )

                # Apply inhibition to the output
                if isinstance(output, tuple) and len(output) > 3:
                    # For models with attention weights at index 3
                    attn_weights = output[3]
                    debug_log(
                        f"Layer {layer_idx}, head {head_idx}: Attention weights shape: {attn_weights.shape}",
                        verbose=bool(verbose),
                    )

                    # Create a scaling tensor that's 1 except for the target head
                    scaling = torch.ones_like(attn_weights)
                    scaling[:, :, head_idx, :] = inhibition_factor
                    debug_log(
                        f"Layer {layer_idx}, head {head_idx}: Scaling tensor shape: {scaling.shape}",
                        verbose=bool(verbose),
                    )

                    # Apply scaling
                    new_weights = attn_weights * scaling
                    debug_log(
                        f"Layer {layer_idx}, head {head_idx}: New weights shape: {new_weights.shape}",
                        verbose=bool(verbose),
                    )

                    # Update the output tuple
                    output_list = list(output)
                    output_list[3] = new_weights
                    # Return the modified output as a tuple
                    return tuple(output_list)

            # If we didn't make any changes, return None
            return None
        except Exception as e:
            debug_log(f"Error applying inhibition: {e}", is_important=True)
            return None
