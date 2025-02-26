"""
Module for generating text with inhibition of the default mode network.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, cast

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from ..model.dmn_identifier import HeadImportanceScore


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
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.top_default_mode_heads = top_default_mode_heads

        # Model dimensions - used for validating head indices
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

    def generate_with_inhibition(
        self,
        prompt: str,
        inhibition_factor: float = 0.5,
        max_new_tokens: int = 200,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> Tuple[str, str]:
        """
        Generate text with and without inhibition of the default mode network.

        Args:
            prompt: Input prompt
            inhibition_factor: Factor by which to scale down the attention weights (0-1)
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            do_sample: Whether to sample or use greedy decoding

        Returns:
            Tuple of (normal_output, inhibited_output)
        """
        # Validate the top_default_mode_heads
        if not self.top_default_mode_heads:
            raise ValueError("Default mode network heads must be provided")

        # Tokenize prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Generate normally (without inhibition)
        print("Generating without inhibition...")
        with torch.no_grad():
            normal_outputs = self.model.generate(
                inputs.input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
            )

        normal_output = self.tokenizer.decode(normal_outputs[0], skip_special_tokens=True)

        # Now generate with inhibition
        print(f"Generating with inhibition factor {inhibition_factor}...")

        # Create attention mask for inhibiting specific heads
        attention_masks: Dict[int, Dict[int, float]] = {}

        try:
            print("Creating attention masks...")
            # Check for any potential layer_idx that might be out of bounds
            max_layer = -1
            for layer_idx, head_idx, _ in self.top_default_mode_heads:
                if layer_idx > max_layer:
                    max_layer = layer_idx

            if max_layer >= self.num_layers:
                print(
                    f"Warning: Some DMN heads reference layers beyond model capacity (max:{max_layer}, model:{self.num_layers})"
                )
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
                print(
                    f"Warning: Some DMN heads reference heads beyond model capacity (model has {self.num_heads} heads per layer)"
                )
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
            prompt, attention_masks, max_new_tokens, temperature, top_p, do_sample
        )

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
        # Tokenize the prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        # Register hooks for attention inhibition
        hooks = []

        def get_inhibition_hook(
            layer_idx: int, head_masks: Dict[int, float]
        ) -> Callable[[Any, Any, Any], Optional[Tuple[Any, ...]]]:
            def hook(module: Any, input: Any, output: Any) -> Optional[Tuple[Any, ...]]:
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
                            return tuple(output_list)

                        # If no modifications were made, return the original output
                        return output
                    # Return original output if output format doesn't match expectations
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
                        return tuple(output_list)

                    # If no modifications were made, return the original output
                    return output

                # Fallback for other model architectures - may require model-specific adjustments
                else:
                    print(
                        f"Warning: Attention format not recognized for layer {layer_idx}. Output type: {type(output)}"
                    )
                    if isinstance(output, tuple):
                        print(f"Output tuple length: {len(output)}")
                    return cast(Optional[Tuple[Any, ...]], output)

            return hook

        # Log model architecture for debugging
        print(
            f"Registering hooks for model with {self.num_layers} layers and {self.num_heads} heads"
        )

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
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=top_p,
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
        return cast(str, generated_text)
