"""
Type definitions for the LLaMa on Acid project.
"""

from typing import Dict, Any, TypeVar, Union, Tuple

# Define a GenerationResult type alias
# Based on the context, it appears to represent the result of text generation
# The actual return type in inhibited_generator.py is Tuple[str, str]
GenerationResult = Tuple[str, str] 