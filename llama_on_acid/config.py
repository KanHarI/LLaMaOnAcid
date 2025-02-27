"""
Configuration settings for the LLaMa on Acid experiment.
"""

import os
from typing import Any, Dict, Optional

import torch

# Default paths
CACHE_DIR = os.path.join(os.getcwd(), "cache")

# Model settings
DEFAULT_MODEL_NAME = "meta-llama/Llama-3-8b"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data settings
DEFAULT_CHUNK_SIZE = 512
MIN_CHUNK_TOKENS = 50
CACHE_TTL_DAYS = {
    "articles": 30,
    "content": 90,
}

# Generation settings
DEFAULT_GENERATION_PARAMS = {
    "max_new_tokens": 200,
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.9,
}

# Experiment settings
DEFAULT_TOP_HEADS = 50
DEFAULT_SAMPLE_SIZE = 100
DEFAULT_INHIBITION_FACTORS = [0.0, 0.3, 0.5, 0.7, 0.9]

# Default queries for experiments when none are provided
DEFAULT_QUERIES = [
    "What is the meaning of life?",
    "Write a creative story about a robot who discovers emotions.",
    "Explain quantum mechanics in simple terms.",
    "Describe a new color that doesn't exist.",
    "What would happen if humans could photosynthesize?",
]

# Fallback articles when API fails
FALLBACK_ARTICLES = [
    "United_States",
    "World_War_II",
    "Albert_Einstein",
    "Climate_change",
    "Artificial_intelligence",
    "COVID-19_pandemic",
    "Quantum_mechanics",
    "World_Wide_Web",
    "William_Shakespeare",
    "Solar_System",
    "Computer",
    "Mathematics",
    "Biology",
    "History",
    "Science",
    "Technology",
    "Art",
    "Music",
    "Film",
    "Literature",
    "Psychology",
    "Economics",
    "Physics",
    "China",
    "India",
    "Europe",
    "Africa",
    "Asia",
    "North_America",
    "South_America",
    "Russia",
    "Germany",
    "United_Kingdom",
    "France",
    "Japan",
    "Australia",
    "Canada",
    "Brazil",
    "Mexico",
    "Italy",
    "Spain",
    "Democracy",
    "Capitalism",
    "Socialism",
    "Internet",
    "Space_exploration",
    "Genetics",
    "Evolution",
]
