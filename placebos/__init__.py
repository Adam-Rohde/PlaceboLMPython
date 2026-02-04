"""
Placebos - Mock Language Model responses for testing.

A lightweight Python package for creating placebo/mock responses
from language models, useful for testing LLM-powered applications
without incurring API costs.
"""

from .core import PlaceboLM, PlaceboResponse, PlaceboConfig
from .providers import OpenAIPlacebo, AnthropicPlacebo, GenericPlacebo

__version__ = "0.1.0"
__all__ = [
    "PlaceboLM",
    "PlaceboResponse",
    "PlaceboConfig",
    "OpenAIPlacebo",
    "AnthropicPlacebo",
    "GenericPlacebo",
]
