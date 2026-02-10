"""Python client for the Transformer Inference API and Ollama."""

from .client import InferClient
from .ollama import OllamaClient

__version__ = "1.5.0"
__all__ = ["InferClient", "OllamaClient"]
