"""
Components package for PEPE embedders.

This package contains refactored components extracted from the BaseEmbedder class
to improve code organization and maintainability.
"""

from .output_manager import OutputManager
from .embedding_processor import EmbeddingProcessor
from .streaming_io import StreamingIO

__all__ = ["OutputManager", "EmbeddingProcessor", "StreamingIO"] 