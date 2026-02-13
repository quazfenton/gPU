"""
Backend implementations for the Notebook ML Orchestrator.

This module contains concrete implementations of the Backend interface
for various cloud compute platforms.
"""

from .modal_backend import ModalBackend
from .huggingface_backend import HuggingFaceBackend
from .kaggle_backend import KaggleBackend
from .colab_backend import ColabBackend

__all__ = ['ModalBackend', 'HuggingFaceBackend', 'KaggleBackend', 'ColabBackend']
