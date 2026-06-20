"""Utility functions."""
from typing import Callable
import torch
import os


def get_device() -> torch.device:
    """Get the appropriate device (GPU if available, else CPU).
    
    Returns:
        torch.device: CUDA device if available, otherwise CPU.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def is_colab() -> bool:
    """Check if running in Google Colab.
    
    Returns:
        bool: True if in Colab environment, False otherwise.
    """
    try:
        import google.colab
        return True
    except ImportError:
        return False


def setup_colab():
    """Setup Google Colab environment."""
    if is_colab():
        from google.colab import drive
        drive.mount('/content/drive')
        print("✓ Google Drive mounted at /content/drive")
    return is_colab()


def get_data_path():
    """Get data path (Colab Drive or local)."""
    if is_colab():
        return "/content/drive/MyDrive/data"
    return "./data"


def count_parameters(model):
    """Count total trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def set_seed(seed=42):
    """Set random seed for reproducibility."""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
