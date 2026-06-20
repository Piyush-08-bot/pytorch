"""PyTorch project package."""

__version__ = "0.1.0"
__author__ = "PyTorch Developer"
__license__ = "MIT"

from . import utils
from . import model
from . import train

__all__ = ['utils', 'model', 'train', '__version__']
