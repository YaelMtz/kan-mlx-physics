"""Shorthand alias for kan_mlx_physics.

Usage:
    import kanx
    from kanx import MultKAN, PINNTrainer, create_dataset
"""

# Re-export everything from kan_mlx_physics
from kan_mlx_physics import *

# Explicitly import module-level attributes
import kan_mlx_physics as _kmp
__version__ = _kmp.__version__
__all__ = _kmp.__all__
