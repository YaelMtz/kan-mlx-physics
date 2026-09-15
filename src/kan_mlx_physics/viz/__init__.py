"""Visualization: two declarative surfaces.

- `paper` — publication figures (static, vector PDF, colorblind-safe, serif fonts).
- `live`  — real-time terminal dashboard for experimenting (rich TUI, headless-safe).

    from kan_mlx_physics.viz import paper, live
"""
from . import paper
from . import live

__all__ = ["paper", "live"]
