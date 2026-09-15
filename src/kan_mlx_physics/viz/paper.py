"""
Publication figures — declarative, paper-grade static plots.

One import gives you a consistent, colorblind-safe, serif/Computer-Modern,
vector-output style plus a few high-level figure helpers for the plots that recur
in physics-informed KAN work: analytic-vs-network cross-sections, 2D phase-space
(Wigner) surfaces, and convergence curves.

    from kan_mlx_physics.viz import paper
    paper.use()                                          # apply the style once
    paper.vs_analytic(r, W_kan, W_true).save("fig.pdf")  # cross-section + error
    paper.wigner_2d(X, P, W).save("wigner.pdf")          # diverging cmap (± quasi-prob)
    paper.convergence(history).save("conv.pdf")

Design notes
------------
- Vector-first: `.save()` defaults to PDF; fonts are Computer-Modern math so
  figures match a LaTeX/Typst manuscript.
- Colorblind-safe categorical palette (Okabe–Ito), with *semantic* roles
  (analytic / native / prior / control) so a figure's colors mean the same thing
  across every experiment.
- The Wigner surface uses a **diverging** colormap centered at zero. A Wigner
  function is a quasi-probability: its sign (positive vs negative) is the physics.
  Diverging-symmetric-about-zero is the correct encoding; a sequential or rainbow
  map would hide the negative regions.
"""
from __future__ import annotations
from typing import Optional, Sequence, Any

# Okabe–Ito–derived, colorblind-safe, with semantic roles.
PALETTE = {
    "analytic": "#1a1a1a",   # near-black — the reference / ground truth
    "native":   "#009e73",   # green      — native / correct basis
    "prior":    "#cc3399",   # magenta    — cross-representation prior (e.g. Hermite)
    "control":  "#d55e00",   # vermillion — generic control (e.g. B-spline)
    "accent":   "#5b5b5b",
    "grid":     "#c9c9c9",
    "note_bg":  "#f7f7f4",
    "note_ec":  "#b9b9b0",
}
# Fixed categorical order (never cycled): reference first, then the informative bases.
CATEGORICAL = [PALETTE["analytic"], PALETTE["native"], PALETTE["prior"], PALETTE["control"]]


def use(grid: bool = True) -> None:
    """Apply the publication rcParams globally. Call once before plotting."""
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
        "savefig.format": "pdf",
        "font.family": "serif",
        "font.serif": ["Palatino", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": 11, "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.labelsize": 11, "axes.edgecolor": "#444444", "axes.linewidth": 0.9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": grid, "grid.color": PALETTE["grid"],
        "grid.linewidth": 0.5, "grid.alpha": 0.5,
        "legend.frameon": True, "legend.framealpha": 0.92,
        "legend.edgecolor": "#d0d0d0", "legend.fontsize": 9,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.color": "#444444", "ytick.color": "#444444",
        "axes.axisbelow": True,
    })


class Figure:
    """Thin wrapper around a matplotlib Figure so helpers can chain `.save()`.

    Escape hatch: `.ax` (or `.axes`) and `.fig` expose the matplotlib objects for
    any custom tweak before saving.
    """
    def __init__(self, fig, ax):
        self.fig = fig
        self.ax = ax
        self.axes = ax  # alias when there are multiple

    def title(self, text: str) -> "Figure":
        (self.ax[0] if _is_seq(self.ax) else self.ax).set_title(text)
        return self

    def caption(self, text: str, y: float = 0.005) -> "Figure":
        """Single italic caption strip along the bottom."""
        self.fig.text(0.5, y, text, ha="center", va="bottom", fontsize=8.4,
                      style="italic", color=PALETTE["accent"])
        return self

    def save(self, path: str, **kwargs) -> "Figure":
        import matplotlib.pyplot as plt
        self.fig.savefig(path, **kwargs)
        plt.close(self.fig)
        return self

    def show(self) -> "Figure":
        import matplotlib.pyplot as plt
        plt.show()
        return self


def _is_seq(x) -> bool:
    return hasattr(x, "__len__") and not hasattr(x, "plot")


def _np(a):
    import numpy as np
    try:
        import mlx.core as mx
        if isinstance(a, mx.array):
            return np.array(a)
    except Exception:
        pass
    return np.asarray(a)


# --------------------------------------------------------------- cross-section
def vs_analytic(x, y_pred, y_true, *, label_pred: str = "KAN",
                label_true: str = "analytic", xlabel: str = "$r^2$",
                ylabel: str = "$W(r^2)$", show_error: bool = True,
                figsize=(4.2, 3.4)) -> Figure:
    """Cross-section: predicted curve vs. the analytic reference, with an optional
    error strip below. The recurring physics figure — one call instead of ~30 lines.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    x, y_pred, y_true = _np(x).ravel(), _np(y_pred).ravel(), _np(y_true).ravel()
    if show_error:
        fig, (ax, axe) = plt.subplots(
            2, 1, figsize=figsize, sharex=True,
            gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08})
    else:
        fig, ax = plt.subplots(figsize=figsize); axe = None
    ax.plot(x, y_true, color=PALETTE["analytic"], lw=2.0, label=label_true, zorder=3)
    ax.plot(x, y_pred, color=PALETTE["native"], lw=1.6, ls="--", label=label_pred, zorder=4)
    ax.set_ylabel(ylabel)
    ax.legend(loc="best")
    if axe is not None:
        err = y_pred - y_true
        axe.plot(x, err, color=PALETTE["control"], lw=1.0)
        axe.fill_between(x, 0, err, color=PALETTE["control"], alpha=0.10, lw=0)
        axe.axhline(0, color=PALETTE["accent"], lw=0.6)
        axe.set_ylabel("residual"); axe.set_xlabel(xlabel)
    else:
        ax.set_xlabel(xlabel)
    return Figure(fig, [ax, axe] if axe is not None else ax)


# --------------------------------------------------------------- Wigner 2D surface
def wigner_2d(X, P, W, *, xlabel: str = "$x$", ylabel: str = "$p$",
              levels: int = 24, colorbar: bool = True, figsize=(4.4, 3.6),
              symmetric: bool = True) -> Figure:
    """2D phase-space (Wigner) surface with a DIVERGING colormap centered at zero.

    A Wigner function is a quasi-probability: the sign is physical, so we encode it
    with a two-hue diverging map symmetric about 0 (blue↔red, neutral at zero) —
    NOT a sequential or rainbow map, which would hide the negative regions.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    X, P, W = _np(X), _np(P), _np(W)
    fig, ax = plt.subplots(figsize=figsize)
    if symmetric:
        vmax = float(np.nanmax(np.abs(W))) or 1.0
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    else:
        norm = None
    cf = ax.contourf(X, P, W, levels=levels, cmap="RdBu_r", norm=norm)
    ax.contour(X, P, W, levels=[0.0], colors=PALETTE["accent"], linewidths=0.6)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.grid(False)
    if colorbar:
        cb = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("$W(x, p)$")
        cb.outline.set_linewidth(0.6)
    return Figure(fig, ax)


# --------------------------------------------------------------- convergence
def convergence(history: Any, *, keys: Optional[Sequence[str]] = None,
                logy: bool = True, xlabel: str = "step",
                figsize=(4.2, 3.0)) -> Figure:
    """Loss / eigenvalue convergence from a TrainingHistory (or a dict of lists)."""
    import numpy as np
    import matplotlib.pyplot as plt
    # accept a TrainingHistory-like object or a plain dict
    def series(name):
        if isinstance(history, dict):
            return history.get(name)
        return getattr(history, name, None)
    fig, ax = plt.subplots(figsize=figsize)
    losses = series("losses")
    if losses is not None:
        steps = np.arange(len(losses))
        ax.plot(steps, _np(losses), color=PALETTE["native"], lw=1.6, label="loss")
        if logy:
            ax.set_yscale("log")
    if keys:
        for name, col in zip(keys, CATEGORICAL[1:]):
            s = series(name)
            if s is not None:
                ax.plot(np.arange(len(s)), _np(s), lw=1.4, color=col, label=name)
    ax.set_xlabel(xlabel); ax.set_ylabel("value")
    ax.legend(loc="best")
    return Figure(fig, ax)


__all__ = ["use", "vs_analytic", "wigner_2d", "convergence", "Figure", "PALETTE",
           "CATEGORICAL"]
