"""
Live experimenting — a declarative real-time terminal dashboard.

While a physics-informed solve runs, `live.monitor(...)` shows a `rich` TUI with
the per-term loss breakdown, the trainable eigenvalue, an optional blind L2 vs an
analytic reference, and a progress bar. It attaches to the solve as a *zero-weight
loss hook*, so it observes every step without changing the training objective.

    from kan_mlx_physics.viz import live
    mon = live.monitor(analytic=W_true, total_steps=10000)
    model, hist = builder.loss(mon).phase(...).solve()   # dashboard updates live

Works headless / over SSH (no display needed). Requires `rich`; if it is not
installed the monitor degrades to a periodic one-line print.
"""
from __future__ import annotations
from typing import Callable, Optional, Sequence
import time

try:
    import mlx.core as mx
except Exception:  # pragma: no cover
    mx = None

# LossTerm is the hook contract; import lazily so viz has no hard pde dependency.
try:
    from ..pde import LossTerm
    _HAS_LOSSTERM = True
except Exception:  # pragma: no cover
    LossTerm = object
    _HAS_LOSSTERM = False


def _np(a):
    import numpy as np
    if mx is not None and isinstance(a, mx.array):
        return np.array(a)
    return np.asarray(a)


class _Dashboard:
    """Rich TUI (or plain-print fallback) that renders monitor state."""
    def __init__(self, title: str, total_steps: Optional[int]):
        self.title = title
        self.total = total_steps
        self._t0 = None
        self.rich = None
        try:
            from rich.live import Live
            self._Live = Live
            self.rich = True
        except Exception:
            self.rich = False

    def _render(self, state: dict):
        from rich.table import Table
        from rich.panel import Panel
        from rich.progress import Progress, BarColumn, TextColumn
        from rich.console import Group
        t = Table.grid(padding=(0, 2))
        t.add_column(justify="right", style="cyan"); t.add_column()
        step = state.get("step", 0)
        t.add_row("step", f"{step}" + (f" / {self.total}" if self.total else ""))
        if state.get("E") is not None:
            t.add_row("eigenvalue E", f"{state['E']:.5f}")
        if state.get("l2") is not None:
            t.add_row("L2 vs analytic", f"{state['l2']:.2f}%")
        for name, val in (state.get("breakdown") or {}).items():
            t.add_row(f"  {name}", f"{val:.3e}")
        if state.get("total_loss") is not None:
            t.add_row("[bold]total loss", f"[bold]{state['total_loss']:.4e}")
        rows = [t]
        if self.total:
            frac = min(1.0, step / max(1, self.total))
            pb = Progress(TextColumn(""), BarColumn(bar_width=32),
                          TextColumn("{task.percentage:>3.0f}%"))
            pb.add_task("", total=1.0, completed=frac)
            rows.append(pb)
        return Panel(Group(*rows), title=self.title, border_style="green")

    def start(self):
        self._t0 = time.time()
        if self.rich:
            self._live = self._Live(refresh_per_second=8, transient=False)
            self._live.start()

    def update(self, state: dict):
        if self.rich:
            self._live.update(self._render(state))
        else:  # plain fallback
            e = f" E={state['E']:.4f}" if state.get("E") is not None else ""
            l2 = f" L2={state['l2']:.1f}%" if state.get("l2") is not None else ""
            print(f"[{self.title}] step {state.get('step',0)}"
                  f" loss={state.get('total_loss', float('nan')):.3e}{e}{l2}", flush=True)

    def stop(self):
        if self.rich and getattr(self, "_live", None):
            self._live.stop()


class MonitorLoss(LossTerm):
    """A zero-weight LossTerm that renders a live dashboard as a side effect.

    Because its weight is 0 it never changes the training objective; it only reads
    the context each step (the model output, the trainable eigenvalue, the
    per-term breakdown if available) and refreshes the dashboard at `every` steps.
    """
    def __init__(self, analytic: Optional[Callable] = None,
                 analytic_grid=None, total_steps: Optional[int] = None,
                 every: int = 50, title: str = "training", **kw):
        # weight 0 → contributes nothing to the loss
        if _HAS_LOSSTERM:
            super().__init__(0.0, **kw)
        self.analytic = analytic
        self.analytic_grid = analytic_grid
        self.every = every
        self._dash = _Dashboard(title, total_steps)
        self._step = 0
        self._started = False

    def compute(self, ctx):
        import numpy as np
        if not self._started:
            self._dash.start(); self._started = True
        self._step += 1
        if self._step % self.every == 0 or self._step == 1:
            state = {"step": self._step}
            # eigenvalue, if the trainer exposes it
            E = None
            tp = getattr(ctx, "trainable_params", None)
            if tp and "E" in tp:
                Ev = _np(tp["E"]); E = float(Ev.reshape(-1)[0])
            elif getattr(ctx, "computed_eigenvalue", None) is not None:
                E = float(_np(ctx.computed_eigenvalue))
            state["E"] = E
            # blind L2 vs analytic, if provided
            if self.analytic is not None and self.analytic_grid is not None:
                Wp = _np(ctx.get_u(self.analytic_grid)).ravel()
                Wt = _np(self.analytic(self.analytic_grid)).ravel()
                num = min(np.sqrt(np.sum((Wp - Wt) ** 2)),
                          np.sqrt(np.sum((Wp + Wt) ** 2)))
                state["l2"] = float(num / (np.sqrt(np.sum(Wt ** 2)) + 1e-12) * 100)
            self._dash.update(state)
        # zero contribution to the loss
        return mx.array(0.0) if mx is not None else 0.0

    def close(self):
        self._dash.stop()


def monitor(analytic: Optional[Callable] = None, analytic_grid=None,
            total_steps: Optional[int] = None, every: int = 50,
            title: str = "training", **kw) -> MonitorLoss:
    """Create a live-dashboard hook to pass to `builder.loss(...)`.

    Parameters
    ----------
    analytic, analytic_grid:
        Optional reference function and the points to score it on, for a live
        blind L2 readout (analytic is used for *display only*, never in the loss).
    total_steps:
        For the progress bar.
    every:
        Refresh cadence in steps.
    """
    return MonitorLoss(analytic=analytic, analytic_grid=analytic_grid,
                       total_steps=total_steps, every=every, title=title, **kw)


__all__ = ["monitor", "MonitorLoss"]
