"""
Minimal, intuitive multi-seed experiment sweeps.

Running a physics-informed *study* (not a single solve) means the same boilerplate
every time: a grid of parameters × seeds, resume-on-restart, append-to-JSONL, and a
grouped summary table. This module collapses all of that into a few lines.

Example — a full resumable multi-seed grid + summary in ~5 lines::

    from kan_mlx_physics.experiment import Sweep

    sweep = Sweep("results.jsonl", grid=[20, 36, 60], n=[1, 2], seed=range(6))

    @sweep.run                          # resumable: skips finished cells
    def cell(grid, n, seed):
        model, E = solve(n, grid, seed)      # your solve
        return {"nodes": count_nodes(model), "l2": l2_error(model, n)}

    sweep.summary(group_by=["grid", "n"], show=["l2"])   # median table

Design goals: read like the experiment you have in mind; never hand-write the
seed loop, the resume logic, the JSONL I/O, or the grouping/median code again.
"""
from __future__ import annotations
import os
import json
import itertools
import time
from typing import Any, Callable, Dict, Iterable, List, Sequence

try:
    import numpy as np
except Exception:  # numpy is a hard dep in practice; guard for import-time safety
    np = None


def _median(xs: Sequence[float]) -> float:
    xs = [x for x in xs if x is not None and x == x]  # drop None / NaN
    if not xs:
        return float("nan")
    if np is not None:
        return float(np.median(xs))
    s = sorted(xs)
    m = len(s) // 2
    return s[m] if len(s) % 2 else (s[m - 1] + s[m]) / 2.0


def _std(xs: Sequence[float]) -> float:
    xs = [x for x in xs if x is not None and x == x]
    if len(xs) < 2:
        return 0.0
    if np is not None:
        return float(np.std(xs))
    mu = sum(xs) / len(xs)
    return (sum((x - mu) ** 2 for x in xs) / len(xs)) ** 0.5


class Sweep:
    """A resumable multi-parameter × multi-seed experiment sweep.

    Parameters
    ----------
    path:
        JSONL file to append results to. Re-running skips already-recorded cells,
        so an interrupted sweep resumes exactly where it stopped.
    **axes:
        Named parameter axes. Each value is an iterable (list, ``range``, etc.).
        The cartesian product of all axes defines the cells. The names become the
        keyword arguments passed to your cell function AND the JSONL record keys.

    Notes
    -----
    A cell is identified by the tuple of its axis values, so your cell function's
    signature must accept exactly the axis names as keyword arguments.
    """

    def __init__(self, path: str, **axes: Iterable[Any]):
        if not axes:
            raise ValueError("Sweep needs at least one named axis, e.g. seed=range(6).")
        self.path = path
        self.axes = {k: list(v) for k, v in axes.items()}
        self.axis_names = list(self.axes.keys())

    # ---- cell enumeration -------------------------------------------------
    def cells(self) -> List[Dict[str, Any]]:
        """All parameter combinations as a list of dicts (cartesian product)."""
        values = [self.axes[k] for k in self.axis_names]
        return [dict(zip(self.axis_names, combo))
                for combo in itertools.product(*values)]

    def _key(self, rec: Dict[str, Any]):
        return tuple(rec[k] for k in self.axis_names)

    def _load_done(self) -> Dict[tuple, Dict[str, Any]]:
        done: Dict[tuple, Dict[str, Any]] = {}
        if os.path.exists(self.path):
            with open(self.path) as f:
                for line in f:
                    try:
                        r = json.loads(line)
                        done[self._key(r)] = r
                    except Exception:
                        pass
        return done

    # ---- running ----------------------------------------------------------
    def run(self, fn: Callable[..., Dict[str, Any]]) -> Callable[..., Dict[str, Any]]:
        """Decorator: run ``fn`` over every not-yet-done cell, append results.

        ``fn(**axis_kwargs)`` must return a dict of result fields (e.g.
        ``{"l2": 3.2, "nodes": 2}``). The axis values are merged into the record
        automatically, so you never repeat them. Resumable, append-only.

        Usage::

            @sweep.run
            def cell(grid, n, seed):
                ...
                return {"l2": ..., "nodes": ...}

        Returns the (unchanged) function, so it stays callable for a single cell.
        """
        done = self._load_done()
        all_cells = self.cells()
        todo = [c for c in all_cells if self._key(c) not in done]
        print(f"[Sweep] {len(all_cells)} cells, {len(done)} done, "
              f"{len(todo)} to run -> {self.path}", flush=True)
        for cell in todo:
            t = time.time()
            try:
                out = fn(**cell)
                if not isinstance(out, dict):
                    raise TypeError(f"cell fn must return a dict, got {type(out)}")
                rec = {**cell, **out, "fail": out.get("fail", False)}
            except Exception as ex:  # a failed cell is recorded, not fatal
                rec = {**cell, "fail": True, "error": str(ex)[:120]}
            with open(self.path, "a") as f:
                f.write(json.dumps(rec, default=_json_default) + "\n")
            label = " ".join(f"{k}={cell[k]}" for k in self.axis_names)
            extra = "" if rec.get("fail") else _fmt_fields(rec, self.axis_names)
            flag = " FAIL" if rec.get("fail") else ""
            print(f"  {label}: {extra}{flag} [{time.time()-t:.0f}s]", flush=True)
        return fn

    # ---- reporting --------------------------------------------------------
    def results(self) -> List[Dict[str, Any]]:
        """All recorded results as a list of dicts."""
        return list(self._load_done().values())

    def summary(self, group_by: Sequence[str] | None = None,
                show: Sequence[str] | None = None,
                agg: str = "median") -> None:
        """Print a grouped summary table.

        Parameters
        ----------
        group_by:
            Axis names to group rows by (default: all axes except the last, which
            is usually the seed).
        show:
            Result fields to aggregate and display. Defaults to every numeric,
            non-axis field found in the records.
        agg:
            ``"median"`` (default) or ``"mean"``; both also print the spread.
        """
        rows = [r for r in self.results() if not r.get("fail")]
        if not rows:
            print("[Sweep] no successful results yet.")
            return
        if group_by is None:
            group_by = self.axis_names[:-1] or self.axis_names
        if show is None:
            axis_set = set(self.axis_names) | {"fail", "error"}
            show = [k for k, v in rows[0].items()
                    if k not in axis_set and isinstance(v, (int, float))]
        # group
        groups: Dict[tuple, List[Dict[str, Any]]] = {}
        for r in rows:
            groups.setdefault(tuple(r.get(k) for k in group_by), []).append(r)

        head = list(group_by) + [f"{f}" for f in show] + ["n"]
        widths = [max(len(str(h)), 8) for h in head]
        print("  " + "  ".join(str(h).rjust(w) for h, w in zip(head, widths)))
        aggf = _median if agg == "median" else (lambda xs: sum(xs) / len(xs))
        for key in sorted(groups, key=lambda k: tuple(str(x) for x in k)):
            g = groups[key]
            cells = [str(x) for x in key]
            for f in show:
                vals = [r[f] for r in g if f in r]
                cells.append(f"{aggf(vals):.2f}±{_std(vals):.2f}")
            cells.append(str(len(g)))
            print("  " + "  ".join(c.rjust(w) for c, w in zip(cells, widths)))


def _is_real(v: Any) -> bool:
    """True for Python floats AND numpy float scalars (which are not `float`)."""
    if isinstance(v, float):
        return True
    if isinstance(v, bool) or isinstance(v, int):
        return False
    try:  # numpy floating scalar without a hard import
        import numpy as _np
        return isinstance(v, _np.floating)
    except Exception:
        return False


def _fmt_fields(rec: Dict[str, Any], axis_names: Sequence[str]) -> str:
    parts = []
    for k, v in rec.items():
        if k in axis_names or k in ("fail", "error"):
            continue
        parts.append(f"{k}={float(v):.3g}" if _is_real(v) else f"{k}={v}")
    return " ".join(parts)


def _json_default(o):
    # make numpy scalars / arrays JSON-serializable without a hard numpy import here
    try:
        import numpy as _np
        if isinstance(o, _np.generic):
            return o.item()
        if isinstance(o, _np.ndarray):
            return o.tolist()
    except Exception:
        pass
    return str(o)


__all__ = ["Sweep"]
