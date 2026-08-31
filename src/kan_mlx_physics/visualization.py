"""PyKAN-identical visualization for MLX-KAN.

This module replicates PyKAN's visualization exactly:
1. Vertical layout: input at bottom, output at top
2. Activation functions embedded as inset axes on edges
3. Transparency controlled by tanh(beta * score)
4. Sum/Mult symbols at nodes
5. Uses actual cached activations (not theoretical curves)
6. Coordinate transformation using DC_to_NFC

The plot() function produces diagrams identical to PyKAN's model.plot()
"""

import mlx.core as mx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import FancyBboxPatch
from pathlib import Path
from typing import Optional, List, Tuple, Literal, TYPE_CHECKING
import tempfile
import os

if TYPE_CHECKING:
    from .multkan import MultKAN

from .spline import coef2curve


def plot_kan(
    model: "MultKAN",
    folder: str = "./figures",
    beta: float = 3.0,
    metric: Literal["forward_n", "forward_u", "backward"] = "backward",
    scale: float = 0.5,
    tick: bool = False,
    sample: bool = False,
    in_vars: Optional[List[str]] = None,
    out_vars: Optional[List[str]] = None,
    title: Optional[str] = None,
    varscale: float = 1.0,
    display: bool = True,
    save: bool = True,
) -> plt.Figure:
    """Plot KAN network - exact PyKAN replica.

    Replicates PyKAN's visualization exactly:
    - Vertical layout: input layer at bottom, output at top
    - Activation functions as embedded inset plots on each edge
    - Uses actual cached activations (requires forward pass first)
    - Sum symbols (Σ) at summation nodes
    - Transparency via tanh(beta * score)
    - Black nodes and edges, red for symbolic

    Args:
        model: MultKAN model to visualize
        folder: Directory to save figure and activation PNGs
        beta: Controls transparency via tanh(beta * l1). Default 3.0
        metric: Score metric ('forward_n', 'forward_u', 'backward')
        scale: Figure size scaling (default 0.5)
        tick: Show axis ticks on activation plots
        sample: Show sample points on activation plots
        in_vars: Names for input variables
        out_vars: Names for output variables
        title: Plot title
        varscale: Size scale for variable labels
        display: If True, display the plot
        save: If True, save to file

    Returns:
        matplotlib Figure
    """
    # Ensure we have activations cached
    if not model._save_act:
        print('Cannot plot since data are not saved. Set _save_act=True first.')
        return None

    if model._acts is None:
        if model._cache_data is None:
            # Do a forward pass with sample data
            x_sample = mx.linspace(-1, 1, 100).reshape(-1, 1)
            x_sample = mx.broadcast_to(x_sample, (100, model.width[0]))
            model(x_sample)
        else:
            model(model._cache_data)

    Path(folder).mkdir(parents=True, exist_ok=True)

    depth = model.depth
    width = model.width

    # PyKAN layout parameters
    A = 1  # Horizontal extent
    y0 = 0.5  # Height between layers (increased for readability)
    z0 = 0.0

    # Total vertical spacing per layer
    layer_height = y0 + z0

    # Compute sizing parameters (PyKAN formulas)
    width_arr = np.array(width)
    min_spacing = A / np.maximum(np.max(width_arr), 5)
    max_neuron = np.max(width_arr)
    max_num_weights = np.max(width_arr[:-1] * width_arr[1:])

    y1 = 0.4 / np.maximum(max_num_weights, 5)  # Size of activation insets
    y2 = 0.15 / np.maximum(max_neuron, 5)  # Size of operation symbols

    # Figure dimensions
    neuron_depth = len(width)
    fig_width = 10 * scale
    fig_height = 10 * scale * (neuron_depth - 1) * layer_height

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Compute edge scores for transparency
    def score2alpha(score):
        return np.tanh(beta * score)

    # Get scores based on metric
    if metric == 'backward':
        # Use edge_scores from layers
        scores = []
        for layer in model.layers:
            layer_scores = np.array(layer.edge_scores())
            scores.append(layer_scores)
    else:
        # Forward metrics - use activation scales
        scores = []
        for l, layer in enumerate(model.layers):
            if model._spline_postacts is not None and l < len(model._spline_postacts):
                postacts = np.array(model._spline_postacts[l])
                # Compute per-edge activation scale
                layer_scores = np.std(postacts, axis=0)  # (out_dim, in_dim)
                if metric == 'forward_n':
                    max_score = layer_scores.max()
                    if max_score > 0:
                        layer_scores = layer_scores / max_score
                scores.append(layer_scores.T)  # Transpose to (in_dim, out_dim)
            else:
                scores.append(np.ones((layer.in_dim, layer.out_dim)))

    alpha_list = [score2alpha(score) for score in scores]

    # First pass: save all activation plots using actual cached activations
    for l in range(depth):
        layer = model.layers[l]
        symbolic_layer = model.symbolic_funs[l] if l < len(model.symbolic_funs) else None

        for i in range(layer.in_dim):
            for j in range(layer.out_dim):
                # Determine color based on masks
                is_symbolic = False
                if symbolic_layer is not None and symbolic_layer.is_symbolic(i, j):
                    is_symbolic = True

                # PyKAN color scheme
                color = "red" if is_symbolic else "black"

                # Save activation plot using cached activations
                _save_activation_png_pykan(
                    model, l, i, j,
                    os.path.join(folder, f"sp_{l}_{i}_{j}.png"),
                    color=color,
                    tick=tick,
                    sample=sample,
                    scale=scale,
                )

    # ---------------------------------------------------------------
    # Layout helpers
    # ---------------------------------------------------------------
    # Node x-position given its index j within a group of n nodes
    def node_x(j, n):
        return 1 / (2 * n) + j / n if n > 0 else 0.5

    # For layer l, compute the x-position and y-position of each node.
    # Nodes sit at y = l * layer_height.
    # For layers with mult nodes (n_sum + n_mult nodes total), the first
    # n_sum are sum nodes, the remaining are mult nodes.
    #
    # Each edge in layer l connects:
    #   source: node i in group l   (y = l * layer_height)
    #   dest  : node j in group l+1 (y = (l+1) * layer_height)
    #
    # The activation inset sits halfway between source and dest, but
    # horizontally grouped near the destination node so it's clear
    # which node the edge feeds into.
    #
    # For mult nodes with mult_arity > 1, each mult node j has
    # mult_arity slot edges coming in from different source nodes.
    # We spread those insets horizontally around the mult node's x.

    # Pre-compute per-layer node x-positions
    node_xs = []
    for l in range(neuron_depth):
        n = width[l]
        node_xs.append([node_x(j, n) for j in range(n)])

    # Set up coordinate transforms
    DC_to_FC = ax.transData.transform
    FC_to_NFC = fig.transFigure.inverted().transform
    DC_to_NFC = lambda x: FC_to_NFC(DC_to_FC(x))

    # ---------------------------------------------------------------
    # Draw edges layer by layer
    # ---------------------------------------------------------------
    # Layout strategy:
    #   - Source node at y_src, destination node at y_dst
    #   - Activation insets sit in the upper third of each gap, just
    #     below their destination node, so it's visually clear which
    #     node each inset feeds into.
    #   - For mult nodes with mult_arity > 1, slot insets are spread
    #     horizontally around the ⊗ node's x-position.
    #   - Lines: source → inset bottom, inset top → destination node.
    inset_positions = {}  # (l, i, j) -> (x_inset, y_inset)

    for l in range(neuron_depth - 1):
        n_src = width[l]
        n_dst = width[l + 1]
        n_sum_dst = model.n_sum[l + 1] if hasattr(model, 'n_sum') else n_dst
        n_mult_dst = model.n_mult[l + 1] if hasattr(model, 'n_mult') else 0
        mult_arity = model.mult_arity if hasattr(model, 'mult_arity') else 1

        y_src = l * layer_height
        y_dst = (l + 1) * layer_height
        gap = y_dst - y_src

        # Insets sit close to their destination node.
        # y_inset_base: vertical center of inset boxes (upper third of gap)
        y_inset_base = y_dst - gap * 0.28

        # Horizontal spread for mult slot insets (in data coords)
        # Use a fixed spacing of ~3 inset widths so insets don't overlap
        slot_spread = max(y1 * 3.0, 0.08 / max(n_dst, 1))

        symbolic_layer = model.symbolic_funs[l] if l < len(model.symbolic_funs) else None

        # --- Sum node edges ---
        for j in range(n_sum_dst):
            x_dst = node_xs[l + 1][j]
            n_edges_to_j = n_src

            for i in range(n_src):
                x_src_pos = node_xs[l][i]

                # Fan insets horizontally around x_dst: spread them evenly
                if n_edges_to_j == 1:
                    x_inset = x_dst
                else:
                    span = slot_spread * (n_edges_to_j - 1)
                    x_inset = x_dst - span / 2 + i * slot_spread

                y_inset = y_inset_base

                is_symbolic = symbolic_layer is not None and symbolic_layer.is_symbolic(i, j)
                color = "red" if is_symbolic else "black"
                alpha_val = float(alpha_list[l][i, j]) if l < len(alpha_list) and alpha_list[l].shape[1] > j else 1.0

                # Line from source node up to bottom of inset
                ax.plot([x_src_pos, x_inset], [y_src, y_inset - y1],
                        color=color, lw=2 * scale, alpha=alpha_val)
                # Line from top of inset up to destination node
                ax.plot([x_inset, x_dst], [y_inset + y1, y_dst],
                        color=color, lw=2 * scale, alpha=alpha_val)
                inset_positions[(l, i, j)] = (x_inset, y_inset)

        # --- Mult node edges ---
        # Node index for mult node m  = n_sum_dst + m
        # Slot edge out_dim index for (m, s) = n_sum_dst + m*mult_arity + s
        for m in range(n_mult_dst):
            node_j = n_sum_dst + m
            x_dst = node_xs[l + 1][node_j]

            for s in range(mult_arity):
                out_j = n_sum_dst + m * mult_arity + s
                i = 0  # single input feeds all slots
                x_src_pos = node_xs[l][i]

                # Spread slot insets symmetrically around x_dst
                if mult_arity == 1:
                    x_inset = x_dst
                else:
                    frac = s / (mult_arity - 1) - 0.5   # -0.5 .. +0.5
                    x_inset = x_dst + frac * slot_spread * (mult_arity - 1)

                y_inset = y_inset_base

                is_symbolic = symbolic_layer is not None and symbolic_layer.is_symbolic(i, out_j)
                color = "red" if is_symbolic else "black"
                alpha_val = float(alpha_list[l][i, out_j]) if l < len(alpha_list) and alpha_list[l].shape[1] > out_j else 1.0

                ax.plot([x_src_pos, x_inset], [y_src, y_inset - y1],
                        color=color, lw=2 * scale, alpha=alpha_val)
                ax.plot([x_inset, x_dst], [y_inset + y1, y_dst],
                        color=color, lw=2 * scale, alpha=alpha_val)
                inset_positions[(l, i, out_j)] = (x_inset, y_inset)

    # Draw input and output nodes (plain dots)
    for l in [0, neuron_depth - 1]:
        n = width[l]
        for i in range(n):
            ax.scatter(node_xs[l][i], l * layer_height,
                       s=min_spacing ** 2 * 10000 * scale ** 2,
                       color='black', zorder=5)

    # Set axis limits before placing insets
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.1 * layer_height, (neuron_depth - 1 + 0.1) * layer_height)
    ax.axis('off')

    # ---------------------------------------------------------------
    # Place activation inset images at computed positions
    # ---------------------------------------------------------------
    for (l, i, j), (x_inset, y_inset) in inset_positions.items():
        img_path = os.path.join(folder, f"sp_{l}_{i}_{j}.png")
        if not os.path.exists(img_path):
            continue
        im = plt.imread(img_path)

        left  = DC_to_NFC([x_inset - y1, 0])[0]
        right = DC_to_NFC([x_inset + y1, 0])[0]
        bot   = DC_to_NFC([0, y_inset - y1])[1]
        top   = DC_to_NFC([0, y_inset + y1])[1]

        alpha_val = float(alpha_list[l][i, j]) if l < len(alpha_list) and alpha_list[l].shape[1] > j else 1.0
        newax = fig.add_axes([left, bot, right - left, top - bot])
        newax.imshow(im, alpha=alpha_val)
        newax.axis('off')

    # ---------------------------------------------------------------
    # Place ⊕ / ⊗ symbols at hidden and output nodes
    # ---------------------------------------------------------------
    sum_symbol_path = os.path.join(os.path.dirname(__file__), "assets/img/sum_symbol.png")
    mult_symbol_path = os.path.join(os.path.dirname(__file__), "assets/img/mult_symbol.png")
    sum_im  = plt.imread(sum_symbol_path)  if os.path.exists(sum_symbol_path)  else None
    mult_im = plt.imread(mult_symbol_path) if os.path.exists(mult_symbol_path) else None

    symbol_size = y2 * 2.0

    for l in range(1, neuron_depth):
        n = width[l]
        n_sum = model.n_sum[l] if hasattr(model, 'n_sum') else n

        for j in range(n):
            x_pos = node_xs[l][j]
            y_pos = l * layer_height
            is_mult = j >= n_sum
            symbol_im = mult_im if is_mult else sum_im
            if symbol_im is None:
                continue

            left  = DC_to_NFC([x_pos - symbol_size, 0])[0]
            right = DC_to_NFC([x_pos + symbol_size, 0])[0]
            bot   = DC_to_NFC([0, y_pos - symbol_size])[1]
            top   = DC_to_NFC([0, y_pos + symbol_size])[1]

            newax = fig.add_axes([left, bot, right - left, top - bot])
            newax.imshow(symbol_im)
            newax.axis('off')

    # Add variable labels
    if in_vars is not None:
        n = width[0]
        for i, var_name in enumerate(in_vars[:n]):
            x_pos = 1 / (2 * n) + i / n if n > 0 else 0.5
            ax.text(x_pos, -0.1, var_name,
                   fontsize=40 * scale * varscale,
                   ha='center', va='center')

    if out_vars is not None:
        n = width[-1]
        for i, var_name in enumerate(out_vars[:n]):
            x_pos = 1 / (2 * n) + i / n if n > 0 else 0.5
            ax.text(x_pos, (neuron_depth - 1) * layer_height + 0.15,
                   var_name, fontsize=40 * scale * varscale,
                   ha='center', va='center')

    if title is not None:
        ax.text(0.5, (neuron_depth - 1) * layer_height + 0.3,
               title, fontsize=40 * scale,
               ha='center', va='center')

    if save:
        plt.savefig(f"{folder}/kan_structure.png", dpi=200,
                   bbox_inches='tight', facecolor='white', edgecolor='none')

    if display:
        plt.show()
    else:
        plt.close()

    return fig


def _save_activation_png_pykan(
    model: "MultKAN",
    layer_idx: int,
    in_idx: int,
    out_idx: int,
    filepath: str,
    color: str = "black",
    tick: bool = False,
    sample: bool = False,
    scale: float = 0.5,
) -> None:
    """Save activation plot using actual cached activations (PyKAN style)."""
    layer = model.layers[layer_idx]
    w_large = 2.0

    fig, ax = plt.subplots(figsize=(w_large, w_large))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # Use cached activations if available
    if (model._acts is not None and
        model._spline_postacts is not None and
        layer_idx < len(model._spline_postacts)):

        # Get input activations and spline outputs
        acts = np.array(model._acts[layer_idx][:, in_idx])
        spline_postacts = np.array(model._spline_postacts[layer_idx][:, out_idx, in_idx])

        # Sort by input value for smooth curve
        rank = np.argsort(acts)
        x_plot = acts[rank]
        y_plot = spline_postacts[rank]

        # Plot the learned activation
        ax.plot(x_plot, y_plot, color=color, lw=5)

        # Optionally show sample points
        if sample:
            ax.scatter(x_plot, y_plot, color=color, s=400 * scale ** 2)

        # Set range based on actual data
        if tick:
            x_min, x_max = x_plot.min(), x_plot.max()
            y_min, y_max = y_plot.min(), y_plot.max()
            ax.tick_params(axis="y", direction="in", pad=-22, labelsize=50)
            ax.tick_params(axis="x", direction="in", pad=-15, labelsize=50)
            ax.set_xticks([x_min, x_max])
            ax.set_yticks([y_min, y_max])
            ax.set_xticklabels([f'{x_min:.2f}', f'{x_max:.2f}'])
            ax.set_yticklabels([f'{y_min:.2f}', f'{y_max:.2f}'])
        else:
            ax.set_xticks([])
            ax.set_yticks([])
    else:
        # Fallback to theoretical curve
        x_range = (-1, 1)
        if hasattr(model, 'grid_range'):
            x_range = model.grid_range

        x_plot = np.linspace(x_range[0], x_range[1], 100)
        x_mx = mx.array(x_plot.reshape(-1, 1).repeat(layer.in_dim, axis=1))
        y_spline = coef2curve(x_mx, layer.grid, layer.coef, layer.k)
        y_plot = np.array(y_spline[:, in_idx, out_idx])

        ax.plot(x_plot, y_plot, color=color, lw=5)

        if not tick:
            ax.set_xticks([])
            ax.set_yticks([])

    # Spine styling (PyKAN style)
    ax.patch.set_edgecolor(color)
    ax.patch.set_linewidth(1.5)
    for spine in ax.spines.values():
        spine.set_color(color)

    plt.savefig(filepath, bbox_inches='tight', dpi=400,
               facecolor='white', edgecolor='none')
    plt.close(fig)


def plot_activations(
    model: "MultKAN",
    layer_idx: int,
    x: Optional[mx.array] = None,
    folder: str = "./figures",
    x_range: Tuple[float, float] = (-1, 1),
    n_points: int = 100,
    display: bool = True,
    save: bool = True,
) -> plt.Figure:
    """Plot all activation functions for a single layer in a grid.

    Args:
        model: MultKAN model
        layer_idx: Index of layer to plot
        x: Optional input data to show actual data range
        folder: Directory to save figure
        x_range: Range for plotting
        n_points: Number of points for smooth curves
        display: If True, display inline or show plot
        save: If True, save to file

    Returns:
        matplotlib Figure
    """
    if save:
        Path(folder).mkdir(parents=True, exist_ok=True)

    layer = model.layers[layer_idx]
    in_dim = layer.in_dim
    out_dim = layer.out_dim

    fig, axes = plt.subplots(out_dim, in_dim, figsize=(2 * in_dim, 2 * out_dim))

    if in_dim == 1 and out_dim == 1:
        axes = np.array([[axes]])
    elif out_dim == 1:
        axes = axes.reshape(1, -1)
    elif in_dim == 1:
        axes = axes.reshape(-1, 1)

    # Get symbolic layer if exists
    symbolic_layer = None
    if hasattr(model, 'symbolic_funs') and model.symbolic_funs is not None:
        symbolic_layer = model.symbolic_funs[layer_idx]

    # Use cached activations if available
    use_cached = (model._acts is not None and
                  model._spline_postacts is not None and
                  layer_idx < len(model._spline_postacts))

    for i in range(in_dim):
        for j in range(out_dim):
            ax = axes[j, i]  # PyKAN uses [out, in] ordering

            # Check if symbolic
            is_symbolic = False
            fn_name = None
            if symbolic_layer is not None and symbolic_layer.is_symbolic(i, j):
                is_symbolic = True
                fn_name = symbolic_layer.fns_name[i][j]

            # Color based on symbolic status
            color = "red" if is_symbolic else "black"

            if use_cached:
                # Use actual activations
                acts = np.array(model._acts[layer_idx][:, i])
                spline_out = np.array(model._spline_postacts[layer_idx][:, j, i])
                rank = np.argsort(acts)
                ax.plot(acts[rank], spline_out[rank], color=color, linewidth=2)
            else:
                # Fallback to theoretical curve
                x_plot = np.linspace(x_range[0], x_range[1], n_points)
                x_mx = mx.array(x_plot.reshape(-1, 1).repeat(in_dim, axis=1))
                y_spline = coef2curve(x_mx, layer.grid, layer.coef, layer.k)
                y_curve = np.array(y_spline[:, i, j])
                ax.plot(x_plot, y_curve, color=color, linewidth=2)

            # Symbolic overlay if exists
            if is_symbolic:
                from .symbolic import SYMBOLIC_REGISTRY
                a = float(symbolic_layer.affine_a[i, j])
                b = float(symbolic_layer.affine_b[i, j])
                c = float(symbolic_layer.affine_c[i, j])
                d = float(symbolic_layer.affine_d[i, j])

                fn = SYMBOLIC_REGISTRY[fn_name].fn_np

                if use_cached:
                    acts = np.array(model._acts[layer_idx][:, i])
                    rank = np.argsort(acts)
                    x_plot = acts[rank]
                else:
                    x_plot = np.linspace(x_range[0], x_range[1], n_points)

                y_symbolic = c * fn(a * x_plot + b) + d
                ax.plot(x_plot, y_symbolic, "g--", linewidth=1.5, alpha=0.8)

            ax.set_title(f"({i},{j})" + (f" {fn_name}" if fn_name else ""), fontsize=8)
            ax.tick_params(labelsize=6)
            ax.grid(True, alpha=0.3)

    fig.suptitle(f"Layer {layer_idx} Activations", fontsize=12)
    plt.tight_layout()

    if save:
        plt.savefig(f"{folder}/activations_layer_{layer_idx}.png", dpi=150, bbox_inches="tight")

    if display:
        plt.show()
    else:
        plt.close()

    return fig


def plot_training_history(
    history: dict,
    folder: str = "./figures",
    display: bool = True,
    save: bool = True,
) -> plt.Figure:
    """Plot training loss history.

    Args:
        history: Dictionary with 'train_loss' and optionally 'test_loss'
        folder: Directory to save figure
        display: If True, display inline or show plot
        save: If True, save to file

    Returns:
        matplotlib Figure
    """
    if save:
        Path(folder).mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    steps = range(len(history["train_loss"]))

    ax.plot(steps, history["train_loss"], "b-", label="Train Loss", linewidth=2)

    if "test_loss" in history and len(history["test_loss"]) > 0:
        ax.plot(steps, history["test_loss"], "r--", label="Test Loss", linewidth=2)

    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training History")
    ax.legend()
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save:
        plt.savefig(f"{folder}/training_history.png", dpi=150, bbox_inches="tight")

    if display:
        plt.show()
    else:
        plt.close()

    return fig


def plot_spline_1d(
    model: "MultKAN",
    layer_idx: int,
    in_idx: int,
    out_idx: int,
    x_range: Tuple[float, float] = (-1, 1),
    n_points: int = 200,
    show_knots: bool = True,
    folder: str = "./figures",
    display: bool = True,
    save: bool = True,
) -> plt.Figure:
    """Plot a single spline activation in detail.

    Args:
        model: MultKAN model
        layer_idx: Layer index
        in_idx: Input neuron index
        out_idx: Output neuron index
        x_range: Range to plot
        n_points: Number of points
        show_knots: Show knot positions
        folder: Directory to save figure
        display: If True, display inline or show plot
        save: If True, save to file

    Returns:
        matplotlib Figure
    """
    if save:
        Path(folder).mkdir(parents=True, exist_ok=True)

    layer = model.layers[layer_idx]

    fig, ax = plt.subplots(figsize=(8, 5))

    # Check if symbolic
    is_symbolic = False
    symbolic_layer = None
    if hasattr(model, 'symbolic_funs') and model.symbolic_funs is not None:
        symbolic_layer = model.symbolic_funs[layer_idx]
        if symbolic_layer.is_symbolic(in_idx, out_idx):
            is_symbolic = True

    color = "red" if is_symbolic else "black"

    # Use cached activations if available
    use_cached = (model._acts is not None and
                  model._spline_postacts is not None and
                  layer_idx < len(model._spline_postacts))

    if use_cached:
        acts = np.array(model._acts[layer_idx][:, in_idx])
        spline_out = np.array(model._spline_postacts[layer_idx][:, out_idx, in_idx])
        rank = np.argsort(acts)
        x_plot = acts[rank]
        y_curve = spline_out[rank]
    else:
        x_plot = np.linspace(x_range[0], x_range[1], n_points)
        x_mx = mx.array(x_plot.reshape(-1, 1).repeat(layer.in_dim, axis=1))
        y_spline = coef2curve(x_mx, layer.grid, layer.coef, layer.k)
        y_curve = np.array(y_spline[:, in_idx, out_idx])

    ax.plot(x_plot, y_curve, color=color, linewidth=2, label="Learned Activation")

    # Show knots
    if show_knots:
        grid = np.array(layer.grid[in_idx])
        x_min, x_max = x_plot.min(), x_plot.max()
        knots = grid[(grid >= x_min) & (grid <= x_max)]
        for knot in knots:
            ax.axvline(knot, color="gray", linestyle=":", alpha=0.5)

    # Symbolic overlay
    if is_symbolic and symbolic_layer is not None:
        from .symbolic import SYMBOLIC_REGISTRY
        fn_name = symbolic_layer.fns_name[in_idx][out_idx]
        a = float(symbolic_layer.affine_a[in_idx, out_idx])
        b = float(symbolic_layer.affine_b[in_idx, out_idx])
        c = float(symbolic_layer.affine_c[in_idx, out_idx])
        d = float(symbolic_layer.affine_d[in_idx, out_idx])

        fn = SYMBOLIC_REGISTRY[fn_name].fn_np
        y_symbolic = c * fn(a * x_plot + b) + d
        ax.plot(x_plot, y_symbolic, "g--", linewidth=2, label=f"Symbolic: {fn_name}")

        # Show formula
        formula = symbolic_layer.get_formula(in_idx, out_idx, "x")
        ax.text(
            0.05, 0.95, f"$f(x) = {formula}$",
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.set_title(f"Layer {layer_idx}, Edge ({in_idx}, {out_idx})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save:
        plt.savefig(f"{folder}/spline_L{layer_idx}_{in_idx}_{out_idx}.png", dpi=150, bbox_inches="tight")

    if display:
        plt.show()
    else:
        plt.close()

    return fig


# Convenience aliases matching PyKAN API
def plot(model: "MultKAN", **kwargs) -> plt.Figure:
    """Alias for plot_kan (PyKAN compatibility)."""
    return plot_kan(model, **kwargs)


# =============================================================================
# LIVE PLOTTER
# =============================================================================

class LivePlotter:
    """Live visualization of the learned function during PDE training.

    Updates a matplotlib figure at regular intervals showing:
    - The current learned function W(x) over the domain
    - An optional analytic reference line
    - Optionally, the KAN network graph at each phase end
    - Optionally, export the whole training as an MP4 / GIF video

    Works in both Jupyter notebooks and scripts.

    Example — basic live plot::

        model, h = (
            PDEBuilder("...")
            .domain([0, 6.0])
            .live_plot(freq=200, analytic_fn=lambda x: np.exp(-x**2)/np.pi,
                       var_name="r²", fn_name="W")
            .solve()
        )

    Example — export video::

        model, h = (
            PDEBuilder("...")
            .domain([0, 6.0])
            .live_plot(freq=200, analytic_fn=lambda x: np.exp(-x**2)/np.pi,
                       var_name="r²", fn_name="W",
                       record=True, video_path="training.mp4", fps=15)
            .solve()
        )
    """

    def __init__(
        self,
        freq: int = 500,
        x_range=None,
        analytic_fn=None,
        n_plot_points: int = 200,
        show_kan: bool = False,
        var_name: str = "x",
        fn_name: str = "W",
        figsize: tuple = (8, 4),
        # Video export
        record: bool = False,
        video_path: str = "training.mp4",
        fps: int = 15,
        dpi: int = 150,
        writer: str = "ffmpeg",  # "ffmpeg" → MP4, "pillow" → GIF
    ):
        self.freq = freq
        self.x_range = x_range
        self.analytic_fn = analytic_fn
        self.n_plot_points = n_plot_points
        self.show_kan = show_kan
        self.var_name = var_name
        self.fn_name = fn_name
        self.figsize = figsize
        self.record = record
        self.video_path = video_path
        self.fps = fps
        self.dpi = dpi
        self.writer = writer

        self._model = None
        self._x_np = None
        self._x_mx = None
        self._viz_freq = freq
        self._fig = None
        self._ax = None
        self._learned_line = None
        self._loss_text = None
        self._attached = False
        self._is_jupyter = self._detect_jupyter()

        # Video state
        self._frames: List[np.ndarray] = []   # RGB uint8 arrays, (H, W, 3)
        self._frame_w: Optional[int] = None
        self._frame_h: Optional[int] = None

    def _detect_jupyter(self) -> bool:
        try:
            from IPython import get_ipython
            ip = get_ipython()
            return ip is not None and hasattr(ip, 'kernel')
        except ImportError:
            return False

    def pre_attach(self, domain) -> None:
        """Resolves x_range from domain if not set. Called before training."""
        if self.x_range is None:
            lo, hi = domain.bounds[0]
            self.x_range = (float(lo), float(hi))

        self._x_np = np.linspace(self.x_range[0], self.x_range[1], self.n_plot_points)
        self._x_mx = mx.array(self._x_np.reshape(-1, 1).astype(np.float32))
        self._viz_freq = self.freq

    def post_attach(self, model) -> None:
        """Creates the figure. Called once at the start of training."""
        self._model = model
        self._setup_figure()

    def _setup_figure(self) -> None:
        """Create figure and draw optional analytic reference."""
        if not self._is_jupyter:
            plt.ion()

        self._fig, self._ax = plt.subplots(figsize=self.figsize)
        self._ax.set_xlabel(self.var_name)
        self._ax.set_ylabel(self.fn_name)
        self._ax.set_title(f"Training: {self.fn_name}({self.var_name})")

        # Analytic reference
        if self.analytic_fn is not None:
            y_analytic = self.analytic_fn(self._x_np)
            self._ax.plot(self._x_np, y_analytic, 'k--', lw=1.5,
                          label=f"{self.fn_name} analytic", alpha=0.7)

        # Learned curve (placeholder at zero)
        self._learned_line, = self._ax.plot(
            self._x_np, np.zeros_like(self._x_np),
            color='#2196F3', lw=2.5, label=f"{self.fn_name} learned"
        )

        # Step / loss annotation in upper-left corner
        self._loss_text = self._ax.text(
            0.02, 0.97, "step 0", transform=self._ax.transAxes,
            fontsize=9, va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7),
        )

        self._ax.legend(loc='upper right', fontsize=9)
        self._ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if self._is_jupyter:
            try:
                from IPython.display import display
                display(self._fig)
            except Exception:
                pass
        else:
            plt.pause(0.001)  # shows window immediately on macOS

        # Capture initial frame
        if self.record:
            self._fig.canvas.draw()
            self._capture_frame()

    def on_step(self, step: int, loss: float) -> None:
        """Called at each logged step. Updates/records the plot every viz_freq steps."""
        if step % self._viz_freq != 0:
            return
        self._update_plot(step, loss)

    def _eval_model(self) -> Optional[np.ndarray]:
        """Run a forward pass and return y as a flat numpy array, or None on error."""
        save_act_prev = getattr(self._model, '_save_act', False)
        self._model._save_act = False
        try:
            y_mx = self._model(self._x_mx)
            mx.eval(y_mx)
            return np.array(y_mx).flatten()
        except Exception:
            return None
        finally:
            self._model._save_act = save_act_prev

    def _update_plot(self, step: int, loss: float) -> None:
        """Forward the model, refresh the curve, optionally capture a frame."""
        if self._model is None or self._x_mx is None:
            return

        y_np = self._eval_model()
        if y_np is None:
            return

        self._learned_line.set_ydata(y_np)
        self._ax.relim()
        self._ax.autoscale_view(scalex=False)
        self._loss_text.set_text(f"step {step:,}   loss {loss:.3e}")

        # Capture frame before interactive display (so recording always works)
        if self.record:
            self._fig.canvas.draw()
            self._capture_frame()

        if self._is_jupyter:
            try:
                from IPython.display import clear_output, display
                clear_output(wait=True)
                display(self._fig)
            except Exception:
                pass
        else:
            # plt.pause draws + flushes events in one call; keep it short
            plt.pause(0.001)

    def _capture_frame(self) -> None:
        """Grab the current figure as an RGB uint8 array and append to frame buffer."""
        self._fig.canvas.draw()
        w, h = self._fig.canvas.get_width_height()
        # buffer_rgba() is available in all recent matplotlib versions
        buf = np.frombuffer(self._fig.canvas.buffer_rgba(), dtype=np.uint8)
        frame = buf.reshape(h, w, 4)[:, :, :3]  # drop alpha → RGB
        if self._frame_w is None:
            self._frame_w, self._frame_h = w, h
        self._frames.append(frame.copy())

    def on_phase_end(self, model, phase_name: str) -> None:
        """Called after each phase. Optionally shows the KAN network graph."""
        if not self.show_kan:
            return
        try:
            save_act_prev = getattr(model, '_save_act', True)
            model._save_act = True
            model(self._x_mx)
            mx.eval()
            model.plot(title=f"KAN after phase: {phase_name}", display=True, save=False)
            model._save_act = save_act_prev
        except Exception as e:
            print(f"[LivePlotter] KAN plot failed after phase '{phase_name}': {e}")

    def close(self) -> None:
        """Called after training ends. Saves video if recording; leaves figure open."""
        # Capture one final frame
        if self.record and self._fig is not None:
            self._capture_frame()

        if self.record and self._frames:
            self._write_video()

        if not self._is_jupyter:
            plt.ioff()
            if self._fig is not None:
                plt.show(block=False)

    def _write_video(self) -> None:
        """Write accumulated frames to video file."""
        n = len(self._frames)
        path = self.video_path
        print(f"[LivePlotter] Writing {n} frames → {path}  ({self.fps} fps) ...")

        ext = path.rsplit('.', 1)[-1].lower() if '.' in path else 'mp4'

        # Choose writer
        use_writer = self.writer
        if use_writer == "ffmpeg":
            from matplotlib.animation import FFMpegWriter
            try:
                FFMpegWriter(fps=self.fps)  # test availability
            except Exception:
                print("[LivePlotter] ffmpeg not found; falling back to pillow (GIF).")
                use_writer = "pillow"
                if not path.endswith('.gif'):
                    path = path.rsplit('.', 1)[0] + '.gif'

        if use_writer == "pillow" or ext == "gif":
            self._write_gif(path)
        else:
            self._write_mp4(path)

    def _write_mp4(self, path: str) -> None:
        """Write frames as MP4 using FFMpegWriter."""
        from matplotlib.animation import FFMpegWriter
        import matplotlib.animation as animation

        # Build a fresh figure from the frames
        h, w = self._frames[0].shape[:2]
        fig_vid, ax_vid = plt.subplots(
            figsize=(w / self.dpi, h / self.dpi), dpi=self.dpi
        )
        ax_vid.axis('off')
        fig_vid.subplots_adjust(0, 0, 1, 1)
        im = ax_vid.imshow(self._frames[0])

        writer = FFMpegWriter(fps=self.fps, metadata={"title": "KAN training"},
                              extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'])

        with writer.saving(fig_vid, path, dpi=self.dpi):
            for frame in self._frames:
                im.set_data(frame)
                writer.grab_frame()

        plt.close(fig_vid)
        print(f"[LivePlotter] Saved MP4: {path}  ({len(self._frames)} frames)")

    def _write_gif(self, path: str) -> None:
        """Write frames as GIF using Pillow."""
        try:
            from PIL import Image
        except ImportError:
            print("[LivePlotter] Pillow not installed. Run: pip install Pillow")
            return

        imgs = [Image.fromarray(f) for f in self._frames]
        duration_ms = int(1000 / self.fps)
        imgs[0].save(
            path,
            save_all=True,
            append_images=imgs[1:],
            duration=duration_ms,
            loop=0,
            optimize=False,
        )
        print(f"[LivePlotter] Saved GIF: {path}  ({len(self._frames)} frames, {duration_ms} ms/frame)")
