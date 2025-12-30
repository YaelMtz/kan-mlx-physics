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
    y0 = 0.3  # Height from input to pre-mult
    z0 = 0.1  # Height from pre-mult to post-mult

    # For simple networks without mult nodes
    # We use y0 + z0 as the total vertical spacing per layer
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

    # Set up coordinate transforms
    DC_to_FC = ax.transData.transform
    FC_to_NFC = fig.transFigure.inverted().transform
    DC_to_NFC = lambda x: FC_to_NFC(DC_to_FC(x))

    # Draw structure: nodes, edges, and inset activation plots
    for l in range(neuron_depth):
        n = width[l]

        # Draw nodes as scatter points
        for i in range(n):
            x_pos = 1 / (2 * n) + i / n if n > 0 else 0.5
            y_pos = l * layer_height
            ax.scatter(x_pos, y_pos, s=min_spacing ** 2 * 10000 * scale ** 2,
                      color='black', zorder=5)

        # Draw edges to next layer
        if l < neuron_depth - 1:
            n_next = width[l + 1]
            N = n * n_next  # Total number of edges

            for i in range(n):
                for j in range(n_next):
                    id_ = i * n_next + j

                    # Check symbolic status
                    symbolic_layer = model.symbolic_funs[l] if l < len(model.symbolic_funs) else None
                    is_symbolic = False
                    if symbolic_layer is not None and symbolic_layer.is_symbolic(i, j):
                        is_symbolic = True

                    # Color and alpha
                    color = "red" if is_symbolic else "black"
                    alpha_val = float(alpha_list[l][i, j]) if l < len(alpha_list) else 1.0

                    # Source position
                    x_src = 1 / (2 * n) + i / n if n > 0 else 0.5
                    y_src = l * layer_height

                    # Inset center position (PyKAN formula)
                    x_inset = 1 / (2 * N) + id_ / N
                    y_inset = l * layer_height + y0 / 2

                    # Target position
                    x_tgt = 1 / (2 * n_next) + j / n_next if n_next > 0 else 0.5
                    y_tgt = l * layer_height + y0

                    # Draw two line segments around the inset (PyKAN style)
                    ax.plot([x_src, x_inset], [y_src, y_inset - y1],
                           color=color, lw=2 * scale, alpha=alpha_val)
                    ax.plot([x_inset, x_tgt], [y_inset + y1, y_tgt],
                           color=color, lw=2 * scale, alpha=alpha_val)

            # Draw connection from pre-mult to post-mult (next layer input)
            # For simple networks, this is just vertical lines
            for j in range(n_next):
                x_pos = 1 / (2 * n_next) + j / n_next if n_next > 0 else 0.5
                y_bottom = l * layer_height + y0
                y_top = (l + 1) * layer_height
                ax.plot([x_pos, x_pos], [y_bottom, y_top],
                       color='black', lw=2 * scale)

    # Set axis limits before placing insets
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.1 * layer_height, (neuron_depth - 1 + 0.1) * layer_height)
    ax.axis('off')

    # Place activation inset images
    for l in range(neuron_depth - 1):
        n = width[l]
        n_next = width[l + 1]
        N = n * n_next

        for i in range(n):
            for j in range(n_next):
                id_ = i * n_next + j

                img_path = os.path.join(folder, f"sp_{l}_{i}_{j}.png")
                if os.path.exists(img_path):
                    im = plt.imread(img_path)

                    # Inset position (PyKAN formula)
                    x_inset = 1 / (2 * N) + id_ / N
                    y_inset = l * layer_height + y0 / 2

                    # Convert to normalized figure coordinates
                    left = DC_to_NFC([x_inset - y1, 0])[0]
                    right = DC_to_NFC([x_inset + y1, 0])[0]
                    bottom = DC_to_NFC([0, y_inset - y1])[1]
                    top = DC_to_NFC([0, y_inset + y1])[1]

                    # Get alpha for this edge
                    alpha_val = float(alpha_list[l][i, j]) if l < len(alpha_list) else 1.0

                    # Create inset axes
                    newax = fig.add_axes([left, bottom, right - left, top - bottom])
                    newax.imshow(im, alpha=alpha_val)
                    newax.axis('off')

    # Place sum/mult symbols at nodes (not input layer, not output layer)
    # These symbols go ON the nodes where edges converge
    sum_symbol_path = os.path.join(os.path.dirname(__file__), "assets/img/sum_symbol.png")
    mult_symbol_path = os.path.join(os.path.dirname(__file__), "assets/img/mult_symbol.png")

    sum_im = plt.imread(sum_symbol_path) if os.path.exists(sum_symbol_path) else None
    mult_im = plt.imread(mult_symbol_path) if os.path.exists(mult_symbol_path) else None

    # Symbol size - make it visible
    symbol_size = y2 * 1.5  # Slightly larger than y2

    for l in range(1, neuron_depth):  # Skip input layer (l=0)
        n = width[l]
        n_sum = model.n_sum[l] if hasattr(model, 'n_sum') else n
        n_mult = model.n_mult[l] if hasattr(model, 'n_mult') else 0

        for j in range(n):
            x_pos = 1 / (2 * n) + j / n if n > 0 else 0.5
            y_pos = l * layer_height  # At the actual node position

            # Determine if this is a sum or mult node
            is_mult = j >= n_sum

            if is_mult and mult_im is not None:
                symbol_im = mult_im
            elif sum_im is not None:
                symbol_im = sum_im
            else:
                continue

            left = DC_to_NFC([x_pos - symbol_size, 0])[0]
            right = DC_to_NFC([x_pos + symbol_size, 0])[0]
            bottom = DC_to_NFC([0, y_pos - symbol_size])[1]
            top = DC_to_NFC([0, y_pos + symbol_size])[1]

            newax = fig.add_axes([left, bottom, right - left, top - bottom])
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
