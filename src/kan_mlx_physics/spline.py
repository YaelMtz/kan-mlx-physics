"""B-spline basis functions for KAN layers in MLX."""

import mlx.core as mx


def extend_grid(grid: mx.array, k_extend: int = 0) -> mx.array:
    """Extend grid by adding boundary points on both ends.

    Args:
        grid: Original grid of shape (in_dim, num_grid + 1)
        k_extend: Number of points to add on each end

    Returns:
        Extended grid of shape (in_dim, num_grid + 1 + 2*k_extend)
    """
    if k_extend == 0:
        return grid

    # Compute step size for each input dimension
    h = (grid[:, -1:] - grid[:, :1]) / (grid.shape[1] - 1)

    # Create extensions on both sides
    left_ext = grid[:, :1] - h * mx.arange(k_extend, 0, -1)
    right_ext = grid[:, -1:] + h * mx.arange(1, k_extend + 1)

    return mx.concatenate([left_ext, grid, right_ext], axis=1)


def B_batch(x: mx.array, grid: mx.array, k: int = 0) -> mx.array:
    """Evaluate B-spline basis functions using de Boor recursion.

    Args:
        x: Input tensor of shape (batch, in_dim)
        grid: Knot positions of shape (in_dim, num_grid + 2*k + 1)
        k: Spline order (0 = step functions, 3 = cubic)

    Returns:
        Basis evaluations of shape (batch, in_dim, num_grid + k)
    """
    # x: (batch, in_dim) -> (batch, in_dim, 1) for broadcasting
    x = mx.expand_dims(x, axis=2)

    # grid: (in_dim, G) where G = num_grid + 2*k + 1
    # We need grid[i] and grid[i+1] for basis computation
    # grid_left: (in_dim, G-1), grid_right: (in_dim, G-1)

    if k == 0:
        # Order 0: Step functions (indicator functions)
        # B_i^0(x) = 1 if grid[i] <= x < grid[i+1], else 0
        grid_left = grid[:, :-1]   # (in_dim, G-1)
        grid_right = grid[:, 1:]   # (in_dim, G-1)

        # Compare with broadcasting: x is (batch, in_dim, 1)
        # grid_left/right need to be (1, in_dim, G-1) for broadcasting
        grid_left = mx.expand_dims(grid_left, axis=0)
        grid_right = mx.expand_dims(grid_right, axis=0)

        # Result: (batch, in_dim, G-1)
        bases = ((x >= grid_left) & (x < grid_right)).astype(mx.float32)
        return bases

    # Recursive case: k > 0
    # B_i^k(x) = w1 * B_i^{k-1}(x) + w2 * B_{i+1}^{k-1}(x)
    # where w1 = (x - t_i) / (t_{i+k} - t_i)
    # and   w2 = (t_{i+k+1} - x) / (t_{i+k+1} - t_{i+1})

    # Get lower order bases recursively
    bases_prev = B_batch(x.squeeze(2), grid, k - 1)  # (batch, in_dim, G-k)

    # Compute weights for the recursion
    # For B_i^k, we need grid points t_i, t_{i+1}, t_{i+k}, t_{i+k+1}
    num_bases = grid.shape[1] - k - 1  # Number of basis functions at order k

    # t_i: grid[:, :num_bases]
    # t_{i+k}: grid[:, k:k+num_bases]
    # t_{i+1}: grid[:, 1:1+num_bases]
    # t_{i+k+1}: grid[:, k+1:k+1+num_bases]

    t_i = mx.expand_dims(grid[:, :num_bases], axis=0)
    t_ik = mx.expand_dims(grid[:, k:k + num_bases], axis=0)
    t_i1 = mx.expand_dims(grid[:, 1:1 + num_bases], axis=0)
    t_ik1 = mx.expand_dims(grid[:, k + 1:k + 1 + num_bases], axis=0)

    # Compute denominators (handle division by zero)
    denom1 = t_ik - t_i
    denom2 = t_ik1 - t_i1

    # Weight 1: (x - t_i) / (t_{i+k} - t_i)
    w1 = mx.where(denom1 != 0, (x - t_i) / denom1, mx.zeros_like(x - t_i))

    # Weight 2: (t_{i+k+1} - x) / (t_{i+k+1} - t_{i+1})
    w2 = mx.where(denom2 != 0, (t_ik1 - x) / denom2, mx.zeros_like(t_ik1 - x))

    # Combine: B_i^k = w1 * B_i^{k-1} + w2 * B_{i+1}^{k-1}
    # bases_prev has num_bases + 1 elements at level k-1
    bases = w1 * bases_prev[:, :, :num_bases] + w2 * bases_prev[:, :, 1:num_bases + 1]

    return bases


def coef2curve(x: mx.array, grid: mx.array, coef: mx.array, k: int) -> mx.array:
    """Evaluate spline curves from B-spline coefficients.

    Args:
        x: Input tensor of shape (batch, in_dim)
        grid: Knot positions of shape (in_dim, num_grid + 2*k + 1)
        coef: B-spline coefficients of shape (in_dim, out_dim, num_grid + k)
        k: Spline order

    Returns:
        Curve values of shape (batch, in_dim, out_dim)
    """
    # Get basis functions: (batch, in_dim, num_grid + k)
    bases = B_batch(x, grid, k)

    # Multiply with coefficients and sum over basis functions
    # bases: (batch, in_dim, num_coef)
    # coef: (in_dim, out_dim, num_coef)
    # We want: (batch, in_dim, out_dim)

    # Expand for broadcasting
    # bases: (batch, in_dim, 1, num_coef)
    # coef: (1, in_dim, out_dim, num_coef)
    bases = mx.expand_dims(bases, axis=2)
    coef = mx.expand_dims(coef, axis=0)

    # Sum over num_coef dimension
    y = mx.sum(bases * coef, axis=3)  # (batch, in_dim, out_dim)

    return y


def curve2coef(x: mx.array, y: mx.array, grid: mx.array, k: int) -> mx.array:
    """Fit B-spline coefficients from curve data using least squares.

    Args:
        x: Input tensor of shape (batch, in_dim)
        y: Output tensor of shape (batch, in_dim, out_dim)
        grid: Knot positions of shape (in_dim, num_grid + 2*k + 1)
        k: Spline order

    Returns:
        B-spline coefficients of shape (in_dim, out_dim, num_grid + k)
    """
    # Get basis functions: (batch, in_dim, num_coef)
    bases = B_batch(x, grid, k)

    batch_size, in_dim, num_coef = bases.shape
    out_dim = y.shape[2]

    # Use CPU stream for linalg operations (not yet supported on GPU in MLX)
    cpu_stream = mx.cpu

    # Solve least squares for each input dimension
    # For each i: bases[:, i, :] @ coef[i, :, :].T = y[:, i, :]
    # Reshape for batch least squares

    coef_list = []
    for i in range(in_dim):
        # A: (batch, num_coef)
        # b: (batch, out_dim)
        A = bases[:, i, :]
        b = y[:, i, :]

        # Solve A @ x = b for x of shape (num_coef, out_dim)
        # Using normal equations: A.T @ A @ x = A.T @ b
        # x = (A.T @ A)^{-1} @ A.T @ b
        AtA = A.T @ A
        Atb = A.T @ b

        # Add small regularization for numerical stability
        AtA = AtA + 1e-8 * mx.eye(num_coef)

        # Solve using direct solve on CPU
        coef_i = mx.linalg.solve(AtA, Atb, stream=cpu_stream)  # (num_coef, out_dim)
        coef_list.append(coef_i)

    # Stack: (in_dim, num_coef, out_dim) -> transpose to (in_dim, out_dim, num_coef)
    coef = mx.stack(coef_list, axis=0)  # (in_dim, num_coef, out_dim)
    coef = mx.transpose(coef, axes=(0, 2, 1))  # (in_dim, out_dim, num_coef)

    return coef
