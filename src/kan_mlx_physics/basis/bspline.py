"""B-spline basis implementation wrapping existing spline.py.

This maintains backwards compatibility with the original KAN implementation
while providing the new pluggable interface.
"""

from typing import Dict, Any, Optional, List, Tuple
import mlx.core as mx

from .base import Basis, BasisConfig
from ..spline import B_batch, extend_grid


class BSplineBasis(Basis):
    """B-spline basis functions.

    Wraps the existing spline.py implementation to provide
    the standard Basis interface.

    The B-spline basis has M = num_grid + k functions, where:
        - num_grid: Number of grid intervals
        - k: Spline polynomial order (3 = cubic)

    Unlike other bases, B-splines require a grid, which is stored
    in the params dictionary.

    Args:
        config: BasisConfig (M will be interpreted as num_grid + k)
        k: Spline order (default 3 for cubic)
        grid_range: Initial grid range (default (-1, 1))
    """

    name = "bspline"

    def __init__(
        self,
        config: BasisConfig,
        k: int = 3,
        grid_range: Tuple[float, float] = (-1.0, 1.0),
    ):
        # For B-splines, M = num_grid + k
        # So num_grid = M - k
        super().__init__(config)
        self.k = k
        self.num_grid = config.M - k
        self.grid_range = grid_range

        if self.num_grid < 1:
            raise ValueError(
                f"M={config.M} too small for k={k}. "
                f"Need M >= k+1 = {k+1}"
            )

        # Store domain info - B-splines are defined on the grid range
        # but don't use the standard domain mapping
        self.config = BasisConfig(
            M=config.M,
            learnable_affine=config.learnable_affine,
            param_mode=config.param_mode,
            domain=grid_range,
            normalize_method="none",  # Grid handles domain
        )

    @classmethod
    def from_grid(
        cls,
        num_grid: int,
        k: int = 3,
        grid_range: Tuple[float, float] = (-1.0, 1.0),
        **kwargs,
    ) -> "BSplineBasis":
        """Create B-spline basis from grid parameters.

        This is the recommended constructor for B-splines as it
        matches the original KAN interface.

        Args:
            num_grid: Number of grid intervals
            k: Spline order
            grid_range: Range for uniform grid
            **kwargs: Additional BasisConfig parameters

        Returns:
            BSplineBasis instance
        """
        M = num_grid + k
        config = BasisConfig(M=M, **kwargs)
        return cls(config, k=k, grid_range=grid_range)

    def init_params(
        self,
        in_dim: int,
        out_dim: Optional[int] = None,
    ) -> Dict[str, mx.array]:
        """Initialize parameters including the grid.

        For B-splines, the grid is stored as a parameter (though typically
        not trained via gradient descent, but updated adaptively).

        Args:
            in_dim: Number of input dimensions
            out_dim: Number of output dimensions

        Returns:
            Dictionary with grid and optional affine params
        """
        # Get standard affine params from base class
        params = super().init_params(in_dim, out_dim)

        # Initialize uniform grid
        grid = mx.linspace(self.grid_range[0], self.grid_range[1], self.num_grid + 1)
        grid = mx.broadcast_to(grid, (in_dim, self.num_grid + 1))
        grid = extend_grid(grid, k_extend=self.k)

        params["grid"] = grid  # Shape: (in_dim, num_grid + 2*k + 1)

        return params

    def features(self, x: mx.array, params: Dict[str, Any]) -> mx.array:
        """Evaluate B-spline basis functions.

        Args:
            x: Input of shape (batch, in_dim)
            params: Parameter dictionary containing grid

        Returns:
            Basis values of shape (batch, in_dim, M)
        """
        grid = params["grid"]
        return B_batch(x, grid, self.k)

    @property
    def symbolic_priority(self) -> List[str]:
        """B-splines can approximate any smooth function."""
        return ["x", "x^2", "x^3", "x^4", "sin", "cos", "exp", "log", "tanh"]

    def symbolic(
        self,
        coeffs: mx.array,
        params: Dict[str, Any],
        var: str = "x",
    ) -> str:
        """Generate symbolic representation.

        B-splines don't have a simple closed-form, so we return
        a description of the spline parameters.
        """
        grid = params.get("grid")
        if grid is not None:
            g_min = float(grid[0, self.k])
            g_max = float(grid[0, -self.k - 1])
            return f"BSpline(k={self.k}, grid=[{g_min:.2f}, {g_max:.2f}], M={self.M})"
        return f"BSpline(k={self.k}, M={self.M})"

    def update_grid(
        self,
        x: mx.array,
        params: Dict[str, mx.array],
        coef: mx.array,
        margin: float = 0.01,
    ) -> Tuple[Dict[str, mx.array], mx.array]:
        """Update grid based on input data distribution.

        This preserves the function values while adapting the grid
        to the data distribution.

        Args:
            x: Input samples of shape (batch, in_dim)
            params: Current parameters with grid
            coef: Current coefficients of shape (in_dim, out_dim, M)
            margin: Margin to extend grid beyond data range

        Returns:
            Tuple of (updated params, updated coefficients)
        """
        from ..spline import coef2curve, curve2coef

        old_grid = params["grid"]

        # Compute data range
        x_min = mx.min(x, axis=0)
        x_max = mx.max(x, axis=0)
        x_range = x_max - x_min
        x_min = x_min - margin * x_range
        x_max = x_max + margin * x_range

        # Create new uniform grid in data range
        new_grids = []
        for i in range(x.shape[1]):
            new_grids.append(
                mx.linspace(
                    float(x_min[i]),
                    float(x_max[i]),
                    self.num_grid + 1,
                )
            )
        new_grid = mx.stack(new_grids, axis=0)
        new_grid = extend_grid(new_grid, k_extend=self.k)

        # Evaluate current spline at sample points
        y = coef2curve(x, old_grid, coef, self.k)

        # Refit coefficients to match function values on new grid
        new_coef = curve2coef(x, y, new_grid, self.k)

        # Update params
        new_params = dict(params)
        new_params["grid"] = new_grid

        return new_params, new_coef
