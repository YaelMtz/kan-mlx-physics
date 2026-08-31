"""
Statistical Performance Comparison: PyKAN vs kan-mlx-physics

Runs multiple trials with different random seeds to get statistical comparison.
"""

import time
import gc
import psutil
import os
import numpy as np
import torch
import mlx.core as mx
from scipy.special import eval_laguerre

process = psutil.Process(os.getpid())

def get_memory_mb():
    """Get current memory usage in MB."""
    return process.memory_info().rss / 1024 / 1024

def analytic_wigner(r2, n=0, hbar_val=1.0):
    """Analytic Wigner function for ground state."""
    normalization = ((-1) ** n) / (np.pi * hbar_val)
    gaussian = np.exp(-r2 / hbar_val)
    laguerre = eval_laguerre(n, 2 * r2 / hbar_val)
    return normalization * gaussian * laguerre

# Configuration
DOMAIN = 3.0
HBAR = 1.0
N_TRIALS = 5

CONFIG = {
    'width': [1, 2, 1],
    'grid': 5,
    'k': 3,
    'phase1_steps': 3000,
    'phase1_lr': 0.001,
    'phase1_points': 1000,
    'phase2_steps': 1000,
    'phase2_lr': 0.0003,
    'phase2_points': 2000,
    'phase3_steps': 200,
    'phase3_lr': 0.0001,
    'phase3_points': 2000,
    'weight_pde': 10.0,
    'weight_norm': 10.0,
    'weight_nontrivial': 20.0,
    'weight_decay': 10.0,
    'weight_smooth': 0.5,
    'weight_anchor': 10.0,
    'weight_eig': 1.0,
    'weight_reg': 0.1,
}

print("=" * 80)
print("STATISTICAL PERFORMANCE COMPARISON")
print("=" * 80)
print(f"\nRunning {N_TRIALS} trials with different random seeds")
print(f"Configuration: {CONFIG['width']}, grid={CONFIG['grid']}, k={CONFIG['k']}")
print(f"Total steps per trial: {CONFIG['phase1_steps'] + CONFIG['phase2_steps'] + CONFIG['phase3_steps']}")

# ============================================================================
# PyKAN Trials
# ============================================================================

print("\n" + "=" * 80)
print("PYKAN TRIALS")
print("=" * 80)

pykan_results = []

for trial in range(N_TRIALS):
    seed = 42 + trial
    print(f"\nTrial {trial+1}/{N_TRIALS} (seed={seed})")
    
    try:
        from kan import KAN
    except ImportError:
        print("ERROR: PyKAN not installed")
        break
    
    # Memory baseline
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    mem_before = get_memory_mb()
    
    # Create model
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    model = KAN(width=CONFIG['width'], grid=CONFIG['grid'], k=CONFIG['k'], seed=seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        model = model.to(device)
    
    # Derivative functions
    def compute_derivatives(model, x):
        h = 1e-4
        y = model(x)
        x_plus = x + h
        x_minus = x - h
        y_plus = model(x_plus)
        y_minus = model(x_minus)
        dy_dx = (y_plus - y_minus) / (2 * h)
        d2y_dx2 = (y_plus - 2 * y + y_minus) / (h ** 2)
        return y, dy_dx, d2y_dx2
    
    def compute_eigenvalue(model, x):
        y, dy, _ = compute_derivatives(model, x)
        r2 = x
        hbar2_4 = HBAR**2 / 4.0
        grad_squared = 4.0 * r2 * (dy**2)
        numerator = torch.mean(r2 * (y**2) + hbar2_4 * grad_squared)
        denominator = 2.0 * torch.mean(y**2) + 1e-8
        return numerator / denominator
    
    def compute_loss(model, x, E):
        r2 = x
        W, dW, d2W = compute_derivatives(model, x)
        
        residual = (r2 - 2*E) * W - (HBAR**2/4) * (4*r2*d2W + 4*dW)
        loss_pde = torch.mean(residual**2) * CONFIG['weight_pde']
        loss_norm = torch.abs(torch.mean(W**2) - 1.0) * CONFIG['weight_norm']
        loss_nontrivial = torch.exp(-torch.mean(W**2) * 10) * CONFIG['weight_nontrivial']
        
        threshold = 0.8 * (2 * DOMAIN**2)
        beyond_mask = r2 > threshold
        loss_decay = torch.mean(W[beyond_mask]**2) * CONFIG['weight_decay'] if beyond_mask.any() else torch.tensor(0.0)
        
        loss_smooth = torch.mean(d2W**2) * CONFIG['weight_smooth']
        anchor_x = torch.tensor([[0.0]])
        W_anchor = model(anchor_x)
        loss_anchor = ((W_anchor - 1.0/(np.pi*HBAR))**2) * CONFIG['weight_anchor']
        loss_eig = torch.abs(E - 0.5) * CONFIG['weight_eig']
        
        params_flat = torch.cat([p.flatten() for p in model.parameters()])
        loss_reg = torch.mean(torch.abs(params_flat)) * CONFIG['weight_reg']
        
        total = loss_pde + loss_norm + loss_nontrivial + loss_decay + loss_smooth + loss_anchor + loss_eig + loss_reg
        return total
    
    # Training
    start_time = time.time()
    
    # Phase 1
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['phase1_lr'])
    for step in range(CONFIG['phase1_steps']):
        r2_vals = np.random.uniform(0, 2 * DOMAIN**2, CONFIG['phase1_points'])
        x = torch.FloatTensor(r2_vals.reshape(-1, 1)).to(device)
        
        with torch.no_grad():
            E = compute_eigenvalue(model, x)
        
        loss = compute_loss(model, x, E)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Phase 2
    x_grid = np.linspace(0, 2 * DOMAIN**2, 1000).reshape(-1, 1)
    model.update_grid_from_samples(torch.FloatTensor(x_grid))
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['phase2_lr'])
    
    for step in range(CONFIG['phase2_steps']):
        r2_vals = np.random.uniform(0, 2 * DOMAIN**2, CONFIG['phase2_points'])
        x = torch.FloatTensor(r2_vals.reshape(-1, 1)).to(device)
        
        with torch.no_grad():
            E = compute_eigenvalue(model, x)
        
        loss = compute_loss(model, x, E)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Phase 3
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['phase3_lr'])
    
    for step in range(CONFIG['phase3_steps']):
        r2_vals = np.random.uniform(0, 2 * DOMAIN**2, CONFIG['phase3_points'])
        x = torch.FloatTensor(r2_vals.reshape(-1, 1)).to(device)
        
        with torch.no_grad():
            E = compute_eigenvalue(model, x)
        
        loss = compute_loss(model, x, E)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    elapsed = time.time() - start_time
    mem_after = get_memory_mb()
    
    # Evaluate
    r2_test = torch.linspace(0, 2 * DOMAIN**2, 1000).reshape(-1, 1)
    W_kan = model(r2_test).detach().cpu().numpy().flatten()
    W_exact = analytic_wigner(r2_test.cpu().numpy().flatten())
    
    mse = np.mean((W_kan - W_exact)**2)
    l2_error = np.sqrt(mse) / (np.sqrt(np.mean(W_exact**2)) + 1e-10)
    
    result = {
        'time': elapsed,
        'memory': mem_after - mem_before,
        'mse': mse,
        'l2_error': l2_error,
        'final_loss': float(loss),
    }
    pykan_results.append(result)
    
    print(f"  Time: {elapsed:.2f}s, MSE: {mse:.2e}, L2: {l2_error*100:.2f}%")
    
    # Cleanup
    del model
    gc.collect()

# ============================================================================
# MLX Trials
# ============================================================================

print("\n" + "=" * 80)
print("MLX TRIALS")
print("=" * 80)

mlx_results = []

for trial in range(N_TRIALS):
    seed = 42 + trial
    print(f"\nTrial {trial+1}/{N_TRIALS} (seed={seed})")
    
    from kan_mlx_physics import register_physics_symbolic
    from kan_mlx_physics.pde import (
        PDEBuilder,
        PDEResidualLoss,
        NormalizationLoss,
        NonTrivialLoss,
        DecayLoss,
        SmoothnessLoss,
        AnchorLoss,
        EigenvalueLoss,
        RegularizationLoss,
    )
    
    register_physics_symbolic()
    
    gc.collect()
    mem_before = get_memory_mb()
    
    def wigner_rayleigh(trainer, x, params):
        hbar = params.get("hbar", HBAR)
        hbar2_4 = hbar**2 / 4.0
        r2 = mx.squeeze(x, axis=-1)
        W = trainer.u(x)
        dW_dr2 = trainer.du(x)
        grad_W_squared = 4.0 * r2 * (dW_dr2**2)
        numerator = mx.mean(r2 * (W**2) + hbar2_4 * grad_W_squared)
        denominator = 2.0 * mx.mean(W**2) + 1e-8
        return numerator / denominator
    
    equation = "(r2 - 2*E)*W - (hbar**2/4)*(4*r2*Derivative(W, r2, 2) + 4*Derivative(W, r2)) = 0"
    
    start_time = time.time()
    
    model, history = (
        PDEBuilder(equation)
        .params(hbar=HBAR)
        .domain([0, 2 * DOMAIN**2])
        .loss(PDEResidualLoss(weight=CONFIG['weight_pde'], normalize=True))
        .loss(NormalizationLoss(weight=CONFIG['weight_norm']))
        .loss(NonTrivialLoss(weight=CONFIG['weight_nontrivial']))
        .loss(DecayLoss(weight=CONFIG['weight_decay'], threshold_ratio=0.8))
        .loss(SmoothnessLoss(weight=CONFIG['weight_smooth']))
        .loss(AnchorLoss(x0=mx.array([[0.0]]), target=1.0/(np.pi*HBAR), weight=CONFIG['weight_anchor']))
        .loss(EigenvalueLoss(weight=CONFIG['weight_eig'], method="rayleigh", rayleigh_fn=wigner_rayleigh))
        .loss(RegularizationLoss(weight=CONFIG['weight_reg'], norm="l1"))
        .phase("initial", steps=CONFIG['phase1_steps'], lr=CONFIG['phase1_lr'],
               n_points=CONFIG['phase1_points'], log_freq=10000)
        .phase("refine", steps=CONFIG['phase2_steps'], lr=CONFIG['phase2_lr'],
               n_points=CONFIG['phase2_points'], grid_update_before=True, log_freq=10000)
        .phase("symbolic", steps=CONFIG['phase3_steps'], lr=CONFIG['phase3_lr'], log_freq=10000)
        .model(
            width=CONFIG['width'],
            grid=CONFIG['grid'],
            k=CONFIG['k'],
            basis="laguerre",
            basis_M=8,
            basis_kwargs={"alpha": 0.0, "weighted": True},
            grid_range=(0, 2 * DOMAIN**2),
            seed=seed,
        )
        .solve(verbose=False)
    )
    
    elapsed = time.time() - start_time
    mem_after = get_memory_mb()
    
    # Evaluate
    r2_test = mx.linspace(0, 2 * DOMAIN**2, 1000).reshape(-1, 1)
    W_kan = np.array(model(r2_test)).flatten()
    W_exact = analytic_wigner(np.array(r2_test).flatten())
    
    mse = np.mean((W_kan - W_exact)**2)
    l2_error = np.sqrt(mse) / (np.sqrt(np.mean(W_exact**2)) + 1e-10)
    
    result = {
        'time': elapsed,
        'memory': mem_after - mem_before,
        'mse': mse,
        'l2_error': l2_error,
        'final_loss': history.final_loss,
    }
    mlx_results.append(result)
    
    print(f"  Time: {elapsed:.2f}s, MSE: {mse:.2e}, L2: {l2_error*100:.2f}%")
    
    # Cleanup
    del model
    gc.collect()

# ============================================================================
# Statistical Analysis
# ============================================================================

print("\n" + "=" * 80)
print("STATISTICAL ANALYSIS")
print("=" * 80)

def compute_stats(results, key):
    values = [r[key] for r in results]
    return {
        'mean': np.mean(values),
        'std': np.std(values),
        'min': np.min(values),
        'max': np.max(values),
    }

print(f"\nResults from {N_TRIALS} trials:\n")

print("TRAINING TIME (seconds):")
pykan_time = compute_stats(pykan_results, 'time')
mlx_time = compute_stats(mlx_results, 'time')
print(f"  PyKAN:  {pykan_time['mean']:.2f} ± {pykan_time['std']:.2f}  (range: {pykan_time['min']:.2f}-{pykan_time['max']:.2f})")
print(f"  MLX:    {mlx_time['mean']:.2f} ± {mlx_time['std']:.2f}  (range: {mlx_time['min']:.2f}-{mlx_time['max']:.2f})")
print(f"  Speedup: {pykan_time['mean']/mlx_time['mean']:.2f}x {'faster' if pykan_time['mean'] < mlx_time['mean'] else 'slower'} (PyKAN)")

print("\nMEMORY USAGE (MB):")
pykan_mem = compute_stats(pykan_results, 'memory')
mlx_mem = compute_stats(mlx_results, 'memory')
print(f"  PyKAN:  {pykan_mem['mean']:.1f} ± {pykan_mem['std']:.1f}  (range: {pykan_mem['min']:.1f}-{pykan_mem['max']:.1f})")
print(f"  MLX:    {mlx_mem['mean']:.1f} ± {mlx_mem['std']:.1f}  (range: {mlx_mem['min']:.1f}-{mlx_mem['max']:.1f})")
print(f"  Difference: {pykan_mem['mean'] - mlx_mem['mean']:.1f} MB {'more' if pykan_mem['mean'] > mlx_mem['mean'] else 'less'} (PyKAN)")

print("\nACCURACY - MSE:")
pykan_mse = compute_stats(pykan_results, 'mse')
mlx_mse = compute_stats(mlx_results, 'mse')
print(f"  PyKAN:  {pykan_mse['mean']:.2e} ± {pykan_mse['std']:.2e}")
print(f"  MLX:    {mlx_mse['mean']:.2e} ± {mlx_mse['std']:.2e}")

print("\nACCURACY - L2 Error (%):")
pykan_l2 = compute_stats(pykan_results, 'l2_error')
mlx_l2 = compute_stats(mlx_results, 'l2_error')
print(f"  PyKAN:  {pykan_l2['mean']*100:.2f} ± {pykan_l2['std']*100:.2f}  (range: {pykan_l2['min']*100:.2f}-{pykan_l2['max']*100:.2f})")
print(f"  MLX:    {mlx_l2['mean']*100:.2f} ± {mlx_l2['std']*100:.2f}  (range: {mlx_l2['min']*100:.2f}-{mlx_l2['max']*100:.2f})")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

speedup = pykan_time['mean'] / mlx_time['mean']
mem_saving = pykan_mem['mean'] - mlx_mem['mean']
mse_ratio = pykan_mse['mean'] / mlx_mse['mean']

print(f"\nBased on {N_TRIALS} trials with different random seeds:")
print(f"  • PyKAN is {speedup:.2f}x {'faster' if speedup > 1 else 'slower'} than MLX")
print(f"  • MLX uses {abs(mem_saving):.1f} MB {'less' if mem_saving > 0 else 'more'} memory")
print(f"  • PyKAN has {mse_ratio:.0f}x {'better' if mse_ratio < 1 else 'worse'} MSE accuracy")
print(f"\nBoth methods show variability due to:")
print(f"  • Different random initializations (PyTorch vs MLX RNG)")
print(f"  • Stochastic training dynamics")
print(f"  • Float32 numerical precision in finite differences")

