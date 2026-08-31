"""Basic usage example for KAN-MLX-Physics.

This example demonstrates:
1. Creating a synthetic dataset
2. Building a MultKAN model
3. Training the model
4. Evaluating and visualizing results

Expected Output:
===============
Creating dataset...
Train samples: 1000
Test samples: 200

Creating MultKAN model...
MultKAN Summary
========================================
Width: [2, 5, 1]
Depth: 2
Grid: 5
Spline order: 3
Speed mode: False
Versions saved: 0

Layers:
  [0] KANLayer: 2 → 5 (100 params)
  [1] KANLayer: 5 → 1 (50 params)

Total parameters: 150

Training...
Step    0 | Train: 0.794273 | Test: 0.899059 | Reg: 0.009918
Step   10 | Train: 0.113492 | Test: 0.117964 | Reg: 0.010431
Step   20 | Train: 0.034868 | Test: 0.034231 | Reg: 0.010094
Step   30 | Train: 0.014045 | Test: 0.015987 | Reg: 0.010026
Step   40 | Train: 0.005436 | Test: 0.005811 | Reg: 0.010075
Step   50 | Train: 0.001775 | Test: 0.001776 | Reg: 0.010006
Step   60 | Train: 0.001164 | Test: 0.000945 | Reg: 0.009942
Step   70 | Train: 0.000700 | Test: 0.000662 | Reg: 0.009937
Step   80 | Train: 0.000491 | Test: 0.000473 | Reg: 0.009910
Step   90 | Train: 0.000380 | Test: 0.000366 | Reg: 0.009886
Step   99 | Train: 0.000317 | Test: 0.000309 | Reg: 0.009870

Final Results:
Final train loss: 0.000317
Final test loss: 0.000309

Pruning edges...
Saving visualization to ./figures/
Saving model checkpoint...
Done!

Performance Notes:
- Training converges rapidly to ~3e-4 loss within 100 steps
- Grid updates occur at steps 0, 20, 40 to improve spline resolution
- Model learns f(x,y) = sin(πx) + y² with high accuracy
- Training time: ~1-2 seconds on Apple Silicon (M1/M2/M3)
- Final test loss close to train loss indicates good generalization
"""

import numpy as np
from kan_mlx_physics import MultKAN, create_dataset


def main():
    # Define a target function: f(x, y) = sin(pi * x) + y^2
    def target_fn(x):
        return np.sin(np.pi * x[:, 0]) + x[:, 1] ** 2

    # Create dataset
    print("Creating dataset...")
    dataset = create_dataset(
        f=target_fn,
        n_var=2,
        ranges=(-1, 1),
        train_num=1000,
        test_num=200,
        seed=42,
    )

    print(f"Train samples: {dataset['train_input'].shape[0]}")
    print(f"Test samples: {dataset['test_input'].shape[0]}")

    # Create KAN model
    # Architecture: 2 inputs -> 5 hidden -> 1 output
    print("\nCreating MultKAN model...")
    model = MultKAN(
        width=[2, 5, 1],
        grid=5,
        k=3,
        seed=42,
    )

    print(model.summary())

    # Train the model
    print("\nTraining...")
    history = model.fit(
        dataset,
        opt="Adam",
        steps=100,
        lr=0.01,
        lamb=0.001,  # L1 regularization
        batch_size=-1,  # Full batch
        update_grid=True,
        grid_update_freq=20,
        stop_grid_update_step=50,
        log_freq=10,
        verbose=True,
    )

    # Final evaluation
    print("\nFinal Results:")
    print(f"Final train loss: {history['train_loss'][-1]:.6f}")
    print(f"Final test loss: {history['test_loss'][-1]:.6f}")

    # Prune edges with low importance
    print("\nPruning edges...")
    model.prune_edges(threshold=0.01)

    # Visualize the network
    print("\nSaving visualization to ./figures/")
    model.plot(folder="./figures")

    # Save checkpoint
    print("\nSaving model checkpoint...")
    model.saveckpt("./checkpoints/model")

    print("\nDone!")


if __name__ == "__main__":
    main()
