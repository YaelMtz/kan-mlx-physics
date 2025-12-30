"""Basic usage example for KAN-MLX-Physics.

This example demonstrates:
1. Creating a synthetic dataset
2. Building a MultKAN model
3. Training the model
4. Evaluating and visualizing results
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
