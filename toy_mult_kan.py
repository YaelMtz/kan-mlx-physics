"""Toy training of [[1,0],[1,0],[0,2],[1,0]] architecture to visualize it."""

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import numpy as np
from kan_mlx_physics import MultKAN, create_dataset


def target_fn(x):
    # Simple 1D function: x * sin(pi*x)
    return x[:, 0] * np.sin(np.pi * x[:, 0])


# Dataset
dataset = create_dataset(
    f=target_fn,
    n_var=1,
    ranges=(-1, 1),
    train_num=500,
    test_num=100,
    seed=0,
)

# Architecture: [[1,0],[1,0],[0,2],[1,0]]
# Layer 0: 1 sum node (input)
# Layer 1: 1 sum node
# Layer 2: 0 sum + 2 mult nodes
# Layer 3: 1 sum node (output)
model = MultKAN(
    width=[[1, 0], [1, 0], [0, 2], [1, 0]],
    grid=5,
    k=3,
    mult_arity=2,
    seed=42,
)

print(model.summary())

history = model.fit(
    dataset,
    opt="Adam",
    steps=100,
    lr=0.01,
    lamb=0.001,
    batch_size=-1,
    log_freq=10,
    verbose=True,
)

print(f"\nFinal train loss: {history['train_loss'][-1]:.6f}")
print(f"Final test  loss: {history['test_loss'][-1]:.6f}")

print("\nSaving visualization to ./figures_toy_mult/")
model.plot(
    folder="./figures_toy_mult",
    in_vars=["x"],
    out_vars=["y"],
    title="[[1,0],[1,0],[0,2],[1,0]] — input at bottom, output at top",
    display=False,
    save=True,
)
print("Done!")
