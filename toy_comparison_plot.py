#!/usr/bin/env python3
"""Render the toy-comparison figure from toy_comparison_results.json.

Panels:
  1. Convergence curves (loss vs step) overlaid for all 4 contenders.
  2. Accuracy bars (L2 error %).
  3. Speed bars (steps/sec).
  4. Fit vs analytic sin(πx) for each contender.
Plus a printed symbolic-recovery table.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "toy_comparison_results.json")
OUT = os.path.join(HERE, "toy_comparison.png")

with open(RESULTS) as f:
    results = json.load(f)

xs = np.linspace(-1, 1, 200)
analytic = np.sin(np.pi * xs)
colors = ["#2563eb", "#16a34a", "#db2777", "#ea580c"]

fig = plt.figure(figsize=(15, 8))

# 1. Convergence curves
ax1 = fig.add_subplot(2, 2, 1)
for r, c in zip(results, colors):
    steps = np.arange(len(r["losses"])) * 10
    ax1.semilogy(steps, r["losses"], label=r["label"], color=c, lw=1.8)
ax1.set(title="Convergence (loss vs step)", xlabel="step", ylabel="MSE loss")
ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)

# 2. Accuracy bars
ax2 = fig.add_subplot(2, 2, 2)
labels = [r["label"].replace(" (compiled)", "\n(compiled)") for r in results]
l2 = [r["l2"] * 100 for r in results]
bars = ax2.bar(range(len(results)), l2, color=colors)
ax2.set(title="Accuracy — L2 error vs analytic (%)", ylabel="L2 error %")
ax2.set_xticks(range(len(results))); ax2.set_xticklabels(labels, fontsize=8)
for b, v in zip(bars, l2):
    ax2.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}%", ha="center",
             va="bottom", fontsize=9)

# 3. Speed bars
ax3 = fig.add_subplot(2, 2, 3)
sps = [r["steps_per_sec"] for r in results]
bars = ax3.bar(range(len(results)), sps, color=colors)
ax3.set(title="Speed — steps/sec (warmed up)", ylabel="steps / sec")
ax3.set_xticks(range(len(results))); ax3.set_xticklabels(labels, fontsize=8)
for b, v in zip(bars, sps):
    ax3.text(b.get_x() + b.get_width() / 2, v, f"{v:.0f}", ha="center",
             va="bottom", fontsize=9)

# 4. Fit vs analytic
ax4 = fig.add_subplot(2, 2, 4)
ax4.plot(xs, analytic, "k-", lw=2.5, label="analytic sin(πx)", alpha=0.7)
for r, c in zip(results, colors):
    ax4.plot(xs, r["y_pred"], "--", color=c, lw=1.3, label=r["label"])
ax4.set(title="Fit vs analytic", xlabel="x", ylabel="f(x)")
ax4.legend(fontsize=8); ax4.grid(True, alpha=0.3)

fig.suptitle("KAN toy comparison — fit f(x)=sin(πx)  [arch 1,1,1 · grid 5 · 1200 steps]",
             fontweight="bold", fontsize=13)
fig.tight_layout()
fig.savefig(OUT, dpi=140)
print(f"Saved figure: {OUT}")

# Symbolic table
print("\nSymbolic recovery:")
for r in results:
    print(f"  {r['label']:26} {r['symbolic']}")
