#!/usr/bin/env python3
"""Figure 3 — Why inversion-based dip extraction fails.

Reads STAR_JAG_benchmark_results.csv and produces a two-panel grouped bar
chart comparing isotropic L2, adaptive anisotropic (STAR), OOD-gated
anisotropic (STAR_OOD), and oracle true-dip (FixedTrueDip) inversions on
(a) target-region model RMSE and (b) model-derived dip error.

Output: figures/Fig3_oracle_dip_negative.png (300 dpi) and .pdf
"""
import csv
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CSV = "STAR_JAG_Benchmark/STAR_JAG_benchmark_results.csv"
OUTDIR = "figures"
os.makedirs(OUTDIR, exist_ok=True)

# Method order and display labels
methods = ["L2", "STAR", "STAR_OOD", "FixedTrueDip"]
mlabels = ["Isotropic L2", "Adaptive (STAR)", "OOD-gated", "Oracle (true dip)"]
# case order and display labels
cases = ["clean25", "covered25", "fault35"]
clabels = ["Clean 25°", "Covered 25°", "Fault 35°"]

# colors (grayscale-friendly, colorblind-safe)
colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B3"]

# read CSV
rmse = {c: {} for c in cases}
diperr = {c: {} for c in cases}
with open(CSV, newline="", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    for row in reader:
        c = row["case"].strip()
        m = row["method"].strip()
        if c in cases and m in methods:
            rmse[c][m] = float(row["target_rmse"])
            diperr[c][m] = float(row["dip_error"])

fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
x = np.arange(len(cases))
width = 0.2

# Panel (a): target-region model RMSE
ax = axes[0]
for i, m in enumerate(methods):
    vals = [rmse[c][m] for c in cases]
    ax.bar(x + (i - 1.5) * width, vals, width, label=mlabels[i],
           color=colors[i], edgecolor="black", linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels(clabels)
ax.set_ylabel("Target-region model RMSE (log₁₀ Ω·m)")
ax.set_title("(a) Structural model recovery")
ax.grid(axis="y", linestyle=":", alpha=0.5)
ax.set_axisbelow(True)

# Panel (b): model-derived dip error
ax = axes[1]
for i, m in enumerate(methods):
    vals = [diperr[c][m] for c in cases]
    ax.bar(x + (i - 1.5) * width, vals, width, label=mlabels[i],
           color=colors[i], edgecolor="black", linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels(clabels)
ax.set_ylabel("Model-derived dip error (°)")
ax.set_title("(b) Dip recovered from inverted model")
ax.grid(axis="y", linestyle=":", alpha=0.5)
ax.set_axisbelow(True)
ax.legend(loc="upper left", frameon=True, fontsize=9)

fig.suptitle("Supplying the exact dip does not improve structural recovery",
             fontsize=12, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.96])

png = os.path.join(OUTDIR, "Fig3_oracle_dip_negative.png")
pdf = os.path.join(OUTDIR, "Fig3_oracle_dip_negative.pdf")
fig.savefig(png, dpi=300, bbox_inches="tight")
fig.savefig(pdf, bbox_inches="tight")
fig.savefig(os.path.join(OUTDIR, "Fig3_oracle_dip_negative.eps"), bbox_inches="tight")
print("saved", png)

# also print the numbers for the caption / sanity check
print("\n--- values used ---")
for c in cases:
    for m in methods:
        print(f"{c:10s} {m:13s} RMSE={rmse[c][m]:.3f}  dip_err={diperr[c][m]:.1f}")
