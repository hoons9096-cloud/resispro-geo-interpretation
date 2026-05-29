#!/usr/bin/env python3
"""Figures 1, 2, 4, 5, 7. Reads geo_library and the result CSVs."""
import csv
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.colors import LogNorm
from matplotlib import cm

sys.path.insert(0, "")
import geo_library as gl

OUT = "figures"
os.makedirs(OUT, exist_ok=True)
RP = ""

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "figure.dpi": 110,
    "savefig.bbox": "tight",
})

RHO_CMAP = "Spectral_r"


class FigMesh:
    """Lightweight mesh compatible with geo_library builders."""
    def __init__(self, x0, x1, nx, z0, z1, nz):
        self.ncx = nx
        self.ncz = nz
        self.n_cells = nx * nz
        self.x_cc = np.linspace(x0, x1, nx)
        self.z_cc = np.linspace(z0, z1, nz)
        self.extent = [x0, x1, z1, z0]


def save(fig, name, eps=False):
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=300)
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    if eps:
        fig.savefig(os.path.join(OUT, name + ".eps"))
    plt.close(fig)
    print("saved", name)


#  FIG 1 — Workflow diagram
def fig1():
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

    boxes = {}

    def box(key, x, y, w, h, text, fc, ec, fs=11):
        b = FancyBboxPatch((x - w/2, y - h/2), w, h,
                           boxstyle="round,pad=0.6,rounding_size=2.2",
                           linewidth=1.8, edgecolor=ec, facecolor=fc, zorder=2)
        ax.add_patch(b)
        ax.text(x, y, text, ha="center", va="center", fontsize=fs,
                fontweight="bold", color="black", zorder=3)
        boxes[key] = dict(x=x, y=y, w=w, h=h)

    def edge(key, side):
        """Return a point on a box edge: 'top','bottom','left','right'."""
        b = boxes[key]
        if side == "top":
            return (b["x"], b["y"] + b["h"]/2)
        if side == "bottom":
            return (b["x"], b["y"] - b["h"]/2)
        if side == "left":
            return (b["x"] - b["w"]/2, b["y"])
        if side == "right":
            return (b["x"] + b["w"]/2, b["y"])

    def arrow(p1, p2, color="#555555"):
        a = FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=22,
                            linewidth=2.0, color=color, zorder=1,
                            shrinkA=3, shrinkB=3)
        ax.add_patch(a)

    # palette
    c_in = "#dbeafe"; e_in = "#2563eb"
    c_diag = "#dcfce7"; e_diag = "#16a34a"
    c_fwd = "#fef9c3"; e_fwd = "#ca8a04"
    c_match = "#fce7f3"; e_match = "#db2777"
    c_gate = "#ede9fe"; e_gate = "#7c3aed"
    c_out = "#ffedd5"; e_out = "#ea580c"

    # boxes
    box("in", 50, 90, 52, 8, "Observed ERT data + survey geometry", c_in, e_in, 12)
    box("diag", 25, 70, 42, 13,
        "Pseudosection diagnostics\n(M1 structure tensor · M2 centroid\nM3 aspect ratio · M4 multi-scale · M5 contour)",
        c_diag, e_diag, 9.5)
    box("fwd", 75, 70, 42, 13,
        "Forward-modeled candidate library\n(9 geological families, 165 templates,\n2.5-D response at same geometry)",
        c_fwd, e_fwd, 9.5)
    box("match", 50, 48, 58, 11,
        "Match & rank in normalized log-ρₐ space\nS = corr − λ·NRMSE  →  softmax weights",
        c_match, e_match, 10.5)
    box("gate", 50, 29, 52, 10,
        "Out-of-distribution confidence gate\n(continuous isotropic ↔ anisotropic trust)",
        c_gate, e_gate, 10.5)
    box("out", 50, 9, 66, 11,
        "Outputs: integrated dip estimate · ranked geological\nhypotheses · support / effective-N · uncertainty & OOD flags\n→ overlay on conventional inversion section",
        c_out, e_out, 9.5)

    # connecting arrows
    arrow((43, edge("in", "bottom")[1]), edge("diag", "top"))
    arrow((57, edge("in", "bottom")[1]), edge("fwd", "top"))
    arrow(edge("diag", "bottom"), (43, edge("match", "top")[1]))
    arrow(edge("fwd", "bottom"), (57, edge("match", "top")[1]))
    arrow(edge("match", "bottom"), edge("gate", "top"))
    arrow(edge("gate", "bottom"), edge("out", "top"))

    ax.set_title("Forward-hypothesis matching workflow",
                 fontsize=14, fontweight="bold", pad=12)
    save(fig, "Fig1_workflow", eps=True)


#  FIG 2 — Non-uniqueness: electrically similar distinct structures
def _find_template(reg, family, pick=0):
    matches = [t for t in reg if t["family"] == family]
    return matches[pick] if matches else None


def fig2():
    reg = gl.build_template_registry()
    mesh = FigMesh(0, 145, 220, 0, 28, 110)
    picks = [
        ("covered_dip", "Covered dipping layer", 4),
        ("groundwater", "Dipping groundwater pathway", 7),
        ("fault_zone", "Conductive fault zone", 2),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    vmin, vmax = 8, 300
    im = None
    for ax, (fam, label, pk) in zip(axes, picks):
        t = _find_template(reg, fam, pk)
        rho = t["builder"](mesh).reshape(mesh.ncz, mesh.ncx)
        im = ax.imshow(rho, extent=mesh.extent, aspect="auto",
                       cmap=RHO_CMAP, norm=LogNorm(vmin=vmin, vmax=vmax))
        ax.set_title(label)
        ax.set_xlabel("Distance (m)")
        ax.set_ylabel("Depth (m)")
    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("Resistivity (Ω·m)")
    fig.suptitle("Distinct geological structures in the candidate library",
                 fontsize=13, fontweight="bold")
    save(fig, "Fig2_nonuniqueness")


#  FIG 4 — Candidate library gallery (9 families)
def fig4():
    reg = gl.build_template_registry()
    mesh = FigMesh(0, 145, 200, 0, 28, 100)
    order = [
        ("clean_dip", "Clean dipping layer", 6),
        ("covered_dip", "Covered dipping layer", 4),
        ("groundwater", "Dipping groundwater", 7),
        ("fault_zone", "Conductive fault zone", 2),
        ("basement", "Dipping basement", 2),
        ("vertical_block", "Vertical block", 2),
        ("lens", "Groundwater lens", 3),
        ("channel", "Buried channel", 3),
        ("composite", "Composite GW–fault", 2),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(14, 9))
    vmin, vmax = 8, 800
    im = None
    for ax, (fam, label, pk) in zip(axes.ravel(), order):
        t = _find_template(reg, fam, pk)
        rho = t["builder"](mesh).reshape(mesh.ncz, mesh.ncx)
        im = ax.imshow(rho, extent=mesh.extent, aspect="auto",
                       cmap=RHO_CMAP, norm=LogNorm(vmin=vmin, vmax=vmax))
        ax.set_title(label, fontsize=11)
        ax.set_xticks([0, 50, 100])
        if ax in axes[:, 0]:
            ax.set_ylabel("Depth (m)")
        if ax in axes[-1, :]:
            ax.set_xlabel("Distance (m)")
    cbar = fig.colorbar(im, ax=axes, fraction=0.018, pad=0.02)
    cbar.set_label("Resistivity (Ω·m)")
    fig.suptitle("Geological candidate library: representative templates per family",
                 fontsize=14, fontweight="bold")
    save(fig, "Fig4_library_gallery")


#  FIG 5 — Dip recovery scatter (matched vs true)
def fig5():
    csv_path = os.path.join(RP, "IndependentForwardValidation",
                            "independent_forward_validation.csv")
    rows = []
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            rows.append(r)

    fig, ax = plt.subplots(figsize=(7.2, 7))
    lim = (5, 45)
    # tolerance bands
    xs = np.array(lim)
    ax.fill_between(xs, xs - 10, xs + 10, color="#bfdbfe", alpha=0.35,
                    label="±10° band", zorder=0)
    ax.fill_between(xs, xs - 5, xs + 5, color="#93c5fd", alpha=0.45,
                    label="±5° band", zorder=0)
    ax.plot(lim, lim, "k--", linewidth=1.6, label="1:1", zorder=1)

    styles = {"FDM": dict(marker="o", color="#2563eb", label="FDM data → FDM templates"),
              "FEM": dict(marker="^", color="#dc2626", label="FEM data → FDM templates")}
    seen = set()
    rng = np.random.default_rng(0)
    for r in rows:
        solver = r["source_solver"].strip()
        td = float(r["true_dip"]); ed = float(r["estimated_dip"])
        # tiny jitter to reveal overlapping points
        jx = rng.uniform(-0.5, 0.5); jy = rng.uniform(-0.5, 0.5)
        st = styles[solver]
        lab = st["label"] if solver not in seen else None
        seen.add(solver)
        ax.scatter(td + jx, ed + jy, marker=st["marker"], s=70,
                   facecolor=st["color"], edgecolor="black", linewidth=0.5,
                   alpha=0.75, label=lab, zorder=3)

    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("True dip (°)")
    ax.set_ylabel("Matched dip (°)")
    ax.set_title("Forward-hypothesis dip recovery under independent forward solvers")
    ax.set_aspect("equal")
    ax.grid(linestyle=":", alpha=0.5)
    ax.legend(loc="lower right", framealpha=0.95)
    save(fig, "Fig5_dip_scatter", eps=True)


#  FIG 7 — Template incompleteness + OOD confidence gate
def fig7():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    # (a) template incompleteness
    ax = axes[0]
    labels = ["Full library", "Correct family\nremoved"]
    vals = [0.984, 0.800]
    bars = ax.bar(labels, vals, color=["#16a34a", "#dc2626"],
                  edgecolor="black", linewidth=0.8, width=0.55)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v + 0.01, f"{v:.3f}",
                ha="center", fontweight="bold")
    ax.axhline(0.85, color="gray", linestyle="--", linewidth=1.2)
    ax.text(1.45, 0.855, "interpretive\nthreshold", fontsize=8, color="gray",
            ha="right", va="bottom")
    ax.set_ylim(0.6, 1.02)
    ax.set_ylabel("Mean best-template correlation")
    ax.set_title("(a) Template-incompleteness is detectable")

    # (b) OOD confidence gate schematic
    ax = axes[1]
    conf = np.linspace(0, 1, 200)
    wmax = 1.0  # anisotropy strength fraction
    gate = conf ** 1.5   # smooth gate
    aniso = gate * wmax
    ax.plot(conf, aniso, color="#7c3aed", linewidth=3, label="anisotropic trust")
    ax.fill_between(conf, 0, aniso, color="#ede9fe", alpha=0.6)
    ax.fill_between(conf, aniso, 1, color="#e0f2fe", alpha=0.6)
    ax.text(0.18, 0.82, "isotropic\n(conservative)", fontsize=10, color="#0369a1",
            ha="center", fontweight="bold")
    ax.text(0.8, 0.18, "structure-\noriented", fontsize=10, color="#6d28d9",
            ha="center", fontweight="bold")
    ax.axvline(0.4, color="gray", linestyle=":", linewidth=1.2)
    ax.text(0.4, 1.02, "OOD →", fontsize=8, color="gray", ha="center")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Dip-estimate confidence (coherence × in-distribution)")
    ax.set_ylabel("Applied anisotropy strength")
    ax.set_title("(b) Out-of-distribution confidence gate")

    fig.suptitle("Uncertainty awareness: library incompleteness and confidence gating",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save(fig, "Fig7_uncertainty", eps=True)


if __name__ == "__main__":
    fig1()
    fig2()
    fig4()
    fig5()
    fig7()
    print("\nAll figures written to", OUT)
