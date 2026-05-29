#!/usr/bin/env python3
"""
Independent forward validation for the geological hypothesis matcher.

This script checks the inverse-crime risk raised during manuscript review.
The candidate library is still the existing FDM-generated template cache, but
the synthetic "observed" data are generated in two ways:

1. FDM: same forward class as the template library (baseline / inverse-crime
   susceptible).
2. FEM: independent triangular-mesh finite-element forward solver with a
   different mesh and discretization.

If the method only works because of self-matching, the FEM-generated cases
should degrade strongly relative to the FDM-generated cases. If the structural
response pattern is genuinely captured, dip errors should remain comparable.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

from RESIS_Pro import DipDipSurvey, Mesh2D, TriMesh, ForwardSolver, ForwardFEM
from geo_library import (
    _build_clean_dip,
    _build_covered_dip,
    _build_groundwater,
    _build_fault_zone,
    _build_basement,
)
from forward_matcher import load_cache, match_observed


ROOT = Path(__file__).resolve().parent
OUTDIR = ROOT / "IndependentForwardValidation"
OUTDIR.mkdir(parents=True, exist_ok=True)

A = 5.0
N_ELEC = 30
N_MAX = 6
NOISE_LEVELS = [0.01, 0.03, 0.05, 0.07, 0.10]

EXPECTED = {
    "Clean dip 28": "clean_dip",
    "Covered dip 32": "covered_dip",
    "Groundwater 18": "groundwater",
    "Fault zone 40": "fault_zone",
    "Basement 18": "basement",
}


def make_survey_meshes():
    elec_x = np.arange(N_ELEC) * A
    survey = DipDipSurvey(
        a=A,
        n_electrodes=N_ELEC,
        n_max=N_MAX,
        electrode_x=elec_x,
        array_type="dipole-dipole",
    )
    # FDM mesh matches the template-building side.
    fdm_mesh = Mesh2D(survey, depth_factor=2.5, dx_factor=0.25)
    # FEM mesh deliberately differs from FDM: triangular and coarser.
    fem_mesh = TriMesh(survey, depth_factor=2.5, dx_factor=0.50)
    return survey, fdm_mesh, fem_mesh


def independent_cases(mesh):
    """Same independent-geometry cases used in the v5 validation discussion."""
    return [
        {
            "name": "Clean dip 28",
            "true_dip": 28.0,
            "rho": _build_clean_dip(
                mesh, 28, x0=48, z0=1.0, thick=7, rho_layer=25, rho_bg=180
            ),
        },
        {
            "name": "Covered dip 32",
            "true_dip": 32.0,
            "rho": _build_covered_dip(
                mesh,
                32,
                x0=42,
                z0=5.5,
                thick=5,
                cover_thick=4.0,
                rho_layer=18,
                rho_cover=55,
                rho_bg=180,
            ),
        },
        {
            "name": "Groundwater 18",
            "true_dip": 18.0,
            "rho": _build_groundwater(
                mesh, 18, x0=38, z0=2.5, width=4, rho_gw=8, rho_bg=350
            ),
        },
        {
            "name": "Fault zone 40",
            "true_dip": 40.0,
            "rho": _build_fault_zone(
                mesh, 40, x0=52, z0=0.5, thick=10, rho_fault=18, rho_bg=220
            ),
        },
        {
            "name": "Basement 18",
            "true_dip": 18.0,
            "rho": _build_basement(
                mesh, 18, x0=35, z0=3.0, rho_above=100, rho_below=600
            ),
        },
    ]


def map_fdm_cells_to_fem_elements(fdm_mesh, fem_mesh, rho_fdm):
    """Nearest-cell transfer from FDM cell centers to FEM element centers."""
    xg, zg = np.meshgrid(fdm_mesh.x_cc, fdm_mesh.z_cc)
    fdm_pts = np.column_stack([xg.ravel(), zg.ravel()])
    fem_depth = fem_mesh.max_elev - fem_mesh.elem_centers[:, 1]
    fem_pts = np.column_stack([fem_mesh.elem_centers[:, 0], fem_depth])
    idx = cKDTree(fdm_pts).query(fem_pts, k=1)[1]
    return np.asarray(rho_fdm, dtype=float)[idx]


def add_noise(rho_a, noise, seed):
    rng = np.random.default_rng(seed)
    return np.maximum(rho_a * (1.0 + noise * rng.standard_normal(len(rho_a))), 1.0)


def forward_fdm(survey, fdm_mesh, rho_fdm):
    return ForwardSolver(fdm_mesh, rho_fdm).compute_data(survey, callback=lambda *_: None)


def forward_fem(survey, fdm_mesh, fem_mesh, rho_fdm):
    rho_fem = map_fdm_cells_to_fem_elements(fdm_mesh, fem_mesh, rho_fdm)
    return ForwardFEM(fem_mesh, rho_fem).compute_data(survey, callback=lambda *_: None)


def evaluate_response(case, source, base_rhoa, cache):
    rows = []
    for noise in NOISE_LEVELS:
        rho_a = add_noise(base_rhoa, noise, seed=int(1000 * noise) + int(case["true_dip"]))
        match = match_observed(rho_a, cache=cache, top_n=10)
        top1 = match["top1"]
        est = top1["dip_deg"]
        err = None if est is None else abs(float(est) - float(case["true_dip"]))
        rows.append({
            "case": case["name"],
            "source_solver": source,
            "noise_pct": noise * 100.0,
            "true_dip": case["true_dip"],
            "estimated_dip": "" if est is None else float(est),
            "abs_error": "" if err is None else float(err),
            "top_family": top1["family"],
            "expected_family": EXPECTED[case["name"]],
            "family_correct": top1["family"] == EXPECTED[case["name"]],
            "top_template": top1["name"],
            "correlation": top1["correlation"],
            "score": top1["score"],
            "weight": top1["weight"],
            "n_eff": match["n_eff"],
        })
    return rows


def summarize(rows):
    out = {}
    for source in sorted(set(r["source_solver"] for r in rows)):
        rr = [r for r in rows if r["source_solver"] == source and r["abs_error"] != ""]
        errs = np.array([float(r["abs_error"]) for r in rr])
        fam = np.array([bool(r["family_correct"]) for r in rr])
        corrs = np.array([float(r["correlation"]) for r in rr])
        out[source] = {
            "mae": float(errs.mean()),
            "within5": float(np.mean(errs <= 5.0) * 100),
            "within10": float(np.mean(errs <= 10.0) * 100),
            "family_acc": float(fam.mean() * 100),
            "mean_corr": float(corrs.mean()),
        }
    return out


def write_csv(rows, path):
    fields = [
        "case", "source_solver", "noise_pct", "true_dip", "estimated_dip",
        "abs_error", "top_family", "expected_family", "family_correct",
        "top_template", "correlation", "score", "weight", "n_eff",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def plot_results(rows, summary):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    colors = {"FDM": "#1b9e77", "FEM": "#d95f02"}

    ax = axes[0, 0]
    for source in ("FDM", "FEM"):
        rr = [r for r in rows if r["source_solver"] == source and r["abs_error"] != ""]
        x = [float(r["true_dip"]) for r in rr]
        y = [float(r["estimated_dip"]) for r in rr]
        ax.scatter(x, y, s=58, alpha=0.78, edgecolor="k",
                   color=colors[source], label=source)
    ax.plot([0, 65], [0, 65], "k--", lw=1)
    ax.set_xlim(0, 65)
    ax.set_ylim(0, 80)
    ax.set_xlabel("True dip (deg)")
    ax.set_ylabel("Matched dip (deg)")
    ax.set_title("(a) Dip recovery")
    ax.grid(alpha=0.25)
    ax.legend()

    ax = axes[0, 1]
    labels = ["FDM", "FEM"]
    mae = [summary[k]["mae"] for k in labels]
    acc = [summary[k]["family_acc"] for k in labels]
    x = np.arange(len(labels))
    ax.bar(x - 0.18, mae, width=0.36, color="#7570b3", edgecolor="k", label="MAE")
    ax2 = ax.twinx()
    ax2.bar(x + 0.18, acc, width=0.36, color="#66a61e", edgecolor="k", label="Family acc.")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("MAE (deg)")
    ax2.set_ylabel("Family accuracy (%)")
    ax.set_title("(b) Solver dependence")
    ax.grid(alpha=0.25, axis="y")

    ax = axes[1, 0]
    for source in ("FDM", "FEM"):
        xs, ys = [], []
        for noise in NOISE_LEVELS:
            rr = [
                r for r in rows
                if r["source_solver"] == source
                and abs(float(r["noise_pct"]) - noise * 100.0) < 1e-9
                and r["abs_error"] != ""
            ]
            xs.append(noise * 100.0)
            ys.append(np.mean([float(r["abs_error"]) for r in rr]))
        ax.plot(xs, ys, "o-", color=colors[source], label=source)
    ax.set_xlabel("Noise level (%)")
    ax.set_ylabel("MAE (deg)")
    ax.set_title("(c) Noise sensitivity")
    ax.grid(alpha=0.25)
    ax.legend()

    ax = axes[1, 1]
    for source in ("FDM", "FEM"):
        xs, ys = [], []
        for noise in NOISE_LEVELS:
            rr = [
                r for r in rows
                if r["source_solver"] == source
                and abs(float(r["noise_pct"]) - noise * 100.0) < 1e-9
            ]
            xs.append(noise * 100.0)
            ys.append(np.mean([float(r["correlation"]) for r in rr]))
        ax.plot(xs, ys, "s-", color=colors[source], label=source)
    ax.set_xlabel("Noise level (%)")
    ax.set_ylabel("Mean top correlation")
    ax.set_ylim(0, 1.05)
    ax.set_title("(d) Correlation stability")
    ax.grid(alpha=0.25)
    ax.legend()

    fig.suptitle("Independent forward validation: FDM templates vs FDM/FEM observations",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = OUTDIR / "independent_forward_validation.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return out


def write_note(summary, path):
    s_fdm = summary["FDM"]
    s_fem = summary["FEM"]
    text = f"""# Independent Forward Validation

Purpose: test inverse-crime sensitivity by matching the same FDM-generated
template library against synthetic observations generated by (1) the same FDM
solver and (2) the independent triangular FEM solver.

## Summary

| Observation solver | MAE (deg) | <=5 deg (%) | <=10 deg (%) | Family accuracy (%) | Mean top correlation |
|---|---:|---:|---:|---:|---:|
| FDM | {s_fdm['mae']:.2f} | {s_fdm['within5']:.1f} | {s_fdm['within10']:.1f} | {s_fdm['family_acc']:.1f} | {s_fdm['mean_corr']:.3f} |
| FEM | {s_fem['mae']:.2f} | {s_fem['within5']:.1f} | {s_fem['within10']:.1f} | {s_fem['family_acc']:.1f} | {s_fem['mean_corr']:.3f} |

## Interpretation

If FEM performance remains close to FDM performance, the method is not solely
explained by exact FDM self-matching. If FEM performance collapses, the current
benchmark is dominated by inverse crime and the manuscript claim should be
reduced or redesigned.
"""
    path.write_text(text, encoding="utf-8")


def main():
    print("=" * 72)
    print("Independent forward validation")
    print("=" * 72)
    print("Loading FDM template cache...")
    cache = load_cache()
    print(f"Templates: {len(cache)}")

    survey, fdm_mesh, fem_mesh = make_survey_meshes()
    cases = independent_cases(fdm_mesh)
    rows = []

    for i, case in enumerate(cases, 1):
        print(f"[{i}/{len(cases)}] {case['name']}")
        base_fdm = forward_fdm(survey, fdm_mesh, case["rho"])
        base_fem = forward_fem(survey, fdm_mesh, fem_mesh, case["rho"])
        rows.extend(evaluate_response(case, "FDM", base_fdm, cache))
        rows.extend(evaluate_response(case, "FEM", base_fem, cache))

    csv_path = OUTDIR / "independent_forward_validation.csv"
    write_csv(rows, csv_path)
    summary = summarize(rows)
    fig_path = plot_results(rows, summary)
    note_path = OUTDIR / "independent_forward_validation_summary.md"
    write_note(summary, note_path)

    print("")
    for source, vals in summary.items():
        print(
            f"{source}: MAE={vals['mae']:.2f} deg, "
            f"<=5={vals['within5']:.1f}%, <=10={vals['within10']:.1f}%, "
            f"family={vals['family_acc']:.1f}%, corr={vals['mean_corr']:.3f}"
        )
    print(f"CSV: {csv_path}")
    print(f"Figure: {fig_path}")
    print(f"Note: {note_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
