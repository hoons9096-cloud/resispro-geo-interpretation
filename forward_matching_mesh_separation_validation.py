#!/usr/bin/env python3
"""
Forward-matching mesh-separation validation.

This is a focused inverse-crime check for the forward-hypothesis matching
manuscript:

    template library: existing FDM cache generated on the standard mesh
                      (Mesh2D dx_factor=0.25)
    observed data:    independent known-dip models generated on either
                      the same standard mesh or a separated finer mesh
                      (Mesh2D dx_factor=0.125)

The geological parameters are also deliberately offset from the template grid,
following the v5 independent-geometry validation. The main question is whether
the reported dip MAE survives when the synthetic observations are not generated
on the exact same model cells as the cached templates.
"""
from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np

ROOT = Path("")
os.environ.setdefault("RESIS_HEADLESS", "1")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from RESIS_Pro import DipDipSurvey, Mesh2D, ForwardSolver
from geo_library import (
    _build_clean_dip,
    _build_covered_dip,
    _build_groundwater,
    _build_fault_zone,
    _build_basement,
)
from forward_matcher import load_cache, match_observed


OUTDIR = ROOT / "ForwardMatchingMeshSeparated"
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


def make_survey():
    elec_x = np.arange(N_ELEC) * A
    return DipDipSurvey(
        a=A,
        n_electrodes=N_ELEC,
        n_max=N_MAX,
        electrode_x=elec_x,
        array_type="dipole-dipole",
    )


def make_mesh(survey, dx_factor):
    return Mesh2D(survey, depth_factor=2.5, dx_factor=dx_factor)


def independent_cases(mesh):
    """Independent geometries not exactly present in the template grid."""
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


def add_noise(rho_a, noise, seed):
    rng = np.random.default_rng(seed)
    return np.maximum(rho_a * (1.0 + noise * rng.standard_normal(len(rho_a))), 1.0)


def forward_fdm(survey, mesh, rho):
    return ForwardSolver(mesh, rho).compute_data(survey, callback=lambda *_: None)


def evaluate_response(case, source, base_rhoa, cache):
    rows = []
    for noise in NOISE_LEVELS:
        rho_a = add_noise(base_rhoa, noise, seed=int(1000 * noise) + int(case["true_dip"]))
        match = match_observed(rho_a, cache=cache, top_n=10)
        top1 = match["top1"]
        est = top1["dip_deg"]
        err = "" if est is None else abs(float(est) - float(case["true_dip"]))
        rows.append({
            "case": case["name"],
            "source_mesh": source,
            "noise_pct": noise * 100.0,
            "true_dip": case["true_dip"],
            "estimated_dip": "" if est is None else float(est),
            "abs_error": err,
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
    summary = {}
    for source in sorted(set(r["source_mesh"] for r in rows)):
        rr = [r for r in rows if r["source_mesh"] == source and r["abs_error"] != ""]
        errs = np.array([float(r["abs_error"]) for r in rr])
        fam = np.array([bool(r["family_correct"]) for r in rr])
        corr = np.array([float(r["correlation"]) for r in rr])
        summary[source] = {
            "mae": float(np.mean(errs)),
            "median_ae": float(np.median(errs)),
            "within5": float(np.mean(errs <= 5.0) * 100.0),
            "within10": float(np.mean(errs <= 10.0) * 100.0),
            "family_acc": float(np.mean(fam) * 100.0),
            "mean_corr": float(np.mean(corr)),
        }
    return summary


def summarize_by_case(rows):
    out = []
    for source in sorted(set(r["source_mesh"] for r in rows)):
        for case in sorted(set(r["case"] for r in rows)):
            rr = [r for r in rows if r["source_mesh"] == source and r["case"] == case]
            errs = [float(r["abs_error"]) for r in rr if r["abs_error"] != ""]
            if not errs:
                continue
            out.append({
                "source_mesh": source,
                "case": case,
                "true_dip": rr[0]["true_dip"],
                "mean_estimated_dip": float(np.mean([float(r["estimated_dip"]) for r in rr if r["estimated_dip"] != ""])),
                "mean_abs_error": float(np.mean(errs)),
                "top_families": ";".join(sorted(set(r["top_family"] for r in rr))),
            })
    return out


def write_csv(rows, path):
    fields = [
        "case", "source_mesh", "noise_pct", "true_dip", "estimated_dip",
        "abs_error", "top_family", "expected_family", "family_correct",
        "top_template", "correlation", "score", "weight", "n_eff",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_case_csv(rows, path):
    fields = ["source_mesh", "case", "true_dip", "mean_estimated_dip",
              "mean_abs_error", "top_families"]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summarize_by_case(rows))


def plot_results(rows, summary):
    colors = {"same_mesh_dx0.25": "#1b9e77", "separated_fine_mesh_dx0.125": "#d95f02"}
    labels = list(summary.keys())
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)

    ax = axes[0, 0]
    for source in labels:
        rr = [r for r in rows if r["source_mesh"] == source and r["abs_error"] != ""]
        ax.scatter([float(r["true_dip"]) for r in rr],
                   [float(r["estimated_dip"]) for r in rr],
                   s=58, alpha=0.78, edgecolor="k",
                   color=colors.get(source, "0.5"), label=source)
    ax.plot([0, 65], [0, 65], "k--", lw=1)
    ax.set_xlim(0, 65)
    ax.set_ylim(0, 65)
    ax.set_xlabel("True dip (deg)")
    ax.set_ylabel("Matched dip (deg)")
    ax.set_title("(a) Dip recovery")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    x = np.arange(len(labels))
    ax.bar(x - 0.18, [summary[k]["mae"] for k in labels], width=0.36,
           color="#7570b3", edgecolor="k", label="MAE")
    ax2 = ax.twinx()
    ax2.bar(x + 0.18, [summary[k]["family_acc"] for k in labels], width=0.36,
            color="#66a61e", edgecolor="k", label="Family acc.")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylabel("MAE (deg)")
    ax2.set_ylabel("Family accuracy (%)")
    ax.set_title("(b) Mesh dependence")
    ax.grid(alpha=0.25, axis="y")

    ax = axes[1, 0]
    for source in labels:
        xs, ys = [], []
        for noise in NOISE_LEVELS:
            rr = [
                r for r in rows
                if r["source_mesh"] == source
                and abs(float(r["noise_pct"]) - noise * 100.0) < 1e-9
                and r["abs_error"] != ""
            ]
            xs.append(noise * 100.0)
            ys.append(np.mean([float(r["abs_error"]) for r in rr]))
        ax.plot(xs, ys, "o-", color=colors.get(source, "0.5"), label=source)
    ax.set_xlabel("Noise level (%)")
    ax.set_ylabel("MAE (deg)")
    ax.set_title("(c) Noise sensitivity")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    by_case = summarize_by_case(rows)
    cases = sorted(set(r["case"] for r in by_case))
    width = 0.38
    x = np.arange(len(cases))
    for i, source in enumerate(labels):
        vals = []
        for case in cases:
            rr = [r for r in by_case if r["source_mesh"] == source and r["case"] == case]
            vals.append(rr[0]["mean_abs_error"] if rr else np.nan)
        ax.bar(x + (i - 0.5) * width, vals, width=width,
               color=colors.get(source, "0.5"), edgecolor="k", label=source)
    ax.set_xticks(x)
    ax.set_xticklabels(cases, rotation=30, ha="right")
    ax.set_ylabel("Mean abs. error (deg)")
    ax.set_title("(d) Case-wise error")
    ax.grid(alpha=0.25, axis="y")
    ax.legend(fontsize=8)

    fig.suptitle("Forward matching validation with separated true-model mesh",
                 fontsize=13, fontweight="bold")
    out = OUTDIR / "forward_matching_mesh_separation.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def write_note(summary, path):
    lines = [
        "# Forward-Matching Mesh-Separation Validation",
        "",
        "The template library is the existing FDM cache generated on the standard",
        "`dx_factor=0.25` mesh. The separated test generates synthetic observations",
        "on a finer `dx_factor=0.125` mesh before matching them to the unchanged",
        "template cache.",
        "",
        "| Observation mesh | MAE (deg) | Median AE (deg) | <=5 deg (%) | <=10 deg (%) | Family acc. (%) | Mean corr. |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, vals in summary.items():
        lines.append(
            f"| {key} | {vals['mae']:.2f} | {vals['median_ae']:.2f} | "
            f"{vals['within5']:.1f} | {vals['within10']:.1f} | "
            f"{vals['family_acc']:.1f} | {vals['mean_corr']:.3f} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    print("=" * 72)
    print("Forward-matching mesh-separation validation")
    print("=" * 72)
    cache = load_cache()
    print(f"Templates: {len(cache)}")

    survey = make_survey()
    same_mesh = make_mesh(survey, dx_factor=0.25)
    fine_mesh = make_mesh(survey, dx_factor=0.125)
    print(f"same_mesh: cells={same_mesh.n_cells}, dx_factor=0.25")
    print(f"fine_mesh: cells={fine_mesh.n_cells}, dx_factor=0.125")

    rows = []
    mesh_defs = [
        ("same_mesh_dx0.25", same_mesh),
        ("separated_fine_mesh_dx0.125", fine_mesh),
    ]
    for source, mesh in mesh_defs:
        cases = independent_cases(mesh)
        for i, case in enumerate(cases, start=1):
            print(f"[{source}] {i}/{len(cases)} {case['name']}")
            base = forward_fdm(survey, mesh, case["rho"])
            rows.extend(evaluate_response(case, source, base, cache))

    csv_path = OUTDIR / "forward_matching_mesh_separation.csv"
    case_csv_path = OUTDIR / "forward_matching_mesh_separation_by_case.csv"
    write_csv(rows, csv_path)
    write_case_csv(rows, case_csv_path)
    summary = summarize(rows)
    fig_path = plot_results(rows, summary)
    note_path = OUTDIR / "forward_matching_mesh_separation_summary.md"
    write_note(summary, note_path)

    print("")
    for key, vals in summary.items():
        print(
            f"{key}: MAE={vals['mae']:.2f} deg, median={vals['median_ae']:.2f}, "
            f"<=5={vals['within5']:.1f}%, <=10={vals['within10']:.1f}%, "
            f"family={vals['family_acc']:.1f}%, corr={vals['mean_corr']:.3f}"
        )
    print(f"CSV: {csv_path}")
    print(f"By case: {case_csv_path}")
    print(f"Figure: {fig_path}")
    print(f"Note: {note_path}")


if __name__ == "__main__":
    main()
