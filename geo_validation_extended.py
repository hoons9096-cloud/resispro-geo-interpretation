#!/usr/bin/env python3
"""
Extended validation for the ERT geological hypothesis-matching interpreter.

This script targets likely reviewer questions:

1. Does the method still work on geometries not exactly used as templates?
2. How sensitive is it to noise?
3. If the correct geological family is missing from the library, does the
   method become uncertain instead of confidently forcing the wrong class?
"""
import csv
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from geo_hypothesis_matching import (
    OUTDIR as HYP_OUTDIR,
    build_library,
    match_hypotheses,
    summarize_top,
)
from geo_synthetic_benchmark import (
    add_alluvium,
    add_dipping_band,
    add_dipping_interface,
    add_fault_zone,
    analyze_case,
    forward_data,
    make_survey_mesh,
)
from geo_structure_interpreter import ROOT, load_ml_model


OUTDIR = ROOT / "SCI_Paper" / "ExtendedValidation"
OUTDIR.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.linewidth"] = 0.8


EXPECTED_FAMILY = {
    "independent_clean": "clean_dipping_layer",
    "independent_covered": "covered_dipping_layer",
    "independent_groundwater": "dipping_groundwater",
    "independent_fault": "conductive_fault_zone",
    "independent_basement": "dipping_basement",
}


def new_rho(mesh, value):
    return np.full(mesh.n_cells, float(value), dtype=float)


def independent_cases(mesh):
    """Cases deliberately offset from the template geometry grid."""
    cases = []

    rho = new_rho(mesh, 620)
    add_dipping_band(rho, mesh, 28, x0=51, z0=4.5, thickness=9, value=38)
    cases.append({
        "name": "independent_clean_28",
        "group": "independent_clean",
        "expected": "clean dipping layer",
        "true_dip": 28.0,
        "rho": rho,
    })

    rho = new_rho(mesh, 760)
    add_dipping_interface(rho, mesh, 32, x0=42, z0=12.5, shallow_value=280, deep_value=980)
    add_dipping_band(rho, mesh, 32, x0=42, z0=15, thickness=7, value=55)
    add_alluvium(rho, mesh, thickness=7.2, value=85)
    cases.append({
        "name": "independent_covered_32",
        "group": "independent_covered",
        "expected": "covered dipping layer",
        "true_dip": 32.0,
        "rho": rho,
    })

    rho = new_rho(mesh, 430)
    add_dipping_band(rho, mesh, 18, x0=57, z0=7.5, thickness=6.5, value=16)
    add_alluvium(rho, mesh, thickness=5.2, value=125)
    cases.append({
        "name": "independent_groundwater_18",
        "group": "independent_groundwater",
        "expected": "dipping groundwater",
        "true_dip": 18.0,
        "rho": rho,
    })

    rho = new_rho(mesh, 650)
    add_fault_zone(rho, mesh, 40, x0=54, z0=2.2, width=9.0, value=28)
    cases.append({
        "name": "independent_fault_40",
        "group": "independent_fault",
        "expected": "conductive fault zone",
        "true_dip": 40.0,
        "rho": rho,
    })

    rho = new_rho(mesh, 260)
    add_dipping_interface(rho, mesh, 18, x0=49, z0=11.5, shallow_value=170, deep_value=1350)
    add_alluvium(rho, mesh, thickness=5.5, value=95)
    cases.append({
        "name": "independent_basement_18",
        "group": "independent_basement",
        "expected": "dipping basement",
        "true_dip": 18.0,
        "rho": rho,
    })

    return cases


def _match_with_optional_family_filter(rho_a, responses, meta, omit_family=None):
    if omit_family is None:
        return match_hypotheses(rho_a, responses, meta, topk=12)
    keep = [i for i, m in enumerate(meta) if m["family"] != omit_family]
    return match_hypotheses(rho_a, responses[keep], [meta[i] for i in keep], topk=12)


def run_extended_validation():
    survey, mesh = make_survey_mesh(n_electrodes=30, a=5.0, n_max=6)
    responses, meta = build_library(survey, mesh, cache=True, level="paper")
    model_dict = load_ml_model()
    cases = independent_cases(mesh)
    noise_levels = [0.01, 0.03, 0.05, 0.07, 0.10]

    rows = []
    missing_rows = []

    print("=" * 72)
    print("Extended validation for geological hypothesis matching")
    print("=" * 72)

    for ci, case in enumerate(cases, 1):
        expected_family = EXPECTED_FAMILY[case["group"]]
        for ni, noise in enumerate(noise_levels, 1):
            rho_a = forward_data(survey, mesh, case["rho"], noise=noise, seed=1000 + ci * 20 + ni)
            interp = analyze_case(case, survey, rho_a, model_dict)

            top = _match_with_optional_family_filter(rho_a, responses, meta)
            hyp = summarize_top(top)
            err = None if hyp["dip_estimate"] is None else abs(hyp["dip_estimate"] - case["true_dip"])
            family_ok = hyp["best_family"] == expected_family

            rows.append({
                "name": case["name"],
                "expected_family": expected_family,
                "noise_pct": noise * 100,
                "true_dip": case["true_dip"],
                "interpreter_est": interp["prediction"]["estimate"],
                "interpreter_err": interp["abs_error"],
                "hyp_family": hyp["best_family"],
                "hyp_dip": hyp["dip_estimate"],
                "hyp_err": err,
                "hyp_corr": hyp["best_corr"],
                "support_weight": hyp["best_prob"],
                "effective_n": hyp["effective_n"],
                "family_ok": family_ok,
            })

            miss_top = _match_with_optional_family_filter(
                rho_a, responses, meta, omit_family=expected_family
            )
            miss = summarize_top(miss_top)
            missing_rows.append({
                "name": case["name"],
                "omitted_family": expected_family,
                "noise_pct": noise * 100,
                "best_wrong_family": miss["best_family"],
                "best_wrong_dip": miss["dip_estimate"],
                "best_wrong_corr": miss["best_corr"],
                "best_wrong_support": miss["best_prob"],
                "effective_n": miss["effective_n"],
            })

        print(f"[{ci}/{len(cases)}] {case['name']} complete")

    write_rows(rows, OUTDIR / "ExtendedValidation_noise_geometry.csv")
    write_rows(missing_rows, OUTDIR / "ExtendedValidation_missing_family.csv")
    fig_path = plot_extended(rows, missing_rows)
    print_summary(rows, missing_rows, fig_path)
    return rows, missing_rows


def write_rows(rows, path):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def plot_extended(rows, missing_rows):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    known = [r for r in rows if r["hyp_err"] is not None]
    noises = sorted(set(r["noise_pct"] for r in known))

    ax = axes[0, 0]
    mae_interp = []
    mae_hyp = []
    for n in noises:
        rr = [r for r in known if r["noise_pct"] == n]
        mae_interp.append(np.mean([r["interpreter_err"] for r in rr]))
        mae_hyp.append(np.mean([r["hyp_err"] for r in rr]))
    ax.plot(noises, mae_interp, "o-", color="#d95f02", label="diagnostic interpreter")
    ax.plot(noises, mae_hyp, "o-", color="#1b9e77", label="forward-hypothesis")
    ax.set_xlabel("Gaussian noise (%)")
    ax.set_ylabel("MAE (deg)")
    ax.set_title("(a) Noise sensitivity")
    ax.grid(alpha=0.25)
    ax.legend()

    ax = axes[0, 1]
    names = sorted(set(r["name"] for r in known))
    vals = []
    labels = []
    for name in names:
        rr = [r for r in known if r["name"] == name]
        vals.append(np.mean([r["hyp_err"] for r in rr]))
        labels.append(name.replace("independent_", "").replace("_", "\n"))
    ax.bar(labels, vals, color="#7570b3", edgecolor="k")
    ax.set_ylabel("Mean absolute error (deg)")
    ax.set_title("(b) Independent geometry error")
    ax.grid(alpha=0.25, axis="y")

    ax = axes[1, 0]
    acc = []
    for n in noises:
        rr = [r for r in known if r["noise_pct"] == n]
        acc.append(np.mean([r["family_ok"] for r in rr]) * 100)
    ax.plot(noises, acc, "s-", color="#386cb0")
    ax.set_ylim(0, 105)
    ax.set_xlabel("Gaussian noise (%)")
    ax.set_ylabel("Top-1 family accuracy (%)")
    ax.set_title("(c) Family recovery under noise")
    ax.grid(alpha=0.25)

    ax = axes[1, 1]
    full_corr = []
    missing_corr = []
    for name in names:
        rr = [r for r in known if r["name"] == name and r["noise_pct"] == 3.0]
        mm = [r for r in missing_rows if r["name"] == name and r["noise_pct"] == 3.0]
        full_corr.append(rr[0]["hyp_corr"])
        missing_corr.append(mm[0]["best_wrong_corr"])
    x = np.arange(len(names))
    w = 0.36
    ax.bar(x - w / 2, full_corr, width=w, color="#1b9e77", edgecolor="k", label="full library")
    ax.bar(x + w / 2, missing_corr, width=w, color="#e7298a", edgecolor="k", label="true family omitted")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Best correlation at 3% noise")
    ax.set_title("(d) Template incompleteness check")
    ax.grid(alpha=0.25, axis="y")
    ax.legend(fontsize=9)

    fig.suptitle("Extended validation of ERT geological hypothesis matching",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = OUTDIR / "ExtendedValidation_summary.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def print_summary(rows, missing_rows, fig_path):
    known = [r for r in rows if r["hyp_err"] is not None]
    interp_mae = float(np.mean([r["interpreter_err"] for r in known]))
    hyp_mae = float(np.mean([r["hyp_err"] for r in known]))
    within5 = float(np.mean([r["hyp_err"] <= 5.0 for r in known]) * 100)
    within10 = float(np.mean([r["hyp_err"] <= 10.0 for r in known]) * 100)
    fam_acc = float(np.mean([r["family_ok"] for r in known]) * 100)
    miss_corr = float(np.mean([r["best_wrong_corr"] for r in missing_rows]))
    full_corr = float(np.mean([r["hyp_corr"] for r in known]))

    print("")
    print(f"Independent/noise diagnostic MAE: {interp_mae:.2f} deg")
    print(f"Independent/noise hypothesis MAE: {hyp_mae:.2f} deg")
    print(f"Hypothesis <=5 deg: {within5:.1f}%")
    print(f"Hypothesis <=10 deg: {within10:.1f}%")
    print(f"Top-1 family accuracy: {fam_acc:.1f}%")
    print(f"Mean corr full library: {full_corr:.3f}")
    print(f"Mean corr with true family omitted: {miss_corr:.3f}")
    print(f"CSV: {OUTDIR / 'ExtendedValidation_noise_geometry.csv'}")
    print(f"Missing-family CSV: {OUTDIR / 'ExtendedValidation_missing_family.csv'}")
    print(f"Figure: {fig_path}")
    print("=" * 72)


if __name__ == "__main__":
    run_extended_validation()
