#!/usr/bin/env python3
"""
Synthetic geological benchmark for the RESIS Pro structure interpreter.

The benchmark creates its own geological models instead of using existing APV
examples. It evaluates whether the interpreter can separate:

- simple dipping structures with a known target dip,
- covered/buried structures where surface response is flattened,
- steep or blocky structures where one dip angle is not a stable target,
- groundwater/lens/channel cases where the output should be a hypothesis,
  not an overconfident angle.
"""
import csv
import os
from pathlib import Path

import numpy as np

ROOT_DIR = Path("")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT_DIR / ".mplconfig"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from RESIS_Pro import DipDipSurvey, Mesh2D, ForwardSolver
from dip_diagnostics import diagnose_all
from geo_structure_interpreter import (
    ROOT,
    load_ml_model,
    robust_dip_estimate,
    adjust_for_buried_structure,
    build_angle_candidates,
    build_geological_hypotheses,
    recommend_strategy,
    resistivity_context,
)


OUTDIR = ROOT / "SyntheticGeoBenchmark"
plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False


def make_survey_mesh(n_electrodes=30, a=5.0, n_max=6):
    elec_x = np.arange(n_electrodes) * a
    survey = DipDipSurvey(
        a=a,
        n_electrodes=n_electrodes,
        n_max=n_max,
        electrode_x=elec_x,
        array_type="dipole-dipole",
    )
    mesh = Mesh2D(survey, depth_factor=2.5, dx_factor=0.25)
    return survey, mesh


def cell_grids(mesh):
    X, Z = np.meshgrid(mesh.x_cc, mesh.z_cc)
    return X, Z


def add_dipping_band(rho, mesh, dip_deg, x0, z0, thickness, value, side="below"):
    X, Z = cell_grids(mesh)
    top = np.tan(np.radians(dip_deg)) * (X - x0) + z0
    if side == "below":
        mask = (Z >= top) & (Z <= top + thickness)
    else:
        mask = (Z <= top) & (Z >= top - thickness)
    rho[mask.ravel()] = value
    return mask


def add_dipping_interface(rho, mesh, dip_deg, x0, z0, shallow_value, deep_value):
    X, Z = cell_grids(mesh)
    boundary = np.tan(np.radians(dip_deg)) * (X - x0) + z0
    rho[:] = shallow_value
    rho[(Z >= boundary).ravel()] = deep_value
    return boundary


def add_fault_zone(rho, mesh, dip_deg, x0, z0, width, value):
    X, Z = cell_grids(mesh)
    line = np.tan(np.radians(dip_deg)) * (X - x0) + z0
    distance = np.abs(Z - line) / np.sqrt(1 + np.tan(np.radians(dip_deg)) ** 2)
    mask = distance <= width / 2
    rho[mask.ravel()] = value
    return mask


def add_alluvium(rho, mesh, thickness, value):
    _, Z = cell_grids(mesh)
    mask = Z <= thickness
    rho[mask.ravel()] = value
    return mask


def add_lens(rho, mesh, xc, zc, rx, rz, value):
    X, Z = cell_grids(mesh)
    mask = ((X - xc) / rx) ** 2 + ((Z - zc) / rz) ** 2 <= 1.0
    rho[mask.ravel()] = value
    return mask


def add_channel(rho, mesh, xc, zc, width, depth, value):
    X, Z = cell_grids(mesh)
    base = zc + depth * ((X - xc) / width) ** 2
    mask = (np.abs(X - xc) <= width) & (Z >= base - 3.0) & (Z <= base + 5.0)
    rho[mask.ravel()] = value
    return mask


def build_cases(mesh):
    cases = []

    def new_rho(value=500.0):
        return np.full(mesh.n_cells, value, dtype=float)

    # 1. Clean dipping conductive layer.
    for dip in (15, 25, 35):
        rho = new_rho(500)
        add_dipping_band(rho, mesh, dip, x0=35, z0=3, thickness=7, value=45)
        cases.append({
            "name": f"clean_dipping_layer_{dip}",
            "group": "known_dip",
            "true_dip": dip,
            "expected": "dipping layer",
            "rho": rho,
        })

    # 2. Alluvium-covered dipping layer.
    for dip in (15, 25, 35):
        rho = new_rho(700)
        add_dipping_interface(rho, mesh, dip, x0=35, z0=10, shallow_value=300, deep_value=900)
        add_dipping_band(rho, mesh, dip, x0=35, z0=12, thickness=6, value=60)
        add_alluvium(rho, mesh, thickness=6, value=90)
        cases.append({
            "name": f"covered_dipping_layer_{dip}",
            "group": "covered_known_dip",
            "true_dip": dip,
            "expected": "covered dipping layer",
            "rho": rho,
        })

    # 3. Dipping groundwater/conductive pathway.
    for dip in (10, 20, 30):
        rho = new_rho(450)
        add_dipping_band(rho, mesh, dip, x0=42, z0=6, thickness=5, value=18)
        add_alluvium(rho, mesh, thickness=4, value=130)
        cases.append({
            "name": f"dipping_groundwater_{dip}",
            "group": "groundwater_known_dip",
            "true_dip": dip,
            "expected": "dipping groundwater",
            "rho": rho,
        })

    # 4. Conductive fault zones, including the difficult 30-60 degree range.
    for dip in (30, 45, 60):
        rho = new_rho(600)
        add_fault_zone(rho, mesh, dip, x0=48, z0=1.5, width=7, value=25)
        cases.append({
            "name": f"conductive_fault_zone_{dip}",
            "group": "fault_known_dip",
            "true_dip": dip,
            "expected": "conductive fault zone",
            "rho": rho,
        })

    # 5. Dipping resistive basement interface.
    for dip in (12, 22):
        rho = new_rho(250)
        add_dipping_interface(rho, mesh, dip, x0=35, z0=9, shallow_value=180, deep_value=1500)
        add_alluvium(rho, mesh, thickness=4, value=100)
        cases.append({
            "name": f"dipping_resistive_basement_{dip}",
            "group": "interface_known_dip",
            "true_dip": dip,
            "expected": "dipping basement",
            "rho": rho,
        })

    # 6. Cases where one dip angle is not the right target.
    rho = new_rho(450)
    rho[cell_grids(mesh)[0].ravel() > 72] = 1200
    rho[cell_grids(mesh)[0].ravel() <= 72] = 180
    add_alluvium(rho, mesh, thickness=5, value=90)
    cases.append({
        "name": "vertical_block_boundary",
        "group": "no_single_dip",
        "true_dip": None,
        "expected": "vertical block/fault boundary",
        "rho": rho,
    })

    rho = new_rho(500)
    add_lens(rho, mesh, xc=70, zc=14, rx=25, rz=6, value=20)
    add_alluvium(rho, mesh, thickness=4, value=110)
    cases.append({
        "name": "perched_groundwater_lens",
        "group": "no_single_dip",
        "true_dip": None,
        "expected": "groundwater lens",
        "rho": rho,
    })

    rho = new_rho(700)
    add_channel(rho, mesh, xc=70, zc=7, width=32, depth=7, value=45)
    add_alluvium(rho, mesh, thickness=3, value=120)
    cases.append({
        "name": "buried_conductive_channel",
        "group": "no_single_dip",
        "true_dip": None,
        "expected": "buried channel",
        "rho": rho,
    })

    rho = new_rho(500)
    add_dipping_band(rho, mesh, 20, x0=28, z0=5, thickness=5, value=35)
    add_fault_zone(rho, mesh, 55, x0=82, z0=2, width=6, value=25)
    add_alluvium(rho, mesh, thickness=5, value=100)
    cases.append({
        "name": "composite_groundwater_fault",
        "group": "no_single_dip",
        "true_dip": None,
        "expected": "composite groundwater/fault",
        "rho": rho,
    })

    return cases


def forward_data(survey, mesh, rho, noise=0.03, seed=42):
    solver = ForwardSolver(mesh, rho)
    rho_a = solver.compute_data(survey, callback=lambda i, n: None)
    rng = np.random.default_rng(seed)
    rho_a = rho_a * (1.0 + noise * rng.standard_normal(len(rho_a)))
    return np.maximum(rho_a, 1.0)


def analyze_case(case, survey, rho_a, model_dict):
    diag = diagnose_all(survey, rho_a, verbose=False)
    pred = robust_dip_estimate(diag, model_dict)
    ctx = resistivity_context(survey, rho_a)
    pred = adjust_for_buried_structure(diag, pred, ctx)
    candidates = build_angle_candidates(diag, pred)
    hypotheses, warnings, disagreement = build_geological_hypotheses(diag, pred, ctx)
    rec = recommend_strategy(pred, diag, ctx)
    err = None
    if case["true_dip"] is not None:
        err = abs(pred["estimate"] - case["true_dip"])
    return {
        "name": case["name"],
        "group": case["group"],
        "expected": case["expected"],
        "true_dip": case["true_dip"],
        "rho": case["rho"],
        "rho_a": rho_a,
        "diag": diag,
        "prediction": pred,
        "context": ctx,
        "candidates": candidates,
        "hypotheses": hypotheses,
        "warnings": warnings,
        "disagreement": disagreement,
        "recommendation": rec,
        "abs_error": err,
    }


def plot_case(result, survey, mesh):
    OUTDIR.mkdir(parents=True, exist_ok=True)
    rho = result["rho"]
    rho_a = result["rho_a"]
    diag = result["diag"]
    pred = result["prediction"]

    X, Z = np.meshgrid(mesh.x_cc, mesh.z_cc)
    xs = np.array([m["x"] for m in survey.measurements])
    zs = np.array([m["z"] for m in survey.measurements])

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    ax = axes[0]
    im = ax.pcolormesh(
        mesh.x_cc,
        mesh.z_cc,
        np.log10(rho.reshape(mesh.ncz, mesh.ncx)),
        shading="auto",
        cmap="turbo",
    )
    ax.set_ylim(45, 0)
    ax.set_title("True synthetic model")
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Depth (m)")
    fig.colorbar(im, ax=ax, shrink=0.85, label="log10 rho")

    ax = axes[1]
    sc = ax.scatter(xs, zs, c=np.log10(rho_a), cmap="turbo", s=36, edgecolor="k", linewidth=0.2)
    ax.set_ylim(max(zs) + survey.a * 0.3, 0)
    ax.set_title("Synthetic pseudosection")
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Pseudodepth (m)")
    fig.colorbar(sc, ax=ax, shrink=0.85, label="log10 apparent rho")

    ax = axes[2]
    labels = ["M1", "M2", "M3", "M4", "M5", "Final"]
    vals = [
        diag["M1"]["theta_med"],
        diag["M2"]["theta"],
        diag["M3"]["dip_proxy"],
        diag["M4"]["mean"],
        diag["M5"]["theta"],
        pred["estimate"],
    ]
    ax.bar(labels, vals, color=["#4c78a8", "#f58518", "#54a24b", "#b279a2", "#e45756", "#222"])
    if result["true_dip"] is not None:
        ax.axhline(result["true_dip"], color="k", linestyle=(0, (6, 3)), label="true")
        ax.legend()
    ax.set_ylim(0, 70)
    ax.set_title(f"Prediction: {pred['estimate']:.1f} ± {pred['uncertainty']:.1f} deg")
    ax.set_ylabel("Dip angle (deg)")
    ax.grid(alpha=0.25, axis="y")

    title = result["name"]
    if result["true_dip"] is not None:
        title += f" | true {result['true_dip']} deg | err {result['abs_error']:.1f} deg"
    else:
        title += " | no single true dip"
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUTDIR / f"{result['name']}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_summary(results):
    OUTDIR.mkdir(parents=True, exist_ok=True)
    path = OUTDIR / "SyntheticGeoBenchmark_summary.csv"
    fields = [
        "name", "group", "expected", "true_dip", "estimate", "uncertainty",
        "abs_error", "method", "confidence", "ood_score", "max_severity",
        "M1", "M2", "M3", "M4", "M5", "recommended_mode", "warning_count",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            d = r["diag"]
            p = r["prediction"]
            writer.writerow({
                "name": r["name"],
                "group": r["group"],
                "expected": r["expected"],
                "true_dip": "" if r["true_dip"] is None else r["true_dip"],
                "estimate": f"{p['estimate']:.3f}",
                "uncertainty": f"{p['uncertainty']:.3f}",
                "abs_error": "" if r["abs_error"] is None else f"{r['abs_error']:.3f}",
                "method": p["method"],
                "confidence": f"{p['confidence']:.3f}",
                "ood_score": f"{p['ood_score']:.3f}",
                "max_severity": f"{p['max_severity']:.3f}",
                "M1": f"{d['M1']['theta_med']:.3f}",
                "M2": f"{d['M2']['theta']:.3f}",
                "M3": f"{d['M3']['dip_proxy']:.3f}",
                "M4": f"{d['M4']['mean']:.3f}",
                "M5": f"{d['M5']['theta']:.3f}",
                "recommended_mode": r["recommendation"]["mode"],
                "warning_count": len(r["warnings"]),
            })
    return path


def plot_summary(results):
    known = [r for r in results if r["true_dip"] is not None]
    no_single = [r for r in results if r["true_dip"] is None]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    ax = axes[0]
    true = np.array([r["true_dip"] for r in known], dtype=float)
    est = np.array([r["prediction"]["estimate"] for r in known], dtype=float)
    unc = np.array([r["prediction"]["uncertainty"] for r in known], dtype=float)
    groups = [r["group"] for r in known]
    colors = {
        "known_dip": "#4c78a8",
        "covered_known_dip": "#f58518",
        "groundwater_known_dip": "#54a24b",
        "fault_known_dip": "#e45756",
        "interface_known_dip": "#b279a2",
    }
    for g in sorted(set(groups)):
        idx = [i for i, gg in enumerate(groups) if gg == g]
        ax.errorbar(true[idx], est[idx], yerr=unc[idx], fmt="o", ms=7, capsize=3,
                    color=colors.get(g, "gray"), label=g, alpha=0.85)
    ax.plot([0, 65], [0, 65], "k--", alpha=0.45)
    ax.set_xlim(0, 65)
    ax.set_ylim(0, 70)
    ax.set_xlabel("True dip (deg)")
    ax.set_ylabel("Estimated structural dip (deg)")
    ax.set_title("Known-dip synthetic cases")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[1]
    names = [r["name"].replace("_", "\n") for r in no_single]
    vals = [r["prediction"]["estimate"] for r in no_single]
    conf = [r["prediction"]["confidence"] for r in no_single]
    x = np.arange(len(no_single))
    bars = ax.bar(x, vals, color="#777777", edgecolor="k", alpha=0.8)
    for b, c in zip(bars, conf):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1,
                f"conf={c:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylim(0, 70)
    ax.set_ylabel("Reported structural candidate (deg)")
    ax.set_title("No-single-dip cases: should be cautious")
    ax.grid(alpha=0.25, axis="y")

    fig.suptitle("Synthetic geological benchmark for ERT structure interpretation", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUTDIR / "SyntheticGeoBenchmark_overview.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    survey, mesh = make_survey_mesh(n_electrodes=30, a=5.0, n_max=6)
    model_dict = load_ml_model()
    cases = build_cases(mesh)

    results = []
    print("=" * 72)
    print("Synthetic geological benchmark")
    print("=" * 72)
    print(f"Cases: {len(cases)}")
    print(f"Output: {OUTDIR}")

    for i, case in enumerate(cases, 1):
        rho_a = forward_data(survey, mesh, case["rho"], noise=0.03, seed=100 + i)
        result = analyze_case(case, survey, rho_a, model_dict)
        results.append(result)
        plot_case(result, survey, mesh)
        td = "none" if case["true_dip"] is None else f"{case['true_dip']:.0f}"
        err = "" if result["abs_error"] is None else f", err={result['abs_error']:.1f}"
        print(
            f"[{i:02d}/{len(cases)}] {case['name']}: true={td}, "
            f"est={result['prediction']['estimate']:.1f}±{result['prediction']['uncertainty']:.1f} "
            f"[{result['prediction']['method']}]{err}"
        )

    summary_path = write_summary(results)
    plot_summary(results)

    known = [r for r in results if r["abs_error"] is not None]
    mae = float(np.mean([r["abs_error"] for r in known]))
    within10 = float(np.mean([r["abs_error"] <= 10.0 for r in known]))
    print("")
    print(f"Known-dip MAE: {mae:.2f} deg")
    print(f"Within 10 deg: {within10 * 100:.1f}%")
    print(f"Summary: {summary_path}")
    print(f"Overview: {OUTDIR / 'SyntheticGeoBenchmark_overview.png'}")
    print("=" * 72)


if __name__ == "__main__":
    main()
