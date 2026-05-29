#!/usr/bin/env python3
"""
Forward-hypothesis matching for ERT geological structure interpretation.

This is a simulation-based alternative to "estimate one dip angle from the
pseudosection". A library of geological hypotheses is forward-modeled with the
same survey geometry, and the observed pseudosection is matched against the
library in normalized log-apparent-resistivity space.

The method is intentionally simple:

- no inversion,
- no external prior image,
- model class + dip candidate + fit score.

It is useful when local diagnostics disagree or when covered/steep structures
are difficult to represent as one smooth dip estimate.
"""
import csv
import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig") if "ROOT" in globals() else "/tmp")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from RESIS_Pro import DipDipSurvey, Mesh2D, ForwardSolver
from geo_synthetic_benchmark import (
    OUTDIR as BENCH_OUTDIR,
    add_alluvium,
    add_channel,
    add_dipping_band,
    add_dipping_interface,
    add_fault_zone,
    add_lens,
    analyze_case,
    build_cases,
    cell_grids,
    forward_data,
    make_survey_mesh,
)
from geo_structure_interpreter import ROOT, load_ml_model


OUTDIR = ROOT / "GeoHypothesisMatching"


def normalized_log_response(rho_a):
    x = np.log10(np.maximum(np.asarray(rho_a, dtype=float), 1.0))
    med = np.median(x)
    scale = np.std(x)
    if scale < 1e-8:
        scale = 1.0
    return (x - med) / scale


def new_rho(mesh, value):
    return np.full(mesh.n_cells, float(value), dtype=float)


def template_models(mesh, level="standard"):
    templates = []

    def add(name, family, dip, rho, expected, params=None):
        templates.append({
            "name": name,
            "family": family,
            "dip": None if dip is None else float(dip),
            "expected": expected,
            "rho": rho,
            "params": params or {},
        })

    dips = [5, 10, 15, 20, 25, 30, 35, 45, 55, 65]
    if level == "paper":
        clean_geoms = [(30, 2, 5), (35, 3, 7), (45, 2, 6), (55, 2, 8)]
        covered_geoms = [(30, 9, 5, 4), (35, 10, 6, 6), (45, 9, 8, 6), (55, 11, 6, 8)]
        groundwater_geoms = [(35, 5, 4), (42, 6, 5), (52, 5, 7)]
        fault_geoms = [(38, 1.5, 5), (48, 1.5, 7), (58, 2.5, 8)]
        basement_geoms = [(30, 8, 3), (35, 9, 4), (45, 10, 5)]
    else:
        clean_geoms = [(35, 3, 7)]
        covered_geoms = [(35, 10, 6, 6)]
        groundwater_geoms = [(42, 6, 5)]
        fault_geoms = [(48, 1.5, 7)]
        basement_geoms = [(35, 9, 4)]

    for dip in dips:
        for x0, z0, thick in clean_geoms:
            rho = new_rho(mesh, 500)
            add_dipping_band(rho, mesh, dip, x0=x0, z0=z0, thickness=thick, value=45)
            add(f"clean_layer_{dip}_x{x0}_t{thick}", "clean_dipping_layer", dip, rho,
                "dipping layer", {"x0": x0, "z0": z0, "thick": thick})

        for x0, z0, thick, cover in covered_geoms:
            rho = new_rho(mesh, 700)
            add_dipping_interface(rho, mesh, dip, x0=x0, z0=z0, shallow_value=300, deep_value=900)
            add_dipping_band(rho, mesh, dip, x0=x0, z0=z0 + 2, thickness=thick, value=60)
            add_alluvium(rho, mesh, thickness=cover, value=90)
            add(f"covered_layer_{dip}_x{x0}_c{cover}_t{thick}", "covered_dipping_layer",
                dip, rho, "covered dipping layer",
                {"x0": x0, "z0": z0, "thick": thick, "cover": cover})

        for x0, z0, thick in groundwater_geoms:
            rho = new_rho(mesh, 450)
            add_dipping_band(rho, mesh, dip, x0=x0, z0=z0, thickness=thick, value=18)
            add_alluvium(rho, mesh, thickness=4, value=130)
            add(f"groundwater_path_{dip}_x{x0}_t{thick}", "dipping_groundwater",
                dip, rho, "dipping groundwater",
                {"x0": x0, "z0": z0, "thick": thick})

        for x0, z0, width in fault_geoms:
            rho = new_rho(mesh, 600)
            add_fault_zone(rho, mesh, dip, x0=x0, z0=z0, width=width, value=25)
            add(f"fault_zone_{dip}_x{x0}_w{width}", "conductive_fault_zone",
                dip, rho, "conductive fault zone",
                {"x0": x0, "z0": z0, "width": width})

        if dip <= 35:
            for x0, z0, cover in basement_geoms:
                rho = new_rho(mesh, 250)
                add_dipping_interface(rho, mesh, dip, x0=x0, z0=z0,
                                      shallow_value=180, deep_value=1500)
                add_alluvium(rho, mesh, thickness=cover, value=100)
                add(f"basement_interface_{dip}_x{x0}_c{cover}", "dipping_basement",
                    dip, rho, "dipping basement",
                    {"x0": x0, "z0": z0, "cover": cover})

    X, _ = cell_grids(mesh)

    rho = new_rho(mesh, 450)
    rho[X.ravel() > 72] = 1200
    rho[X.ravel() <= 72] = 180
    add_alluvium(rho, mesh, thickness=5, value=90)
    add("vertical_block_boundary", "vertical_block_boundary", None, rho, "vertical block boundary")

    rho = new_rho(mesh, 500)
    add_lens(rho, mesh, xc=70, zc=14, rx=25, rz=6, value=20)
    add_alluvium(rho, mesh, thickness=4, value=110)
    add("groundwater_lens", "groundwater_lens", None, rho, "groundwater lens")

    rho = new_rho(mesh, 700)
    add_channel(rho, mesh, xc=70, zc=7, width=32, depth=7, value=45)
    add_alluvium(rho, mesh, thickness=3, value=120)
    add("buried_channel", "buried_channel", None, rho, "buried conductive channel")

    rho = new_rho(mesh, 500)
    add_dipping_band(rho, mesh, 20, x0=28, z0=5, thickness=5, value=35)
    add_fault_zone(rho, mesh, 55, x0=82, z0=2, width=6, value=25)
    add_alluvium(rho, mesh, thickness=5, value=100)
    add("composite_groundwater_fault", "composite", None, rho, "composite groundwater/fault")

    return templates


def cache_key(survey, level="standard"):
    n_counts = []
    for n in range(1, survey.n_max + 1):
        n_counts.append(str(sum(1 for m in survey.measurements if m["n"] == n)))
    counts = "-".join(n_counts)
    return f"{level}_ne{survey.n_electrodes}_a{survey.a:g}_n{survey.n_max}_d{survey.n_data}_{counts}"


def build_library(survey, mesh, cache=True, level="standard"):
    OUTDIR.mkdir(parents=True, exist_ok=True)
    key = cache_key(survey, level=level)
    npz_path = OUTDIR / f"template_library_{key}.npz"
    meta_path = OUTDIR / f"template_library_{key}.json"

    if cache and npz_path.exists() and meta_path.exists():
        data = np.load(npz_path)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return data["responses"], meta

    templates = template_models(mesh, level=level)
    responses = []
    meta = []
    print(f"Building hypothesis library: {len(templates)} templates")
    for i, t in enumerate(templates, 1):
        solver = ForwardSolver(mesh, t["rho"])
        rho_a = solver.compute_data(survey, callback=lambda j, n: None)
        responses.append(normalized_log_response(rho_a))
        meta.append({k: t[k] for k in ("name", "family", "dip", "expected", "params")})
        if i % 10 == 0:
            print(f"  {i}/{len(templates)}")

    responses = np.asarray(responses, dtype=float)
    np.savez_compressed(npz_path, responses=responses)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return responses, meta


def _softmax(values, temperature=0.08):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return values
    z = (values - np.max(values)) / max(temperature, 1e-6)
    z = np.clip(z, -80, 0)
    p = np.exp(z)
    return p / max(np.sum(p), 1e-30)


def match_hypotheses(rho_a, responses, meta, topk=5, temperature=0.08):
    obs = normalized_log_response(rho_a)
    scores = []
    for i, tpl in enumerate(responses):
        corr = float(np.corrcoef(obs, tpl)[0, 1])
        rmse = float(np.sqrt(np.mean((obs - tpl) ** 2)))
        score = corr - 0.25 * rmse
        rec = dict(meta[i])
        rec.update({"corr": corr, "rmse": rmse, "score": score})
        scores.append(rec)
    probs = _softmax([r["score"] for r in scores], temperature=temperature)
    for r, p in zip(scores, probs):
        r["prob"] = float(p)
    scores.sort(key=lambda r: r["score"], reverse=True)
    return scores[:topk]


def summarize_top(top):
    best = top[0]
    families = {}
    family_probs = {}
    for r in top:
        families[r["family"]] = max(families.get(r["family"], -999), r["score"])
        family_probs[r["family"]] = family_probs.get(r["family"], 0.0) + r.get("prob", 0.0)
    best_family = max(families, key=families.get)
    dip_vals = [
        r["dip"] for r in top
        if r["family"] == best_family
        and r["dip"] is not None
        and r["score"] > best["score"] - 0.10
    ]
    if dip_vals:
        dip_est = float(np.median(dip_vals))
        dip_spread = float(np.std(dip_vals))
    else:
        dip_est = None
        dip_spread = None
    pvals = np.array([r.get("prob", 0.0) for r in top], dtype=float)
    pnorm = pvals / max(pvals.sum(), 1e-30)
    entropy = float(-np.sum(pnorm * np.log(np.maximum(pnorm, 1e-30)))) if len(pnorm) else 0.0
    eff_n = float(np.exp(entropy)) if len(pnorm) else 0.0
    return {
        "best_family": best_family,
        "best_name": best["name"],
        "best_dip": best["dip"],
        "dip_estimate": dip_est,
        "dip_spread": dip_spread,
        "best_score": best["score"],
        "best_corr": best["corr"],
        "best_prob": best.get("prob", 0.0),
        "family_probs": dict(sorted(family_probs.items(), key=lambda kv: kv[1], reverse=True)),
        "entropy": entropy,
        "effective_n": eff_n,
        "top": top,
    }


def run_benchmark():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    survey, mesh = make_survey_mesh(n_electrodes=30, a=5.0, n_max=6)
    responses, meta = build_library(survey, mesh, cache=True, level="paper")
    model_dict = load_ml_model()
    cases = build_cases(mesh)

    rows = []
    print("=" * 72)
    print("Forward-hypothesis matching benchmark")
    print("=" * 72)
    for i, case in enumerate(cases, 1):
        rho_a = forward_data(survey, mesh, case["rho"], noise=0.03, seed=100 + i)
        interp = analyze_case(case, survey, rho_a, model_dict)
        top = match_hypotheses(rho_a, responses, meta, topk=12)
        hyp = summarize_top(top)

        if case["true_dip"] is not None and hyp["dip_estimate"] is not None:
            hyp_err = abs(hyp["dip_estimate"] - case["true_dip"])
        else:
            hyp_err = None

        rows.append({
            "name": case["name"],
            "group": case["group"],
            "true_dip": case["true_dip"],
            "interpreter_est": interp["prediction"]["estimate"],
            "interpreter_err": interp["abs_error"],
            "hyp_family": hyp["best_family"],
            "hyp_name": hyp["best_name"],
            "hyp_dip": hyp["dip_estimate"],
            "hyp_spread": hyp["dip_spread"],
            "hyp_err": hyp_err,
            "hyp_score": hyp["best_score"],
            "hyp_corr": hyp["best_corr"],
            "hyp_prob": hyp["best_prob"],
            "effective_n": hyp["effective_n"],
            "family_probs": hyp["family_probs"],
            "top": top,
        })

        td = "none" if case["true_dip"] is None else f"{case['true_dip']:.0f}"
        hyp_dip = "none" if hyp["dip_estimate"] is None else f"{hyp['dip_estimate']:.1f}"
        err = "" if hyp_err is None else f", hyp_err={hyp_err:.1f}"
        print(
            f"[{i:02d}/{len(cases)}] {case['name']}: true={td}, "
            f"interp={interp['prediction']['estimate']:.1f}, "
            f"hyp={hyp['best_family']}:{hyp_dip}{err}, corr={hyp['best_corr']:.2f}"
        )

    path = OUTDIR / "HypothesisMatching_benchmark.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        fields = [
            "name", "group", "true_dip", "interpreter_est", "interpreter_err",
            "hyp_family", "hyp_name", "hyp_dip", "hyp_spread", "hyp_err",
            "hyp_score", "hyp_corr", "top1", "top2", "top3",
            "hyp_prob", "effective_n", "family_probs",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({
                "name": r["name"],
                "group": r["group"],
                "true_dip": "" if r["true_dip"] is None else r["true_dip"],
                "interpreter_est": f"{r['interpreter_est']:.3f}",
                "interpreter_err": "" if r["interpreter_err"] is None else f"{r['interpreter_err']:.3f}",
                "hyp_family": r["hyp_family"],
                "hyp_name": r["hyp_name"],
                "hyp_dip": "" if r["hyp_dip"] is None else f"{r['hyp_dip']:.3f}",
                "hyp_spread": "" if r["hyp_spread"] is None else f"{r['hyp_spread']:.3f}",
                "hyp_err": "" if r["hyp_err"] is None else f"{r['hyp_err']:.3f}",
                "hyp_score": f"{r['hyp_score']:.3f}",
                "hyp_corr": f"{r['hyp_corr']:.3f}",
                "hyp_prob": f"{r['hyp_prob']:.4f}",
                "effective_n": f"{r['effective_n']:.3f}",
                "family_probs": json.dumps(r["family_probs"], ensure_ascii=False),
                "top1": r["top"][0]["name"],
                "top2": r["top"][1]["name"] if len(r["top"]) > 1 else "",
                "top3": r["top"][2]["name"] if len(r["top"]) > 2 else "",
            })

    fig_path = plot_benchmark_summary(rows)

    known = [r for r in rows if r["hyp_err"] is not None]
    interp_known = [r for r in rows if r["interpreter_err"] is not None]
    hyp_mae = float(np.mean([r["hyp_err"] for r in known]))
    interp_mae = float(np.mean([r["interpreter_err"] for r in interp_known]))
    expected_family = {
        "known_dip": "clean_dipping_layer",
        "covered_known_dip": "covered_dipping_layer",
        "groundwater_known_dip": "dipping_groundwater",
        "fault_known_dip": "conductive_fault_zone",
        "interface_known_dip": "dipping_basement",
    }
    top_family_acc = np.mean([
        r["hyp_family"] == expected_family.get(r["group"], "")
        for r in known
    ])
    within5 = np.mean([r["hyp_err"] <= 5.0 for r in known])
    within10 = np.mean([r["hyp_err"] <= 10.0 for r in known])
    print("")
    print(f"Interpreter MAE: {interp_mae:.2f} deg")
    print(f"Hypothesis matching MAE: {hyp_mae:.2f} deg")
    print(f"Hypothesis <=5 deg: {within5*100:.1f}%")
    print(f"Hypothesis <=10 deg: {within10*100:.1f}%")
    print(f"Approx. family accuracy: {top_family_acc*100:.1f}%")
    print(f"CSV: {path}")
    print(f"Figure: {fig_path}")
    print("=" * 72)


def plot_benchmark_summary(rows):
    known = [r for r in rows if r["true_dip"] is not None and r["hyp_dip"] is not None]
    no_single = [r for r in rows if r["true_dip"] is None]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    true = np.array([r["true_dip"] for r in known], dtype=float)
    interp = np.array([r["interpreter_est"] for r in known], dtype=float)
    hyp = np.array([r["hyp_dip"] for r in known], dtype=float)
    spread = np.array([0.0 if r["hyp_spread"] is None else r["hyp_spread"] for r in known], dtype=float)
    ax.plot([0, 65], [0, 65], "k--", alpha=0.45, label="1:1")
    ax.scatter(true, interp, s=70, color="#d95f02", edgecolor="k",
               label="multi-evidence interpreter", alpha=0.82)
    ax.errorbar(true, hyp, yerr=spread, fmt="o", ms=7, capsize=3,
                color="#1b9e77", ecolor="#1b9e77", label="forward hypothesis", alpha=0.90)
    ax.set_xlabel("True dip (deg)")
    ax.set_ylabel("Estimated dip (deg)")
    ax.set_xlim(0, 65)
    ax.set_ylim(0, 70)
    ax.set_title("(a) Dip recovery")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)

    ax = axes[0, 1]
    interp_err = np.array([abs(r["interpreter_err"]) for r in known], dtype=float)
    hyp_err = np.array([abs(r["hyp_err"]) for r in known], dtype=float)
    bins = np.arange(0, max(20, hyp_err.max() + 5), 2.5)
    ax.hist(interp_err, bins=bins, alpha=0.65, color="#d95f02",
            edgecolor="k", label=f"interpreter MAE={interp_err.mean():.2f} deg")
    ax.hist(hyp_err, bins=bins, alpha=0.65, color="#1b9e77",
            edgecolor="k", label=f"hypothesis MAE={hyp_err.mean():.2f} deg")
    ax.axvline(5, color="k", linestyle=(0, (5, 3)), lw=1)
    ax.axvline(10, color="k", linestyle=":", lw=1)
    ax.set_xlabel("Absolute dip error (deg)")
    ax.set_ylabel("Count")
    ax.set_title("(b) Error distribution")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)

    ax = axes[1, 0]
    families = sorted(set(r["hyp_family"] for r in known))
    counts = [sum(r["hyp_family"] == f for r in known) for f in families]
    ax.barh(families, counts, color="#7570b3", edgecolor="k")
    ax.set_xlabel("Top-1 count")
    ax.set_title("(c) Recovered geological families")
    ax.grid(alpha=0.25, axis="x")

    ax = axes[1, 1]
    names = [r["name"].replace("_", "\n") for r in no_single]
    probs = []
    labels = []
    for r in no_single:
        fp = r.get("family_probs", {})
        if fp:
            fam, prob = sorted(fp.items(), key=lambda kv: kv[1], reverse=True)[0]
            probs.append(prob)
            labels.append(fam)
        else:
            probs.append(0.0)
            labels.append(r["hyp_family"])
    x = np.arange(len(no_single))
    ax.bar(x, probs, color="#66a61e", edgecolor="k", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8)
    for i, (p, lab) in enumerate(zip(probs, labels)):
        ax.text(i, p + 0.02, lab, rotation=90, ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Best family support weight")
    ax.set_title("(d) No-single-dip cases")
    ax.grid(alpha=0.25, axis="y")

    fig.suptitle("Forward-hypothesis matching benchmark", fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = OUTDIR / "HypothesisMatching_benchmark.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


if __name__ == "__main__":
    run_benchmark()
