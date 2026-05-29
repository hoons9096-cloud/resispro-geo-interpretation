#!/usr/bin/env python3
"""
JAG-oriented STAR benchmark.

Purpose
-------
Evaluate whether OOD-gated STAR improves 2-D ERT model recovery under a
reduced inverse-crime protocol:

1. Synthetic observations are generated on a fine forward mesh.
2. Inversions are run on a separate coarser mesh.
3. Methods are compared by model-space log error, structural target-zone error,
   and recovered conductive-anomaly dip where applicable.

This script is intentionally separate from the earlier forward-hypothesis
matching manuscript. It supports the STAR/regularization paper route.
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
from scipy.ndimage import binary_dilation, binary_erosion, gaussian_filter

from RESIS_Pro import DipDipSurvey, Mesh2D, ForwardSolver, Inversion2D, run_star_inversion


OUTDIR = ROOT / "STAR_JAG_Benchmark"
OUTDIR.mkdir(parents=True, exist_ok=True)

A = 5.0
N_ELEC = 30
N_MAX = 6

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False


def make_survey():
    ex = np.arange(N_ELEC) * A
    return DipDipSurvey(
        a=A,
        n_electrodes=N_ELEC,
        n_max=N_MAX,
        electrode_x=ex,
        array_type="dipole-dipole",
    )


def grids(mesh):
    return np.meshgrid(mesh.x_cc, mesh.z_cc)


def model_and_mask(mesh, kind, p):
    """Return resistivity model and the known target-structure mask."""
    X, Z = grids(mesh)
    rho = np.full(mesh.n_cells, p.get("rho_bg", 500.0), dtype=float)
    mask = np.zeros((mesh.ncz, mesh.ncx), dtype=bool)

    if kind == "clean_dip":
        top = np.tan(np.radians(p["dip"])) * (X - p["x0"]) + p["z0"]
        mask = (Z >= top) & (Z <= top + p["thick"])
        rho[mask.ravel()] = p.get("rho_target", 35.0)

    elif kind == "covered_dip":
        rho[:] = p.get("rho_bg", 700.0)
        rho[(Z <= p["cover"]).ravel()] = p.get("rho_cover", 100.0)
        top = np.tan(np.radians(p["dip"])) * (X - p["x0"]) + p["z0"]
        mask = (Z >= top) & (Z <= top + p["thick"])
        rho[mask.ravel()] = p.get("rho_target", 45.0)

    elif kind == "groundwater":
        rho[:] = p.get("rho_bg", 450.0)
        if p.get("cover", 0.0) > 0:
            rho[(Z <= p["cover"]).ravel()] = p.get("rho_cover", 130.0)
        top = np.tan(np.radians(p["dip"])) * (X - p["x0"]) + p["z0"]
        mask = (Z >= top) & (Z <= top + p["thick"])
        rho[mask.ravel()] = p.get("rho_target", 18.0)

    elif kind == "fault_zone":
        line = np.tan(np.radians(p["dip"])) * (X - p["x0"]) + p["z0"]
        dist = np.abs(Z - line) / np.sqrt(1.0 + np.tan(np.radians(p["dip"])) ** 2)
        mask = dist <= p["width"] / 2.0
        rho[mask.ravel()] = p.get("rho_target", 25.0)

    elif kind == "basement":
        boundary = np.tan(np.radians(p["dip"])) * (X - p["x0"]) + p["z0"]
        mask = Z >= boundary
        rho[:] = p.get("rho_above", 180.0)
        rho[mask.ravel()] = p.get("rho_below", 1200.0)
        if p.get("cover", 0.0) > 0:
            rho[(Z <= p["cover"]).ravel()] = p.get("rho_cover", 100.0)

    else:
        raise ValueError(f"Unknown model kind: {kind}")

    return rho, mask.ravel()


def benchmark_cases(quick=True):
    cases = [
        ("clean25", "clean_dip", dict(dip=25.0, x0=38.0, z0=2.0, thick=7.0,
                                      rho_bg=500.0, rho_target=35.0)),
        ("covered25", "covered_dip", dict(dip=25.0, x0=38.0, z0=9.0, thick=6.0,
                                           cover=5.0, rho_bg=700.0,
                                           rho_cover=100.0, rho_target=45.0)),
        ("fault35", "fault_zone", dict(dip=35.0, x0=48.0, z0=1.0, width=8.0,
                                        rho_bg=600.0, rho_target=25.0)),
    ]
    if not quick:
        cases.extend([
            ("clean15", "clean_dip", dict(dip=15.0, x0=38.0, z0=2.0, thick=7.0,
                                          rho_bg=500.0, rho_target=35.0)),
            ("clean35", "clean_dip", dict(dip=35.0, x0=40.0, z0=1.0, thick=7.0,
                                          rho_bg=500.0, rho_target=35.0)),
            ("groundwater20", "groundwater", dict(dip=20.0, x0=42.0, z0=6.0,
                                                  thick=5.0, cover=4.0,
                                                  rho_bg=450.0, rho_cover=130.0,
                                                  rho_target=18.0)),
            ("fault45", "fault_zone", dict(dip=45.0, x0=50.0, z0=0.5, width=8.0,
                                            rho_bg=600.0, rho_target=25.0)),
            ("basement22", "basement", dict(dip=22.0, x0=35.0, z0=8.0,
                                             cover=4.0, rho_above=180.0,
                                             rho_below=1400.0, rho_cover=100.0)),
        ])
    return cases


def add_noise(rho_a, pct=0.03, seed=0):
    rng = np.random.default_rng(seed)
    return np.maximum(rho_a * (1.0 + pct * rng.standard_normal(len(rho_a))), 1.0)


def core_mask(mesh, survey):
    X, Z = grids(mesh)
    return ((X >= survey.electrode_x[0]) &
            (X <= survey.electrode_x[-1]) &
            (Z <= survey.n_max * survey.a * 1.25)).ravel()


def model_metrics(rho_inv, rho_true, target_mask, core):
    li = np.log10(np.clip(rho_inv, 1.0, 1e6))
    lt = np.log10(np.clip(rho_true, 1.0, 1e6))
    cm = core
    tm = target_mask & core
    bg = (~target_mask) & core
    rmse = float(np.sqrt(np.mean((li[cm] - lt[cm]) ** 2)))
    target_rmse = float(np.sqrt(np.mean((li[tm] - lt[tm]) ** 2))) if np.any(tm) else np.nan
    bg_rmse = float(np.sqrt(np.mean((li[bg] - lt[bg]) ** 2))) if np.any(bg) else np.nan
    corr = float(np.corrcoef(li[cm], lt[cm])[0, 1]) if cm.sum() > 3 else np.nan
    return rmse, target_rmse, bg_rmse, corr


def estimate_conductive_dip(rho, mesh, survey, low_pct=25):
    """Weighted centroid tracking for conductive anomalies in the inverted model."""
    r2 = np.asarray(rho).reshape(mesh.ncz, mesh.ncx)
    x_pts, z_pts = [], []
    max_depth = survey.n_max * survey.a * 1.15
    for ix, x in enumerate(mesh.x_cc):
        if x < survey.electrode_x[1] or x > survey.electrode_x[-2]:
            continue
        col = r2[:, ix]
        valid = mesh.z_cc <= max_depth
        if valid.sum() < 4:
            continue
        thresh = np.percentile(col[valid], low_pct)
        sel = valid & (col <= thresh)
        if sel.sum() < 2:
            continue
        w = 1.0 / np.maximum(col[sel], 1.0)
        zc = float(np.sum(mesh.z_cc[sel] * w) / np.sum(w))
        x_pts.append(float(x)); z_pts.append(zc)
    if len(x_pts) < 8:
        return np.nan, 0.0
    x = np.array(x_pts); z = np.array(z_pts)
    slope, intercept = np.polyfit(x, z, 1)
    pred = slope * x + intercept
    ss_res = np.sum((z - pred) ** 2)
    ss_tot = np.sum((z - np.mean(z)) ** 2)
    r2fit = max(0.0, 1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    return float(np.degrees(np.arctan(abs(slope)))), float(r2fit)


def angle_error_deg(estimate, truth):
    if not np.isfinite(estimate):
        return np.nan
    err = abs(float(estimate) - float(truth))
    return min(err, 180.0 - err)


def structure_tensor_field(rho, mesh, smooth_sigma=1.5):
    """Return dip-angle, coherence, and gradient magnitude fields from log rho."""
    arr = np.log(np.clip(np.asarray(rho).reshape(mesh.ncz, mesh.ncx), 0.1, 1e6))
    gx = np.gradient(arr, mesh.x_cc, axis=1)
    gz = np.gradient(arr, mesh.z_cc, axis=0)
    Jxx = gaussian_filter(gx * gx, sigma=smooth_sigma)
    Jzz = gaussian_filter(gz * gz, sigma=smooth_sigma)
    Jxz = gaussian_filter(gx * gz, sigma=smooth_sigma)
    trace = Jxx + Jzz
    det = Jxx * Jzz - Jxz ** 2
    disc = np.sqrt(np.maximum((trace / 2.0) ** 2 - det, 0.0))
    lam1 = trace / 2.0 + disc
    lam2 = trace / 2.0 - disc
    vx = Jxz
    vz = lam1 - Jxx
    norm = np.sqrt(vx ** 2 + vz ** 2) + 1e-10
    flip = vz < 0
    vx = np.where(flip, -vx, vx)
    vz = np.where(flip, -vz, vz)
    theta = np.degrees(np.arctan2(-vx / norm, vz / norm))
    theta = np.abs(theta)
    theta = np.where(theta > 90.0, 180.0 - theta, theta)
    coherence = np.where(lam1 > 1e-10, (lam1 - lam2) / (lam1 + 1e-10), 0.0)
    grad_mag = np.sqrt(gx * gx + gz * gz)
    return theta, coherence, grad_mag


def true_boundary_mask(target_mask, mesh):
    mask = np.asarray(target_mask, dtype=bool).reshape(mesh.ncz, mesh.ncx)
    return binary_dilation(mask, iterations=2) ^ binary_erosion(mask, iterations=1)


def structural_metrics(rho_inv, mesh, survey, target_mask, core, true_dip):
    """Metrics that target geometry rather than pixel-amplitude RMSE."""
    theta, coh, grad = structure_tensor_field(rho_inv, mesh)
    boundary = true_boundary_mask(target_mask, mesh)
    core2 = np.asarray(core, dtype=bool).reshape(mesh.ncz, mesh.ncx)
    eval_mask = boundary & core2
    if eval_mask.sum() < 8:
        return dict(st_dip=np.nan, st_dip_error=np.nan, st_coh=np.nan,
                    boundary_gradient=np.nan, edge_overlap=np.nan)

    # Focus on cells where the inverted model actually contains a structural edge.
    grad_core = grad[core2]
    edge_thr = np.percentile(grad_core, 85) if grad_core.size else 0.0
    oriented = eval_mask & (grad >= edge_thr)
    if oriented.sum() < 8:
        oriented = eval_mask

    weights = grad[oriented] * np.clip(coh[oriented], 0.0, 1.0)
    if np.sum(weights) > 1e-12:
        st_dip = float(np.average(theta[oriented], weights=weights))
    else:
        st_dip = float(np.median(theta[oriented]))
    st_err = angle_error_deg(st_dip, true_dip)

    model_edge = (grad >= edge_thr) & core2
    boundary_dilated = binary_dilation(boundary, iterations=2) & core2
    inter = np.logical_and(model_edge, boundary_dilated).sum()
    denom = model_edge.sum() + boundary_dilated.sum()
    edge_overlap = float(2.0 * inter / denom) if denom > 0 else np.nan

    off_boundary = core2 & (~boundary_dilated)
    bg_grad = np.median(grad[off_boundary]) if np.any(off_boundary) else 0.0
    boundary_gradient = float(np.mean(grad[eval_mask]) / (bg_grad + 1e-12))

    return dict(
        st_dip=st_dip,
        st_dip_error=st_err,
        st_coh=float(np.mean(coh[oriented])),
        boundary_gradient=boundary_gradient,
        edge_overlap=edge_overlap,
    )


def run_l2(survey, mesh, d_obs, ref, max_iter):
    inv = Inversion2D(
        survey, mesh, rho_ref=ref, alpha=1.0, max_iter=max_iter, tol=0.05,
        solver_type="FDM", use_blocks=False, reg_type="L2",
        noise_floor=0.001, pct_error=0.05, target_chi2=1.0, cooling_factor=0.7,
    )
    rho, _, _ = inv.run(d_obs, callback=lambda *_: None, auto_alpha=True)
    return rho


def run_fixed_dip(survey, mesh, d_obs, ref, dip, max_iter):
    inv = Inversion2D(
        survey, mesh, rho_ref=ref, alpha=1.0, max_iter=max_iter, tol=0.05,
        solver_type="FDM", use_blocks=False, reg_type="L2",
        dip_angle=float(dip), dip_weight=6.0,
        noise_floor=0.001, pct_error=0.05, target_chi2=1.0, cooling_factor=0.7,
    )
    rho, _, _ = inv.run(d_obs, callback=lambda *_: None, auto_alpha=True)
    return rho


def run_star(survey, mesh, d_obs, ref, max_iter, gated):
    logs = []
    rho, dips, rms = run_star_inversion(
        survey, mesh, d_obs, ref,
        n_outer=2, st_dip_weight=6.0, st_smooth_sigma=2.5,
        st_coherence_thresh=0.15, use_mgs_init=False,
        max_iter=max_iter, tol=0.05,
        noise_floor=0.001, pct_error=0.05,
        target_chi2=1.0, cooling_factor=0.7,
        callback=logs.append,
        init_dip_method="ensemble",
        use_ood_gate=gated,
    )
    gate_line = next((s for s in logs if "[OOD gate]" in s), "")
    return rho, gate_line


def plot_case(case_name, true_rho, models, inv_mesh, survey):
    fig, axes = plt.subplots(1, len(models) + 1, figsize=(4.0 * (len(models) + 1), 3.2),
                             constrained_layout=True)
    all_vals = [true_rho] + [m[1] for m in models]
    vmin = np.percentile(np.concatenate(all_vals), 2)
    vmax = np.percentile(np.concatenate(all_vals), 98)
    panels = [("True", true_rho)] + models
    for ax, (title, rho) in zip(axes, panels):
        arr = np.asarray(rho).reshape(inv_mesh.ncz, inv_mesh.ncx)
        im = ax.pcolormesh(inv_mesh.x_nodes, inv_mesh.z_nodes, arr,
                           shading="auto", cmap="turbo", vmin=vmin, vmax=vmax)
        ax.invert_yaxis()
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("x (m)")
    axes[0].set_ylabel("depth (m)")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.82, label="Resistivity (ohm m)")
    out = OUTDIR / f"{case_name}_sections.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)


def plot_summary(rows):
    methods = ["L2", "STAR", "STAR_OOD", "FixedTrueDip"]
    colors = ["0.55", "#c77cff", "#00a087", "#4c78a8"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for ax, metric, ylabel in [
        (axes[0, 0], "log_rmse", "Core log10 RMSE"),
        (axes[0, 1], "target_rmse", "Target-zone log10 RMSE"),
        (axes[0, 2], "dip_error", "Centroid dip error (deg)"),
        (axes[1, 0], "st_dip_error", "Boundary ST dip error (deg)"),
        (axes[1, 1], "edge_overlap", "Edge overlap Dice"),
        (axes[1, 2], "boundary_gradient", "Boundary sharpness ratio"),
    ]:
        vals = []
        for m in methods:
            mm = [float(r[metric]) for r in rows
                  if r["method"] == m and r[metric] not in ("", "nan")]
            vals.append(np.nanmean(mm) if mm else np.nan)
        ax.bar(methods, vals, color=colors, edgecolor="k", linewidth=0.8)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("STAR JAG benchmark: independent forward/inversion mesh", fontsize=13)
    out = OUTDIR / "STAR_JAG_benchmark_summary.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)


def main():
    quick = os.environ.get("STAR_BENCH_FULL", "0") != "1"
    max_iter = 4 if quick else 7
    noise = 0.03
    survey = make_survey()
    fwd_mesh = Mesh2D(survey, depth_factor=2.5, dx_factor=0.25)
    inv_mesh = Mesh2D(survey, depth_factor=2.5, dx_factor=0.50)
    core = core_mask(inv_mesh, survey)

    print("=" * 72)
    print("STAR JAG benchmark")
    print(f"  mode={'quick' if quick else 'full'}  cases={len(benchmark_cases(quick))}  max_iter={max_iter}")
    print("  forward mesh dx_factor=0.25, inversion mesh dx_factor=0.50")
    print("=" * 72)

    rows = []
    for ci, (name, kind, params) in enumerate(benchmark_cases(quick), start=1):
        print(f"\n[{ci}] {name} ({kind}, true dip={params.get('dip', '')})")
        rho_fwd, _ = model_and_mask(fwd_mesh, kind, params)
        rho_true_inv, target_inv = model_and_mask(inv_mesh, kind, params)
        d_clean = ForwardSolver(fwd_mesh, rho_fwd).compute_data(survey, callback=lambda *_: None)
        d_obs = add_noise(d_clean, pct=noise, seed=100 + ci)
        ref = float(np.mean(d_obs))

        method_models = []
        methods = [
            ("L2", lambda: (run_l2(survey, inv_mesh, d_obs, ref, max_iter), "")),
            ("STAR", lambda: run_star(survey, inv_mesh, d_obs, ref, max_iter, gated=False)),
            ("STAR_OOD", lambda: run_star(survey, inv_mesh, d_obs, ref, max_iter, gated=True)),
            ("FixedTrueDip", lambda: (run_fixed_dip(survey, inv_mesh, d_obs, ref, params["dip"], max_iter), "")),
        ]
        for method, runner in methods:
            print(f"  - {method}")
            rho_inv, note = runner()
            rmse, trmse, brmse, corr = model_metrics(rho_inv, rho_true_inv, target_inv, core)
            est_dip, fit_r2 = estimate_conductive_dip(rho_inv, inv_mesh, survey)
            dip_err = abs(est_dip - params["dip"]) if np.isfinite(est_dip) else np.nan
            sm = structural_metrics(rho_inv, inv_mesh, survey, target_inv, core, params["dip"])
            rows.append({
                "case": name,
                "kind": kind,
                "true_dip": params["dip"],
                "method": method,
                "log_rmse": rmse,
                "target_rmse": trmse,
                "background_rmse": brmse,
                "log_corr": corr,
                "estimated_dip": est_dip,
                "dip_fit_R2": fit_r2,
                "dip_error": dip_err,
                "st_dip": sm["st_dip"],
                "st_dip_error": sm["st_dip_error"],
                "st_coherence": sm["st_coh"],
                "boundary_gradient": sm["boundary_gradient"],
                "edge_overlap": sm["edge_overlap"],
                "note": note,
            })
            method_models.append((method, rho_inv))
            print(f"    logRMSE={rmse:.3f}, target={trmse:.3f}, corr={corr:.3f}, "
                  f"centroidDip={est_dip:.1f} (err={dip_err:.1f}, R2={fit_r2:.2f}), "
                  f"STdip={sm['st_dip']:.1f} (err={sm['st_dip_error']:.1f}), "
                  f"edge={sm['edge_overlap']:.2f}")
            if note:
                print(f"    {note}")
        if ci == 1:
            plot_case(name, rho_true_inv, method_models, inv_mesh, survey)

    csv_path = OUTDIR / "STAR_JAG_benchmark_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    plot_summary(rows)

    print("\nSummary by method")
    for method in ["L2", "STAR", "STAR_OOD", "FixedTrueDip"]:
        rr = [r for r in rows if r["method"] == method]
        print(f"  {method:12s}  logRMSE={np.mean([r['log_rmse'] for r in rr]):.3f}  "
              f"targetRMSE={np.mean([r['target_rmse'] for r in rr]):.3f}  "
              f"centroidErr={np.nanmean([r['dip_error'] for r in rr]):.2f}°  "
              f"STerr={np.nanmean([r['st_dip_error'] for r in rr]):.2f}°  "
              f"edgeDice={np.nanmean([r['edge_overlap'] for r in rr]):.2f}")
    print(f"\nSaved: {csv_path}")
    print(f"Saved: {OUTDIR / 'STAR_JAG_benchmark_summary.png'}")


if __name__ == "__main__":
    main()
