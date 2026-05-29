#!/usr/bin/env python3
"""
run_occam_comparison.py
=======================
Occam L2 inversion + structure tensor dip estimation comparison.

Compares three dip-estimation methods on the same synthetic test cases:
  1. Diagnostic interpreter  (Diag) — pseudosection ST + ML [existing]
  2. Forward-hypothesis matching   (FWD) — template library [existing]
  3. Occam L2 inversion + model tensor (Occam) — THIS SCRIPT

Table 1 cases  : 3 known-dip groups (5 groups × 2-3 cases)
Table 2 cases  : 5 independent geometries (different offsets/params from Table 1)

Results saved to /tmp/occam_comparison_results.txt
"""

import sys
import os

sys.path.insert(0, '')

import matplotlib
matplotlib.use('Agg')

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata

# ── RESIS_Pro imports ──
from RESIS_Pro import (
    DipDipSurvey, Mesh2D, ForwardSolver, Inversion2D, filter_bad_data
)

# ── Survey / mesh constants ──
A = 5.0
N_ELEC = 30
N_MAX = 6

RESULTS_PATH = '/tmp/occam_comparison_results.txt'


# ════════════════════════════════════════════════════════════
#  Survey and mesh factory
# ════════════════════════════════════════════════════════════

def make_survey_mesh():
    elec_x = np.arange(N_ELEC) * A
    survey = DipDipSurvey(
        a=A, n_electrodes=N_ELEC, n_max=N_MAX,
        electrode_x=elec_x, array_type='dipole-dipole',
    )
    mesh = Mesh2D(survey, depth_factor=2.5, dx_factor=0.25)
    return survey, mesh


# ════════════════════════════════════════════════════════════
#  Model builders (mirror geo_synthetic_benchmark.py exactly)
# ════════════════════════════════════════════════════════════

def cell_grids(mesh):
    X, Z = np.meshgrid(mesh.x_cc, mesh.z_cc)
    return X, Z


def _new_rho(mesh, value=500.0):
    return np.full(mesh.n_cells, float(value))


def add_dipping_band(rho, mesh, dip_deg, x0, z0, thickness, value):
    X, Z = cell_grids(mesh)
    top = np.tan(np.radians(dip_deg)) * (X - x0) + z0
    mask = (Z >= top) & (Z <= top + thickness)
    rho[mask.ravel()] = value
    return rho


def add_dipping_interface(rho, mesh, dip_deg, x0, z0, shallow_value, deep_value):
    X, Z = cell_grids(mesh)
    boundary = np.tan(np.radians(dip_deg)) * (X - x0) + z0
    rho[:] = shallow_value
    rho[(Z >= boundary).ravel()] = deep_value
    return rho


def add_fault_zone(rho, mesh, dip_deg, x0, z0, width, value):
    X, Z = cell_grids(mesh)
    line = np.tan(np.radians(dip_deg)) * (X - x0) + z0
    dist = np.abs(Z - line) / np.sqrt(1 + np.tan(np.radians(dip_deg))**2)
    mask = dist <= width / 2
    rho[mask.ravel()] = value
    return rho


def add_alluvium(rho, mesh, thickness, value):
    _, Z = cell_grids(mesh)
    mask = Z <= thickness
    rho[mask.ravel()] = value
    return rho


# ── Table 1 model builders ──

def build_clean_dip(mesh, dip, x0=35, z0=3, thickness=7,
                     rho_layer=45, rho_bg=500):
    rho = _new_rho(mesh, rho_bg)
    add_dipping_band(rho, mesh, dip, x0=x0, z0=z0,
                     thickness=thickness, value=rho_layer)
    return rho


def build_covered_dip(mesh, dip, x0=35, z0=10, thickness=6,
                       rho_layer=60, rho_bg=700):
    rho = _new_rho(mesh, rho_bg)
    add_dipping_interface(rho, mesh, dip, x0=x0, z0=z0,
                          shallow_value=300, deep_value=900)
    add_dipping_band(rho, mesh, dip, x0=x0, z0=z0+2,
                     thickness=thickness, value=rho_layer)
    add_alluvium(rho, mesh, thickness=6, value=90)
    return rho


def build_groundwater_dip(mesh, dip, x0=42, z0=6, width=5,
                           rho_gw=18, rho_bg=450):
    rho = _new_rho(mesh, rho_bg)
    add_dipping_band(rho, mesh, dip, x0=x0, z0=z0,
                     thickness=width, value=rho_gw)
    add_alluvium(rho, mesh, thickness=4, value=130)
    return rho


def build_fault_zone(mesh, dip, x0=48, z0=1.5, width=7,
                      rho_fault=25, rho_bg=600):
    rho = _new_rho(mesh, rho_bg)
    add_fault_zone(rho, mesh, dip, x0=x0, z0=z0,
                   width=width, value=rho_fault)
    return rho


def build_basement_dip(mesh, dip, x0=35, z0=9,
                        rho_above=180, rho_below=1500):
    rho = _new_rho(mesh, 250)
    add_dipping_interface(rho, mesh, dip, x0=x0, z0=z0,
                          shallow_value=rho_above, deep_value=rho_below)
    add_alluvium(rho, mesh, thickness=4, value=100)
    return rho


# ── Table 2 independent model builders ──

def build_indep_clean(mesh, dip=28):
    rho = _new_rho(mesh, 620)
    add_dipping_band(rho, mesh, dip, x0=51, z0=4.5, thickness=9, value=38)
    return rho


def build_indep_covered(mesh, dip=32):
    rho = _new_rho(mesh, 760)
    add_dipping_interface(rho, mesh, dip, x0=42, z0=12.5,
                          shallow_value=280, deep_value=980)
    add_dipping_band(rho, mesh, dip, x0=42, z0=15, thickness=7, value=55)
    add_alluvium(rho, mesh, thickness=7.2, value=85)
    return rho


def build_indep_groundwater(mesh, dip=18):
    rho = _new_rho(mesh, 430)
    add_dipping_band(rho, mesh, dip, x0=57, z0=7.5, thickness=6.5, value=16)
    add_alluvium(rho, mesh, thickness=5.2, value=125)
    return rho


def build_indep_fault(mesh, dip=40):
    rho = _new_rho(mesh, 650)
    add_fault_zone(rho, mesh, dip, x0=54, z0=2.2, width=9.0, value=28)
    return rho


def build_indep_basement(mesh, dip=18):
    rho = _new_rho(mesh, 260)
    add_dipping_interface(rho, mesh, dip, x0=49, z0=11.5,
                          shallow_value=170, deep_value=1350)
    add_alluvium(rho, mesh, thickness=5.5, value=95)
    return rho


# ════════════════════════════════════════════════════════════
#  Forward data generator
# ════════════════════════════════════════════════════════════

def forward_data(survey, mesh, rho, noise=0.03, seed=42):
    solver = ForwardSolver(mesh, rho)
    rho_a = solver.compute_data(survey, callback=lambda i, n: None)
    rng = np.random.default_rng(seed)
    rho_a = rho_a * (1.0 + noise * rng.standard_normal(len(rho_a)))
    return np.maximum(rho_a, 1.0)


# ════════════════════════════════════════════════════════════
#  Occam L2 block inversion
# ════════════════════════════════════════════════════════════

def run_occam_inversion(survey, mesh, rho_a, verbose=False):
    """
    Run Occam (L2 block) inversion on apparent resistivity data.

    Returns
    -------
    rho_inv  : array, length = mesh.n_cells (block values mapped back to cells)
    inv      : Inversion2D object (to access inv.inv_blocks.blocks)
    """
    ref = float(np.mean(rho_a))

    cb = (lambda msg: print(f'    {msg}')) if verbose else None

    inv = Inversion2D(
        survey, mesh,
        rho_ref=ref,
        alpha=1.0,
        max_iter=5,
        tol=0.05,
        solver_type='FDM',
        use_blocks=True,
        reg_type='L2',
        noise_floor=0.001,
        pct_error=0.05,
        target_chi2=1.0,
        cooling_factor=0.7,
    )

    rho_inv, hist, dcalc = inv.run(
        rho_a,
        callback=cb,
        auto_alpha=True,
        robust=False,
    )

    return rho_inv, inv


# ════════════════════════════════════════════════════════════
#  Structure tensor dip extraction from inverted model
# ════════════════════════════════════════════════════════════

def occam_dip_from_model(rho_inv, inv, mesh, smooth_sigma=2.0):
    """
    Extract dominant dip angle from the Occam-inverted resistivity model.

    Strategy: fit a line through the anomalous block centres in (x, z) space.
    This is robust for the block-mode Occam model where each block has a
    single resistivity value and the dipping anomaly creates a spatially
    shifting low-resistivity pattern across depth rows.

    Three sub-methods are tried and a coherence-weighted estimate returned:

    A. **Anomaly-centre tracking** — for each depth row, find the x-centroid
       of the most anomalous blocks; regress cz vs cx to get slope → dip.

    B. **Weighted PCA on block centres** — blocks weighted by |log_rho - median|;
       PCA first axis gives the primary trend direction.

    C. **Structure tensor on the gridded block values** — interpolate block
       values to a regular x-z grid and apply the standard tensor.

    The final estimate is the weighted median of the three.

    Parameters
    ----------
    rho_inv   : array (mesh.n_cells,)  Inverted resistivity (linear scale).
    inv       : Inversion2D
    mesh      : Mesh2D
    smooth_sigma : float

    Returns
    -------
    dip_deg : float   (0 = horizontal, positive = dipping)
    coherence : float (quality indicator)
    """
    ib = inv.inv_blocks

    if ib is None:
        # Cell mode: use built-in ST
        m_log = np.log(np.maximum(rho_inv, 1.0))
        dip_st, coh = inv.estimate_dip_from_model(m_log, smooth_sigma=smooth_sigma)
        return abs(dip_st), coh

    # ── Extract per-block resistivity ──
    block_rho = np.array([
        rho_inv[ib.cell_to_block == bi].mean()
        if (ib.cell_to_block == bi).any()
        else float(np.median(rho_inv))
        for bi in range(ib.n_blocks)
    ])
    cx_arr = np.array([b['cx'] for b in ib.blocks])
    cz_arr = np.array([b['cz'] for b in ib.blocks])
    row_arr = np.array([b['row'] for b in ib.blocks])
    log_rho = np.log(np.maximum(block_rho, 1.0))
    med_log = np.median(log_rho)

    # ── Method A: Row-by-row anomaly centre tracking ──
    anomaly_sign = 1 if med_log > np.mean(log_rho) else -1  # conductive or resistive anomaly
    row_cx = []
    row_cz = []
    n_rows = int(row_arr.max()) + 1
    for r in range(n_rows):
        mask_r = row_arr == r
        if mask_r.sum() < 2:
            continue
        cx_r = cx_arr[mask_r]
        cz_r = cz_arr[mask_r]
        lr_r = log_rho[mask_r]
        # anomaly weight: deviation from median (always positive weight)
        w_r = np.abs(lr_r - med_log)
        if w_r.sum() < 1e-10:
            continue
        x_centroid = float(np.sum(cx_r * w_r) / np.sum(w_r))
        z_centroid = float(np.mean(cz_r))
        row_cx.append(x_centroid)
        row_cz.append(z_centroid)

    if len(row_cx) >= 3:
        # Fit line: z = m*x + b → slope = dz/dx → dip = arctan(|slope|)
        p = np.polyfit(row_cx, row_cz, 1)
        dip_track = float(np.degrees(np.arctan(abs(p[0]))))
        ss_res = np.sum((np.array(row_cz) - np.polyval(p, row_cx))**2)
        ss_tot = np.sum((np.array(row_cz) - np.mean(row_cz))**2)
        r2_track = max(0.0, 1.0 - ss_res / (ss_tot + 1e-10))
    else:
        dip_track = 0.0
        r2_track = 0.0

    # ── Method B: Weighted PCA on block centres ──
    weights_pca = np.abs(log_rho - med_log)
    w_max = weights_pca.max()
    if w_max > 1e-6:
        weights_pca /= w_max
    else:
        weights_pca = np.ones_like(weights_pca)

    X_bl = np.column_stack([cx_arr, cz_arr])
    mu = np.average(X_bl, axis=0, weights=weights_pca)
    Xc = (X_bl - mu) * np.sqrt(weights_pca[:, np.newaxis])
    _, S_pca, Vt = np.linalg.svd(Xc, full_matrices=False)
    pc1 = Vt[0]  # direction of maximum spread
    dip_pca = float(np.degrees(np.arctan2(abs(pc1[1]), abs(pc1[0]))))
    # PCA coherence: ratio of first to second singular value
    pca_coh = float(S_pca[0] / max(S_pca[1], 1e-10)) if len(S_pca) > 1 else 1.0

    # ── Method C: Structure tensor on interpolated block grid ──
    x1 = float(cx_arr.min()); x2 = float(cx_arr.max())
    z1 = float(cz_arr.min()); z2 = float(cz_arr.max())
    xi = np.linspace(x1, x2, max(int((x2 - x1) / A) + 1, 10))
    zi = np.linspace(z1, z2, max(n_rows, 4))
    XI, ZI = np.meshgrid(xi, zi)
    try:
        grid = griddata((cx_arr, cz_arr), log_rho, (XI, ZI), method='linear')
        nm = np.isnan(grid)
        if nm.any():
            grid[nm] = griddata((cx_arr, cz_arr), log_rho,
                                 (XI[nm], ZI[nm]), method='nearest')
    except Exception:
        grid = griddata((cx_arr, cz_arr), log_rho, (XI, ZI), method='nearest')

    gz, gx = np.gradient(grid)
    sig = max(smooth_sigma, 1.0)
    Jxx = gaussian_filter(gx * gx, sigma=sig)
    Jxz = gaussian_filter(gx * gz, sigma=sig)
    Jzz = gaussian_filter(gz * gz, sigma=sig)
    Txx = float(np.nanmean(Jxx))
    Tzz = float(np.nanmean(Jzz))
    Txz = float(np.nanmean(Jxz))
    trace = Txx + Tzz
    det = Txx * Tzz - Txz * Txz
    disc = max(trace * trace * 0.25 - det, 0.0)
    lam_max = trace * 0.5 + np.sqrt(disc)
    lam_min = max(trace * 0.5 - np.sqrt(disc), 1e-30)
    if abs(Txx - Tzz) < 1e-30 and abs(Txz) < 1e-30:
        theta_grad_rad = 0.0
    else:
        theta_grad_rad = 0.5 * np.arctan2(2.0 * Txz, Txx - Tzz)
    feature_rad = theta_grad_rad + np.pi * 0.5
    feature_deg = float(np.degrees(feature_rad))
    while feature_deg > 90:
        feature_deg -= 180
    while feature_deg < -90:
        feature_deg += 180
    dip_st = abs(feature_deg)
    coh_st = float(np.sqrt(lam_max / lam_min))

    # ── Combine: weighted average using quality metrics ──
    # r2_track and pca_coh are quality indicators for each method.
    # ST coherence is not reliable on coarse block grids — use it only if high.
    w_a = float(r2_track)            # row-tracking: weight by R²
    w_b = float(min(pca_coh / 10.0, 1.0))  # PCA: clamp to [0,1]
    w_c = float(max(coh_st - 1.0, 0.0) / 5.0)  # ST: meaningful only when high

    total_w = w_a + w_b + w_c
    if total_w < 1e-6:
        # All methods uncertain — use simple average
        dip_final = (dip_track + dip_pca + dip_st) / 3.0
        coh_final = 1.0
    else:
        dip_final = (w_a * dip_track + w_b * dip_pca + w_c * dip_st) / total_w
        coh_final = total_w / (w_a + w_b + w_c + 1e-10) * max(r2_track, pca_coh / 10.0)

    return float(dip_final), float(coh_st)


# ════════════════════════════════════════════════════════════
#  PCA-based fallback dip estimator (for noisy block models)
# ════════════════════════════════════════════════════════════

def pca_dip_from_blocks(rho_inv, inv, threshold_pct=50):
    """
    Simple PCA on the block centres weighted by |log_rho - median|.
    Returns absolute dip angle in degrees.
    """
    ib = inv.inv_blocks
    if ib is None:
        return 0.0

    cx_arr = np.array([b['cx'] for b in ib.blocks])
    cz_arr = np.array([b['cz'] for b in ib.blocks])

    # Get per-block resistivity from cell rho via lookup
    block_rho = np.array([
        rho_inv[ib.cell_to_block == bi].mean()
        if (ib.cell_to_block == bi).any()
        else np.median(rho_inv)
        for bi in range(ib.n_blocks)
    ])

    log_rho = np.log(np.maximum(block_rho, 1.0))
    med = np.median(log_rho)
    weights = np.abs(log_rho - med)
    weights = weights / (weights.max() + 1e-10)

    # Stack weighted centres
    X = np.column_stack([cx_arr, cz_arr])
    w = weights[:, np.newaxis]
    mu = np.average(X, axis=0, weights=weights)
    Xc = (X - mu) * np.sqrt(w)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    # First principal component (largest singular value)
    pc1 = Vt[0]  # [dx, dz]
    dip_rad = np.arctan2(abs(pc1[1]), abs(pc1[0]))
    return float(np.degrees(dip_rad))


# ════════════════════════════════════════════════════════════
#  Diagnostic interpreter (Diag) — pseudosection ST
# ════════════════════════════════════════════════════════════

def pseudosection_ST_dip(survey_obj, rho_a, smooth_sigma=2.5, coh_thresh=0.10):
    """
    Pseudosection structure tensor dip estimate (RESIS_Pro validated algorithm).
    Returns median absolute dip in degrees.
    """
    measurements = survey_obj.measurements
    rho_a = np.asarray(rho_a)
    xs = np.array([m['x'] for m in measurements])
    zs = np.array([m['z'] for m in measurements])
    log_ra = np.log10(np.maximum(rho_a, 1.0))

    x1 = survey_obj.electrode_x[0]
    x2 = survey_obj.electrode_x[-1]
    z_max = survey_obj.n_max * survey_obj.a
    xi = np.linspace(x1, x2, 80)
    zi = np.linspace(0, z_max, 40)
    XI, ZI = np.meshgrid(xi, zi)
    grid = griddata((xs, zs), log_ra, (XI, ZI), method='linear')
    nm = np.isnan(grid)
    if nm.any():
        grid[nm] = griddata((xs, zs), log_ra, (XI[nm], ZI[nm]), method='nearest')

    dx = np.gradient(grid, xi, axis=1)
    dz = np.gradient(grid, zi, axis=0)
    sig = max(smooth_sigma, 2.0)
    Jxx = gaussian_filter(dx * dx, sigma=sig)
    Jzz = gaussian_filter(dz * dz, sigma=sig)
    Jxz = gaussian_filter(dx * dz, sigma=sig)

    trace = Jxx + Jzz
    det = Jxx * Jzz - Jxz ** 2
    disc = np.sqrt(np.maximum((trace / 2) ** 2 - det, 0.0))
    lam1 = trace / 2 + disc
    lam2 = trace / 2 - disc

    vx = Jxz
    vz = lam1 - Jxx
    norm = np.sqrt(vx ** 2 + vz ** 2) + 1e-10
    flip = vz < 0
    vx_c = np.where(flip, -vx, vx)
    vz_c = np.where(flip, -vz, vz)
    theta_grid = np.arctan2(-vx_c / norm, vz_c / norm)

    coh = np.where(lam1 > 1e-10, (lam1 - lam2) / (lam1 + 1e-10), 0.0)
    scale = np.clip((coh - coh_thresh) / (1.0 - coh_thresh), 0.0, 1.0)

    core = np.degrees(theta_grid) * scale
    nonzero = core[np.abs(core) > 0.5]
    if len(nonzero) == 0:
        return 0.0
    return float(np.median(np.abs(nonzero)))


# ════════════════════════════════════════════════════════════
#  Forward-hypothesis matching (simplified weighted-dip)
# ════════════════════════════════════════════════════════════

def fwd_dip_estimate(survey_obj, rho_a):
    """
    Use the cached template library to estimate dip angle.
    Falls back to None if cache unavailable.
    """
    try:
        import pickle
        CACHE_PATH = 'geo_template_cache.pkl'
        with open(CACHE_PATH, 'rb') as f:
            cache = pickle.load(f)

        d_obs = _normalize_rho(rho_a)
        SOFTMAX_T = 0.08
        LAMBDA = 0.3
        scores = []
        dips = []
        for entry in cache:
            d_tmpl = entry['rho_a_norm']
            n = len(d_obs)
            if len(d_tmpl) != n:
                idx_i = np.linspace(0, len(d_tmpl) - 1, n)
                d_tmpl = np.interp(idx_i, np.arange(len(d_tmpl)), d_tmpl)
            r = float(np.corrcoef(d_obs, d_tmpl)[0, 1])
            if not np.isfinite(r):
                r = -1.0
            nrmse = float(np.linalg.norm(d_obs - d_tmpl) /
                          (np.linalg.norm(d_obs) + 1e-9))
            scores.append(r - LAMBDA * nrmse)
            dips.append(entry.get('dip_deg', 0.0) or 0.0)

        scores = np.array(scores, dtype=float)
        dips = np.array(dips, dtype=float)
        s_shift = scores - scores.max()
        exp_s = np.exp(s_shift / SOFTMAX_T)
        weights = exp_s / (exp_s.sum() + 1e-30)

        # weighted mean dip (only entries with dip_deg > 0)
        valid = dips > 0
        if valid.sum() == 0:
            return 0.0
        w_valid = weights[valid]
        d_valid = dips[valid]
        return float(np.sum(w_valid * d_valid) / np.sum(w_valid))

    except Exception as e:
        print(f'  [FWD] cache error: {e}')
        return None


def _normalize_rho(vec):
    v = np.log10(np.maximum(vec, 1e-6))
    mu = v.mean()
    sigma = v.std()
    if sigma < 1e-9:
        return v - mu
    return (v - mu) / sigma


# ════════════════════════════════════════════════════════════
#  Case runner
# ════════════════════════════════════════════════════════════

def run_case(name, true_dip, rho_model, survey_obj, mesh, noise=0.03, seed=42):
    """
    Run all three methods for a single case.
    Returns dict with dip estimates and errors.
    """
    print(f'\n  [{name}] true={true_dip}°', flush=True)

    # Forward data with noise
    rho_a = forward_data(survey_obj, mesh, rho_model, noise=noise, seed=seed)

    # --- Method 1: Diag (pseudosection ST) ---
    diag_dip = pseudosection_ST_dip(survey_obj, rho_a)
    diag_err = abs(diag_dip - true_dip)
    print(f'    Diag={diag_dip:.1f}° (err={diag_err:.1f}°)', flush=True)

    # --- Method 2: FWD (template matching) ---
    fwd_dip = fwd_dip_estimate(survey_obj, rho_a)
    if fwd_dip is None:
        fwd_err = None
        fwd_str = 'N/A'
    else:
        fwd_err = abs(fwd_dip - true_dip)
        fwd_str = f'{fwd_dip:.1f}° (err={fwd_err:.1f}°)'
    print(f'    FWD={fwd_str}', flush=True)

    # --- Method 3: Occam L2 block inversion ---
    try:
        rho_inv, inv_obj = run_occam_inversion(survey_obj, mesh, rho_a, verbose=False)
        occam_dip, coh = occam_dip_from_model(rho_inv, inv_obj, mesh)
        occam_err = abs(occam_dip - true_dip)
        print(f'    Occam={occam_dip:.1f}° (coh={coh:.2f}, err={occam_err:.1f}°)', flush=True)
    except Exception as e:
        print(f'    Occam ERROR: {e}', flush=True)
        import traceback; traceback.print_exc()
        occam_dip = None
        occam_err = None

    return {
        'name': name,
        'true_dip': true_dip,
        'diag_dip': diag_dip,
        'diag_err': diag_err,
        'fwd_dip': fwd_dip,
        'fwd_err': fwd_err,
        'occam_dip': occam_dip,
        'occam_err': occam_err,
    }


# ════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════

def main():
    import time
    t0 = time.time()

    print('=' * 70)
    print('  Occam L2 Inversion Dip Comparison')
    print('  Survey: N_ELEC=30, a=5m, N_MAX=6')
    print('=' * 70)

    global survey, mesh  # make accessible to occam_dip_from_model
    survey, mesh = make_survey_mesh()

    lines = []
    lines.append('=' * 70)
    lines.append('Occam L2 Inversion Dip Comparison — Results')
    lines.append('=' * 70)

    # ─────────────────────────────────────────────────────
    #  TABLE 1 CASES
    # ─────────────────────────────────────────────────────
    print('\n' + '─' * 50)
    print('TABLE 1 CASES')
    print('─' * 50)
    lines.append('\n--- TABLE 1 CASES ---')

    table1_cases = []

    # 1a. Clean dipping layer (dip=15, 25, 35)
    for dip in (15, 25, 35):
        rho = build_clean_dip(mesh, dip)
        r = run_case(f'Clean {dip}°', dip, rho, survey, mesh, noise=0.03, seed=42)
        table1_cases.append(r)

    # 1b. Covered dipping layer (dip=15, 25, 35)
    for dip in (15, 25, 35):
        rho = build_covered_dip(mesh, dip)
        r = run_case(f'Covered {dip}°', dip, rho, survey, mesh, noise=0.03, seed=42)
        table1_cases.append(r)

    # 1c. Groundwater (dip=10, 20, 30)
    for dip in (10, 20, 30):
        rho = build_groundwater_dip(mesh, dip)
        r = run_case(f'Groundwater {dip}°', dip, rho, survey, mesh, noise=0.03, seed=42)
        table1_cases.append(r)

    # 1d. Fault zone (dip=30, 45, 60)
    for dip in (30, 45, 60):
        rho = build_fault_zone(mesh, dip)
        r = run_case(f'Fault {dip}°', dip, rho, survey, mesh, noise=0.03, seed=42)
        table1_cases.append(r)

    # 1e. Resistive basement (dip=12, 22)
    for dip in (12, 22):
        rho = build_basement_dip(mesh, dip)
        r = run_case(f'Basement {dip}°', dip, rho, survey, mesh, noise=0.03, seed=42)
        table1_cases.append(r)

    # ─────────────────────────────────────────────────────
    #  TABLE 2 CASES (independent, 3 % noise)
    # ─────────────────────────────────────────────────────
    print('\n' + '─' * 50)
    print('TABLE 2 INDEPENDENT CASES')
    print('─' * 50)
    lines.append('\n--- TABLE 2 INDEPENDENT CASES ---')

    table2_specs = [
        ('Indep Clean 28°',        28, build_indep_clean),
        ('Indep Covered 32°',      32, build_indep_covered),
        ('Indep Groundwater 18°',  18, build_indep_groundwater),
        ('Indep Fault 40°',        40, build_indep_fault),
        ('Indep Basement 18°',     18, build_indep_basement),
    ]

    table2_cases = []
    for name, true_dip, builder in table2_specs:
        rho = builder(mesh, true_dip)
        r = run_case(name, true_dip, rho, survey, mesh, noise=0.03, seed=101)
        table2_cases.append(r)

    # ─────────────────────────────────────────────────────
    #  Summary table
    # ─────────────────────────────────────────────────────

    def fmt_row(r):
        diag_s = f"{r['diag_dip']:.1f}°"
        fwd_s  = f"{r['fwd_dip']:.1f}°"  if r['fwd_dip']   is not None else 'N/A'
        occ_s  = f"{r['occam_dip']:.1f}°" if r['occam_dip'] is not None else 'N/A'
        return f"  {r['name']:<26} true={r['true_dip']:>3}°  Diag={diag_s:>6}  FWD={fwd_s:>6}  Occam={occ_s:>6}"

    header = f"  {'Case':<26} {'true':>7}  {'Diag':>10}  {'FWD':>9}  {'Occam':>9}"
    sep = '  ' + '-' * 65

    print('\n' + '=' * 70)
    print('SUMMARY TABLE')
    print('=' * 70)
    print(header)
    print(sep)

    lines.append('\nSUMMARY TABLE')
    lines.append(header)
    lines.append(sep)

    for r in table1_cases:
        row = fmt_row(r)
        print(row)
        lines.append(row)

    lines.append(sep)
    for r in table2_cases:
        row = fmt_row(r)
        print(row)
        lines.append(row)

    # ─────────────────────────────────────────────────────
    #  MAE calculation
    # ─────────────────────────────────────────────────────

    def mae(cases, key):
        errs = [r[key] for r in cases if r[key] is not None]
        return float(np.mean(errs)) if errs else float('nan')

    t1_diag_mae   = mae(table1_cases, 'diag_err')
    t1_fwd_mae    = mae(table1_cases, 'fwd_err')
    t1_occam_mae  = mae(table1_cases, 'occam_err')

    t2_diag_mae   = mae(table2_cases, 'diag_err')
    t2_fwd_mae    = mae(table2_cases, 'fwd_err')
    t2_occam_mae  = mae(table2_cases, 'occam_err')

    mae1_line = (f"Table1 MAE:  Diag={t1_diag_mae:.2f}°  "
                 f"FWD={t1_fwd_mae:.2f}°  Occam={t1_occam_mae:.2f}°")
    mae2_line = (f"Table2 MAE:  Diag={t2_diag_mae:.2f}°  "
                 f"FWD={t2_fwd_mae:.2f}°  Occam={t2_occam_mae:.2f}°")

    print()
    print(mae1_line)
    print(mae2_line)
    print(f'\nTotal elapsed: {time.time()-t0:.0f} s')

    lines.append('')
    lines.append(mae1_line)
    lines.append(mae2_line)
    lines.append(f'\nTotal elapsed: {time.time()-t0:.0f} s')

    # ─────────────────────────────────────────────────────
    #  Per-case error detail
    # ─────────────────────────────────────────────────────
    lines.append('\n--- PER-CASE ERRORS (absolute, degrees) ---')
    err_hdr = f"  {'Case':<26} {'Diag':>8}  {'FWD':>8}  {'Occam':>8}"
    lines.append(err_hdr)
    lines.append('  ' + '-' * 53)
    for r in table1_cases + table2_cases:
        de = f"{r['diag_err']:.1f}" if r['diag_err']  is not None else 'N/A'
        fe = f"{r['fwd_err']:.1f}"  if r['fwd_err']   is not None else 'N/A'
        oe = f"{r['occam_err']:.1f}" if r['occam_err'] is not None else 'N/A'
        lines.append(f"  {r['name']:<26} {de:>8}  {fe:>8}  {oe:>8}")

    # Save results
    with open(RESULTS_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'\nResults saved to {RESULTS_PATH}')

    return {
        'table1': table1_cases,
        'table2': table2_cases,
        'table1_mae': {'diag': t1_diag_mae, 'fwd': t1_fwd_mae, 'occam': t1_occam_mae},
        'table2_mae': {'diag': t2_diag_mae, 'fwd': t2_fwd_mae, 'occam': t2_occam_mae},
    }


if __name__ == '__main__':
    main()
