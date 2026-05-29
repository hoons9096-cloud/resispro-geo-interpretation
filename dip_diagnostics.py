#!/usr/bin/env python3
"""
경사각 자동 추정 — 진단 도구 모음 (Phase 1)

역산 없이 의사단면도만으로 빠르게 다양한 방법으로 경사 추정 + 신뢰도 평가.

방법:
  M1: 의사단면도 ST (구조 텐서, 단일 스케일)
  M2: n-레벨 이상체 중심추적 (선형 회귀)
  M3: n-레벨 이상체 폭/깊이비 → 종횡비 자동 분류
  M4: 다중 스케일 ST (sigma=1.5, 2.5, 4.0, 6.0)
  M5: 저비저항 등치선 추적 (contour 평균 기울기)

각 방법에 신뢰도 지표 동봉.
"""
import sys, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, '')
from RESIS_Pro import DipDipSurvey, Mesh2D, ForwardSolver

plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False
OUTDIR = ''

A = 5.0; N_ELEC = 30


# ════════════════════════════════════════════════════════
# 공통 유틸: 의사단면도 정규 격자 보간
# ════════════════════════════════════════════════════════
def build_pseudosection_grid(survey, rho_a, nx=80, nz=40):
    measurements = survey.measurements
    xs = np.array([m['x'] for m in measurements])
    zs = np.array([m['z'] for m in measurements])
    log_ra = np.log10(np.maximum(np.asarray(rho_a), 1.0))

    x1 = survey.electrode_x[0]; x2 = survey.electrode_x[-1]
    z_max = survey.n_max * survey.a
    xi = np.linspace(x1, x2, nx)
    zi = np.linspace(0, z_max, nz)
    XI, ZI = np.meshgrid(xi, zi)
    grid = griddata((xs, zs), log_ra, (XI, ZI), method='linear')
    nm = np.isnan(grid)
    if nm.any():
        grid[nm] = griddata((xs, zs), log_ra, (XI[nm], ZI[nm]), method='nearest')
    return xi, zi, grid


# ════════════════════════════════════════════════════════
# M1: 의사단면도 ST (단일 스케일)
# ════════════════════════════════════════════════════════
def method1_pseudosection_ST(survey, rho_a, smooth_sigma=2.5, coh_thresh=0.10):
    xi, zi, grid = build_pseudosection_grid(survey, rho_a)
    dx = np.gradient(grid, xi, axis=1)
    dz = np.gradient(grid, zi, axis=0)
    sig = max(smooth_sigma, 2.0)
    Jxx = gaussian_filter(dx*dx, sigma=sig)
    Jzz = gaussian_filter(dz*dz, sigma=sig)
    Jxz = gaussian_filter(dx*dz, sigma=sig)

    trace = Jxx + Jzz
    det = Jxx*Jzz - Jxz**2
    disc = np.sqrt(np.maximum((trace/2)**2 - det, 0.0))
    lam1 = trace/2 + disc; lam2 = trace/2 - disc

    vx = Jxz; vz = lam1 - Jxx
    norm = np.sqrt(vx**2 + vz**2) + 1e-10
    flip = vz < 0
    vx_c = np.where(flip, -vx, vx)
    vz_c = np.where(flip, -vz, vz)
    theta = np.arctan2(-vx_c/norm, vz_c/norm)

    coh = np.where(lam1 > 1e-10, (lam1-lam2)/(lam1+1e-10), 0.0)
    scale = np.clip((coh - coh_thresh)/(1.0 - coh_thresh), 0.0, 1.0)
    theta_scaled = theta * scale

    deg = np.degrees(theta_scaled)
    sig_mask = np.abs(deg) > 0.5
    if sig_mask.sum() == 0:
        return {'theta_med': 0.0, 'theta_p75': 0.0, 'coh_mean': 0.0,
                'theta_grid': theta, 'coh_grid': coh, 'n_valid': 0}
    return {
        'theta_med': float(np.median(np.abs(deg[sig_mask]))),
        'theta_p75': float(np.percentile(np.abs(deg[sig_mask]), 75)),
        'coh_mean': float(np.mean(coh[sig_mask])),
        'theta_grid': theta, 'coh_grid': coh,
        'n_valid': int(sig_mask.sum()),
    }


# ════════════════════════════════════════════════════════
# M2: n-레벨 이상체 중심추적
# ════════════════════════════════════════════════════════
def method2_nlevel_centroid(survey, rho_a, low_pct=30):
    measurements = survey.measurements
    rho_a = np.asarray(rho_a)
    a = survey.a; n_max = survey.n_max

    centroids = []
    for n in range(1, n_max + 1):
        mask = np.array([m['n'] == n for m in measurements])
        if mask.sum() < 4: continue
        x_pts = np.array([m['x'] for m in measurements])[mask]
        rho_pts = rho_a[mask]
        thresh = np.percentile(rho_pts, low_pct)
        sel = rho_pts <= thresh
        if sel.sum() < 3: continue
        x_sel = x_pts[sel]; rho_sel = rho_pts[sel]
        w = 1.0 / (rho_sel + 1e-6)
        x_c = np.sum(x_sel * w) / np.sum(w)
        z_p = n * a * 0.519
        centroids.append((n, x_c, z_p))

    if len(centroids) < 3:
        return {'theta': 0.0, 'R2': 0.0, 'n_levels': len(centroids), 'centroids': centroids}

    xs = np.array([c[1] for c in centroids])
    zs = np.array([c[2] for c in centroids])
    slope, icpt = np.polyfit(xs, zs, 1)
    theta = np.degrees(np.arctan(np.abs(slope)))
    x_pred = (zs - icpt)/slope if abs(slope) > 1e-6 else xs
    ss_res = np.sum((xs - x_pred)**2)
    ss_tot = np.sum((xs - np.mean(xs))**2)
    R2 = max(0.0, 1.0 - ss_res/ss_tot) if ss_tot > 0 else 0.0
    return {
        'theta': float(theta), 'R2': float(R2),
        'n_levels': len(centroids), 'centroids': centroids,
    }


# ════════════════════════════════════════════════════════
# M3: n-레벨 폭/깊이비 분석 (Aspect Ratio Detector)
# ════════════════════════════════════════════════════════
def method3_aspect_ratio(survey, rho_a, low_pct=30):
    """
    저비저항 이상체의 폭(W)과 깊이방향 변화(ΔX) 비로 단층 종류 자동 분류.

    저각 단층:  W 큼, ΔX 큼 (W/ΔX ≈ 적당)
    중각 단층:  W 작음, ΔX 큼
    고각 단층:  W 매우 작음, ΔX 매우 큼
    """
    measurements = survey.measurements
    rho_a = np.asarray(rho_a)
    a = survey.a; n_max = survey.n_max

    widths = []      # 각 n-level에서 이상체 폭
    centers = []     # 각 n-level의 x_center
    z_levels = []

    for n in range(1, n_max + 1):
        mask = np.array([m['n'] == n for m in measurements])
        if mask.sum() < 4: continue
        x_pts = np.array([m['x'] for m in measurements])[mask]
        rho_pts = rho_a[mask]
        thresh = np.percentile(rho_pts, low_pct)
        sel = rho_pts <= thresh
        if sel.sum() < 2: continue
        x_sel = np.sort(x_pts[sel])
        width = x_sel[-1] - x_sel[0]                  # 이상체 전체 폭
        center = (x_sel[-1] + x_sel[0]) / 2
        widths.append(width)
        centers.append(center)
        z_levels.append(n * a * 0.519)

    if len(widths) < 3:
        return {'category': 'unknown', 'dip_proxy': 0.0,
                'mean_width': 0.0, 'shift_rate': 0.0, 'WtoH_ratio': 0.0}

    widths = np.array(widths); centers = np.array(centers); z_levels = np.array(z_levels)
    mean_width = float(np.mean(widths))

    # 중심 이동률 (ΔX/Δz) → 경사
    if len(centers) >= 2:
        slope_center, _ = np.polyfit(z_levels, centers, 1)
        shift_rate = float(abs(slope_center))   # dx/dz
    else:
        shift_rate = 0.0

    # 폭/총 깊이 변화
    total_depth = z_levels[-1] - z_levels[0] + 1e-6
    WtoH = mean_width / total_depth

    # 자동 분류
    if WtoH > 4.0:
        category = 'low'      # 저각 (수평에 가까운 띠)
    elif WtoH > 1.5:
        category = 'mid'      # 중각
    else:
        category = 'high'     # 고각 (수직에 가까운 블록)

    # dip proxy from shift_rate (간접 추정)
    # shift_rate = dx/dz → tan(90-dip)=dx/dz → dip = arctan(1/shift_rate)
    if shift_rate > 0.1:
        dip_proxy = float(np.degrees(np.arctan(1.0 / shift_rate)))
        dip_proxy = min(dip_proxy, 89.0)
    else:
        dip_proxy = 0.0   # 거의 수평

    return {
        'category': category,
        'dip_proxy': dip_proxy,
        'mean_width': mean_width,
        'shift_rate': shift_rate,
        'WtoH_ratio': float(WtoH),
        'widths': widths.tolist(),
        'centers': centers.tolist(),
        'z_levels': z_levels.tolist(),
    }


# ════════════════════════════════════════════════════════
# M4: 다중 스케일 의사단면도 ST
# ════════════════════════════════════════════════════════
def method4_multiscale_ST(survey, rho_a, sigmas=(1.5, 2.5, 4.0, 6.0)):
    estimates = []
    for s in sigmas:
        r = method1_pseudosection_ST(survey, rho_a, smooth_sigma=s)
        estimates.append({'sigma': s, 'theta_med': r['theta_med'],
                           'coh_mean': r['coh_mean']})

    # 스케일 간 일관성 평가 (표준편차)
    thetas = [e['theta_med'] for e in estimates if e['theta_med'] > 0.5]
    if len(thetas) >= 2:
        consistency = 1.0 / (1.0 + np.std(thetas))   # 0~1
        ms_mean = float(np.mean(thetas))
        ms_std = float(np.std(thetas))
    else:
        consistency = 0.0; ms_mean = 0.0; ms_std = 0.0

    return {'estimates': estimates, 'mean': ms_mean,
            'std': ms_std, 'consistency': float(consistency)}


# ════════════════════════════════════════════════════════
# M5: 저비저항 등치선 평균 기울기
# ════════════════════════════════════════════════════════
def method5_contour_slope(survey, rho_a, contour_pct=25):
    xi, zi, grid = build_pseudosection_grid(survey, rho_a)
    # 저비저항 등치선 (contour_pct percentile)
    threshold = np.percentile(grid, contour_pct)
    mask = grid <= threshold

    if mask.sum() < 5:
        return {'theta': 0.0, 'n_pts': 0, 'R2': 0.0}

    # 마스크된 점들의 (x, z) 좌표
    ZZ, XX = np.meshgrid(zi, xi, indexing='ij')
    x_pts = XX[mask]; z_pts = ZZ[mask]

    # 선형 회귀: z = a*x + b
    if len(x_pts) < 3:
        return {'theta': 0.0, 'n_pts': len(x_pts), 'R2': 0.0}
    slope, icpt = np.polyfit(x_pts, z_pts, 1)
    theta = np.degrees(np.arctan(np.abs(slope)))
    z_pred = slope * x_pts + icpt
    ss_res = np.sum((z_pts - z_pred)**2)
    ss_tot = np.sum((z_pts - np.mean(z_pts))**2)
    R2 = max(0.0, 1.0 - ss_res/ss_tot) if ss_tot > 0 else 0.0

    return {'theta': float(theta), 'n_pts': int(mask.sum()),
            'R2': float(R2)}


# ════════════════════════════════════════════════════════
# 통합 진단 함수
# ════════════════════════════════════════════════════════
def diagnose_all(survey, rho_a, verbose=False):
    """모든 진단 방법 실행, 결과 dict 반환."""
    r1 = method1_pseudosection_ST(survey, rho_a)
    r2 = method2_nlevel_centroid(survey, rho_a)
    r3 = method3_aspect_ratio(survey, rho_a)
    r4 = method4_multiscale_ST(survey, rho_a)
    r5 = method5_contour_slope(survey, rho_a)

    if verbose:
        print(f'  M1 의사단면 ST:     θ={r1["theta_med"]:5.1f}°  '
              f'(coh={r1["coh_mean"]:.2f})')
        print(f'  M2 n-레벨 중심:     θ={r2["theta"]:5.1f}°  '
              f'(R²={r2["R2"]:.2f})')
        print(f'  M3 종횡비 분석:     카테고리={r3["category"]:5s}  '
              f'W/H={r3["WtoH_ratio"]:.2f}  proxy={r3["dip_proxy"]:5.1f}°')
        print(f'  M4 다중 스케일 ST:  평균={r4["mean"]:5.1f}°  '
              f'(일관성={r4["consistency"]:.2f}, σ={r4["std"]:.1f}°)')
        print(f'  M5 등치선 기울기:   θ={r5["theta"]:5.1f}°  '
              f'(R²={r5["R2"]:.2f}, n={r5["n_pts"]})')

    return {'M1': r1, 'M2': r2, 'M3': r3, 'M4': r4, 'M5': r5}


def build_dip_model(mesh, true_dip_deg, rho_bg=200.0, rho_fault=20.0,
                    fault_thick=6.0, fault_x0=40.0, fault_z0=2.0):
    rho = np.full(mesh.n_cells, rho_bg)
    d = np.radians(true_dip_deg)
    for iz in range(mesh.ncz):
        for ix in range(mesh.ncx):
            xc = mesh.x_cc[ix]; zc = mesh.z_cc[iz]
            zt = np.tan(d)*(xc - fault_x0) + fault_z0
            if 0 <= zt <= 80 and zt <= zc <= zt + fault_thick:
                rho[iz*mesh.ncx + ix] = rho_fault
    return rho


# ════════════════════════════════════════════════════════
# 검증: 다양한 진 경사에서 진단 도구 실행
# ════════════════════════════════════════════════════════
def main():
    print('='*72)
    print(' 경사 진단 도구 (Phase 1) — 5가지 방법 비교 (역산 없이)')
    print('='*72)

    elec_x = np.arange(N_ELEC) * A
    survey = DipDipSurvey(a=A, n_electrodes=N_ELEC, n_max=6,
                          electrode_x=elec_x, array_type='dipole-dipole')
    mesh = Mesh2D(survey, depth_factor=2.5, dx_factor=0.25)

    test_dips = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
    results = {}
    np.random.seed(42)

    for td in test_dips:
        print(f'\n진 {td}°:')
        rho_true = build_dip_model(mesh, td, fault_x0=40, fault_z0=2.0, fault_thick=6.0)
        solver = ForwardSolver(mesh, rho_true)
        rho_a = solver.compute_data(survey, callback=lambda i,n: None)
        rho_a = rho_a * (1 + 0.03*np.random.randn(len(rho_a)))
        diag = diagnose_all(survey, rho_a, verbose=True)
        results[td] = diag

    # ── 시각화 ──
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    # (a) 진 경사 vs 각 방법 추정
    ax = axes[0, 0]
    td_arr = np.array(test_dips)
    m1_vals = [results[td]['M1']['theta_med'] for td in test_dips]
    m2_vals = [results[td]['M2']['theta'] for td in test_dips]
    m3_vals = [results[td]['M3']['dip_proxy'] for td in test_dips]
    m4_vals = [results[td]['M4']['mean'] for td in test_dips]
    m5_vals = [results[td]['M5']['theta'] for td in test_dips]

    ax.plot([0, 65], [0, 65], 'k--', alpha=0.4, label='완벽 (y=x)')
    ax.plot(td_arr, m1_vals, 'o-', label='M1: 의사단면 ST', color='#1f77b4', lw=2, ms=7)
    ax.plot(td_arr, m2_vals, 's-', label='M2: n-레벨 중심', color='#d62728', lw=2, ms=7)
    ax.plot(td_arr, m3_vals, '^-', label='M3: 종횡비 proxy', color='#9467bd', lw=2, ms=7)
    ax.plot(td_arr, m4_vals, 'D-', label='M4: 다중 스케일 ST', color='#2ca02c', lw=2, ms=7)
    ax.plot(td_arr, m5_vals, 'v-', label='M5: 등치선 기울기', color='#ff7f0e', lw=2, ms=7)
    ax.set_xlabel('진 경사각 (°)', fontsize=11)
    ax.set_ylabel('추정 경사각 (°)', fontsize=11)
    ax.set_title('(a) 진 vs 추정 — 5가지 방법', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 65); ax.set_ylim(0, 65)

    # (b) 신뢰도 지표
    ax = axes[0, 1]
    coh1 = [results[td]['M1']['coh_mean'] for td in test_dips]
    R2_2 = [results[td]['M2']['R2'] for td in test_dips]
    cons4 = [results[td]['M4']['consistency'] for td in test_dips]
    R2_5 = [results[td]['M5']['R2'] for td in test_dips]
    ax.plot(td_arr, coh1, 'o-', label='M1 Coherence', color='#1f77b4', lw=2)
    ax.plot(td_arr, R2_2, 's-', label='M2 R²', color='#d62728', lw=2)
    ax.plot(td_arr, cons4, 'D-', label='M4 Consistency', color='#2ca02c', lw=2)
    ax.plot(td_arr, R2_5, 'v-', label='M5 R²', color='#ff7f0e', lw=2)
    ax.set_xlabel('진 경사각 (°)', fontsize=11)
    ax.set_ylabel('신뢰도 지표 (0-1)', fontsize=11)
    ax.set_title('(b) 신뢰도 지표', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.1)

    # (c) 종횡비 자동 분류
    ax = axes[1, 0]
    WtoH = [results[td]['M3']['WtoH_ratio'] for td in test_dips]
    cats = [results[td]['M3']['category'] for td in test_dips]
    colors = ['#1a9641' if c=='low' else '#ffbf00' if c=='mid' else '#d7191c' if c=='high' else 'gray'
              for c in cats]
    bars = ax.bar(td_arr, WtoH, color=colors, edgecolor='k')
    for i, (td, c) in enumerate(zip(td_arr, cats)):
        ax.text(td, WtoH[i]+0.2, c, ha='center', fontsize=8, fontweight='bold')
    ax.axhline(4.0, color='#1a9641', ls=':', alpha=0.7, label='저각 경계 (W/H>4)')
    ax.axhline(1.5, color='#d7191c', ls=':', alpha=0.7, label='고각 경계 (W/H<1.5)')
    ax.set_xlabel('진 경사각 (°)', fontsize=11)
    ax.set_ylabel('이상체 W/H 비', fontsize=11)
    ax.set_title('(c) M3: 종횡비 자동 분류', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis='y')

    # (d) 오차 (각 방법별 |추정-진|)
    ax = axes[1, 1]
    err1 = np.abs(np.array(m1_vals) - td_arr)
    err2 = np.abs(np.array(m2_vals) - td_arr)
    err3 = np.abs(np.array(m3_vals) - td_arr)
    err4 = np.abs(np.array(m4_vals) - td_arr)
    err5 = np.abs(np.array(m5_vals) - td_arr)
    ax.plot(td_arr, err1, 'o-', label=f'M1 MAE={np.mean(err1):.1f}°', color='#1f77b4', lw=2)
    ax.plot(td_arr, err2, 's-', label=f'M2 MAE={np.mean(err2):.1f}°', color='#d62728', lw=2)
    ax.plot(td_arr, err3, '^-', label=f'M3 MAE={np.mean(err3):.1f}°', color='#9467bd', lw=2)
    ax.plot(td_arr, err4, 'D-', label=f'M4 MAE={np.mean(err4):.1f}°', color='#2ca02c', lw=2)
    ax.plot(td_arr, err5, 'v-', label=f'M5 MAE={np.mean(err5):.1f}°', color='#ff7f0e', lw=2)
    ax.set_xlabel('진 경사각 (°)', fontsize=11)
    ax.set_ylabel('|추정 − 진| (°)', fontsize=11)
    ax.set_title('(d) 절대 오차', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    fig.suptitle('Phase 1 진단 도구 — 5가지 방법 정확도/신뢰도 비교 (역산 없음)',
                 fontsize=13, fontweight='bold', y=1.00)
    fig.tight_layout()
    out = os.path.join(OUTDIR, 'DipDiagnostics_Phase1.png')
    fig.savefig(out, dpi=140, bbox_inches='tight')
    print(f'\n  → 그림 저장: {out}')

    # CSV 저장
    csv_path = os.path.join(OUTDIR, 'DipDiagnostics_table.csv')
    import csv
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['TrueDip', 'M1_ST', 'M1_coh', 'M2_centroid', 'M2_R2',
                    'M3_dipProxy', 'M3_WtoH', 'M3_category',
                    'M4_meanST', 'M4_cons', 'M5_contour', 'M5_R2'])
        for td in test_dips:
            r = results[td]
            w.writerow([td,
                        f'{r["M1"]["theta_med"]:.2f}', f'{r["M1"]["coh_mean"]:.3f}',
                        f'{r["M2"]["theta"]:.2f}', f'{r["M2"]["R2"]:.3f}',
                        f'{r["M3"]["dip_proxy"]:.2f}', f'{r["M3"]["WtoH_ratio"]:.2f}',
                        r["M3"]["category"],
                        f'{r["M4"]["mean"]:.2f}', f'{r["M4"]["consistency"]:.3f}',
                        f'{r["M5"]["theta"]:.2f}', f'{r["M5"]["R2"]:.3f}'])
    print(f'  → CSV 저장: {csv_path}')

    # 최고 방법 식별 (각도 구간별)
    print('\n' + '='*72)
    print(' 각도 구간별 가장 정확한 방법')
    print('='*72)
    method_errs = {'M1': err1, 'M2': err2, 'M3': err3, 'M4': err4, 'M5': err5}
    method_vals = {'M1': m1_vals, 'M2': m2_vals, 'M3': m3_vals, 'M4': m4_vals, 'M5': m5_vals}
    for i, td in enumerate(test_dips):
        errs_at_i = {k: v[i] for k, v in method_errs.items()}
        best = min(errs_at_i, key=errs_at_i.get)
        print(f'  진 {td:3d}°  →  최적: {best} (오차 {errs_at_i[best]:.1f}°, '
              f'추정 {method_vals[best][i]:.1f}°)')

    print('\n' + '='*72)
    print(' 완료!')
    print('='*72)


if __name__ == '__main__':
    main()
