#!/usr/bin/env python3
"""
경사각 추정 향상 알고리즘:
  아이디어 A: n-레벨별 이상체 중심 x 위치 추적 → 직접 경사각 계산
  아이디어 B: 순방향 모델링 보정 곡선 → θ_pseudo → θ_true 변환

목표: 30-60° 중경사 단층의 자동 경사각 추정 정확도 향상
"""
import sys, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit

sys.path.insert(0, '')
from RESIS_Pro import (DipDipSurvey, Mesh2D, ForwardSolver, filter_bad_data)

plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False
OUTDIR = ''

A = 5.0; N_ELEC = 30


# ════════════════════════════════════════════════════════
# 아이디어 A: n-레벨별 이상체 중심 추적
# ════════════════════════════════════════════════════════
def track_centroid_by_nlevel(survey, rho_a, low_pct=30):
    """
    각 n-level의 저비저항 이상체 x 중심 위치 추적.
    저비저항 가중 중심 vs pseudodepth → 회귀 → 겉보기 경사

    Returns:
        apparent_dip_deg: 회귀로 얻은 겉보기 경사각 (°)
        centroids: [(n, x_centroid, z_pseudo), ...]
        fit_R2: 회귀 결정계수
    """
    measurements = survey.measurements
    rho_a = np.asarray(rho_a)
    n_max = survey.n_max
    a = survey.a

    centroids = []
    for n in range(1, n_max + 1):
        mask = np.array([m['n'] == n for m in measurements])
        if mask.sum() < 4:
            continue
        x_pts = np.array([m['x'] for m in measurements])[mask]
        rho_pts = rho_a[mask]

        # 저비저항 가중치: 1/ρ - 그러나 저비저항 백분위 이하만 사용
        thresh = np.percentile(rho_pts, low_pct)
        sel = rho_pts <= thresh
        if sel.sum() < 3:
            continue
        x_sel = x_pts[sel]
        rho_sel = rho_pts[sel]
        weights = 1.0 / (rho_sel + 1e-6)
        x_centroid = np.sum(x_sel * weights) / np.sum(weights)

        # 의사깊이 (dipole-dipole: z_p = n·a·0.519, Edwards 1977)
        z_p = n * a * 0.519
        centroids.append((n, x_centroid, z_p))

    if len(centroids) < 3:
        return None, centroids, 0.0

    ns = np.array([c[0] for c in centroids])
    xs = np.array([c[1] for c in centroids])
    zs = np.array([c[2] for c in centroids])

    # z = slope·x + intercept → tan(θ) = slope = dz/dx
    slope, intercept = np.polyfit(xs, zs, 1)
    apparent_dip_rad = np.arctan(np.abs(slope))
    apparent_dip_deg = np.degrees(apparent_dip_rad)

    # R²
    x_pred = (zs - intercept) / slope if slope != 0 else xs
    ss_res = np.sum((xs - x_pred)**2)
    ss_tot = np.sum((xs - np.mean(xs))**2)
    R2 = 1.0 - ss_res/ss_tot if ss_tot > 0 else 0.0

    return apparent_dip_deg, centroids, R2


# ════════════════════════════════════════════════════════
# 아이디어 B: 순방향 보정 곡선
# ════════════════════════════════════════════════════════
def build_dip_model(mesh, true_dip_deg, rho_bg=200.0, rho_fault=20.0,
                    fault_thick=6.0, fault_x0=40.0, fault_z0=2.0):
    rho = np.full(mesh.n_cells, rho_bg)
    d = np.radians(true_dip_deg)
    for iz in range(mesh.ncz):
        for ix in range(mesh.ncx):
            xc = mesh.x_cc[ix]; zc = mesh.z_cc[iz]
            zt = np.tan(d) * (xc - fault_x0) + fault_z0
            if 0 <= zt <= 80 and zt <= zc <= zt + fault_thick:
                rho[iz * mesh.ncx + ix] = rho_fault
    return rho


def pseudosection_st_median(survey, rho_a, smooth_sigma=2.5, coh_thresh=0.10):
    """의사단면도 구조 텐서 중앙값 계산 (STAR 내부와 동일)."""
    from scipy.interpolate import griddata
    measurements = survey.measurements
    rho_a = np.asarray(rho_a)
    a = survey.a
    n_max = survey.n_max
    x1 = survey.electrode_x[0]; x2 = survey.electrode_x[-1]

    xs = np.array([m['x'] for m in measurements])
    zs = np.array([m['z'] for m in measurements])
    log_rho = np.log10(rho_a + 1e-6)

    xi = np.linspace(x1, x2, 80)
    zi = np.linspace(0.5*a, n_max*a*1.05, 40)
    XX, ZZ = np.meshgrid(xi, zi)

    grid = griddata((xs, zs), log_rho, (XX, ZZ), method='linear', fill_value=np.nanmean(log_rho))
    grid = gaussian_filter(grid, sigma=smooth_sigma)

    # 구조 텐서
    gx = np.gradient(grid, axis=1)
    gz = np.gradient(grid, axis=0)
    Jxx = gaussian_filter(gx*gx, sigma=smooth_sigma)
    Jzz = gaussian_filter(gz*gz, sigma=smooth_sigma)
    Jxz = gaussian_filter(gx*gz, sigma=smooth_sigma)

    trace = Jxx + Jzz
    det = Jxx*Jzz - Jxz**2
    disc = np.sqrt(np.maximum(trace**2 - 4*det, 0))
    lam1 = 0.5*(trace + disc)
    lam2 = 0.5*(trace - disc)
    coh = (lam1 - lam2) / (lam1 + 1e-12)

    theta = 0.5 * np.arctan2(2*Jxz, Jxx - Jzz)
    # 경사 방향 (수직 그래디언트 방향이 아닌 구조 따라가는 방향)
    theta_struct = theta + np.pi/2
    theta_struct = ((theta_struct + np.pi/2) % np.pi) - np.pi/2

    # 유의미한 추정
    sig = (coh > coh_thresh) & (np.abs(np.degrees(theta_struct)) > 0.5)
    theta_deg = np.degrees(theta_struct)
    if sig.sum() < 5:
        return 0.0, 0.0
    med = np.median(np.abs(theta_deg[sig]))
    p75 = np.percentile(np.abs(theta_deg[sig]), 75)
    return med, p75


def build_calibration_curve(survey,
                             dip_angles_true=None,
                             fault_x0=40.0, fault_z0=2.0, fault_thick=6.0):
    """
    순방향 모델링으로 θ_true → θ_pseudo, θ_centroid 매핑 생성.
    """
    if dip_angles_true is None:
        dip_angles_true = list(range(5, 71, 5))   # 5, 10, 15, ..., 70

    mesh = Mesh2D(survey, depth_factor=2.5, dx_factor=0.25)

    pseudo_meds = []
    centroid_dips = []
    centroid_R2s = []

    print(f'\n── 보정 곡선 구축 (총 {len(dip_angles_true)} 각도) ──')
    print(f'   서베이: N_ELEC={survey.n_electrodes}, n_max={survey.n_max}, a={survey.a}m')
    print(f'   단층: x0={fault_x0}, z0={fault_z0}, t={fault_thick}m')

    for dip in dip_angles_true:
        rho_true = build_dip_model(mesh, dip,
                                    fault_x0=fault_x0,
                                    fault_z0=fault_z0,
                                    fault_thick=fault_thick)
        solver = ForwardSolver(mesh, rho_true)
        rho_a = solver.compute_data(survey, callback=lambda i,n: None)

        # 의사단면도 ST
        med, p75 = pseudosection_st_median(survey, rho_a)
        # n-레벨 중심 추적
        dip_c, _, R2 = track_centroid_by_nlevel(survey, rho_a, low_pct=30)
        if dip_c is None:
            dip_c = 0.0; R2 = 0.0

        pseudo_meds.append(med)
        centroid_dips.append(dip_c)
        centroid_R2s.append(R2)

        print(f'   진={dip:3d}°  →  의사단면도ST={med:5.1f}°  중심추적={dip_c:5.1f}° (R²={R2:.2f})')

    return (np.array(dip_angles_true), np.array(pseudo_meds),
            np.array(centroid_dips), np.array(centroid_R2s))


# ════════════════════════════════════════════════════════
# 메인: 30° 케이스에 적용
# ════════════════════════════════════════════════════════
def main():
    print('='*70)
    print(' 경사각 추정 향상 — n-레벨 추적 + 보정 곡선')
    print('='*70)

    # ── 서베이 설정 (n_max=6 중앙 배치)
    n_max = 6
    elec_x = np.arange(N_ELEC) * A
    survey = DipDipSurvey(a=A, n_electrodes=N_ELEC, n_max=n_max,
                          electrode_x=elec_x, array_type='dipole-dipole')

    # ── 보정 곡선 생성
    print('\n[Step 1] 순방향 보정 곡선 구축 중...')
    dips_true, dips_pseudo, dips_centroid, R2s = build_calibration_curve(survey)

    # ── 보정 함수 피팅 (의사단면도 ST 기반)
    # θ_true = f(θ_pseudo) → 2차 또는 3차 다항식
    valid_pseudo = dips_pseudo > 0
    coefs_pseudo = np.polyfit(dips_pseudo[valid_pseudo], dips_true[valid_pseudo], 3)

    # 중심추적도 보정 (R²>0.3인 점만)
    valid_c = (dips_centroid > 0) & (R2s > 0.3)
    if valid_c.sum() >= 3:
        coefs_centroid = np.polyfit(dips_centroid[valid_c], dips_true[valid_c], 2)
    else:
        coefs_centroid = None

    print('\n[Step 2] 보정 함수 피팅 완료')
    print(f'   의사단면도 ST 보정 (3차): {coefs_pseudo}')
    if coefs_centroid is not None:
        print(f'   중심추적 보정 (2차): {coefs_centroid}')

    # ── 30° 케이스에 적용 검증
    print('\n[Step 3] 30° 케이스 적용 (fault_x0=40, fault_z0=1, t=8m)')
    mesh = Mesh2D(survey, depth_factor=2.5, dx_factor=0.25)
    rho_30 = build_dip_model(mesh, 30.0, fault_x0=40.0, fault_z0=1.0, fault_thick=8.0)
    solver = ForwardSolver(mesh, rho_30)
    rho_a_30 = solver.compute_data(survey, callback=lambda i,n: None)
    # 노이즈 추가
    np.random.seed(42)
    rho_a_30 = rho_a_30 * (1 + 0.03*np.random.randn(len(rho_a_30)))

    med_30, _ = pseudosection_st_median(survey, rho_a_30)
    dip_c_30, centroids_30, R2_30 = track_centroid_by_nlevel(survey, rho_a_30)

    # 보정 적용
    med_30_corrected = np.polyval(coefs_pseudo, med_30)
    if coefs_centroid is not None and dip_c_30 is not None:
        dip_c_30_corrected = np.polyval(coefs_centroid, dip_c_30)
    else:
        dip_c_30_corrected = None

    print(f'\n   [원시]    의사단면도 ST = {med_30:.1f}°  → 진 30° 오차 {30-med_30:+.1f}°')
    if dip_c_30 is not None:
        print(f'   [원시]    중심 추적     = {dip_c_30:.1f}°  (R²={R2_30:.2f}) → 진 30° 오차 {30-dip_c_30:+.1f}°')
    print(f'   [보정후]  의사단면도 ST = {med_30_corrected:.1f}°  → 진 30° 오차 {30-med_30_corrected:+.1f}°')
    if dip_c_30_corrected is not None:
        print(f'   [보정후]  중심 추적     = {dip_c_30_corrected:.1f}°  → 진 30° 오차 {30-dip_c_30_corrected:+.1f}°')

    # ── 시각화
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # (a) 보정 곡선
    ax = axes[0, 0]
    ax.plot([0, 70], [0, 70], 'k--', alpha=0.4, label='완벽 추정 (y=x)')
    ax.plot(dips_true, dips_pseudo, 'bo-', ms=7, lw=2, label='의사단면도 ST 중앙값')
    ax.plot(dips_true, dips_centroid, 'r^-', ms=7, lw=2, alpha=0.7,
            label='n-레벨 중심 추적')
    ax.set_xlabel('진 경사각 (°)', fontsize=11)
    ax.set_ylabel('추정 경사각 (°)', fontsize=11)
    ax.set_title('(a) 보정 전 — 진 경사 vs 추정 경사', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 70); ax.set_ylim(0, 70)

    # (b) 보정 함수 시각화
    ax = axes[0, 1]
    x_test = np.linspace(5, 50, 100)
    y_corr_pseudo = np.polyval(coefs_pseudo, x_test)
    ax.plot(x_test, y_corr_pseudo, 'b-', lw=2, label='의사단면도 ST 보정 (3차)')
    ax.scatter(dips_pseudo, dips_true, c='blue', s=50, marker='o',
               label='원본 데이터점')
    if coefs_centroid is not None:
        y_corr_c = np.polyval(coefs_centroid, x_test)
        ax.plot(x_test, y_corr_c, 'r-', lw=2, label='중심추적 보정 (2차)')
        ax.scatter(dips_centroid[valid_c], dips_true[valid_c], c='red',
                   s=50, marker='^', alpha=0.7, label='중심추적 데이터점')
    ax.plot([0, 70], [0, 70], 'k--', alpha=0.4)
    ax.set_xlabel('측정 추정각 (°)', fontsize=11)
    ax.set_ylabel('보정된 진 경사각 (°)', fontsize=11)
    ax.set_title('(b) 보정 함수 (역변환)', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (c) 30° 케이스 — 보정 효과
    ax = axes[1, 0]
    labels = ['진값', '의사단면도ST\n원시', '의사단면도ST\n보정', '중심추적\n원시', '중심추적\n보정']
    values = [30.0, med_30, med_30_corrected,
              dip_c_30 if dip_c_30 else 0.0,
              dip_c_30_corrected if dip_c_30_corrected else 0.0]
    colors = ['#2166ac', '#7fbf7f', '#1a9641', '#fdae61', '#d7191c']
    bars = ax.bar(labels, values, color=colors, edgecolor='k')
    ax.axhline(30, color='blue', ls='--', alpha=0.5, lw=1)
    for b, v in zip(bars, values):
        ax.text(b.get_x()+b.get_width()/2, v+0.5, f'{v:.1f}°',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_ylabel('추정 경사각 (°)', fontsize=11)
    ax.set_title('(c) 30° 단층 — 보정 전후 비교', fontsize=11, fontweight='bold')
    ax.set_ylim(0, 40)
    ax.grid(alpha=0.3, axis='y')

    # (d) 중심 추적 산점도 (30° 케이스)
    ax = axes[1, 1]
    if centroids_30:
        ns = np.array([c[0] for c in centroids_30])
        xs = np.array([c[1] for c in centroids_30])
        zs = np.array([c[2] for c in centroids_30])
        sc = ax.scatter(xs, zs, c=ns, cmap='viridis', s=80, edgecolors='k',
                        label='n-레벨 중심')
        # 회귀선
        if len(centroids_30) >= 2:
            slope, intercept = np.polyfit(xs, zs, 1)
            x_line = np.linspace(xs.min(), xs.max(), 50)
            z_line = slope*x_line + intercept
            ax.plot(x_line, z_line, 'r-', lw=2,
                    label=f'회귀: θ_app={dip_c_30:.1f}°, R²={R2_30:.2f}')
        # 진 30° 기준선
        x_ref = np.linspace(xs.min(), xs.max(), 50)
        z_ref = np.tan(np.radians(30.0))*(x_ref - xs.mean()) + zs.mean()
        ax.plot(x_ref, z_ref, 'b--', lw=2, alpha=0.6, label='진 30° 기울기')
        plt.colorbar(sc, ax=ax, label='n-level')
        ax.invert_yaxis()
        ax.set_xlabel('이상체 x 중심 (m)', fontsize=11)
        ax.set_ylabel('의사깊이 (m)', fontsize=11)
        ax.set_title('(d) n-레벨별 이상체 중심 추적', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8, loc='upper left')
        ax.grid(alpha=0.3)

    fig.suptitle('경사각 보정 — n-레벨 추적 + 순방향 보정 곡선', fontsize=13, fontweight='bold')
    fig.tight_layout()
    out = os.path.join(OUTDIR, 'DipCalibration_results.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f'\n   → 그림 저장: DipCalibration_results.png')

    # 보정 함수 저장 (numpy 파일)
    np.savez(os.path.join(OUTDIR, 'dip_calibration_coefs.npz'),
             coefs_pseudo=coefs_pseudo,
             coefs_centroid=coefs_centroid if coefs_centroid is not None else np.array([]),
             dips_true=dips_true,
             dips_pseudo=dips_pseudo,
             dips_centroid=dips_centroid,
             n_max=n_max, N_ELEC=N_ELEC, A=A)
    print(f'   → 보정 계수 저장: dip_calibration_coefs.npz')

    print('\n' + '='*70)
    print(' 완료!')
    print('='*70)


if __name__ == '__main__':
    main()
