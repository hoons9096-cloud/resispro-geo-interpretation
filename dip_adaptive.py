#!/usr/bin/env python3
"""
경사각 자동 추정 — Phase 2 적응형 통합기

Phase 1 진단 결과 (DipDiagnostics_table.csv) 기반 결정 트리:

[Step 1] 저각 자동 검출
  M5(등치선) 추정이 다른 방법들의 중앙값보다 10°+ 낮고 R²>0.1 → 저각
  → M5 신뢰

[Step 2] 중각 영역 신뢰
  M2(n-레벨) R²이 0.3~0.85 → M2 신뢰

[Step 3] 안정성 우선
  M4(다중스케일) consistency ≥ 0.5 → M4 사용

[Step 4] 기본값
  M4 (가장 안정적 평균)
"""
import sys, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv

sys.path.insert(0, '')
from RESIS_Pro import DipDipSurvey, Mesh2D, ForwardSolver
from dip_diagnostics import (diagnose_all, build_dip_model,
                              method1_pseudosection_ST, method2_nlevel_centroid,
                              method3_aspect_ratio, method4_multiscale_ST,
                              method5_contour_slope)

plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False
OUTDIR = ''

A = 5.0; N_ELEC = 30


# ════════════════════════════════════════════════════════
# 적응형 결정 함수
# ════════════════════════════════════════════════════════
def adaptive_dip_estimate(diag, verbose=False):
    """
    5가지 진단 결과를 받아 가장 신뢰할 만한 단일 추정값 반환.

    Returns:
        (theta_est, decision_path, confidence)
    """
    M1 = diag['M1']['theta_med']
    M1_coh = diag['M1']['coh_mean']
    M2 = diag['M2']['theta']
    M2_R2 = diag['M2']['R2']
    M3p = diag['M3']['dip_proxy']
    M3_WtoH = diag['M3']['WtoH_ratio']
    M4 = diag['M4']['mean']
    M4_cons = diag['M4']['consistency']
    M5 = diag['M5']['theta']
    M5_R2 = diag['M5']['R2']

    log = []

    # ── Step 1: 저각 자동 검출 ──
    others = [M1, M2, M4]
    others_med = np.median(others)
    M5_diff = others_med - M5
    log.append(f"Step1: M5={M5:.1f}, others_med={others_med:.1f}, diff={M5_diff:.1f}")
    if M5_diff > 10 and M5_R2 > 0.05 and M5 > 0.5:
        decision = 'low_dip_via_M5'
        theta = M5
        confidence = 0.7 + 0.3 * M5_R2
        if verbose: print(f"  [{decision}] θ={theta:.1f}° conf={confidence:.2f}")
        return theta, decision, confidence

    # ── Step 2: 중각 영역 — M2 신뢰 (R² 0.3~0.85) ──
    log.append(f"Step2: M2={M2:.1f}, R²={M2_R2:.2f}")
    if 0.3 <= M2_R2 <= 0.85 and 15 < M2 < 50:
        decision = 'mid_dip_via_M2'
        theta = M2
        confidence = 0.6 + 0.3 * M2_R2
        if verbose: print(f"  [{decision}] θ={theta:.1f}° conf={confidence:.2f}")
        return theta, decision, confidence

    # ── Step 3: 일관성 우선 — M4 consistency ≥ 0.5 ──
    log.append(f"Step3: M4={M4:.1f}, cons={M4_cons:.2f}")
    if M4_cons >= 0.5:
        decision = 'stable_via_M4'
        theta = M4
        confidence = 0.5 + 0.4 * M4_cons
        if verbose: print(f"  [{decision}] θ={theta:.1f}° conf={confidence:.2f}")
        return theta, decision, confidence

    # ── Step 4: 클러스터 평균 (M1, M2, M4 중 비슷한 것들) ──
    log.append(f"Step4: clustering [M1={M1:.1f}, M2={M2:.1f}, M4={M4:.1f}]")
    estimates = np.array([M1, M2, M4])
    med = np.median(estimates)
    close = estimates[np.abs(estimates - med) < 8]
    if len(close) >= 2:
        theta = float(np.mean(close))
        decision = f'cluster_avg_{len(close)}'
        confidence = 0.4 + 0.1 * len(close)
        if verbose: print(f"  [{decision}] θ={theta:.1f}° conf={confidence:.2f}")
        return theta, decision, confidence

    # ── Default: M4 ──
    decision = 'fallback_M4'
    theta = M4
    confidence = 0.3
    if verbose: print(f"  [{decision}] θ={theta:.1f}° conf={confidence:.2f}")
    return theta, decision, confidence


# ════════════════════════════════════════════════════════
# 전체 평가
# ════════════════════════════════════════════════════════
def main():
    print('='*72)
    print(' 경사각 자동 추정 — Phase 2 적응형 통합기')
    print('='*72)
    print(' 사전 정보 없이 어떤 진 경사가 와도 자동 판단')
    print('='*72)

    elec_x = np.arange(N_ELEC) * A
    survey = DipDipSurvey(a=A, n_electrodes=N_ELEC, n_max=6,
                          electrode_x=elec_x, array_type='dipole-dipole')
    mesh = Mesh2D(survey, depth_factor=2.5, dx_factor=0.25)

    # 두 가지 잡음 시나리오로 강건성 평가
    scenarios = [
        ('seed42_3pct', 42, 0.03),
        ('seed7_5pct',  7, 0.05),
    ]
    test_dips = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]

    all_results = {}
    for scen_name, seed, noise_lvl in scenarios:
        print(f'\n── 시나리오: {scen_name} (seed={seed}, noise={noise_lvl*100:.0f}%) ──')
        scen_results = []
        np.random.seed(seed)
        for td in test_dips:
            rho_true = build_dip_model(mesh, td, fault_x0=40, fault_z0=2.0, fault_thick=6.0)
            solver = ForwardSolver(mesh, rho_true)
            rho_a = solver.compute_data(survey, callback=lambda i,n: None)
            rho_a = rho_a * (1 + noise_lvl*np.random.randn(len(rho_a)))

            diag = diagnose_all(survey, rho_a, verbose=False)
            theta_est, decision, conf = adaptive_dip_estimate(diag, verbose=False)
            err = theta_est - td
            print(f'  진 {td:3d}°  →  {theta_est:5.1f}°  err={err:+6.1f}°  '
                  f'({decision},  conf={conf:.2f})')
            scen_results.append({
                'true_dip': td, 'est': theta_est, 'err': err,
                'decision': decision, 'conf': conf,
                'M1': diag['M1']['theta_med'],
                'M2': diag['M2']['theta'],
                'M3p': diag['M3']['dip_proxy'],
                'M4': diag['M4']['mean'],
                'M5': diag['M5']['theta'],
            })
        mae = np.mean([abs(r['err']) for r in scen_results])
        print(f'  {scen_name} MAE = {mae:.2f}°')
        all_results[scen_name] = {'data': scen_results, 'mae': mae}

    # ── 요약: 각도 구간별 성능 ──
    print('\n' + '='*72)
    print(' 각도 구간별 성능 요약')
    print('='*72)
    for scen_name in all_results:
        data = all_results[scen_name]['data']
        low = [r for r in data if r['true_dip'] <= 20]
        mid = [r for r in data if 20 < r['true_dip'] <= 40]
        high = [r for r in data if r['true_dip'] > 40]
        mae_low = np.mean([abs(r['err']) for r in low])
        mae_mid = np.mean([abs(r['err']) for r in mid])
        mae_high = np.mean([abs(r['err']) for r in high])
        print(f'  [{scen_name}] 저각(≤20°): {mae_low:.1f}°  '
              f'중각(20-40°): {mae_mid:.1f}°  고각(40-60°): {mae_high:.1f}°')

    # ── 시각화 ──
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    colors_scen = ['#1f77b4', '#d62728']
    for ax_idx, scen_name in enumerate(scenarios):
        sname = scen_name[0]
        data = all_results[sname]['data']
        td = np.array([r['true_dip'] for r in data])
        est = np.array([r['est'] for r in data])
        err = np.array([r['err'] for r in data])

        # (a/b) 진 vs 추정 (각 시나리오)
        ax = axes[0, ax_idx]
        ax.plot([0, 65], [0, 65], 'k--', alpha=0.4, label='완벽 (y=x)')
        # 모든 방법 점선
        for mk in ['M1', 'M2', 'M3p', 'M4', 'M5']:
            vals = [r[mk] for r in data]
            ax.plot(td, vals, '-', alpha=0.25, lw=1, label=mk)
        # 적응형 결과 강조
        ax.plot(td, est, 'o-', color='red', lw=2.5, ms=10,
                label=f'적응형 (MAE={all_results[sname]["mae"]:.1f}°)',
                markeredgecolor='k')
        # 결정 경로별 색상
        for r in data:
            if 'M5' in r['decision']:
                ax.scatter(r['true_dip'], r['est'], s=120,
                            facecolor='none', edgecolor='blue', lw=2)
            elif 'M2' in r['decision']:
                ax.scatter(r['true_dip'], r['est'], s=120,
                            facecolor='none', edgecolor='green', lw=2)
            elif 'M4' in r['decision']:
                ax.scatter(r['true_dip'], r['est'], s=120,
                            facecolor='none', edgecolor='orange', lw=2)
        ax.set_xlabel('진 경사각 (°)', fontsize=11)
        ax.set_ylabel('추정 경사각 (°)', fontsize=11)
        ax.set_title(f'({chr(97+ax_idx)}) {sname}\n'
                     '동그라미: 파랑=M5, 초록=M2, 주황=M4',
                     fontsize=10, fontweight='bold')
        ax.legend(fontsize=8, loc='upper left')
        ax.grid(alpha=0.3)
        ax.set_xlim(0, 65); ax.set_ylim(-5, 65)

    # (c) 오차 비교
    ax = axes[1, 0]
    for i, scen_name in enumerate(scenarios):
        sname = scen_name[0]
        data = all_results[sname]['data']
        td = np.array([r['true_dip'] for r in data])
        err = np.array([r['err'] for r in data])
        ax.plot(td, err, '-o', label=f'{sname} (MAE={all_results[sname]["mae"]:.1f}°)',
                color=colors_scen[i], lw=2, ms=8)
    ax.axhline(0, color='k', alpha=0.4)
    ax.axhspan(-5, 5, color='green', alpha=0.1, label='|err|<5° 영역')
    ax.set_xlabel('진 경사각 (°)', fontsize=11)
    ax.set_ylabel('오차 (추정 − 진) (°)', fontsize=11)
    ax.set_title('(c) 잡음 강건성 — 두 시나리오 오차', fontsize=11, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    # (d) 결정 경로 사용 빈도
    ax = axes[1, 1]
    all_decisions = []
    for sname in all_results:
        all_decisions.extend([r['decision'] for r in all_results[sname]['data']])
    unique, counts = np.unique(all_decisions, return_counts=True)
    colors_d = plt.cm.Set2(np.linspace(0, 1, len(unique)))
    bars = ax.bar(range(len(unique)), counts, color=colors_d, edgecolor='k')
    ax.set_xticks(range(len(unique)))
    ax.set_xticklabels([u.replace('_', '\n') for u in unique], rotation=0, fontsize=9)
    ax.set_ylabel('사용 빈도', fontsize=11)
    ax.set_title('(d) 결정 경로 통계', fontsize=11, fontweight='bold')
    for b, c in zip(bars, counts):
        ax.text(b.get_x()+b.get_width()/2, c+0.2, str(c),
                ha='center', fontsize=10, fontweight='bold')
    ax.grid(alpha=0.3, axis='y')

    fig.suptitle('Phase 2 적응형 통합기 — 사전 정보 없이 자동 추정',
                 fontsize=13, fontweight='bold', y=1.00)
    fig.tight_layout()
    out = os.path.join(OUTDIR, 'DipAdaptive_Phase2.png')
    fig.savefig(out, dpi=140, bbox_inches='tight')
    print(f'\n  → 그림 저장: {out}')

    # CSV
    csv_path = os.path.join(OUTDIR, 'DipAdaptive_table.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['Scenario', 'TrueDip', 'Adaptive', 'Error',
                    'Decision', 'Confidence', 'M1', 'M2', 'M3p', 'M4', 'M5'])
        for sname in all_results:
            for r in all_results[sname]['data']:
                w.writerow([sname, r['true_dip'], f'{r["est"]:.2f}',
                            f'{r["err"]:.2f}', r['decision'],
                            f'{r["conf"]:.3f}',
                            f'{r["M1"]:.1f}', f'{r["M2"]:.1f}',
                            f'{r["M3p"]:.1f}', f'{r["M4"]:.1f}', f'{r["M5"]:.1f}'])
    print(f'  → CSV 저장: {csv_path}')

    print('\n' + '='*72)
    print(' 완료!')
    print('='*72)


if __name__ == '__main__':
    main()
