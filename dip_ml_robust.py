#!/usr/bin/env python3
"""
경사각 자동 추정 — Phase 3 견고화 (Robust ML with OOD detection)

전략:
  1) OOD (Out-of-Distribution) 검출
     - 학습 데이터의 feature 분포 통계
     - 새 sample이 분포 안에 있는지 percentile boundary로 확인
  2) Fallback 메커니즘
     - In-distribution → ML 예측 + 불확실성
     - OOD → 보수적 방법 (M2 R² 적정 시 M2, 그 외 M1 또는 평균)
  3) Confidence interval (RF 트리 분산 활용)
"""
import sys, os, pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '')
from RESIS_Pro import (DipDipSurvey, Mesh2D, ForwardSolver,
                        parse_apv, apv_to_survey_data, filter_bad_data)
from dip_diagnostics import diagnose_all, build_dip_model
from dip_ml_train import extract_features, FEATURE_NAMES

plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False
OUTDIR = ''

A = 5.0; N_ELEC = 30


# ════════════════════════════════════════════════════════
# OOD 검출
# ════════════════════════════════════════════════════════
def compute_ood_thresholds(X_train, percentile=98):
    """학습 데이터 feature별 percentile boundary."""
    lo = np.percentile(X_train, (100-percentile)/2, axis=0)
    hi = np.percentile(X_train, 100-(100-percentile)/2, axis=0)
    return lo, hi


def ood_score(X_new, lo, hi):
    """샘플별 OOD 점수 (0=완전 OOD, 1=완전 ID)."""
    in_range = (X_new >= lo) & (X_new <= hi)
    return float(np.mean(in_range))


def ood_severity(X_new, lo, hi, scale):
    """가장 OOD가 심한 feature와 그 정도."""
    deviation = np.zeros_like(X_new, dtype=float)
    below = X_new < lo
    above = X_new > hi
    deviation[below] = (lo[below] - X_new[below]) / (scale[below] + 1e-6)
    deviation[above] = (X_new[above] - hi[above]) / (scale[above] + 1e-6)
    return deviation


# ════════════════════════════════════════════════════════
# RF 트리 분산 기반 불확실성
# ════════════════════════════════════════════════════════
def rf_uncertainty(rf_model, X_sample):
    """RF의 개별 트리 예측 표준편차 → 1σ 불확실성."""
    predictions = np.array([tree.predict(X_sample.reshape(1,-1))[0]
                              for tree in rf_model.estimators_])
    return float(np.std(predictions)), float(np.mean(predictions))


# ════════════════════════════════════════════════════════
# Robust 통합 예측기
# ════════════════════════════════════════════════════════
def robust_predict(diag, model_dict, ood_thresholds=None, verbose=False):
    """
    OOD 검출 + Fallback 통합 예측.

    Returns:
        {'estimate': θ, 'method': str, 'confidence': 0-1,
         'uncertainty': σ°, 'ood_score': float, 'severity': 가장큰편차}
    """
    feat = extract_features(diag)
    rf_model = model_dict['rf']
    X_train = model_dict['X_train']

    # OOD 임계값
    if ood_thresholds is None:
        lo, hi = compute_ood_thresholds(X_train, percentile=98)
        scale = np.std(X_train, axis=0)
    else:
        lo, hi, scale = ood_thresholds

    # OOD 점수
    ood = ood_score(feat, lo, hi)
    sev = ood_severity(feat, lo, hi, scale)
    max_sev = float(np.max(np.abs(sev)))
    worst_feat = FEATURE_NAMES[int(np.argmax(np.abs(sev)))]

    # ML 예측 + 불확실성
    ml_pred = float(rf_model.predict(feat.reshape(1,-1))[0])
    ml_unc, _ = rf_uncertainty(rf_model, feat)

    # ── Fallback 결정 ──
    # OOD 점수 ≥ 0.85: 학습 분포 내 → ML 사용
    # OOD 점수 < 0.85 OR max_sev > 2.0: Fallback
    use_ml = (ood >= 0.85) and (max_sev <= 2.0)

    if use_ml:
        method = 'ML'
        estimate = ml_pred
        # 불확실성 = RF 표준편차 (보통 2-5°)
        uncertainty = max(ml_unc, 2.0)
        confidence = ood   # 0.85~1.0
    else:
        # Fallback: M2 R² 적정 → M2
        M2_theta = diag['M2']['theta']; M2_R2 = diag['M2']['R2']
        M1_theta = diag['M1']['theta_med']; M1_coh = diag['M1']['coh_mean']
        if 0.3 < M2_R2 < 0.85 and 5 < M2_theta < 50:
            method = 'Fallback_M2'
            estimate = M2_theta
            uncertainty = 5.0
            confidence = 0.5
        elif M1_coh > 0.7 and 5 < M1_theta < 50:
            method = 'Fallback_M1'
            estimate = M1_theta
            uncertainty = 6.0
            confidence = 0.4
        else:
            # 마지막 수단: 보수적 평균
            method = 'Fallback_avg'
            vals = [v for v in [M1_theta, M2_theta, diag['M4']['mean']] if 5 < v < 50]
            estimate = float(np.mean(vals)) if vals else 0.0
            uncertainty = 8.0
            confidence = 0.3

    if verbose:
        print(f'    OOD 점수={ood:.2f}, 최대 편차={max_sev:.2f}σ ({worst_feat})')
        print(f'    ML 예측={ml_pred:.1f}° (±{ml_unc:.1f}°)')
        print(f'    → 최종: {estimate:.1f}° via {method}, conf={confidence:.2f}')

    return {
        'estimate': estimate, 'method': method,
        'confidence': confidence, 'uncertainty': uncertainty,
        'ood_score': ood, 'max_severity': max_sev,
        'worst_feature': worst_feat,
        'ml_pred': ml_pred, 'ml_unc': ml_unc,
    }


# ════════════════════════════════════════════════════════
# 검증
# ════════════════════════════════════════════════════════
def main():
    print('='*72)
    print(' Robust ML — OOD 검출 + Fallback')
    print('='*72)

    with open(os.path.join(OUTDIR, 'dip_ml_model.pkl'), 'rb') as f:
        model_dict = pickle.load(f)

    elec_x = np.arange(N_ELEC) * A
    survey = DipDipSurvey(a=A, n_electrodes=N_ELEC, n_max=6,
                          electrode_x=elec_x, array_type='dipole-dipole')
    mesh = Mesh2D(survey, depth_factor=2.5, dx_factor=0.25)

    X_train = model_dict['X_train']
    lo, hi = compute_ood_thresholds(X_train, percentile=98)
    scale = np.std(X_train, axis=0)
    ood_thresh = (lo, hi, scale)

    print('\n학습 데이터 feature 범위 (1%~99% percentile):')
    for i, name in enumerate(FEATURE_NAMES):
        print(f'  {name:18s}: [{lo[i]:7.2f}, {hi[i]:7.2f}]  '
              f'(평균={X_train[:,i].mean():.2f})')

    # ── Test 1: Phase 1 동일 (in-distribution 예상) ──
    print('\n' + '='*72)
    print(' Test 1: Phase 1 동일 조건 (학습 분포 내)')
    print('='*72)
    test_dips = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
    r1 = []
    np.random.seed(42)
    for td in test_dips:
        rho_true = build_dip_model(mesh, td, fault_x0=40, fault_z0=2, fault_thick=6)
        solver = ForwardSolver(mesh, rho_true)
        rho_a = solver.compute_data(survey, callback=lambda i,n: None)
        rho_a = rho_a * (1 + 0.03*np.random.randn(len(rho_a)))
        diag = diagnose_all(survey, rho_a, verbose=False)
        res = robust_predict(diag, model_dict, ood_thresh)
        err = res['estimate'] - td
        print(f'  진 {td:3d}°  →  {res["estimate"]:5.1f}°  err={err:+5.1f}°  '
              f'[{res["method"]:14s}]  OOD={res["ood_score"]:.2f}  '
              f'sev={res["max_severity"]:.1f}σ')
        r1.append({'td': td, 'est': res['estimate'], 'err': err,
                    'method': res['method'], 'ood': res['ood_score'],
                    'sev': res['max_severity'], 'unc': res['uncertainty']})
    mae1 = np.mean([abs(r['err']) for r in r1])
    print(f'  Test 1 MAE = {mae1:.2f}°')

    # ── Test 2: 학습 안 본 노이즈 (OOD 예상) ──
    print('\n' + '='*72)
    print(' Test 2: 학습 안 본 노이즈 1%, 7%, 10% (OOD 검출 예상)')
    print('='*72)
    test_dips_n = [15, 25, 35, 45]
    noises = [0.01, 0.07, 0.10]
    r2 = []
    for noise in noises:
        print(f'\n  노이즈 {noise*100:.0f}%:')
        for td in test_dips_n:
            np.random.seed(101)
            rho_true = build_dip_model(mesh, td, fault_x0=40, fault_z0=2, fault_thick=6)
            solver = ForwardSolver(mesh, rho_true)
            rho_a = solver.compute_data(survey, callback=lambda i,n: None)
            rho_a = rho_a * (1 + noise*np.random.randn(len(rho_a)))
            diag = diagnose_all(survey, rho_a, verbose=False)
            res = robust_predict(diag, model_dict, ood_thresh)
            err = res['estimate'] - td
            print(f'    진 {td:3d}°  →  {res["estimate"]:5.1f}°  err={err:+5.1f}°  '
                  f'[{res["method"]:14s}]  OOD={res["ood_score"]:.2f}')
            r2.append({'noise': noise, 'td': td, 'est': res['estimate'],
                        'err': err, 'method': res['method'], 'ood': res['ood_score']})
    mae2 = np.mean([abs(r['err']) for r in r2])
    print(f'\n  Test 2 MAE = {mae2:.2f}°')

    # ── Test 3: 현장 데이터 ──
    print('\n' + '='*72)
    print(' Test 3: 현장 데이터 (field_site_1/제당)')
    print('='*72)
    sites = [
        ('field_site_1', 'field_site_1.APV'),
        ('field_site_2', 'field_site_2.APV'),
    ]
    field_results = []
    for site_name, apv_path in sites:
        if not os.path.exists(apv_path): continue
        print(f'\n  [{site_name}]')
        data = parse_apv(apv_path)
        electrodes, measurements, rho_a = apv_to_survey_data(data)
        survey_f = DipDipSurvey(a=data['a'], n_electrodes=data['n_electrodes'],
                                n_max=data['n_max'],
                                electrode_x=electrodes, measurements=measurements)
        s2, ra, _ = filter_bad_data(survey_f, rho_a, verbose=False)
        diag = diagnose_all(s2, ra, verbose=False)

        # 학습 모델은 N_ELEC=30, n_max=6용. 현장 환경 다르면 OOD 가능성 큼.
        # 일단 같은 feature 추출 후 분석
        res = robust_predict(diag, model_dict, ood_thresh, verbose=True)
        field_results.append({'site': site_name, **res, 'diag': diag})

    # ── 요약 ──
    print('\n' + '='*72)
    print(' 종합 결과')
    print('='*72)
    print(f'  Test 1 (in-distribution): MAE = {mae1:.2f}°')
    print(f'  Test 2 (OOD noise):       MAE = {mae2:.2f}°')

    # 방법별 사용 빈도
    methods = [r['method'] for r in r1+r2]
    from collections import Counter
    cnt = Counter(methods)
    print('\n  방법 사용 빈도:')
    for m, c in cnt.most_common():
        print(f'    {m:18s}: {c}')

    # ── 시각화 ──
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    # (a) Test 1: in-dist
    ax = axes[0, 0]
    td_arr = np.array([r['td'] for r in r1])
    est_arr = np.array([r['est'] for r in r1])
    unc_arr = np.array([r['unc'] for r in r1])
    colors_m = ['#1f77b4' if r['method']=='ML' else '#d62728' for r in r1]
    ax.errorbar(td_arr, est_arr, yerr=unc_arr, fmt='o', ms=10,
                 capsize=4, color='gray', alpha=0.5)
    for i, (td, est, m) in enumerate(zip(td_arr, est_arr, [r['method'] for r in r1])):
        ax.scatter(td, est, s=120, c=colors_m[i], edgecolor='k', zorder=5,
                    label=m if m not in [c.get_label() for c in ax.collections[:-1]] else None)
    ax.plot([0, 65], [0, 65], 'k--', alpha=0.4)
    ax.set_xlabel('진 경사각 (°)', fontsize=11)
    ax.set_ylabel('Robust 예측 ± σ (°)', fontsize=11)
    ax.set_title(f'(a) In-distribution MAE={mae1:.2f}°', fontsize=11, fontweight='bold')
    ax.grid(alpha=0.3); ax.set_xlim(0, 65); ax.set_ylim(-5, 65)

    # (b) Test 2: OOD
    ax = axes[0, 1]
    colors_n = {0.01: '#2ca02c', 0.07: '#ff7f0e', 0.10: '#d62728'}
    for noise in [0.01, 0.07, 0.10]:
        sub = [r for r in r2 if r['noise']==noise]
        td = [r['td'] for r in sub]; est = [r['est'] for r in sub]
        ax.plot(td, est, '-o', label=f'노이즈 {noise*100:.0f}%',
                 color=colors_n[noise], ms=10, lw=2)
    ax.plot([0, 60], [0, 60], 'k--', alpha=0.4)
    ax.set_xlabel('진 경사각 (°)', fontsize=11)
    ax.set_ylabel('Robust 예측 (°)', fontsize=11)
    ax.set_title(f'(b) 학습 안 본 노이즈 MAE={mae2:.2f}°',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=10); ax.grid(alpha=0.3)

    # (c) OOD 점수 분포
    ax = axes[1, 0]
    all_ood = [r['ood'] for r in r1] + [r['ood'] for r in r2]
    all_td = [r['td'] for r in r1] + [r['td'] for r in r2]
    all_method = [r['method'] for r in r1] + [r['method'] for r in r2]
    colors = ['blue' if 'ML' in m else 'red' for m in all_method]
    ax.scatter(all_td, all_ood, c=colors, s=70, edgecolor='k', alpha=0.7)
    ax.axhline(0.85, color='red', ls='--', label='OOD 임계값 (0.85)')
    ax.set_xlabel('진 경사각 (°)', fontsize=11)
    ax.set_ylabel('OOD 점수 (1=완전 ID)', fontsize=11)
    ax.set_title('(c) OOD 검출 — 파랑=ML 사용, 빨강=Fallback',
                  fontsize=11, fontweight='bold')
    ax.legend(fontsize=9); ax.grid(alpha=0.3); ax.set_ylim(0, 1.05)

    # (d) 현장 데이터 진단
    ax = axes[1, 1]
    if field_results:
        names = [r['site'] for r in field_results]
        ml_preds = [r['ml_pred'] for r in field_results]
        finals = [r['estimate'] for r in field_results]
        oods = [r['ood_score'] for r in field_results]
        methods = [r['method'] for r in field_results]
        x = np.arange(len(names))
        w = 0.35
        ax.bar(x-w/2, ml_preds, w, label='ML 예측 (raw)', color='#1f77b4', edgecolor='k')
        ax.bar(x+w/2, finals, w, label='Robust 최종', color='#d62728', edgecolor='k')
        for i, (p, f, m, o) in enumerate(zip(ml_preds, finals, methods, oods)):
            ax.text(i-w/2, p+1, f'{p:.1f}°', ha='center', fontsize=9)
            ax.text(i+w/2, f+1, f'{f:.1f}°\n[{m}]\nOOD={o:.2f}',
                    ha='center', fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=11)
        ax.set_ylabel('추정 경사각 (°)', fontsize=11)
        ax.set_title('(d) 현장 데이터 — ML vs Robust', fontsize=11, fontweight='bold')
        ax.legend(fontsize=10); ax.grid(alpha=0.3, axis='y')

    fig.suptitle('Phase 3 Robust ML — OOD 검출 + Fallback',
                 fontsize=13, fontweight='bold', y=1.0)
    fig.tight_layout()
    out = os.path.join(OUTDIR, 'DipML_robust.png')
    fig.savefig(out, dpi=140, bbox_inches='tight')
    print(f'\n  → 그림 저장: {out}')

    print('\n' + '='*72)
    print(' 완료!')
    print('='*72)


if __name__ == '__main__':
    main()
