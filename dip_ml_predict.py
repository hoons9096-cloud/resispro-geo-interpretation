#!/usr/bin/env python3
"""
경사각 자동 추정 — Phase 2-B ML 모델 검증

학습된 RF 모델을 학습 데이터에 없는 새 시나리오에서 테스트:
  1. 학습 안 본 fault 기하 (extrapolation)
  2. 학습 안 본 노이즈 수준
  3. 학습 시 사용한 시나리오와 동일 조건 (Phase 1, 2와 비교)
  4. field site 현장 데이터에 적용
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


def load_model():
    with open(os.path.join(OUTDIR, 'dip_ml_model.pkl'), 'rb') as f:
        return pickle.load(f)


def predict_dip(diag, model_dict):
    feat = extract_features(diag).reshape(1, -1)
    return float(model_dict['model'].predict(feat)[0])


# ════════════════════════════════════════════════════════
# 검증 시나리오
# ════════════════════════════════════════════════════════
def test_unseen_geometry():
    """학습 안 본 fault 기하 + 진 경사 조합."""
    print('\n' + '='*72)
    print(' [Test 1] 학습 안 본 fault 기하')
    print('='*72)

    elec_x = np.arange(N_ELEC) * A
    survey = DipDipSurvey(a=A, n_electrodes=N_ELEC, n_max=6,
                          electrode_x=elec_x, array_type='dipole-dipole')
    mesh = Mesh2D(survey, depth_factor=2.5, dx_factor=0.25)

    model_dict = load_model()

    # 학습 데이터: x0 ∈ {30, 35, 40, 45, 50, 60, 75}, z0 ∈ {1,2,3}, thick ∈ {4,6,8}
    # 검증: 안 본 조합
    test_set = [
        # (true_dip, x0, z0, thick, rho_f)
        (12, 55, 1.5, 5, 18),
        (18, 38, 2.5, 7, 22),
        (28, 42, 1.5, 6, 25),
        (33, 48, 2.0, 5, 18),
        (42, 52, 2.5, 7, 22),
        (52, 40, 1.0, 8, 30),
    ]

    np.random.seed(99)   # 학습과 다른 seed
    results = []
    for (td, x0, z0, thick, rho_f) in test_set:
        rho_true = build_dip_model(mesh, td, fault_x0=x0, fault_z0=z0,
                                    fault_thick=thick, rho_fault=rho_f)
        solver = ForwardSolver(mesh, rho_true)
        rho_a = solver.compute_data(survey, callback=lambda i,n: None)
        rho_a = rho_a * (1 + 0.04*np.random.randn(len(rho_a)))   # 4% 노이즈
        diag = diagnose_all(survey, rho_a, verbose=False)
        pred = predict_dip(diag, model_dict)
        err = pred - td
        print(f'  진 {td:3d}°  x0={x0} z0={z0} t={thick} ρf={rho_f}  →  '
              f'예측 {pred:5.1f}°  (오차 {err:+.1f}°)')
        results.append({'td': td, 'pred': pred, 'err': err,
                        'x0': x0, 'z0': z0, 't': thick, 'rho_f': rho_f})
    mae = np.mean([abs(r['err']) for r in results])
    print(f'  Test 1 MAE = {mae:.2f}°')
    return results, mae


def test_unseen_noise():
    """학습 안 본 노이즈 (1%, 7%, 10%)."""
    print('\n' + '='*72)
    print(' [Test 2] 학습 안 본 노이즈 수준')
    print('='*72)

    elec_x = np.arange(N_ELEC) * A
    survey = DipDipSurvey(a=A, n_electrodes=N_ELEC, n_max=6,
                          electrode_x=elec_x, array_type='dipole-dipole')
    mesh = Mesh2D(survey, depth_factor=2.5, dx_factor=0.25)
    model_dict = load_model()

    # 학습 노이즈: 2%, 3%, 5%
    # 검증: 1%, 7%, 10%
    test_dips = [15, 25, 35, 45]
    noises = [0.01, 0.07, 0.10]

    rows = []
    for noise in noises:
        print(f'\n  노이즈 {noise*100:.0f}%:')
        for td in test_dips:
            np.random.seed(101)
            rho_true = build_dip_model(mesh, td, fault_x0=40, fault_z0=2, fault_thick=6)
            solver = ForwardSolver(mesh, rho_true)
            rho_a = solver.compute_data(survey, callback=lambda i,n: None)
            rho_a = rho_a * (1 + noise*np.random.randn(len(rho_a)))
            diag = diagnose_all(survey, rho_a, verbose=False)
            pred = predict_dip(diag, model_dict)
            err = pred - td
            print(f'    진 {td:3d}°  →  예측 {pred:5.1f}°  (오차 {err:+.1f}°)')
            rows.append({'noise': noise, 'td': td, 'pred': pred, 'err': err})
    mae = np.mean([abs(r['err']) for r in rows])
    print(f'\n  Test 2 MAE = {mae:.2f}°')
    return rows, mae


def test_phase1_comparison():
    """Phase 1과 동일 조건에서 비교 (학습 시나리오 포함)."""
    print('\n' + '='*72)
    print(' [Test 3] Phase 1과 동일 조건 비교')
    print('='*72)

    elec_x = np.arange(N_ELEC) * A
    survey = DipDipSurvey(a=A, n_electrodes=N_ELEC, n_max=6,
                          electrode_x=elec_x, array_type='dipole-dipole')
    mesh = Mesh2D(survey, depth_factor=2.5, dx_factor=0.25)
    model_dict = load_model()

    test_dips = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
    rows = []
    np.random.seed(42)   # Phase 1과 동일
    for td in test_dips:
        rho_true = build_dip_model(mesh, td, fault_x0=40, fault_z0=2, fault_thick=6)
        solver = ForwardSolver(mesh, rho_true)
        rho_a = solver.compute_data(survey, callback=lambda i,n: None)
        rho_a = rho_a * (1 + 0.03*np.random.randn(len(rho_a)))
        diag = diagnose_all(survey, rho_a, verbose=False)
        pred = predict_dip(diag, model_dict)
        err = pred - td
        print(f'  진 {td:3d}°  →  예측 {pred:5.1f}°  (오차 {err:+.1f}°)')
        rows.append({'td': td, 'pred': pred, 'err': err})
    mae = np.mean([abs(r['err']) for r in rows])
    print(f'  Test 3 MAE = {mae:.2f}°')
    return rows, mae


def test_field_data():
    """현장 데이터 (field_site_1/제당)에 ML 적용."""
    print('\n' + '='*72)
    print(' [Test 4] 현장 데이터 — field site')
    print('='*72)
    model_dict = load_model()

    sites = [
        ('field_site_1', 'field_site_1.APV'),
        ('field_site_2', 'field_site_2.APV'),
    ]
    field_results = []
    for site_name, apv_path in sites:
        if not os.path.exists(apv_path):
            print(f'  ⚠️ {apv_path} 없음, 건너뜀')
            continue
        data = parse_apv(apv_path)
        electrodes, measurements, rho_a = apv_to_survey_data(data)
        survey_f = DipDipSurvey(a=data['a'], n_electrodes=data['n_electrodes'],
                                n_max=data['n_max'],
                                electrode_x=electrodes, measurements=measurements)
        s2, ra, _ = filter_bad_data(survey_f, rho_a, verbose=False)
        diag = diagnose_all(s2, ra, verbose=True)
        pred = predict_dip(diag, model_dict)
        print(f'  → {site_name}: ML 예측 경사각 = {pred:.1f}°')
        field_results.append({'site': site_name, 'pred': pred,
                              'M1': diag['M1']['theta_med'],
                              'M2': diag['M2']['theta'],
                              'M3p': diag['M3']['dip_proxy'],
                              'M4': diag['M4']['mean'],
                              'M5': diag['M5']['theta']})
    return field_results


# ════════════════════════════════════════════════════════
# 메인
# ════════════════════════════════════════════════════════
def main():
    print('='*72)
    print(' Phase 2-B ML 모델 검증 — 학습 데이터 외부에서 평가')
    print('='*72)

    r1, mae1 = test_unseen_geometry()
    r2, mae2 = test_unseen_noise()
    r3, mae3 = test_phase1_comparison()
    rf = test_field_data()

    # 통합 요약
    print('\n' + '='*72)
    print(' 종합 결과')
    print('='*72)
    print(f'  Test 1 (학습 안 본 기하):  MAE = {mae1:.2f}°')
    print(f'  Test 2 (학습 안 본 노이즈): MAE = {mae2:.2f}°')
    print(f'  Test 3 (Phase 1 동일 조건): MAE = {mae3:.2f}°')

    print(f'\n  📊 기존 방법 대비:')
    print(f'     의사단면도 ST 단독: MAE 12.81°')
    print(f'     Phase 2 결정 트리: MAE 15.51°')
    print(f'     ML 회귀 (RF):     MAE {mae3:.2f}°')

    # ── 시각화 ──
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    # (a) Test 1
    ax = axes[0, 0]
    td_arr = np.array([r['td'] for r in r1])
    pred_arr = np.array([r['pred'] for r in r1])
    ax.scatter(td_arr, pred_arr, s=100, c='blue', edgecolor='k', label='ML 예측')
    ax.plot([0, 60], [0, 60], 'k--', alpha=0.5, label='완벽 (y=x)')
    ax.set_xlabel('진 경사각 (°)', fontsize=11)
    ax.set_ylabel('ML 예측 (°)', fontsize=11)
    ax.set_title(f'(a) 학습 안 본 기하  MAE={mae1:.2f}°', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    ax.set_xlim(0, 60); ax.set_ylim(0, 60)

    # (b) Test 2 — 노이즈별
    ax = axes[0, 1]
    colors = {0.01: '#2ca02c', 0.07: '#ff7f0e', 0.10: '#d62728'}
    for noise in [0.01, 0.07, 0.10]:
        sub = [r for r in r2 if r['noise'] == noise]
        td = [r['td'] for r in sub]; pr = [r['pred'] for r in sub]
        ax.plot(td, pr, '-o', label=f'노이즈 {noise*100:.0f}%',
                 color=colors[noise], ms=10, lw=2)
    ax.plot([0, 60], [0, 60], 'k--', alpha=0.5)
    ax.set_xlabel('진 경사각 (°)', fontsize=11)
    ax.set_ylabel('ML 예측 (°)', fontsize=11)
    ax.set_title(f'(b) 학습 안 본 노이즈 MAE={mae2:.2f}°', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    ax.set_xlim(0, 60); ax.set_ylim(0, 60)

    # (c) Test 3 — Phase 1 비교
    ax = axes[1, 0]
    td_arr = np.array([r['td'] for r in r3])
    pred_arr = np.array([r['pred'] for r in r3])
    ax.plot(td_arr, pred_arr, 'o-', color='red', ms=10, lw=2,
            label=f'ML (MAE={mae3:.2f}°)')
    ax.plot([0, 65], [0, 65], 'k--', alpha=0.5, label='완벽')
    # 기존 단일 방법 대비
    ax.set_xlabel('진 경사각 (°)', fontsize=11)
    ax.set_ylabel('예측 경사각 (°)', fontsize=11)
    ax.set_title(f'(c) Phase 1 동일 조건  MAE={mae3:.2f}° (이전 12.8°)',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=10); ax.grid(alpha=0.3)
    ax.set_xlim(0, 65); ax.set_ylim(0, 65)

    # (d) 종합 오차 막대
    ax = axes[1, 1]
    labels = ['Test 1\n(기하)', 'Test 2\n(노이즈)', 'Test 3\n(Phase 1)']
    maes = [mae1, mae2, mae3]
    bars = ax.bar(labels, maes, color=['#1f77b4', '#ff7f0e', '#2ca02c'], edgecolor='k')
    for b, m in zip(bars, maes):
        ax.text(b.get_x()+b.get_width()/2, m+0.2, f'{m:.2f}°',
                ha='center', fontsize=12, fontweight='bold')
    ax.axhline(12.81, color='gray', ls='--', label='기존 의사단면도 ST (12.81°)')
    ax.axhline(15.51, color='red', ls=':', label='Phase 2 결정 트리 (15.51°)')
    ax.set_ylabel('MAE (°)', fontsize=11)
    ax.set_title('(d) 검증 결과 종합', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis='y')
    ax.set_ylim(0, 20)

    fig.suptitle('Phase 2-B ML 모델 — 일반화 능력 검증',
                 fontsize=13, fontweight='bold', y=1.00)
    fig.tight_layout()
    out = os.path.join(OUTDIR, 'DipML_validation.png')
    fig.savefig(out, dpi=140, bbox_inches='tight')
    print(f'\n  → 그림 저장: {out}')

    if rf:
        print('\n  📍 현장 데이터 결과:')
        for r in rf:
            print(f'     {r["site"]}: ML={r["pred"]:.1f}° '
                  f'(M1={r["M1"]:.1f}, M2={r["M2"]:.1f}, M4={r["M4"]:.1f})')

    print('\n' + '='*72)
    print(' 완료!')
    print('='*72)


if __name__ == '__main__':
    main()
