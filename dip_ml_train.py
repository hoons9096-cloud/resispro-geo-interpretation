#!/usr/bin/env python3
"""
경사각 자동 추정 — Phase 2 옵션 B (ML 회귀 모델)

다양한 fault 기하 + 노이즈 조합으로 훈련 데이터 생성 → Random Forest 회귀 학습

훈련 데이터 차원:
  진 경사: 5° ~ 60° (2.5° 간격, 23개)
  fault_x0: 30, 45, 60, 75 m (4개)
  fault_z0: 1, 2, 3 m (3개)
  fault_thick: 4, 6, 8 m (3개)
  rho_contrast: 10, 20 (2개)
  noise+seed: 3% (seed 42, 7, 123)
  → 총 23 × 4 × 3 × 3 × 2 × 3 = 4968 샘플 (×5초 ≈ 7시간)

실용적 축소:
  진 경사: 16개 (5, 8, 11, ..., 50, 55, 60)
  기하: 6개 대표 조합
  noise/seed: 3개
  → 16 × 6 × 3 = 288 샘플 (×3초 ≈ 14분)

Feature (입력 X):
  M1_theta, M1_coh
  M2_theta, M2_R2, M2_n_levels
  M3_dip_proxy, M3_WtoH, M3_shift_rate, M3_mean_width
  M4_mean, M4_std, M4_consistency
  M5_theta, M5_R2, M5_n_pts

Target (y): 진 경사각 (°)
"""
import sys, os, pickle, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv

sys.path.insert(0, '')
from RESIS_Pro import DipDipSurvey, Mesh2D, ForwardSolver
from dip_diagnostics import (diagnose_all, build_dip_model)

plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False
OUTDIR = ''

A = 5.0; N_ELEC = 30


def extract_features(diag):
    """진단 결과 → 14차원 feature vector (M2_n_levels 제거: 탐사 설계 파라미터)"""
    return np.array([
        diag['M1']['theta_med'],
        diag['M1']['coh_mean'],
        diag['M2']['theta'],
        diag['M2']['R2'],
        diag['M3']['dip_proxy'],
        diag['M3']['WtoH_ratio'],
        diag['M3']['shift_rate'],
        diag['M3']['mean_width'],
        diag['M4']['mean'],
        diag['M4']['std'],
        diag['M4']['consistency'],
        diag['M5']['theta'],
        diag['M5']['R2'],
        float(diag['M5']['n_pts']),
    ])

FEATURE_NAMES = [
    'M1_theta', 'M1_coh',
    'M2_theta', 'M2_R2',
    'M3_dip_proxy', 'M3_WtoH', 'M3_shift_rate', 'M3_mean_width',
    'M4_mean', 'M4_std', 'M4_consistency',
    'M5_theta', 'M5_R2', 'M5_n_pts',
]


def generate_training_data():
    """다양한 fault 기하 × 진 경사 × 노이즈 조합으로 훈련 데이터 생성."""
    print('='*72)
    print(' ML 훈련 데이터 생성 중...')
    print('='*72)

    elec_x = np.arange(N_ELEC) * A
    survey = DipDipSurvey(a=A, n_electrodes=N_ELEC, n_max=6,
                          electrode_x=elec_x, array_type='dipole-dipole')
    mesh = Mesh2D(survey, depth_factor=2.5, dx_factor=0.25)

    # 진 경사 (촘촘하게)
    true_dips = list(np.arange(5, 61, 3.5))   # 5, 8.5, 12, ..., 56.5, 60 (약 17개)

    # 대표 기하 조합
    geometries = [
        # (x0, z0, thick, rho_fault)
        (40, 2, 6, 20),    # 표준
        (40, 1, 8, 20),    # 얕은 두꺼운
        (50, 2, 5, 30),    # 중앙우, 얇은
        (35, 3, 6, 15),    # 좌측, 깊은 시작
        (60, 2, 7, 25),    # 우측 (좌우 대칭성)
        (45, 2, 6, 50),    # 약한 대비
    ]

    # 노이즈 시나리오 (1%~10% 폭넓게 커버)
    noise_seeds = [
        (0.01, 99),   # 1%
        (0.02, 11),   # 2%
        (0.03, 42),   # 3%
        (0.05, 7),    # 5%
        (0.07, 33),   # 7%
        (0.10, 55),   # 10%
    ]

    X_all = []; y_all = []; meta = []
    total = len(true_dips) * len(geometries) * len(noise_seeds)
    t0 = time.time()
    idx = 0
    for td in true_dips:
        for (x0, z0, thick, rho_f) in geometries:
            for (noise, seed) in noise_seeds:
                idx += 1
                np.random.seed(seed)
                rho_true = build_dip_model(mesh, td,
                                            fault_x0=x0, fault_z0=z0,
                                            fault_thick=thick,
                                            rho_fault=rho_f)
                solver = ForwardSolver(mesh, rho_true)
                rho_a = solver.compute_data(survey, callback=lambda i,n: None)
                rho_a = rho_a * (1 + noise*np.random.randn(len(rho_a)))
                diag = diagnose_all(survey, rho_a, verbose=False)
                feat = extract_features(diag)
                X_all.append(feat); y_all.append(td)
                meta.append({'td': td, 'x0': x0, 'z0': z0, 'thick': thick,
                             'rho_f': rho_f, 'noise': noise, 'seed': seed})
                if idx % 20 == 0:
                    elapsed = time.time() - t0
                    eta = elapsed/idx * (total-idx)
                    print(f'  [{idx:4d}/{total}]  진={td:.1f}° x0={x0} 기하 진행 중  '
                          f'(경과 {elapsed:.0f}s, ETA {eta:.0f}s)')

    X = np.array(X_all); y = np.array(y_all)
    print(f'\n  완료: X shape={X.shape}, y shape={y.shape}, 시간={time.time()-t0:.0f}s')
    return X, y, meta


def train_and_evaluate(X, y, meta):
    """Random Forest 훈련 + 평가."""
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.model_selection import train_test_split, KFold
    from sklearn.metrics import mean_absolute_error, r2_score

    print('\n' + '='*72)
    print(' 모델 훈련 시작')
    print('='*72)

    # 데이터 분할 (80/20)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f'   훈련: {len(X_tr)}샘플, 테스트: {len(X_te)}샘플')

    # 모델 1: Random Forest
    rf = RandomForestRegressor(n_estimators=300, max_depth=15,
                                 min_samples_leaf=2, random_state=42, n_jobs=-1)
    print('\n   Random Forest 훈련...')
    t0 = time.time()
    rf.fit(X_tr, y_tr)
    print(f'   완료 ({time.time()-t0:.1f}s)')
    y_pred_rf = rf.predict(X_te)
    mae_rf = mean_absolute_error(y_te, y_pred_rf)
    r2_rf = r2_score(y_te, y_pred_rf)

    # 모델 2: Gradient Boosting
    gb = GradientBoostingRegressor(n_estimators=300, max_depth=5,
                                     learning_rate=0.05, random_state=42)
    print('\n   Gradient Boosting 훈련...')
    t0 = time.time()
    gb.fit(X_tr, y_tr)
    print(f'   완료 ({time.time()-t0:.1f}s)')
    y_pred_gb = gb.predict(X_te)
    mae_gb = mean_absolute_error(y_te, y_pred_gb)
    r2_gb = r2_score(y_te, y_pred_gb)

    print(f'\n  Random Forest:    MAE={mae_rf:.2f}°, R²={r2_rf:.3f}')
    print(f'  Gradient Boosting: MAE={mae_gb:.2f}°, R²={r2_gb:.3f}')

    # K-Fold (5)
    print('\n   5-Fold Cross Validation (RF):')
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    maes = []
    for fi, (i_tr, i_te) in enumerate(kf.split(X)):
        rf_k = RandomForestRegressor(n_estimators=200, max_depth=15,
                                      min_samples_leaf=2, random_state=42, n_jobs=-1)
        rf_k.fit(X[i_tr], y[i_tr])
        y_p = rf_k.predict(X[i_te])
        m = mean_absolute_error(y[i_te], y_p)
        maes.append(m)
        print(f'    Fold {fi+1}: MAE={m:.2f}°')
    print(f'   평균 MAE: {np.mean(maes):.2f}° ± {np.std(maes):.2f}°')

    # Feature importance
    print('\n   Feature 중요도 (RF):')
    importances = rf.feature_importances_
    order = np.argsort(importances)[::-1]
    for i in order[:10]:
        print(f'     {FEATURE_NAMES[i]:18s}: {importances[i]:.4f}')

    # 베스트 모델 선택 (MAE가 낮은 것)
    best_model_name = 'RF' if mae_rf <= mae_gb else 'GB'
    best_model = rf if mae_rf <= mae_gb else gb
    best_mae = min(mae_rf, mae_gb)
    print(f'\n  → 최종 채택: {best_model_name} (MAE={best_mae:.2f}°)')

    # 모델 저장
    pkl_path = os.path.join(OUTDIR, 'dip_ml_model.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump({'model': best_model, 'feature_names': FEATURE_NAMES,
                     'rf': rf, 'gb': gb, 'X_train': X_tr, 'y_train': y_tr,
                     'kfold_mae_mean': float(np.mean(maes)),
                     'kfold_mae_std': float(np.std(maes))}, f)
    print(f'  → 모델 저장: {pkl_path}')

    # 시각화
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) 예측 vs 진값 (RF)
    ax = axes[0, 0]
    ax.scatter(y_te, y_pred_rf, alpha=0.5, s=30, c='blue', edgecolors='k', label='RF 예측')
    ax.plot([0, 65], [0, 65], 'k--', alpha=0.5, label='완벽 (y=x)')
    ax.set_xlabel('진 경사각 (°)', fontsize=11)
    ax.set_ylabel('RF 예측 (°)', fontsize=11)
    ax.set_title(f'(a) Random Forest  MAE={mae_rf:.2f}°, R²={r2_rf:.3f}',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    ax.set_xlim(0, 65); ax.set_ylim(0, 65)

    # (b) 예측 vs 진값 (GB)
    ax = axes[0, 1]
    ax.scatter(y_te, y_pred_gb, alpha=0.5, s=30, c='green', edgecolors='k', label='GB 예측')
    ax.plot([0, 65], [0, 65], 'k--', alpha=0.5)
    ax.set_xlabel('진 경사각 (°)', fontsize=11)
    ax.set_ylabel('GB 예측 (°)', fontsize=11)
    ax.set_title(f'(b) Gradient Boosting  MAE={mae_gb:.2f}°, R²={r2_gb:.3f}',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    ax.set_xlim(0, 65); ax.set_ylim(0, 65)

    # (c) Feature importance
    ax = axes[1, 0]
    sorted_idx = np.argsort(importances)
    ax.barh(range(len(FEATURE_NAMES)), importances[sorted_idx],
            color='steelblue', edgecolor='k')
    ax.set_yticks(range(len(FEATURE_NAMES)))
    ax.set_yticklabels([FEATURE_NAMES[i] for i in sorted_idx], fontsize=9)
    ax.set_xlabel('중요도', fontsize=11)
    ax.set_title('(c) Feature 중요도 (RF)', fontsize=11, fontweight='bold')
    ax.grid(alpha=0.3, axis='x')

    # (d) 오차 히스토그램
    ax = axes[1, 1]
    err_rf = y_pred_rf - y_te
    err_gb = y_pred_gb - y_te
    ax.hist(err_rf, bins=20, alpha=0.6, label=f'RF (MAE {mae_rf:.1f}°)',
            color='blue', edgecolor='k')
    ax.hist(err_gb, bins=20, alpha=0.6, label=f'GB (MAE {mae_gb:.1f}°)',
            color='green', edgecolor='k')
    ax.axvline(0, color='k', ls='--')
    ax.set_xlabel('예측 오차 (°)', fontsize=11)
    ax.set_ylabel('빈도', fontsize=11)
    ax.set_title('(d) 오차 분포', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    fig.suptitle('Phase 2-B: ML 회귀 모델 훈련 결과',
                 fontsize=13, fontweight='bold', y=1.0)
    fig.tight_layout()
    out = os.path.join(OUTDIR, 'DipML_training_results.png')
    fig.savefig(out, dpi=140, bbox_inches='tight')
    print(f'  → 그림 저장: {out}')

    return best_model, X_tr, y_tr, X_te, y_te


def main():
    # 1. 훈련 데이터 생성
    X, y, meta = generate_training_data()

    # 2. 훈련 + 평가
    model, X_tr, y_tr, X_te, y_te = train_and_evaluate(X, y, meta)

    print('\n' + '='*72)
    print(' 훈련 완료! 다음 단계: dip_ml_predict.py 실행')
    print(' (다양한 시나리오에서 모델 예측 검증)')
    print('='*72)


if __name__ == '__main__':
    main()
