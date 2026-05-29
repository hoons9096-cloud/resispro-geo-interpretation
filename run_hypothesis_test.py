#!/usr/bin/env python3
"""
Forward-Hypothesis Matching 검증 — Table 1 & Table 2 재현

Table 1: 통제 합성 벤치마크 (known-dip + no-single-dip)
Table 2: 독립 기하 + 노이즈 민감도 (1%, 3%, 5%, 7%, 10%)
"""

import sys, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '')
from RESIS_Pro import DipDipSurvey, Mesh2D, ForwardSolver
from geo_library import (_build_clean_dip, _build_covered_dip,
                          _build_groundwater, _build_fault_zone,
                          _build_basement, _build_vertical_block,
                          _build_lens, _build_channel, _build_composite)
from dip_diagnostics import diagnose_all, build_dip_model
from forward_matcher import load_cache, match_observed, integrated_dip_estimate

plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False
OUTDIR = ''

A = 5.0; N_ELEC = 30; N_MAX = 6

# ─────────────────────────────────────────────────────────
#  공통: 탐사 설정
# ─────────────────────────────────────────────────────────
def make_survey_mesh():
    elec_x = np.arange(N_ELEC) * A
    survey = DipDipSurvey(a=A, n_electrodes=N_ELEC, n_max=N_MAX,
                          electrode_x=elec_x, array_type='dipole-dipole')
    mesh = Mesh2D(survey, depth_factor=2.5, dx_factor=0.25)
    return survey, mesh


def forward_noise(mesh, survey, rho_true, noise_pct=0.03, seed=42):
    np.random.seed(seed)
    solver = ForwardSolver(mesh, rho_true)
    rho_a = solver.compute_data(survey, callback=lambda a, b: None)
    rho_a = rho_a * (1 + noise_pct * np.random.randn(len(rho_a)))
    return rho_a


def diag_dip(survey, rho_a):
    """진단 방법 통합 추정 (M1 중앙값 우선)."""
    diag = diagnose_all(survey, rho_a, verbose=False)
    # 간단한 가중 평균: M1이 신뢰할 만하면 M1, 아니면 M2
    m1 = diag['M1']['theta_med']
    m1_coh = diag['M1']['coh_mean']
    m2 = diag['M2']['theta']
    m2_r2 = diag['M2']['R2']
    if m1_coh > 0.3 and m1 > 2:
        return float(m1)
    elif m2_r2 > 0.3 and m2 > 2:
        return float(m2)
    else:
        return float(diag['M4']['mean'])


# ═══════════════════════════════════════════════════════════
#  Table 1: 통제 합성 벤치마크
# ═══════════════════════════════════════════════════════════

def run_table1(survey, mesh, cache):
    print('='*72)
    print(' Table 1: 통제 합성 벤치마크')
    print('='*72)

    NOISE = 0.03
    SEED = 42
    rows = []

    # ── (a) clean dipping layer ─────────────────────────────
    for dip in [15, 25, 35]:
        rho_true = _build_clean_dip(mesh, dip, x0=40, z0=1.5, thick=6,
                                     rho_layer=20, rho_bg=200)
        rho_a = forward_noise(mesh, survey, rho_true, NOISE, SEED)
        d_est = diag_dip(survey, rho_a)
        mres = match_observed(rho_a, cache)
        int_res = integrated_dip_estimate(mres, d_est, verbose=False)
        rows.append(dict(
            case=f'Clean dipping layer',
            true_dip=dip, diag=d_est,
            fwd_family=int_res['family'],
            fwd_dip=int_res['dip'] if int_res['method'] == 'forward_hypothesis' else mres['top1']['dip_deg'],
            corr=mres['best_corr'],
        ))

    # ── (b) covered dipping layer ───────────────────────────
    for dip in [15, 25, 35]:
        rho_true = _build_covered_dip(mesh, dip, x0=40, z0=4.5, thick=6,
                                       cover_thick=3.0, rho_layer=15,
                                       rho_cover=60, rho_bg=200)
        rho_a = forward_noise(mesh, survey, rho_true, NOISE, SEED)
        d_est = diag_dip(survey, rho_a)
        mres = match_observed(rho_a, cache)
        int_res = integrated_dip_estimate(mres, d_est, verbose=False)
        rows.append(dict(
            case='Covered dipping layer',
            true_dip=dip, diag=d_est,
            fwd_family=int_res['family'],
            fwd_dip=int_res['dip'] if int_res['method'] == 'forward_hypothesis' else mres['top1']['dip_deg'],
            corr=mres['best_corr'],
        ))

    # ── (c) dipping groundwater ─────────────────────────────
    for dip in [10, 20, 30]:
        rho_true = _build_groundwater(mesh, dip, x0=40, z0=2.0, width=3,
                                       rho_gw=10, rho_bg=300)
        rho_a = forward_noise(mesh, survey, rho_true, NOISE, SEED)
        d_est = diag_dip(survey, rho_a)
        mres = match_observed(rho_a, cache)
        int_res = integrated_dip_estimate(mres, d_est, verbose=False)
        rows.append(dict(
            case='Dipping groundwater',
            true_dip=dip, diag=d_est,
            fwd_family=int_res['family'],
            fwd_dip=int_res['dip'] if int_res['method'] == 'forward_hypothesis' else mres['top1']['dip_deg'],
            corr=mres['best_corr'],
        ))

    # ── (d) conductive fault zone ───────────────────────────
    for dip in [30, 45, 60]:
        rho_true = _build_fault_zone(mesh, dip, x0=40, z0=0.5, thick=8,
                                      rho_fault=20, rho_bg=200)
        rho_a = forward_noise(mesh, survey, rho_true, NOISE, SEED)
        d_est = diag_dip(survey, rho_a)
        mres = match_observed(rho_a, cache)
        int_res = integrated_dip_estimate(mres, d_est, verbose=False)
        rows.append(dict(
            case='Conductive fault zone',
            true_dip=dip, diag=d_est,
            fwd_family=int_res['family'],
            fwd_dip=int_res['dip'] if int_res['method'] == 'forward_hypothesis' else mres['top1']['dip_deg'],
            corr=mres['best_corr'],
        ))

    # ── (e) dipping basement ────────────────────────────────
    for dip in [12, 22]:
        rho_true = _build_basement(mesh, dip, x0=40, z0=4.0,
                                    rho_above=80, rho_below=800)
        rho_a = forward_noise(mesh, survey, rho_true, NOISE, SEED)
        d_est = diag_dip(survey, rho_a)
        mres = match_observed(rho_a, cache)
        int_res = integrated_dip_estimate(mres, d_est, verbose=False)
        rows.append(dict(
            case='Dipping basement',
            true_dip=dip, diag=d_est,
            fwd_family=int_res['family'],
            fwd_dip=int_res['dip'] if int_res['method'] == 'forward_hypothesis' else mres['top1']['dip_deg'],
            corr=mres['best_corr'],
        ))

    # ── 출력 ────────────────────────────────────────────────
    print(f'\n{"케이스":22s}  {"진":>5}  {"진단":>6}  {"FWD 패밀리":20s}  '
          f'{"FWD경사":>7}  {"오차":>6}  {"상관":>6}')
    print('-' * 80)
    diag_errs = []; fwd_errs = []
    for r in rows:
        ferr = abs(r['fwd_dip'] - r['true_dip']) if r['fwd_dip'] else 999
        derr = abs(r['diag'] - r['true_dip'])
        diag_errs.append(derr); fwd_errs.append(ferr)
        fd = f"{r['fwd_dip']:.1f}°" if r['fwd_dip'] else 'N/A'
        print(f"  {r['case']:20s}  {r['true_dip']:>5.0f}°  "
              f"{r['diag']:>5.1f}°  {r['fwd_family']:20s}  "
              f"{fd:>7}  {ferr:>5.1f}°  {r['corr']:>5.3f}")

    diag_mae = np.mean(diag_errs)
    fwd_mae  = np.mean(fwd_errs)
    w5  = np.mean(np.array(fwd_errs) <= 5) * 100
    w10 = np.mean(np.array(fwd_errs) <= 10) * 100
    print('-' * 80)
    print(f'  진단 MAE = {diag_mae:.2f}°   FWD MAE = {fwd_mae:.2f}°')
    print(f'  5°이내 = {w5:.1f}%   10°이내 = {w10:.1f}%')

    # ── no-single-dip ────────────────────────────────────────
    print('\n  [비단일경사 케이스]')
    nsd_cases = [
        ('Vertical block',       _build_vertical_block(mesh, 45, 10, 20, 200)),
        ('Groundwater lens',     _build_lens(mesh, 65, 4, 20, 3, 15, 200)),
        ('Buried channel',       _build_channel(mesh, 70, 3, 20, 4, 15, 200)),
        ('Composite structure',  _build_composite(mesh, 15, 35, 1.5, 3,
                                                   55, 6, 10, 25, 200)),
    ]
    EXPECTED_FAMILY = {
        'Vertical block': 'vertical_block',
        'Groundwater lens': 'lens',
        'Buried channel': 'channel',
        'Composite structure': 'composite',
    }
    fam_correct = 0
    for name, rho_true in nsd_cases:
        rho_a = forward_noise(mesh, survey, rho_true, NOISE, SEED)
        mres = match_observed(rho_a, cache)
        top1 = mres['top1']
        correct = '✅' if top1['family'] == EXPECTED_FAMILY[name] else '❌'
        print(f'    {name:22s}  top1={top1["family"]:16s}  '
              f'corr={top1["correlation"]:.3f}  {correct}')
        if top1['family'] == EXPECTED_FAMILY[name]:
            fam_correct += 1
    print(f'  패밀리 정확도: {fam_correct}/{len(nsd_cases)}')

    return rows, diag_mae, fwd_mae


# ═══════════════════════════════════════════════════════════
#  Table 2: 독립 기하 + 노이즈 민감도
# ═══════════════════════════════════════════════════════════

def run_table2(survey, mesh, cache):
    print('\n' + '='*72)
    print(' Table 2: 독립 기하 + 노이즈 민감도')
    print('='*72)

    # 독립 케이스 (학습/캐시에 없는 기하)
    independent_cases = [
        ('Clean dip 28°',   28,
         _build_clean_dip(mesh, 28, x0=48, z0=1.0, thick=7, rho_layer=25, rho_bg=180)),
        ('Covered dip 32°', 32,
         _build_covered_dip(mesh, 32, x0=42, z0=5.5, thick=5,
                             cover_thick=4.0, rho_layer=18, rho_cover=55, rho_bg=180)),
        ('Groundwater 18°', 18,
         _build_groundwater(mesh, 18, x0=38, z0=2.5, width=4, rho_gw=8, rho_bg=350)),
        ('Fault zone 40°',  40,
         _build_fault_zone(mesh, 40, x0=52, z0=0.5, thick=10, rho_fault=18, rho_bg=220)),
        ('Basement 18°',    18,
         _build_basement(mesh, 18, x0=35, z0=3.0, rho_above=100, rho_below=600)),
    ]
    noise_levels = [0.01, 0.03, 0.05, 0.07, 0.10]

    print(f'\n{"케이스":20s}  {"노이즈":>6}  {"진":>5}  {"진단":>6}  '
          f'{"FWD경사":>7}  {"FWD오차":>7}  {"상관":>6}  {"패밀리":s}')
    print('-' * 85)

    diag_errs_all = []
    fwd_errs_all  = []
    noise_stats = {nl: {'diag': [], 'fwd': []} for nl in noise_levels}
    fam_correct = 0; total = 0

    EXPECTED = {
        'Clean dip 28°':   'clean_dip',
        'Covered dip 32°': 'covered_dip',
        'Groundwater 18°': 'groundwater',
        'Fault zone 40°':  'fault_zone',
        'Basement 18°':    'basement',
    }

    for case_name, true_dip, rho_true in independent_cases:
        for nl in noise_levels:
            rho_a = forward_noise(mesh, survey, rho_true, nl, seed=77)
            d_est = diag_dip(survey, rho_a)
            mres = match_observed(rho_a, cache)
            top1 = mres['top1']
            fwd_dip = top1['dip_deg'] if top1['dip_deg'] else d_est

            derr = abs(d_est - true_dip)
            ferr = abs(fwd_dip - true_dip) if fwd_dip else 999
            diag_errs_all.append(derr); fwd_errs_all.append(ferr)
            noise_stats[nl]['diag'].append(derr)
            noise_stats[nl]['fwd'].append(ferr)

            correct = '✅' if top1['family'] == EXPECTED[case_name] else '❌'
            if top1['family'] == EXPECTED[case_name]:
                fam_correct += 1
            total += 1

            fd = f'{fwd_dip:.1f}°' if fwd_dip else 'N/A'
            print(f'  {case_name:18s}  {nl*100:4.0f}%  '
                  f'{true_dip:>5.0f}°  {d_est:>5.1f}°  '
                  f'{fd:>7}  {ferr:>6.1f}°  {top1["correlation"]:>5.3f}  '
                  f'{top1["family"]} {correct}')

    print('-' * 85)
    diag_mae = np.mean(diag_errs_all)
    fwd_mae  = np.mean(fwd_errs_all)
    w5   = np.mean(np.array(fwd_errs_all) <= 5) * 100
    w10  = np.mean(np.array(fwd_errs_all) <= 10) * 100
    print(f'\n  전체 진단 MAE = {diag_mae:.2f}°   FWD MAE = {fwd_mae:.2f}°')
    print(f'  FWD: 5°이내 = {w5:.1f}%   10°이내 = {w10:.1f}%')
    print(f'  패밀리 정확도: {fam_correct}/{total} = {fam_correct/total*100:.1f}%')

    print('\n  [노이즈 수준별 MAE]')
    print(f'  {"노이즈":>6}  {"진단 MAE":>9}  {"FWD MAE":>8}')
    print('  ' + '-'*30)
    for nl in noise_levels:
        dm = np.mean(noise_stats[nl]['diag'])
        fm = np.mean(noise_stats[nl]['fwd'])
        print(f'  {nl*100:5.0f}%  {dm:>9.2f}°  {fm:>8.2f}°')

    return diag_mae, fwd_mae


# ═══════════════════════════════════════════════════════════
#  현장 데이터 검증
# ═══════════════════════════════════════════════════════════

def run_field(cache):
    from RESIS_Pro import parse_apv, apv_to_survey_data, filter_bad_data

    sites = [
        ('field_site_1', 'field_site_1.APV'),
        ('field_site_2', 'field_site_2.APV'),
    ]

    print('\n' + '='*72)
    print(' 현장 데이터 — Forward-Hypothesis Matching')
    print('='*72)

    for site_name, apv_path in sites:
        try:
            data = parse_apv(apv_path)
            electrodes, measurements, rho_a = apv_to_survey_data(data)
            survey = DipDipSurvey(
                a=data['a'], n_electrodes=data['n_electrodes'],
                n_max=data['n_max'], electrode_x=electrodes,
                measurements=measurements)
            survey, rho_a, _ = filter_bad_data(survey, rho_a, verbose=False)

            diag = diagnose_all(survey, np.array(rho_a), verbose=False)
            d_est = diag['M1']['theta_med']

            mres = match_observed(np.array(rho_a), cache)
            int_res = integrated_dip_estimate(mres, d_est, verbose=False)

            top1 = mres['top1']
            print(f'\n  [{site_name}]')
            print(f'    M1={diag["M1"]["theta_med"]:.1f}°  M4={diag["M4"]["mean"]:.1f}°')
            print(f'    top1: {top1["name"]}  corr={top1["correlation"]:.3f}  '
                  f'family={top1["family"]}  dip={top1["dip_deg"]}°')
            print(f'    최종: {int_res["dip"]:.1f}°  [{int_res["method"]}]')
            if int_res['warning']:
                print(f'    ⚠  {int_res["warning"]}')
        except Exception as e:
            print(f'  [{site_name}] 오류: {e}')


# ═══════════════════════════════════════════════════════════
#  메인
# ═══════════════════════════════════════════════════════════

def main():
    print('캐시 로드 중...')
    cache = load_cache()
    print(f'  {len(cache)}개 템플릿 로드 완료.')

    survey, mesh = make_survey_mesh()

    t1_rows, t1_diag_mae, t1_fwd_mae = run_table1(survey, mesh, cache)
    t2_diag_mae, t2_fwd_mae = run_table2(survey, mesh, cache)
    run_field(cache)

    print('\n' + '='*72)
    print(' 종합')
    print('='*72)
    print(f'  Table 1 (통제 벤치마크):  진단 MAE={t1_diag_mae:.2f}°  FWD MAE={t1_fwd_mae:.2f}°')
    print(f'  Table 2 (독립 + 노이즈):  진단 MAE={t2_diag_mae:.2f}°  FWD MAE={t2_fwd_mae:.2f}°')
    print('='*72)


if __name__ == '__main__':
    main()
