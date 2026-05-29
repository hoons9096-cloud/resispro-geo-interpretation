#!/usr/bin/env python3
"""
Forward-Hypothesis Matching 엔진

관측 의사단면도 → 165개 지질 후보 라이브러리와 정규화 log-ρa 공간 비교
→ Pearson r + NRMSE 기반 스코어 → softmax 지지도 가중치

사용법:
  from forward_matcher import build_library_cache, match_observed

  # 1회 캐시 빌드 (약 8분):
  build_library_cache()

  # 매칭 (즉각):
  result = match_observed(survey, rho_a_obs)
"""

import sys, os, pickle, time
import numpy as np

sys.path.insert(0, '')
from RESIS_Pro import DipDipSurvey, Mesh2D, ForwardSolver
from geo_library import build_template_registry

CACHE_PATH = 'geo_template_cache.pkl'
A = 5.0; N_ELEC = 30; N_MAX = 6

# softmax 온도 (논문 T=0.08)
SOFTMAX_T = 0.08
# NRMSE 페널티 가중치
LAMBDA = 0.3


# ═══════════════════════════════════════════════════════════
#  정규화
# ═══════════════════════════════════════════════════════════

def _normalize(vec):
    """로그 겉보기비저항 → 평균0 표준편차1 정규화."""
    v = np.log10(np.maximum(vec, 1e-6))
    mu = v.mean(); sigma = v.std()
    if sigma < 1e-9:
        return v - mu
    return (v - mu) / sigma


# ═══════════════════════════════════════════════════════════
#  캐시 빌드
# ═══════════════════════════════════════════════════════════

def build_library_cache(force=False):
    """
    모든 템플릿의 forward response를 계산하고 캐시 파일에 저장.

    Parameters
    ----------
    force : bool
        True이면 기존 캐시를 무시하고 재계산.
    """
    if os.path.exists(CACHE_PATH) and not force:
        print(f'캐시 이미 존재: {CACHE_PATH}')
        print('  재계산하려면 build_library_cache(force=True)')
        return

    print('='*68)
    print(' 지질 후보 라이브러리 forward 계산 시작')
    print('='*68)

    elec_x = np.arange(N_ELEC) * A
    survey = DipDipSurvey(a=A, n_electrodes=N_ELEC, n_max=N_MAX,
                          electrode_x=elec_x, array_type='dipole-dipole')
    mesh = Mesh2D(survey, depth_factor=2.5, dx_factor=0.25)

    templates = build_template_registry()
    cache = []
    t0 = time.time()

    for i, tmpl in enumerate(templates):
        rho_true = tmpl['builder'](mesh)
        solver = ForwardSolver(mesh, rho_true)
        rho_a = solver.compute_data(survey, callback=lambda a, b: None)
        rho_a_norm = _normalize(rho_a)

        entry = {
            'family':   tmpl['family'],
            'name':     tmpl['name'],
            'dip_deg':  tmpl['dip_deg'],
            'params':   tmpl['params'],
            'rho_a':    rho_a,
            'rho_a_norm': rho_a_norm,
        }
        cache.append(entry)

        if (i + 1) % 20 == 0 or (i + 1) == len(templates):
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(templates) - i - 1)
            print(f'  [{i+1:3d}/{len(templates)}]  {tmpl["name"]:30s}  '
                  f'경과 {elapsed:.0f}s  ETA {eta:.0f}s')

    with open(CACHE_PATH, 'wb') as f:
        pickle.dump(cache, f)

    print(f'\n캐시 저장: {CACHE_PATH}  ({len(cache)}개 템플릿, {time.time()-t0:.0f}s)')
    return cache


def load_cache():
    """캐시 로드. 없으면 빌드."""
    if not os.path.exists(CACHE_PATH):
        print('캐시 없음. build_library_cache() 실행 필요.')
        build_library_cache()
    with open(CACHE_PATH, 'rb') as f:
        return pickle.load(f)


# ═══════════════════════════════════════════════════════════
#  매칭 함수
# ═══════════════════════════════════════════════════════════

def match_observed(rho_a_obs, cache=None, top_n=10):
    """
    관측 겉보기비저항 벡터를 캐시된 템플릿과 매칭.

    Parameters
    ----------
    rho_a_obs : array_like
        관측 겉보기비저항 벡터 (원래 단위, Ω·m).
    cache : list or None
        미리 로드한 캐시. None이면 load_cache() 호출.
    top_n : int
        반환할 상위 후보 수.

    Returns
    -------
    result : dict
        {
          'top1': {family, name, dip_deg, correlation, score, weight},
          'top_n': [...],
          'family_support': {family: sum_weight},
          'n_eff': float,          # 유효 후보 수 (엔트로피 기반)
          'all_scores': array,
          'all_weights': array,
        }
    """
    if cache is None:
        cache = load_cache()

    d_obs = _normalize(rho_a_obs)
    n = len(d_obs)

    scores = []
    for entry in cache:
        d_tmpl = entry['rho_a_norm']
        # 길이가 다르면 보간 (필드 데이터 n_max 불일치 대응)
        if len(d_tmpl) != n:
            idx = np.linspace(0, len(d_tmpl) - 1, n)
            d_tmpl = np.interp(idx, np.arange(len(d_tmpl)), d_tmpl)

        # Pearson 상관
        r = float(np.corrcoef(d_obs, d_tmpl)[0, 1])
        if not np.isfinite(r):
            r = -1.0

        # NRMSE
        nrmse = float(np.linalg.norm(d_obs - d_tmpl) /
                      (np.linalg.norm(d_obs) + 1e-9))

        s = r - LAMBDA * nrmse
        scores.append(s)

    scores = np.array(scores)

    # Softmax 지지도 가중치
    s_shift = scores - scores.max()   # 수치 안정화
    exp_s = np.exp(s_shift / SOFTMAX_T)
    weights = exp_s / (exp_s.sum() + 1e-30)

    # 유효 후보 수 (엔트로피 기반)
    eps = 1e-30
    h = -np.sum(weights * np.log(weights + eps))
    n_eff = float(np.exp(h))

    # 상위 후보 정렬
    order = np.argsort(scores)[::-1]
    top_list = []
    for idx in order[:top_n]:
        entry = cache[idx]
        d_tmpl = entry['rho_a_norm']
        if len(d_tmpl) != n:
            idx_interp = np.linspace(0, len(d_tmpl) - 1, n)
            d_tmpl = np.interp(idx_interp, np.arange(len(d_tmpl)), d_tmpl)
        r = float(np.corrcoef(d_obs, d_tmpl)[0, 1])
        top_list.append({
            'family':       entry['family'],
            'name':         entry['name'],
            'dip_deg':      entry['dip_deg'],
            'score':        float(scores[idx]),
            'correlation':  r,
            'weight':       float(weights[idx]),
        })

    # 패밀리별 지지도 합산
    family_support = {}
    for i, entry in enumerate(cache):
        fam = entry['family']
        family_support[fam] = family_support.get(fam, 0.0) + float(weights[i])

    return {
        'top1':           top_list[0],
        'top_n':          top_list,
        'family_support': family_support,
        'n_eff':          n_eff,
        'all_scores':     scores,
        'all_weights':    weights,
        'best_corr':      top_list[0]['correlation'],
    }


# ═══════════════════════════════════════════════════════════
#  통합 dip 추정: 매칭 + 진단 결합
# ═══════════════════════════════════════════════════════════

def integrated_dip_estimate(match_result, diag_estimate,
                             corr_threshold=0.85, verbose=True):
    """
    Forward-hypothesis 매칭 결과와 진단 추정치를 결합.

    corr >= corr_threshold  →  matched dip 우선
    corr <  corr_threshold  →  진단 추정치 (conservative)

    Returns
    -------
    dict: {dip, method, confidence, family, correlation, warning}
    """
    top1 = match_result['top1']
    corr = top1['correlation']
    matched_dip = top1['dip_deg']
    family = top1['family']
    fam_sup = match_result['family_support'].get(family, 0.0)
    n_eff = match_result['n_eff']

    if corr >= corr_threshold and matched_dip is not None:
        method = 'forward_hypothesis'
        dip = matched_dip
        confidence = min(1.0, corr * fam_sup * 10)
        warning = None
    else:
        method = 'diagnostic_fallback'
        dip = diag_estimate
        confidence = 0.40
        if corr < 0.5:
            warning = (f'낮은 템플릿 상관 ({corr:.3f}). '
                       f'현재 라이브러리가 현장 응답을 충분히 설명하지 못함.')
        else:
            warning = f'중간 상관 ({corr:.3f}). 진단 추정치 사용.'

    if verbose:
        print(f'  → 최종 경사 추정: {dip:.1f}°  [{method}]')
        print(f'     top1: {top1["name"]}  corr={corr:.3f}  '
              f'weight={top1["weight"]:.4f}  n_eff={n_eff:.1f}')
        print(f'     패밀리 지지도: {family}={fam_sup:.3f}')
        if warning:
            print(f'     ⚠  {warning}')

    return {
        'dip':         dip,
        'method':      method,
        'confidence':  confidence,
        'family':      family,
        'correlation': corr,
        'n_eff':       n_eff,
        'fam_support': fam_sup,
        'warning':     warning,
        'top1_name':   top1['name'],
    }


# ═══════════════════════════════════════════════════════════
#  단독 실행: 캐시 빌드
# ═══════════════════════════════════════════════════════════

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--force', action='store_true', help='캐시 강제 재계산')
    args = ap.parse_args()
    build_library_cache(force=args.force)
