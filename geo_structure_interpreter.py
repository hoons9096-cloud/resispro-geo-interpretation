#!/usr/bin/env python3
"""
Geological structure interpreter for RESIS Pro.

This module does not try to force the inversion model to recover one exact
dip angle. It reads ERT apparent resistivity data, runs several fast
pseudosection-based structural diagnostics, checks whether the ML dip
estimator is within its training distribution, and writes an interpretation
report with uncertainty and recommended inversion strategy.
"""
import argparse
import csv
import glob
import os
import pickle
import re
from pathlib import Path

import numpy as np

ROOT = Path("")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from RESIS_Pro import DipDipSurvey, Mesh2D, parse_apv, apv_to_survey_data, filter_bad_data
from dip_diagnostics import diagnose_all
from dip_ml_train import FEATURE_NAMES, extract_features
from dip_ml_robust import compute_ood_thresholds, ood_score, ood_severity, rf_uncertainty


MODEL_PATH = ROOT / "dip_ml_model.pkl"

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False


def safe_name(text):
    """Filename-safe label while keeping Korean characters readable."""
    stem = Path(text).stem
    stem = re.sub(r"\s+", "_", stem)
    stem = re.sub(r"[^\w가-힣°.-]+", "_", stem)
    return stem.strip("_") or "dataset"


def finite(value, default=0.0):
    try:
        if value is None or not np.isfinite(value):
            return default
        return float(value)
    except Exception:
        return default


def load_ml_model(model_path=MODEL_PATH):
    if not Path(model_path).exists():
        return None
    with open(model_path, "rb") as f:
        return pickle.load(f)


def load_apv(path, apply_filter=True):
    data = parse_apv(str(path))
    electrodes, measurements, rho_a = apv_to_survey_data(data)
    survey = DipDipSurvey(
        a=data["a"],
        n_electrodes=data["n_electrodes"],
        n_max=data["n_max"],
        electrode_x=electrodes,
        measurements=measurements,
    )
    if apply_filter:
        survey, rho_a, _ = filter_bad_data(survey, rho_a, verbose=False)
    return data, survey, np.asarray(rho_a, dtype=float)


def robust_dip_estimate(diag, model_dict):
    """
    Predict structural dip with OOD detection.

    The best saved model is used for the raw prediction. RF tree scatter is
    used only as an uncertainty proxy because Gradient Boosting has no native
    ensemble scatter in this script.
    """
    feat = extract_features(diag)

    if model_dict is None:
        return fallback_dip_estimate(diag, feat, None, reason="no_ml_model")

    X_train = model_dict["X_train"]
    lo, hi = compute_ood_thresholds(X_train, percentile=98)
    scale = np.std(X_train, axis=0)
    scale[scale <= 1e-9] = 1.0

    ood = ood_score(feat, lo, hi)
    sev = ood_severity(feat, lo, hi, scale)
    max_sev = float(np.max(np.abs(sev)))
    worst_idx = int(np.argmax(np.abs(sev)))
    worst_feature = FEATURE_NAMES[worst_idx]

    model = model_dict.get("model", model_dict.get("rf"))
    ml_pred = float(model.predict(feat.reshape(1, -1))[0])
    ml_unc = 4.0
    if "rf" in model_dict:
        ml_unc, _ = rf_uncertainty(model_dict["rf"], feat)

    use_ml = (ood >= 0.85) and (max_sev <= 2.0)
    if use_ml:
        return {
            "estimate": ml_pred,
            "uncertainty": max(float(ml_unc), 2.0),
            "method": "ML",
            "confidence": min(0.92, max(0.55, ood)),
            "ood_score": float(ood),
            "max_severity": max_sev,
            "worst_feature": worst_feature,
            "ml_pred": ml_pred,
            "ml_unc": float(ml_unc),
            "is_ood": False,
            "fallback_reason": "",
            "features": feat,
        }

    result = fallback_dip_estimate(diag, feat, model_dict, reason="ood")
    result.update({
        "ood_score": float(ood),
        "max_severity": max_sev,
        "worst_feature": worst_feature,
        "ml_pred": ml_pred,
        "ml_unc": float(ml_unc),
        "is_ood": True,
        "features": feat,
    })
    return result


def fallback_dip_estimate(diag, feat, model_dict, reason):
    m1 = diag["M1"]
    m2 = diag["M2"]
    m4 = diag["M4"]

    m1_theta = finite(m1.get("theta_med"))
    m1_coh = finite(m1.get("coh_mean"))
    m2_theta = finite(m2.get("theta"))
    m2_r2 = finite(m2.get("R2"))
    m4_mean = finite(m4.get("mean"))
    m4_cons = finite(m4.get("consistency"))

    st_consensus = (
        5.0 <= m1_theta <= 55.0
        and 5.0 <= m4_mean <= 55.0
        and abs(m1_theta - m4_mean) <= 5.0
        and m1_coh >= 0.60
        and m4_cons >= 0.15
    )

    if st_consensus:
        vals = np.array([m1_theta, m4_mean])
        estimate = float(np.mean(vals))
        method = "Fallback_ST_consensus"
        uncertainty = max(5.0, float(np.std(vals)) * 2.0)
        confidence = 0.55
    elif 0.30 < m2_r2 < 0.90 and 5.0 <= m2_theta <= 55.0:
        estimate = m2_theta
        method = "Fallback_M2_nlevel"
        uncertainty = 6.0
        confidence = 0.52
    elif m1_coh >= 0.65 and 5.0 <= m1_theta <= 55.0:
        estimate = m1_theta
        method = "Fallback_M1_ST"
        uncertainty = 7.0
        confidence = 0.43
    else:
        vals = []
        weights = []
        if 5.0 <= m1_theta <= 55.0:
            vals.append(m1_theta)
            weights.append(max(m1_coh, 0.20))
        if 5.0 <= m2_theta <= 55.0:
            vals.append(m2_theta)
            weights.append(max(m2_r2, 0.20))
        if 5.0 <= m4_mean <= 55.0:
            vals.append(m4_mean)
            weights.append(max(m4_cons, 0.20))

        if vals:
            estimate = float(np.average(vals, weights=weights))
            spread = float(np.std(vals))
            uncertainty = max(8.0, spread)
            method = "Fallback_weighted"
            confidence = 0.35
        else:
            estimate = 0.0
            uncertainty = 12.0
            method = "Unresolved"
            confidence = 0.15

    return {
        "estimate": float(estimate),
        "uncertainty": float(uncertainty),
        "method": method,
        "confidence": float(confidence),
        "ood_score": 0.0,
        "max_severity": 0.0,
        "worst_feature": "",
        "ml_pred": np.nan,
        "ml_unc": np.nan,
        "is_ood": True,
        "fallback_reason": reason,
        "features": feat,
    }


def resistivity_context(survey, rho_a):
    rho_a = np.asarray(rho_a, dtype=float)
    med = float(np.median(rho_a))
    p10 = float(np.percentile(rho_a, 10))
    p90 = float(np.percentile(rho_a, 90))
    contrast = p90 / max(p10, 1e-9)

    n_values = np.array([m["n"] for m in survey.measurements])
    n1 = rho_a[n_values == 1]
    ndeep = rho_a[n_values == survey.n_max]
    shallow_med = float(np.median(n1)) if len(n1) else med
    deep_med = float(np.median(ndeep)) if len(ndeep) else med

    xs = np.array([m["x"] for m in survey.measurements])
    x_mid = float((survey.electrode_x[0] + survey.electrode_x[-1]) / 2)
    left_med = float(np.median(rho_a[xs < x_mid])) if np.any(xs < x_mid) else med
    right_med = float(np.median(rho_a[xs >= x_mid])) if np.any(xs >= x_mid) else med
    lateral_ratio = max(left_med, right_med) / max(min(left_med, right_med), 1e-9)

    return {
        "median": med,
        "p10": p10,
        "p90": p90,
        "contrast_ratio": float(contrast),
        "shallow_median": shallow_med,
        "deep_median": deep_med,
        "shallow_to_deep": shallow_med / max(deep_med, 1e-9),
        "left_median": left_med,
        "right_median": right_med,
        "lateral_ratio": float(lateral_ratio),
    }


def build_angle_candidates(diag, pred):
    """Collect separate angle indicators instead of hiding them in one number."""
    cands = []
    m1 = diag["M1"]
    m2 = diag["M2"]
    m3 = diag["M3"]
    m4 = diag["M4"]
    m5 = diag["M5"]

    cands.append({
        "name": "surface_ST",
        "angle": finite(m1.get("theta_med")),
        "quality": finite(m1.get("coh_mean")),
        "meaning": "의사단면 구조 텐서가 보는 주 경사",
    })
    cands.append({
        "name": "nlevel_centroid",
        "angle": finite(m2.get("theta")),
        "quality": finite(m2.get("R2")),
        "meaning": "n-level별 저비저항 중심 이동",
    })
    cands.append({
        "name": "buried_geometry",
        "angle": finite(m3.get("dip_proxy")),
        "quality": 1.0 / (1.0 + abs(finite(m3.get("WtoH_ratio")) - 4.0) / 6.0),
        "meaning": "저비저항대 폭/깊이비 기반 매몰 구조 후보",
    })
    cands.append({
        "name": "multiscale_ST",
        "angle": finite(m4.get("mean")),
        "quality": finite(m4.get("consistency")),
        "meaning": "여러 smoothing scale에서 일관된 ST 경사",
    })
    cands.append({
        "name": "contour_slope",
        "angle": finite(m5.get("theta")),
        "quality": finite(m5.get("R2")),
        "meaning": "저비저항 등치선 평균 기울기",
    })
    cands.append({
        "name": "raw_ML",
        "angle": finite(pred.get("ml_pred"), np.nan),
        "quality": 0.0 if pred.get("is_ood") else finite(pred.get("confidence")),
        "meaning": "훈련 분포 내일 때만 신뢰하는 ML 예측",
    })
    return cands


def adjust_for_buried_structure(diag, pred, ctx):
    """
    When conductive cover masks a dipping target, ST/M4 often collapse toward
    shallow angles. Keep the conservative estimate, but promote a geometry
    candidate as the final structural hypothesis with larger uncertainty.
    """
    adjusted = dict(pred)
    adjusted["conservative_estimate"] = pred["estimate"]
    adjusted["conservative_method"] = pred["method"]
    adjusted["buried_adjusted"] = False

    m1_theta = finite(diag["M1"].get("theta_med"))
    m1_coh = finite(diag["M1"].get("coh_mean"))
    m2_theta = finite(diag["M2"].get("theta"))
    m2_r2 = finite(diag["M2"].get("R2"))
    m3_theta = finite(diag["M3"].get("dip_proxy"))
    m3_wtoh = finite(diag["M3"].get("WtoH_ratio"))
    m4_theta = finite(diag["M4"].get("mean"))
    m4_cons = finite(diag["M4"].get("consistency"))

    st_consensus = (
        5.0 <= m1_theta <= 55.0
        and 5.0 <= m4_theta <= 55.0
        and abs(m1_theta - m4_theta) <= 5.0
        and m1_coh >= 0.60
        and m4_cons >= 0.12
    )
    st_est = 0.5 * (m1_theta + m4_theta)
    override_upward = (
        pred["estimate"] < st_est - 5.0
        and pred["estimate"] >= 15.0
    )
    override_downward = (
        pred["estimate"] > st_est + 10.0
        and not (
            pred["method"] == "ML"
            and pred["confidence"] >= 0.80
            and 15.0 <= pred["estimate"] <= 35.0
            and m2_theta >= st_est + 4.0
        )
    )
    if st_consensus and (override_upward or override_downward):
        adjusted["estimate"] = float(st_est)
        adjusted["uncertainty"] = max(5.0, abs(m1_theta - m4_theta), pred["uncertainty"])
        adjusted["method"] = "ST_consensus_override"
        adjusted["confidence"] = max(pred["confidence"], 0.58)
        adjusted["buried_adjusted"] = False
        adjusted["buried_basis"] = (
            f"ML/기타 추정({pred['estimate']:.1f}°)과 달리 M1/M4 구조텐서가 "
            f"{st_est:.1f}° 부근에서 일치하여 ST 합의를 우선했습니다."
        )
        return adjusted

    conductive_cover = ctx["shallow_to_deep"] < 0.75
    cover_nlevel_candidate = (
        conductive_cover
        and 10.0 <= m2_theta <= 45.0
        and (m2_r2 >= 0.15 or (m3_theta >= 45.0 and m3_wtoh >= 5.0))
        and max(m1_theta, m4_theta, pred["estimate"]) < (m2_theta - 7.0)
    )
    if cover_nlevel_candidate:
        vals = [m2_theta]
        weights = [max(0.50, m2_r2)]
        if 5.0 <= m3_theta <= 65.0:
            vals.append(min(m3_theta, 45.0))
            weights.append(0.25)
        if 5.0 <= m1_theta <= 55.0 and m1_coh > 0.50:
            vals.append(m1_theta)
            weights.append(0.10)
        if 5.0 <= m4_theta <= 55.0 and m4_cons > 0.20:
            vals.append(m4_theta)
            weights.append(0.10)

        new_est = float(np.average(vals, weights=weights))
        spread = float(np.std(vals)) if len(vals) >= 2 else 8.0
        adjusted["estimate"] = new_est
        adjusted["uncertainty"] = max(pred["uncertainty"], spread, 8.0)
        adjusted["method"] = "Covered_nlevel_hybrid"
        adjusted["confidence"] = min(max(pred["confidence"], 0.48), 0.56)
        adjusted["buried_adjusted"] = True
        adjusted["buried_basis"] = (
            "전도성 표층/충적층 가능성이 있고 ST가 저각으로 눌렸지만, "
            f"n-level 중심 이동이 {m2_theta:.1f}°를 지시합니다."
        )
        return adjusted

    geometry_candidate = 10.0 <= m3_theta <= 45.0 and m3_wtoh >= 2.0
    surface_flattened = max(m1_theta, m4_theta, pred["estimate"]) < (m3_theta - 6.0)
    fallback_state = pred["method"].startswith("Fallback") or pred["method"] == "Unresolved"

    if conductive_cover and geometry_candidate and surface_flattened and fallback_state:
        vals = [m3_theta]
        weights = [0.50]
        if 5.0 <= m2_theta <= 55.0 and m2_r2 > 0.15:
            vals.append(m2_theta)
            weights.append(0.25 + 0.25 * min(m2_r2, 1.0))
        if 5.0 <= m1_theta <= 55.0 and m1_coh > 0.50:
            vals.append(m1_theta)
            weights.append(0.12)
        if 5.0 <= m4_theta <= 55.0 and m4_cons > 0.20:
            vals.append(m4_theta)
            weights.append(0.13)

        new_est = float(np.average(vals, weights=weights))
        spread = float(np.std(vals)) if len(vals) >= 2 else 8.0
        adjusted["estimate"] = new_est
        adjusted["uncertainty"] = max(pred["uncertainty"], spread, 8.0)
        adjusted["method"] = "Buried_geometry_hybrid"
        adjusted["confidence"] = min(max(pred["confidence"], 0.40), 0.52)
        adjusted["buried_adjusted"] = True
        adjusted["buried_basis"] = (
            "전도성 표층/충적층 가능성이 있고, ST 계열은 저각으로 눌리지만 "
            f"종횡비 기반 매몰 구조 후보가 {m3_theta:.1f}°를 지시합니다."
        )
    else:
        adjusted["buried_basis"] = ""

    steep_st_saturation = (
        38.0 <= m1_theta <= 52.0
        and 38.0 <= m4_theta <= 52.0
        and abs(m1_theta - m4_theta) <= 8.0
        and diag["M3"].get("category") in ("mid", "high")
    )
    if steep_st_saturation:
        adjusted["uncertainty"] = max(adjusted["uncertainty"], 14.0)
        adjusted["method"] = adjusted["method"] + "_steep_range"
        adjusted["buried_basis"] = (
            adjusted.get("buried_basis", "")
            + " ST가 40-50° 부근에 포화되어 실제 고각 단층은 30-60° 후보 범위로 해석해야 합니다."
        ).strip()

    return adjusted


def angle_class(theta):
    theta = finite(theta)
    if theta < 7:
        return "near_horizontal", "수평 또는 완만한 층상 구조"
    if theta < 20:
        return "low_dip", "저각 경사 구조"
    if theta < 35:
        return "medium_dip", "중각 경사 구조"
    if theta < 55:
        return "steep_dip", "급경사 단층/전도성 파쇄대 가능"
    return "very_steep", "고각 단층 또는 블록 경계 가능"


def build_geological_hypotheses(diag, pred, ctx):
    theta = pred["estimate"]
    cls, cls_label = angle_class(theta)
    m1 = diag["M1"]
    m2 = diag["M2"]
    m3 = diag["M3"]
    m4 = diag["M4"]

    hypotheses = []
    warnings = []

    contrast = ctx["contrast_ratio"]
    conductive_cover = (
        ctx["shallow_to_deep"] < 0.85
        and ctx["shallow_median"] < ctx["median"] * 1.05
    )
    strong_anomaly = contrast >= 2.0
    coherent_st = finite(m1.get("coh_mean")) >= 0.55
    nlevel_good = finite(m2.get("R2")) >= 0.35
    multiscale_good = finite(m4.get("consistency")) >= 0.06
    wtoh = finite(m3.get("WtoH_ratio"))

    if conductive_cover:
        hypotheses.append({
            "name": "충적층 또는 전도성 표층 피복",
            "likelihood": "중",
            "basis": "n=1 겉보기비저항 중앙값이 심부 n-level보다 낮습니다.",
        })

    if cls in ("low_dip", "medium_dip") and coherent_st:
        hypotheses.append({
            "name": "경사 지층 또는 경사 지하수대",
            "likelihood": "상" if pred["confidence"] >= 0.55 else "중",
            "basis": f"구조 텐서와 통합 추정이 {theta:.1f}° 부근의 연속 경사를 지시합니다.",
        })

    if cls in ("medium_dip", "steep_dip") and (nlevel_good or wtoh < 4.0):
        hypotheses.append({
            "name": "단층대 또는 전도성 파쇄대",
            "likelihood": "중",
            "basis": "n-level 이상체 중심 이동 또는 종횡비가 경사성 불연속을 지시합니다.",
        })

    if cls in ("steep_dip", "very_steep"):
        hypotheses.append({
            "name": "고각 단층/블록 경계",
            "likelihood": "중",
            "basis": "추정 경사가 커서 경계성 구조 가능성이 있으나 ERT 단독 각도 분해능은 낮습니다.",
        })

    if strong_anomaly and cls in ("low_dip", "medium_dip", "steep_dip"):
        hypotheses.append({
            "name": "지하수 또는 점토질 전도성 이상대",
            "likelihood": "중",
            "basis": f"겉보기비저항 p90/p10 대비가 {contrast:.1f}배로 전도성 이상대가 뚜렷합니다.",
        })

    if not hypotheses:
        hypotheses.append({
            "name": cls_label,
            "likelihood": "저",
            "basis": "진단 방법 간 일치도가 낮아 구조 해석은 보수적으로 보아야 합니다.",
        })

    method_values = np.array([
        finite(m1.get("theta_med"), np.nan),
        finite(m2.get("theta"), np.nan),
        finite(m4.get("mean"), np.nan),
        finite(pred.get("estimate"), np.nan),
    ], dtype=float)
    method_values = method_values[np.isfinite(method_values) & (method_values > 0)]
    disagreement = float(np.std(method_values)) if len(method_values) >= 2 else 0.0

    if pred["is_ood"]:
        warnings.append(
            f"ML 입력 특성이 학습 범위를 벗어났습니다. 최악 feature={pred['worst_feature']}, "
            f"편차={pred['max_severity']:.1f}σ."
        )
    if disagreement > 12.0:
        warnings.append(
            f"진단 방법 간 경사 추정 차이가 큽니다(표준편차 {disagreement:.1f}°). "
            "단일 각도보다 후보 구조군으로 해석해야 합니다."
        )
    if theta >= 35.0:
        warnings.append(
            "35° 이상 구조는 ERT 의사단면/역산에서 수평화 편향과 DOI 한계가 커집니다. "
            "각도 값보다 단층대 위치와 연속성 판단을 우선해야 합니다."
        )
    if not multiscale_good:
        warnings.append("다중 스케일 구조 텐서 일관성이 낮습니다. 노이즈 또는 복합 구조 가능성이 있습니다.")
    if pred.get("buried_adjusted"):
        warnings.append(
            "최종 경사는 보수 ST 추정이 아니라 매몰 구조 후보까지 반영한 값입니다. "
            "리포트의 conservative estimate도 함께 확인해야 합니다."
        )

    return hypotheses, warnings, disagreement


def run_hypothesis_matching(survey, mesh, rho_a):
    """
    Lazy wrapper to avoid a hard import cycle.

    Returns None if the hypothesis library cannot be built for any reason.
    """
    try:
        from geo_hypothesis_matching import build_library, match_hypotheses, summarize_top
        responses, meta = build_library(survey, mesh, cache=True)
        top = match_hypotheses(rho_a, responses, meta, topk=8)
        return summarize_top(top)
    except Exception as exc:
        return {"error": str(exc)}


def match_likelihood(corr):
    corr = finite(corr)
    if corr >= 0.85:
        return "상"
    if corr >= 0.60:
        return "중"
    return "저"


def recommend_strategy(pred, diag, ctx):
    theta = pred["estimate"]
    cls, _ = angle_class(theta)

    if pred["confidence"] >= 0.70 and not pred["is_ood"]:
        confidence_label = "높음"
    elif pred["confidence"] >= 0.45:
        confidence_label = "중간"
    else:
        confidence_label = "낮음"

    if cls in ("near_horizontal", "low_dip"):
        mode = "Cell+L2 또는 약한 STAR"
        action = "저각/완만 구조로 보고 경사 평활 강도는 낮게 두고, 표층 피복 여부를 먼저 확인합니다."
    elif cls == "medium_dip":
        mode = "STAR + MGS 보조"
        action = "20-35° 후보 경사를 중심으로 STAR를 적용하고 MGS로 경계 위치를 대조합니다."
    elif cls == "steep_dip":
        mode = "MGS + STAR 후보각 비교"
        action = "단일 각도 복원보다 단층대 위치/폭/연속성을 우선하고, 30-60° 후보군을 비교합니다."
    else:
        mode = "MGS/블록성 해석 + DOI 검토"
        action = "고각 구조는 블록 경계로 표현될 가능성이 높으므로 각도 수치 해석은 제한적으로 사용합니다."

    if pred["is_ood"]:
        action += " 현재 자료는 ML 학습 분포 밖이므로 자동 각도는 보조 지표로만 사용합니다."

    return {
        "confidence_label": confidence_label,
        "mode": mode,
        "action": action,
        "dip_range": (
            max(0.0, theta - pred["uncertainty"]),
            min(89.0, theta + pred["uncertainty"]),
        ),
    }


def interpret_survey_data(survey, rho_a, outdir, name="dataset", model_dict=None,
                          mesh=None, use_hypothesis=True, source_path=""):
    if mesh is None:
        mesh = Mesh2D(survey)
    diag = diagnose_all(survey, rho_a, verbose=False)
    pred = robust_dip_estimate(diag, model_dict)
    ctx = resistivity_context(survey, rho_a)
    pred = adjust_for_buried_structure(diag, pred, ctx)
    candidates = build_angle_candidates(diag, pred)
    hypothesis_match = run_hypothesis_matching(survey, mesh, rho_a) if use_hypothesis else None
    hypotheses, warnings, disagreement = build_geological_hypotheses(diag, pred, ctx)
    if hypothesis_match and not hypothesis_match.get("error"):
        best_family = hypothesis_match.get("best_family", "")
        best_dip = hypothesis_match.get("dip_estimate")
        if best_dip is not None:
            hypotheses.insert(0, {
                "name": f"Forward-matched {best_family}",
                "likelihood": match_likelihood(hypothesis_match.get("best_corr", 0)),
                "basis": (
                    f"후보 forward response 매칭 결과 {best_family}, "
                    f"경사 후보 {best_dip:.1f}°가 가장 높은 상관을 보였습니다 "
                    f"(corr={hypothesis_match.get('best_corr', 0):.2f})."
                ),
            })
        else:
            hypotheses.insert(0, {
                "name": f"Forward-matched {best_family}",
                "likelihood": match_likelihood(hypothesis_match.get("best_corr", 0)),
                "basis": (
                    f"후보 forward response 매칭 결과 {best_family} 계열이 가장 유사했습니다 "
                    f"(corr={hypothesis_match.get('best_corr', 0):.2f})."
                ),
            })
    elif hypothesis_match and hypothesis_match.get("error"):
        warnings.append(f"Forward hypothesis matching 실패: {hypothesis_match['error']}")
    rec = recommend_strategy(pred, diag, ctx)

    name = safe_name(name)
    result = {
        "name": name,
        "path": str(source_path),
        "n_electrodes": survey.n_electrodes,
        "a": survey.a,
        "n_max": survey.n_max,
        "n_data": len(rho_a),
        "diag": diag,
        "prediction": pred,
        "context": ctx,
        "candidates": candidates,
        "hypothesis_match": hypothesis_match,
        "hypotheses": hypotheses,
        "warnings": warnings,
        "disagreement": disagreement,
        "recommendation": rec,
    }

    write_markdown_report(result, outdir)
    plot_interpretation(result, survey, rho_a, outdir)
    return result


def interpret_dataset(apv_path, outdir, model_dict=None, apply_filter=True, use_hypothesis=True):
    data, survey, rho_a = load_apv(apv_path, apply_filter=apply_filter)
    mesh = Mesh2D(survey)
    return interpret_survey_data(
        survey, rho_a, outdir,
        name=Path(apv_path).stem,
        model_dict=model_dict,
        mesh=mesh,
        use_hypothesis=use_hypothesis,
        source_path=str(apv_path),
    )


def write_markdown_report(result, outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    pred = result["prediction"]
    rec = result["recommendation"]
    diag = result["diag"]
    ctx = result["context"]

    lines = [
        f"# Geological Structure Interpretation: {result['name']}",
        "",
        "## Data",
        f"- File: `{result['path']}`",
        f"- Electrodes: {result['n_electrodes']}, spacing a={result['a']} m, n_max={result['n_max']}",
        f"- Data count after filtering: {result['n_data']}",
        "",
        "## Structural Dip Estimate",
        f"- Final estimate: **{pred['estimate']:.1f}° ± {pred['uncertainty']:.1f}°**",
        f"- Method: `{pred['method']}`",
        f"- Confidence: {rec['confidence_label']} ({pred['confidence']:.2f})",
        f"- OOD score: {pred['ood_score']:.2f}, max severity: {pred['max_severity']:.1f}σ ({pred['worst_feature']})",
        f"- Raw ML prediction: {pred['ml_pred']:.1f}° ± {pred['ml_unc']:.1f}°",
        f"- Conservative estimate: {pred.get('conservative_estimate', pred['estimate']):.1f}° ({pred.get('conservative_method', pred['method'])})",
        "",
    ]
    hm = result.get("hypothesis_match")
    if hm:
        lines.append("## Forward Hypothesis Matching")
        if hm.get("error"):
            lines.append(f"- Error: {hm['error']}")
        else:
            dip = hm.get("dip_estimate")
            dip_txt = "no single dip" if dip is None else f"{dip:.1f}°"
            spread = hm.get("dip_spread")
            spread_txt = "" if spread is None else f" ± {spread:.1f}°"
            lines.extend([
                f"- Best family: **{hm.get('best_family', '')}**",
                f"- Best template: `{hm.get('best_name', '')}`",
                f"- Matched dip: **{dip_txt}{spread_txt}**",
                f"- Correlation: {hm.get('best_corr', 0):.3f}, score={hm.get('best_score', 0):.3f}",
                f"- Relative probability: {hm.get('best_prob', 0):.3f}, effective candidates={hm.get('effective_n', 0):.2f}",
                f"- Family probabilities: {format_family_probs(hm.get('family_probs', {}))}",
                "- Top candidates:",
            ])
            for item in hm.get("top", [])[:5]:
                idip = item.get("dip")
                idip_txt = "none" if idip is None else f"{idip:.1f}°"
                lines.append(
                    f"  - {item.get('name')} ({item.get('family')}), "
                    f"dip={idip_txt}, corr={item.get('corr', 0):.3f}, p={item.get('prob', 0):.3f}"
                )
        lines.append("")

    lines.append("## Angle Candidates")
    for c in result["candidates"]:
        angle = c["angle"]
        angle_txt = "nan" if not np.isfinite(angle) else f"{angle:.1f}°"
        lines.append(f"- {c['name']}: {angle_txt}, quality={c['quality']:.2f} - {c['meaning']}")

    lines.extend([
        "",
        "## Diagnostic Methods",
        f"- M1 pseudosection ST: {diag['M1']['theta_med']:.1f}° (coherence={diag['M1']['coh_mean']:.2f})",
        f"- M2 n-level centroid: {diag['M2']['theta']:.1f}° (R2={diag['M2']['R2']:.2f})",
        f"- M3 aspect ratio: {diag['M3']['category']} (W/H={diag['M3']['WtoH_ratio']:.2f}, proxy={diag['M3']['dip_proxy']:.1f}°)",
        f"- M4 multiscale ST: {diag['M4']['mean']:.1f}° (std={diag['M4']['std']:.1f}°, consistency={diag['M4']['consistency']:.2f})",
        f"- M5 contour slope: {diag['M5']['theta']:.1f}° (R2={diag['M5']['R2']:.2f})",
        "",
        "## Resistivity Context",
        f"- Median apparent resistivity: {ctx['median']:.1f} ohm-m",
        f"- p10-p90 contrast ratio: {ctx['contrast_ratio']:.2f}",
        f"- Shallow/deep median ratio: {ctx['shallow_to_deep']:.2f}",
        f"- Left/right median contrast ratio: {ctx['lateral_ratio']:.2f}",
        "",
        "## Geological Hypotheses",
    ])
    for h in result["hypotheses"]:
        lines.append(f"- **{h['name']}** ({h['likelihood']}): {h['basis']}")

    lines.extend([
        "",
        "## Recommended Processing",
        f"- Mode: **{rec['mode']}**",
        f"- Candidate dip range: {rec['dip_range'][0]:.1f}° - {rec['dip_range'][1]:.1f}°",
        f"- Action: {rec['action']}",
    ])

    if result["warnings"]:
        lines.extend(["", "## Cautions"])
        for w in result["warnings"]:
            lines.append(f"- {w}")

    report_path = outdir / f"GeoInterp_report_{result['name']}.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_interpretation(result, survey, rho_a, outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    diag = result["diag"]
    pred = result["prediction"]
    rec = result["recommendation"]

    xs = np.array([m["x"] for m in survey.measurements])
    zs = np.array([m["z"] for m in survey.measurements])
    ns = np.array([m["n"] for m in survey.measurements])
    log_rho = np.log10(np.maximum(rho_a, 1.0))

    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1.0], height_ratios=[1.0, 1.0])

    ax = fig.add_subplot(gs[:, 0])
    sc = ax.scatter(xs, zs, c=log_rho, s=52, cmap="turbo", edgecolor="k", linewidth=0.25)
    for c in diag["M2"].get("centroids", []):
        ax.plot(c[1], c[2], "wo", ms=7, mec="k", mew=1.0)
    ax.set_title("Pseudosection and n-level anomaly centers", fontweight="bold")
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Pseudodepth (m)")
    ax.set_ylim(max(zs) + survey.a * 0.4, 0)
    cb = fig.colorbar(sc, ax=ax, shrink=0.88)
    cb.set_label("log10 apparent resistivity")
    ax.grid(alpha=0.25)

    ax = fig.add_subplot(gs[0, 1])
    labels = ["M1 ST", "M2 n-level", "M3 aspect", "M4 multi-ST", "M5 contour", "Final"]
    vals = [
        finite(diag["M1"].get("theta_med")),
        finite(diag["M2"].get("theta")),
        finite(diag["M3"].get("dip_proxy")),
        finite(diag["M4"].get("mean")),
        finite(diag["M5"].get("theta")),
        finite(pred.get("estimate")),
    ]
    colors = ["#4c78a8", "#f58518", "#54a24b", "#b279a2", "#e45756", "#222222"]
    ax.barh(np.arange(len(vals)), vals, color=colors, edgecolor="k", alpha=0.88)
    ax.errorbar(pred["estimate"], len(vals) - 1, xerr=pred["uncertainty"], color="white",
                ecolor="black", capsize=4, lw=2)
    ax.set_yticks(np.arange(len(vals)))
    ax.set_yticklabels(labels)
    ax.set_xlim(0, max(65, max(vals) + 10))
    ax.set_xlabel("Dip estimate (deg)")
    ax.set_title("Structural diagnostics", fontweight="bold")
    ax.grid(alpha=0.25, axis="x")

    ax = fig.add_subplot(gs[1, 1])
    ax.axis("off")
    h_lines = [f"- {h['name']} ({h['likelihood']})" for h in result["hypotheses"][:4]]
    warning = result["warnings"][0] if result["warnings"] else "No major caution beyond ERT non-uniqueness."
    text = (
        f"{result['name']}\n\n"
        f"Final structural dip: {pred['estimate']:.1f}° ± {pred['uncertainty']:.1f}°\n"
        f"Method: {pred['method']} | Confidence: {rec['confidence_label']}\n"
        f"Conservative: {pred.get('conservative_estimate', pred['estimate']):.1f}° "
        f"({pred.get('conservative_method', pred['method'])})\n"
        f"OOD: {pred['is_ood']} | score={pred['ood_score']:.2f}, worst={pred['worst_feature']}\n\n"
        f"{hypothesis_text(result)}\n\n"
        "Geological hypotheses:\n" + "\n".join(h_lines) + "\n\n"
        f"Recommended mode:\n{rec['mode']}\n"
        f"Candidate range: {rec['dip_range'][0]:.1f}° - {rec['dip_range'][1]:.1f}°\n\n"
        f"Caution:\n{warning}"
    )
    ax.text(0.02, 0.98, text, va="top", ha="left", fontsize=11, linespacing=1.35)

    fig.suptitle("ERT Geological Structure Interpreter", fontsize=15, fontweight="bold")
    fig.tight_layout()
    out = outdir / f"GeoInterp_{result['name']}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def hypothesis_text(result):
    hm = result.get("hypothesis_match")
    if not hm:
        return "Forward match: not used"
    if hm.get("error"):
        return f"Forward match error: {hm['error']}"
    dip = hm.get("dip_estimate")
    dip_txt = "no single dip" if dip is None else f"{dip:.1f}°"
    return (
        f"Forward match: {hm.get('best_family', '')}\n"
        f"Matched dip: {dip_txt} | corr={hm.get('best_corr', 0):.2f} | p={hm.get('best_prob', 0):.2f}"
    )


def format_family_probs(family_probs):
    if not family_probs:
        return "none"
    pairs = sorted(family_probs.items(), key=lambda kv: kv[1], reverse=True)[:4]
    return ", ".join(f"{k}={v:.2f}" for k, v in pairs)


def write_summary_csv(results, outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "GeoInterp_summary.csv"
    fields = [
        "name", "n_electrodes", "a", "n_max", "n_data",
        "dip_estimate", "dip_uncertainty", "method", "confidence",
        "ood_score", "max_severity", "worst_feature",
        "M1_theta", "M1_coh", "M2_theta", "M2_R2", "M3_category",
        "M3_proxy", "M3_WtoH", "M4_mean", "M4_std", "M5_theta", "M5_R2",
        "contrast_ratio", "shallow_to_deep", "lateral_ratio",
        "conservative_estimate", "conservative_method",
        "hyp_family", "hyp_dip", "hyp_corr", "hyp_prob", "hyp_eff_n", "hyp_template",
        "recommended_mode", "dip_range_min", "dip_range_max",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            d = r["diag"]
            p = r["prediction"]
            c = r["context"]
            rec = r["recommendation"]
            hm = r.get("hypothesis_match") or {}
            hyp_dip = hm.get("dip_estimate") if not hm.get("error") else None
            writer.writerow({
                "name": r["name"],
                "n_electrodes": r["n_electrodes"],
                "a": r["a"],
                "n_max": r["n_max"],
                "n_data": r["n_data"],
                "dip_estimate": f"{p['estimate']:.3f}",
                "dip_uncertainty": f"{p['uncertainty']:.3f}",
                "method": p["method"],
                "confidence": f"{p['confidence']:.3f}",
                "ood_score": f"{p['ood_score']:.3f}",
                "max_severity": f"{p['max_severity']:.3f}",
                "worst_feature": p["worst_feature"],
                "M1_theta": f"{d['M1']['theta_med']:.3f}",
                "M1_coh": f"{d['M1']['coh_mean']:.3f}",
                "M2_theta": f"{d['M2']['theta']:.3f}",
                "M2_R2": f"{d['M2']['R2']:.3f}",
                "M3_category": d["M3"]["category"],
                "M3_proxy": f"{d['M3']['dip_proxy']:.3f}",
                "M3_WtoH": f"{d['M3']['WtoH_ratio']:.3f}",
                "M4_mean": f"{d['M4']['mean']:.3f}",
                "M4_std": f"{d['M4']['std']:.3f}",
                "M5_theta": f"{d['M5']['theta']:.3f}",
                "M5_R2": f"{d['M5']['R2']:.3f}",
                "contrast_ratio": f"{c['contrast_ratio']:.3f}",
                "shallow_to_deep": f"{c['shallow_to_deep']:.3f}",
                "lateral_ratio": f"{c['lateral_ratio']:.3f}",
                "conservative_estimate": f"{p.get('conservative_estimate', p['estimate']):.3f}",
                "conservative_method": p.get("conservative_method", p["method"]),
                "hyp_family": "" if hm.get("error") else hm.get("best_family", ""),
                "hyp_dip": "" if hyp_dip is None else f"{hyp_dip:.3f}",
                "hyp_corr": "" if hm.get("error") else f"{hm.get('best_corr', 0):.3f}",
                "hyp_prob": "" if hm.get("error") else f"{hm.get('best_prob', 0):.4f}",
                "hyp_eff_n": "" if hm.get("error") else f"{hm.get('effective_n', 0):.3f}",
                "hyp_template": "" if hm.get("error") else hm.get("best_name", ""),
                "recommended_mode": rec["mode"],
                "dip_range_min": f"{rec['dip_range'][0]:.3f}",
                "dip_range_max": f"{rec['dip_range'][1]:.3f}",
            })
    return path


def collect_apv_files(args):
    if args.apv:
        return [Path(p) for p in args.apv]

    return sorted((Path(p) for p in glob.glob(str(ROOT / "*.APV"))), key=lambda p: p.name)


def main():
    parser = argparse.ArgumentParser(description="ERT geological structure interpreter")
    parser.add_argument("--apv", nargs="*", help="APV files to analyze. Default: synthetic and field APVs in project.")
    parser.add_argument("--outdir", default=str(ROOT / "GeoInterp_outputs"), help="Output directory")
    parser.add_argument("--no-filter", action="store_true", help="Disable bad-data filtering")
    parser.add_argument("--no-ml", action="store_true", help="Disable ML model and use diagnostic fallback only")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    model_dict = None if args.no_ml else load_ml_model()
    apv_files = collect_apv_files(args)
    if not apv_files:
        raise SystemExit("No APV files found.")

    results = []
    print("=" * 72)
    print("ERT Geological Structure Interpreter")
    print("=" * 72)
    print(f"Output directory: {outdir}")
    print(f"ML model: {'disabled' if model_dict is None else MODEL_PATH}")
    print("")

    for path in apv_files:
        print(f"[{len(results)+1}/{len(apv_files)}] {path.name}")
        try:
            result = interpret_dataset(
                path,
                outdir=outdir,
                model_dict=model_dict,
                apply_filter=not args.no_filter,
            )
            p = result["prediction"]
            rec = result["recommendation"]
            print(
                f"  dip={p['estimate']:.1f}±{p['uncertainty']:.1f}° "
                f"[{p['method']}], conf={p['confidence']:.2f}, "
                f"OOD={p['ood_score']:.2f}, mode={rec['mode']}"
            )
            results.append(result)
        except Exception as exc:
            print(f"  ERROR: {exc}")

    summary_path = write_summary_csv(results, outdir)
    print("")
    print(f"Summary CSV: {summary_path}")
    print(f"Reports and figures: {outdir}")
    print("=" * 72)


if __name__ == "__main__":
    main()
