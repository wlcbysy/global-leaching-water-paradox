#!/usr/bin/env python3
"""
Pillar 2B: Arm A Brackish Irrigation Safe Window (ECw*), Duration x Depth Salt Accumulation
vs. Sustainable Desalination, and Irrigation/Drainage/Amendment Moderation Model.

Reads armA_extracted_current.csv, enforces 12-Gate PASS_READY / PASS_NO_VARIANCE filtering
with explicit finite ECw_c and ECw_t, fits:
  - Legacy delta-ECw quadratic dose-response models (retained for backwards key compatibility)
  - Absolute ECw Maas-Hoffman piecewise models + cluster bootstrap 95% CIs (B=1000, seed=20261124)
  - Logistic van Genuchten-Hoffman sensitivity fits (free p and sigmoidal p >= 1)
  - Monotonic absolute-ECw soil EC accumulation thresholds (+50%, +100% with 95% CIs)
  - 7-group crop ECw* thresholds mapped against FAO-29 / SPAM ECe90 (empirical k_emp vs 0.67)
  - Covariate-adjusted scheduling & irrigation technology CR1 regressions
  - Bridge moderator tests (amendments, mulching, subsurface drainage) on yield & soil EC
  - Within-study family-FE year x depth soil EC dynamics
  - Site-matched global grid crop_ece90 reference (ece90_ref_sites)
"""

from __future__ import annotations

from config import HARMONIZED, OUT_TABLES, grid_input, prepare_outputs

import json
import math
import re
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import optimize, stats

ARMA_CSV = HARMONIZED / "armA_extracted_current.csv"
P1_GRID = grid_input("pillar1_grid05_baseline_current.parquet")
OUT_RESULTS_JSON = OUT_TABLES / "pillar2_armA_ecw_results_current.json"
OUT_STRATA_CSV = OUT_TABLES / "pillar2_armA_strata_summary_current.csv"

SEED = 20261124
N_BOOT = 1000

# Consistent with pillar1_global_leaching_paradox.py CROP_SALT_PARAMS (ECe_threshold, slope b % per dS/m)
CROP_SALT_PARAMS: Dict[str, Tuple[float, float]] = {
    "BARL": (8.0, 5.0), "COTT": (7.7, 5.2), "SUGB": (7.0, 5.9), "SORG": (6.8, 16.0),
    "RAPE": (6.5, 13.0), "WHEA": (6.0, 7.1), "OCER": (5.5, 7.0), "SOYB": (5.0, 20.0),
    "OFIB": (5.0, 10.0), "COWP": (4.9, 12.0), "SUNF": (4.8, 7.9), "PMIL": (4.0, 16.0),
    "MILL": (4.0, 16.0), "CNUT": (4.0, 10.0), "OOIL": (3.5, 12.0), "GROU": (3.2, 29.0),
    "RICE": (3.0, 12.0), "OILP": (3.0, 10.0), "SESA": (3.0, 12.0), "TOMA": (2.5, 9.9),
    "REST": (2.5, 12.0), "TOBA": (2.0, 12.0), "RUBB": (2.0, 12.0), "VEGE": (1.8, 12.0),
    "TROF": (1.8, 14.0), "MAIZ": (1.7, 12.0), "SUGC": (1.7, 5.9), "POTA": (1.7, 12.0),
    "CITR": (1.7, 16.0), "TEMF": (1.6, 15.0), "SWPO": (1.5, 11.0), "CASS": (1.5, 14.0),
    "YAMS": (1.5, 14.0), "ORTS": (1.5, 14.0), "CHIC": (1.5, 16.0), "BANA": (1.5, 15.0),
    "PLNT": (1.5, 15.0), "COFF": (1.5, 15.0), "RCOF": (1.5, 15.0), "COCO": (1.5, 15.0),
    "TEAS": (1.5, 15.0), "PIGE": (1.3, 15.0), "LENT": (1.3, 15.0), "OPUL": (1.3, 16.0),
    "ONIO": (1.2, 16.0), "BEAN": (1.0, 19.0),
}

CROP7_TO_SPAM: Dict[str, List[str]] = {
    "wheat": ["WHEA"],
    "cotton": ["COTT"],
    "maize": ["MAIZ"],
    "vegetables": ["VEGE", "TOMA", "ONIO"],
    "fruit": ["TEMF", "TROF", "CITR", "BANA", "PLNT", "CNUT"],
    "oilseed_legume": ["SOYB", "GROU", "SUNF", "RAPE", "SESA", "BEAN", "LENT", "CHIC", "PIGE", "COWP", "OPUL", "OOIL"],
    "other_cereals": ["BARL", "RICE", "SORG", "MILL", "PMIL", "OCER"],
}


def wls_cluster_summary(y: np.ndarray, w: np.ndarray, clusters: np.ndarray) -> Dict[str, Any]:
    """Cluster-robust weighted mean with CR1 small-sample t-CI across family_id clusters."""
    ok = np.isfinite(y) & np.isfinite(w) & (w > 0)
    y, w, clusters = y[ok], w[ok], clusters[ok]
    k = len(y)
    if k < 3:
        return {"k": int(k), "n_clusters": int(len(set(clusters))), "status": "INSUFFICIENT_K"}
    uniq = np.unique(clusters)
    n_cl = len(uniq)
    sw = float(np.sum(w))
    mu = float(np.sum(w * y) / sw)
    if n_cl == 1:
        se = math.sqrt(1.0 / sw)
        return {
            "k": int(k), "n_clusters": 1, "mu_lnrr": round(mu, 4),
            "pct_change": round((math.exp(mu) - 1.0) * 100.0, 2),
            "status": "SINGLE_CLUSTER",
        }
    meat = sum(float(np.sum(w[clusters == c] * (y[clusters == c] - mu))) ** 2 for c in uniq)
    var_cr = (n_cl / max(n_cl - 1, 1)) * meat / (sw ** 2)
    se_cr = math.sqrt(max(var_cr, 1e-10))
    df_t = max(n_cl - 1, 1)
    t_crit = float(stats.t.ppf(0.975, df=df_t))
    p_val = float(2.0 * stats.t.sf(abs(mu / se_cr), df=df_t))
    return {
        "k": int(k), "n_clusters": int(n_cl), "mu_lnrr": round(mu, 4),
        "se_cr": round(se_cr, 4),
        "ci95": [round(mu - t_crit * se_cr, 4), round(mu + t_crit * se_cr, 4)],
        "pct_change": round((math.exp(mu) - 1.0) * 100.0, 2),
        "p_val": round(p_val, 6), "status": "OK",
    }


def fit_ecw_dose_response_cluster_wls(df_sub: pd.DataFrame, direction: str = "negative_drop") -> Dict[str, Any]:
    """Legacy delta-ECw quadratic regression retained for backwards JSON compatibility."""
    sub = df_sub.dropna(subset=["lnRR", "delta_ecw", "w_obs"]).copy()
    sub = sub[sub["delta_ecw"] > 0]
    k = len(sub)
    n_cl = int(sub["family_id"].nunique())
    if k < 8 or n_cl < 3:
        return {"k": int(k), "n_clusters": n_cl, "status": "INSUFFICIENT_DATA_FOR_DOSE_CURVE"}

    x = sub["delta_ecw"].values
    y = sub["lnRR"].values
    w = sub["w_obs"].values
    cl = sub["family_id"].values

    X = np.column_stack([x, x ** 2])
    XtWX = X.T @ (w[:, None] * X) + 1e-8 * np.eye(2)
    beta = np.linalg.solve(XtWX, X.T @ (w * y))
    resid = y - X @ beta

    XtWX_inv = np.linalg.inv(XtWX)
    meat = sum(np.outer(X[cl == c].T @ (w[cl == c] * resid[cl == c]), X[cl == c].T @ (w[cl == c] * resid[cl == c])) for c in np.unique(cl))
    cov_cr = (n_cl / max(n_cl - 1, 1)) * (XtWX_inv @ meat @ XtWX_inv)
    se_cr = np.sqrt(np.maximum(np.diag(cov_cr), 1e-12))

    slp_lin = float(np.sum(w * x * y) / max(np.sum(w * (x ** 2)), 1e-9))
    res_lin = y - slp_lin * x
    meat_lin = sum(float(np.sum(w[cl == c] * x[cl == c] * res_lin[cl == c])) ** 2 for c in np.unique(cl))
    se_lin = math.sqrt((n_cl / max(n_cl - 1, 1)) * meat_lin / max(float(np.sum(w * (x ** 2))) ** 2, 1e-12))
    med_ctrl = float(sub["ecw_control_dsm"].median()) if sub["ecw_control_dsm"].notna().any() else 0.5

    def solve_thresh(target_lnrr: float, is_upper: bool = False) -> float | None:
        b1, b2 = float(beta[0]), float(beta[1])
        grid = np.linspace(0.05, 25.0, 2500)
        pred = b1 * grid + b2 * (grid ** 2)
        hit = np.where(pred >= target_lnrr)[0] if is_upper else np.where(pred <= target_lnrr)[0]
        if len(hit) > 0:
            return round(med_ctrl + float(grid[hit[0]]), 2)
        if (is_upper and slp_lin > 1e-4) or ((not is_upper) and slp_lin < -1e-4):
            return round(med_ctrl + target_lnrr / slp_lin, 2)
        return None

    out = {
        "k": int(k), "n_clusters": n_cl, "median_ctrl_ecw_dsm": round(med_ctrl, 2),
        "linear_slope_per_dsm": round(slp_lin, 4), "linear_se_cr": round(se_lin, 4),
        "quad_b1": round(float(beta[0]), 4), "quad_b2": round(float(beta[1]), 4),
        "quad_se_cr": [round(float(se_cr[0]), 4), round(float(se_cr[1]), 4)],
        "status": "OK",
    }
    if direction == "positive_accumulation":
        out["ecw_star_plus50pct_dsm"] = solve_thresh(math.log(1.50), is_upper=True)
        out["ecw_star_plus100pct_dsm"] = solve_thresh(math.log(2.00), is_upper=True)
    else:
        out["ecw_star_90pct_dsm"] = solve_thresh(math.log(0.90), is_upper=False)
        out["ecw_star_80pct_dsm"] = solve_thresh(math.log(0.80), is_upper=False)
    return out


def _fit_mh_arrays(
    ec_t: np.ndarray,
    ec_c: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    starts: List[Tuple[float, float]] | None = None,
) -> Tuple[float, float, bool]:
    """Fit Maas-Hoffman piecewise RY(EC) = max(1 - b*max(0, EC - t), 0.05) on lnRR = ln(RY_t/RY_c)."""
    sw = np.sqrt(w)
    if starts is None:
        starts = [(t0, b0) for t0 in (0.2, 0.8, 1.5, 2.5, 4.0) for b0 in (0.02, 0.05, 0.10)]

    def resid_fn(p: np.ndarray) -> np.ndarray:
        t_v, b_v = float(p[0]), float(p[1])
        ry_t = np.maximum(1.0 - b_v * np.maximum(0.0, ec_t - t_v), 0.05)
        ry_c = np.maximum(1.0 - b_v * np.maximum(0.0, ec_c - t_v), 0.05)
        return sw * (y - np.log(ry_t / ry_c))

    best = None
    for t0, b0 in starts:
        res = optimize.least_squares(resid_fn, x0=[t0, b0], bounds=([0.0, 1e-4], [15.0, 0.50]))
        if best is None or res.cost < best.cost:
            best = res
    assert best is not None
    return float(best.x[0]), float(best.x[1]), bool(best.success)


def fit_maas_hoffman_abs(
    df_sub: pd.DataFrame,
    rng: np.random.Generator,
    n_boot: int = N_BOOT,
    curve_grid: np.ndarray | None = None,
) -> Dict[str, Any]:
    """Fit absolute-ECw Maas-Hoffman piecewise model with cluster-bootstrap 95% CIs."""
    sub = df_sub.dropna(subset=["lnRR", "ecw_treatment_dsm", "ecw_control_dsm", "w_obs"]).copy()
    sub = sub[sub["ecw_treatment_dsm"] > sub["ecw_control_dsm"]]
    k = int(len(sub))
    uniq_cl = sub["family_id"].unique()
    n_cl = int(len(uniq_cl))
    if k < 8 or n_cl < 3:
        return {"k": k, "n_clusters": n_cl, "status": "INSUFFICIENT_DATA"}

    ec_t = sub["ecw_treatment_dsm"].values
    ec_c = sub["ecw_control_dsm"].values
    y = sub["lnRR"].values
    w = sub["w_obs"].values
    t_hat, b_hat, converged = _fit_mh_arrays(ec_t, ec_c, y, w)
    ecw90 = t_hat + 0.10 / b_hat
    ecw80 = t_hat + 0.20 / b_hat

    # Cluster bootstrap
    cl_arrays = [
        (g["ecw_treatment_dsm"].values, g["ecw_control_dsm"].values, g["lnRR"].values, g["w_obs"].values)
        for _, g in sub.groupby("family_id")
    ]
    boot_starts = [(t_hat, b_hat), (0.3, 0.03), (2.0, 0.03)]
    ts, bs, e90s, e80s = [], [], [], []
    n_conv = 0
    for _ in range(n_boot):
        idx = rng.integers(0, n_cl, size=n_cl)
        bt = np.concatenate([cl_arrays[i][0] for i in idx])
        bc = np.concatenate([cl_arrays[i][1] for i in idx])
        by = np.concatenate([cl_arrays[i][2] for i in idx])
        bw = np.concatenate([cl_arrays[i][3] for i in idx])
        tb, bb, okb = _fit_mh_arrays(bt, bc, by, bw, starts=boot_starts)
        if okb:
            n_conv += 1
        ts.append(tb)
        bs.append(bb)
        e90s.append(tb + 0.10 / bb)
        e80s.append(tb + 0.20 / bb)

    ts_a, bs_a, e90_a, e80_a = map(np.asarray, (ts, bs, e90s, e80s))
    # Cap summary CI arrays at physical 30 dS/m only if extreme flat slope occurred, but record exact boundary hit rate!
    t_hit_lower_pct = round(float((ts_a < 0.01).mean() * 100.0), 1)
    t_hit_upper_pct = round(float((ts_a > 14.9).mean() * 100.0), 1)
    b_hit_lower_pct = round(float((bs_a < 2e-4).mean() * 100.0), 1)
    ecw90_hit_bound_pct = round(float(((bs_a < 2e-4) | (e90_a > 30.0)).mean() * 100.0), 1)

    curve = {}
    if curve_grid is not None:
        predictions = np.maximum(1.0 - bs_a[:, None] * np.maximum(0, curve_grid - ts_a[:, None]), 0.05)
        curve = {"curve_ecw_dsm": curve_grid.tolist(),
                 "curve_lo": np.percentile(predictions, 2.5, axis=0).tolist(),
                 "curve_hi": np.percentile(predictions, 97.5, axis=0).tolist()}
    return {
        **curve,
        "k": k,
        "n_clusters": n_cl,
        "converged": converged,
        "n_boot": n_boot,
        "boot_converged_pct": round(n_conv / n_boot * 100.0, 1),
        "t_dsm": round(t_hat, 3),
        "t_ci95": [round(float(np.percentile(ts_a, 2.5)), 3), round(float(np.percentile(ts_a, 97.5)), 3)],
        "b_per_dsm": round(b_hat, 4),
        "b_ci95": [round(float(np.percentile(bs_a, 2.5)), 4), round(float(np.percentile(bs_a, 97.5)), 4)],
        "ecw_star_90pct_dsm": round(ecw90, 3),
        "ecw_star_90pct_ci95": [round(float(np.percentile(e90_a, 2.5)), 3), round(float(np.percentile(e90_a, 97.5)), 3)],
        "ecw_star_80pct_dsm": round(ecw80, 3),
        "ecw_star_80pct_ci95": [round(float(np.percentile(e80_a, 2.5)), 3), round(float(np.percentile(e80_a, 97.5)), 3)],
        "t_hit_lower_pct": t_hit_lower_pct,
        "t_hit_upper_pct": t_hit_upper_pct,
        "b_hit_lower_pct": b_hit_lower_pct,
        "ecw90_hit_bound_pct": ecw90_hit_bound_pct,
        "status": "OK",
    }


def fit_logistic_sensitivity(df_sub: pd.DataFrame, p_lower_bound: float = 1.0) -> Dict[str, Any]:
    """van Genuchten-Hoffman logistic sensitivity: RY(EC) = 1 / (1 + (EC / EC50)^p)."""
    sub = df_sub.dropna(subset=["lnRR", "ecw_treatment_dsm", "ecw_control_dsm", "w_obs"]).copy()
    sub = sub[sub["ecw_treatment_dsm"] > sub["ecw_control_dsm"]]
    if len(sub) < 8 or sub["family_id"].nunique() < 3:
        return {"k": int(len(sub)), "status": "INSUFFICIENT_DATA"}
    ec_t = np.maximum(sub["ecw_treatment_dsm"].values, 1e-4)
    ec_c = np.maximum(sub["ecw_control_dsm"].values, 1e-4)
    y = sub["lnRR"].values
    sw = np.sqrt(sub["w_obs"].values)

    def res_log(p: np.ndarray) -> np.ndarray:
        ec50, pw = float(p[0]), float(p[1])
        ry_t = 1.0 / (1.0 + (ec_t / ec50) ** pw)
        ry_c = 1.0 / (1.0 + (ec_c / ec50) ** pw)
        return sw * (y - np.log(ry_t / ry_c))

    best = None
    for e0 in (5.0, 12.0, 25.0):
        for p0 in (max(1.2, p_lower_bound), 2.5):
            r = optimize.least_squares(res_log, [e0, p0], bounds=([0.5, p_lower_bound], [60.0, 10.0]))
            if best is None or r.cost < best.cost:
                best = r
    ec50, pw = float(best.x[0]), float(best.x[1])
    ecw90 = ec50 * ((1.0 / 0.90 - 1.0) ** (1.0 / pw))
    ecw80 = ec50 * ((1.0 / 0.80 - 1.0) ** (1.0 / pw))
    return {
        "k": int(len(sub)), "n_clusters": int(sub["family_id"].nunique()),
        "p_lower_bound": p_lower_bound, "ec50_dsm": round(ec50, 3), "p_shape": round(pw, 3),
        "ecw_star_90pct_dsm": round(ecw90, 3), "ecw_star_80pct_dsm": round(ecw80, 3),
        "cost": round(float(best.cost), 4), "status": "OK",
    }


def fit_soil_ec_absolute(df_sub: pd.DataFrame, rng: np.random.Generator, n_boot: int = N_BOOT) -> Dict[str, Any]:
    """Physical power-law ln(ECs_t/ECs_c) = beta * ln(ECw_t/ECw_c) for soil_EC with cluster bootstrap CIs."""
    sub = df_sub.dropna(subset=["lnRR", "ecw_treatment_dsm", "ecw_control_dsm", "w_obs"]).copy()
    sub = sub[sub["ecw_treatment_dsm"] > sub["ecw_control_dsm"]]
    k = int(len(sub))
    n_cl = int(sub["family_id"].nunique())
    if k < 8 or n_cl < 3:
        return {"k": k, "n_clusters": n_cl, "status": "INSUFFICIENT_DATA"}

    ec_t = sub["ecw_treatment_dsm"].values
    ec_c = sub["ecw_control_dsm"].values
    ec_t_safe = np.maximum(ec_t, 0.05)
    ec_c_safe = np.maximum(ec_c, 0.05)
    ln_r = np.log(ec_t_safe / ec_c_safe)
    dx = ec_t - ec_c
    y = sub["lnRR"].values
    w = sub["w_obs"].values
    ecc_med = float(sub["ecw_control_dsm"].median())

    b_pow = float(np.sum(w * ln_r * y) / max(np.sum(w * (ln_r ** 2)), 1e-9))
    sst = float(np.sum(w * (y - np.average(y, weights=w)) ** 2))
    r2_pow = 1.0 - float(np.sum(w * (y - b_pow * ln_r) ** 2)) / max(sst, 1e-12)

    c_lin = float(np.sum(w * dx * y) / max(np.sum(w * (dx ** 2)), 1e-9))
    r2_exp = 1.0 - float(np.sum(w * (y - c_lin * dx) ** 2)) / max(sst, 1e-12)

    ec50 = ecc_med * (1.50 ** (1.0 / max(b_pow, 1e-4)))
    ec100 = ecc_med * (2.00 ** (1.0 / max(b_pow, 1e-4)))
    exp_ec50 = ecc_med + math.log(1.50) / max(c_lin, 1e-4)
    exp_ec100 = ecc_med + math.log(2.00) / max(c_lin, 1e-4)

    cl_arrays = [
        (
            np.log(np.maximum(g["ecw_treatment_dsm"].values, 0.05) / np.maximum(g["ecw_control_dsm"].values, 0.05)),
            g["lnRR"].values,
            g["w_obs"].values,
        )
        for _, g in sub.groupby("family_id")
    ]
    b_boots, e50_b, e100_b = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n_cl, size=n_cl)
        blnr = np.concatenate([cl_arrays[i][0] for i in idx])
        by = np.concatenate([cl_arrays[i][1] for i in idx])
        bw = np.concatenate([cl_arrays[i][2] for i in idx])
        bb = max(float(np.sum(bw * blnr * by) / max(np.sum(bw * (blnr ** 2)), 1e-9)), 1e-4)
        b_boots.append(bb)
        e50_b.append(ecc_med * (1.50 ** (1.0 / bb)))
        e100_b.append(ecc_med * (2.00 ** (1.0 / bb)))

    return {
        "k": k, "n_clusters": n_cl, "median_ctrl_ecw_dsm": round(ecc_med, 3),
        "model_form": "power_law_log_ratio",
        "power_exponent_b": round(b_pow, 4),
        "power_exponent_b_ci95": [round(float(np.percentile(b_boots, 2.5)), 4), round(float(np.percentile(b_boots, 97.5)), 4)],
        "r2_power_law": round(r2_pow, 4),
        "ecw_plus50pct_dsm": round(ec50, 3),
        "ecw_plus50pct_ci95": [round(float(np.percentile(e50_b, 2.5)), 3), round(float(np.percentile(e50_b, 97.5)), 3)],
        "ecw_plus100pct_dsm": round(ec100, 3),
        "ecw_plus100pct_ci95": [round(float(np.percentile(e100_b, 2.5)), 3), round(float(np.percentile(e100_b, 97.5)), 3)],
        "linear_slope_per_dsm": round(c_lin, 4),
        "r2_exponential_legacy": round(r2_exp, 4),
        "exp_legacy_ecw_plus50pct_dsm": round(exp_ec50, 3),
        "exp_legacy_ecw_plus100pct_dsm": round(exp_ec100, 3),
        "status": "OK",
    }


def _cr1_ols(y: np.ndarray, X: np.ndarray, w: np.ndarray, cl: np.ndarray, names: List[str]) -> Dict[str, Any]:
    uniq = np.unique(cl)
    n_cl = len(uniq)
    XtWX = X.T @ (w[:, None] * X) + 1e-8 * np.eye(X.shape[1])
    beta = np.linalg.solve(XtWX, X.T @ (w * y))
    resid = y - X @ beta
    XtWX_inv = np.linalg.pinv(XtWX)
    meat = np.zeros((X.shape[1], X.shape[1]))
    for c in uniq:
        m = cl == c
        sc = X[m].T @ (w[m] * resid[m])
        meat += np.outer(sc, sc)
    cov_cr = (n_cl / max(n_cl - 1, 1)) * (XtWX_inv @ meat @ XtWX_inv)
    se_cr = np.sqrt(np.maximum(np.diag(cov_cr), 1e-12))
    df_t = max(n_cl - 1, 1)
    terms: Dict[str, Any] = {}
    for i, nm in enumerate(names):
        t_v = float(beta[i] / se_cr[i])
        p_v = float(2.0 * stats.t.sf(abs(t_v), df=df_t))
        terms[nm] = {
            "coef": round(float(beta[i]), 5),
            "se_cr": round(float(se_cr[i]), 5),
            "p_val": round(p_v, 5),
        }
    return {"k": int(len(y)), "n_clusters": int(n_cl), "beta": beta, "cov_cr": cov_cr, "terms": terms}


def fit_interaction_slope_cr1(
    df_sub: pd.DataFrame,
    group_col: str,
    ref_level: str,
    comp_levels: List[str],
) -> Dict[str, Any]:
    """Legacy CR1 interaction regression retained for backwards compatibility."""
    sub = df_sub[df_sub[group_col].isin([ref_level] + comp_levels)].dropna(subset=["lnRR", "delta_ecw", "w_obs"]).copy()
    sub = sub[sub["delta_ecw"] > 0]
    active_comps = [lvl for lvl in comp_levels if (sub[group_col] == lvl).sum() >= 3]
    if (sub[group_col] == ref_level).sum() < 5 or not active_comps or sub["family_id"].nunique() < 4:
        return {"status": "INSUFFICIENT_FOR_INTERACTION", "k": int(len(sub)), "n_clusters": int(sub["family_id"].nunique())}

    cols = [sub["delta_ecw"].values]
    names = [f"slope_{ref_level}"]
    for lvl in active_comps:
        cols.append(sub["delta_ecw"].values * (sub[group_col].values == lvl).astype(float))
        names.append(f"delta_slope_{lvl}_vs_{ref_level}")

    res = _cr1_ols(sub["lnRR"].values, np.column_stack(cols), sub["w_obs"].values, sub["family_id"].values, names)
    terms = {
        nm: {"coef_per_dsm": v["coef"], "se_cr": v["se_cr"], "p_val": v["p_val"]}
        for nm, v in res["terms"].items()
    }
    return {"status": "OK", "k": res["k"], "n_clusters": res["n_clusters"], "terms": terms}


def fit_adjusted_scheduling_and_tech(
    yield_all: pd.DataFrame,
    soil_ec: pd.DataFrame,
    t_pooled: float,
    rng: np.random.Generator,
    n_boot: int = N_BOOT,
) -> Dict[str, Any]:
    """
    Step 1d: Covariate-adjusted CR1 regression:
      lnRR ~ excess * factor + crop_group7 + aridity_cls + dur_cls + experiment_type_cls
    where excess = max(0, ECt - t_pooled) - max(0, ECc - t_pooled).
    Also fits origin-constrained slope-interaction sensitivity (excess * covariates)
    and derives cluster-bootstrapped CIs for adjusted ECw*90 (continuous vs cyclic).
    """
    df_a = yield_all.dropna(subset=["lnRR", "ecw_treatment_dsm", "ecw_control_dsm", "w_obs"]).copy()
    df_a = df_a[df_a["ecw_treatment_dsm"] > df_a["ecw_control_dsm"]]
    df_a["excess"] = np.maximum(0.0, df_a["ecw_treatment_dsm"] - t_pooled) - np.maximum(0.0, df_a["ecw_control_dsm"] - t_pooled)

    cov_dum = pd.get_dummies(df_a[["crop_group7", "aridity_cls", "duration_stratum", "exp_type_cls"]], drop_first=True, dtype=float)
    cov_cols = [cov_dum[c].values for c in cov_dum.columns]
    cov_names = list(cov_dum.columns)

    ex = df_a["excess"].values
    cyc = (df_a["irrigation_scheduling"] == "cyclic_alternating").astype(float).values
    bln = (df_a["irrigation_scheduling"] == "blended_conjunctive").astype(float).values
    y = df_a["lnRR"].values
    w = df_a["w_obs"].values
    cl = df_a["family_id"].values

    # Primary model (intercept + additive covariates)
    X_sched = np.column_stack([np.ones(len(df_a)), ex, ex * cyc, ex * bln] + cov_cols)
    nm_sched = ["intercept", "excess_continuous", "excess_x_cyclic", "excess_x_blended"] + cov_names
    res_sched = _cr1_ols(y, X_sched, w, cl, nm_sched)

    # Origin-constrained slope-interaction sensitivity (no intercept, covariates interact with excess)
    X_sens = np.column_stack([ex, ex * cyc, ex * bln] + [ex * c for c in cov_cols])
    nm_sens = ["excess_continuous", "excess_x_cyclic", "excess_x_blended"] + [f"excess_x_{c}" for c in cov_names]
    res_sens = _cr1_ols(y, X_sens, w, cl, nm_sens)

    b_cont = -float(res_sched["beta"][1])
    b_cyc = -float(res_sched["beta"][1] + res_sched["beta"][2])
    ecw90_cont_adj = t_pooled + 0.10 / max(b_cont, 1e-4)
    ecw90_cyc_adj = t_pooled + 0.10 / max(b_cyc, 1e-4) if b_cyc > 1e-4 else float("nan")
    diff_adj = ecw90_cyc_adj - ecw90_cont_adj if math.isfinite(ecw90_cyc_adj) else float("nan")

    # Cluster bootstrap for adjusted continuous/cyclic ECw*90 and their difference
    uniq_cl = df_a["family_id"].unique()
    n_cl = len(uniq_cl)
    cl_packs = []
    for c in uniq_cl:
        m = cl == c
        cl_packs.append((y[m], X_sched[m], w[m]))
    b_cont_list, b_cyc_list, diff_list, coef_cyc_list = [], [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n_cl, size=n_cl)
        by = np.concatenate([cl_packs[i][0] for i in idx])
        bX = np.vstack([cl_packs[i][1] for i in idx])
        bw = np.concatenate([cl_packs[i][2] for i in idx])
        XtWX = bX.T @ (bw[:, None] * bX) + 1e-8 * np.eye(bX.shape[1])
        bb = np.linalg.solve(XtWX, bX.T @ (bw * by))
        bc = -float(bb[1])
        byc = -float(bb[1] + bb[2])
        coef_cyc_list.append(float(bb[2]))
        if bc > 1e-4 and byc > 1e-4:
            ec = t_pooled + 0.10 / bc
            ey = t_pooled + 0.10 / byc
            b_cont_list.append(ec)
            b_cyc_list.append(ey)
            diff_list.append(ey - ec)

    # Technology mode adjusted CR1 model (ref = surface_flood_furrow)
    m_dm = (df_a["irrigation_mode_std"] == "drip_mulched").astype(float).values
    m_du = (df_a["irrigation_mode_std"] == "drip_unmulched").astype(float).values
    m_sdi = (df_a["irrigation_mode_std"] == "subsurface_drip").astype(float).values
    m_oth = (~df_a["irrigation_mode_std"].isin(["surface_flood_furrow", "drip_mulched", "drip_unmulched", "subsurface_drip"])).astype(float).values
    X_tech = np.column_stack([np.ones(len(df_a)), ex, ex * m_dm, ex * m_du, ex * m_sdi, ex * m_oth] + cov_cols)
    nm_tech = ["intercept", "excess_surface_flood", "excess_x_drip_mulched", "excess_x_drip_unmulched", "excess_x_subsurface_drip", "excess_x_other_mode"] + cov_names
    res_tech = _cr1_ols(y, X_tech, w, cl, nm_tech)

    # FAO-48 / Minhas (2020) orthogonal delivery CR1 model on yield (ref = surface_flood_furrow; surface_drip = drip_mulched + drip_unmulched)
    m_sd = (df_a["irrigation_delivery_fao48"] == "surface_drip").astype(float).values
    m_sdi_f = (df_a["irrigation_delivery_fao48"] == "subsurface_drip").astype(float).values
    m_oth_f = (~df_a["irrigation_delivery_fao48"].isin(["surface_flood_furrow", "surface_drip", "subsurface_drip"])).astype(float).values
    X_fao = np.column_stack([np.ones(len(df_a)), ex, ex * m_sd, ex * m_sdi_f, ex * m_oth_f] + cov_cols)
    nm_fao = ["intercept", "excess_surface_flood", "excess_x_surface_drip", "excess_x_subsurface_drip", "excess_x_other_mode"] + cov_names
    res_fao = _cr1_ols(y, X_fao, w, cl, nm_fao)

    # Matching covariate-adjusted CR1 models on Soil EC (dECw)
    df_se = soil_ec.dropna(subset=["lnRR", "ecw_treatment_dsm", "ecw_control_dsm", "w_obs"]).copy()
    df_se = df_se[df_se["ecw_treatment_dsm"] > df_se["ecw_control_dsm"]]
    cov_s_dum = pd.get_dummies(df_se[["crop_group7", "aridity_cls", "duration_stratum", "exp_type_cls"]], drop_first=True, dtype=float)
    cov_s_cols = [cov_s_dum[c].values for c in cov_s_dum.columns]
    cov_s_names = list(cov_s_dum.columns)
    dx_s = (df_se["ecw_treatment_dsm"] - df_se["ecw_control_dsm"]).values
    y_s = df_se["lnRR"].values
    w_s = df_se["w_obs"].values
    cl_s = df_se["family_id"].values
    cyc_s = (df_se["irrigation_scheduling"] == "cyclic_alternating").astype(float).values
    bln_s = (df_se["irrigation_scheduling"] == "blended_conjunctive").astype(float).values
    X_se_sched = np.column_stack([np.ones(len(df_se)), dx_s, dx_s * cyc_s, dx_s * bln_s] + cov_s_cols)
    nm_se_sched = ["intercept", "dECw_continuous", "dECw_x_cyclic", "dECw_x_blended"] + cov_s_names
    res_se_sched = _cr1_ols(y_s, X_se_sched, w_s, cl_s, nm_se_sched)

    msd_s = (df_se["irrigation_delivery_fao48"] == "surface_drip").astype(float).values
    msdi_s = (df_se["irrigation_delivery_fao48"] == "subsurface_drip").astype(float).values
    moth_s = (~df_se["irrigation_delivery_fao48"].isin(["surface_flood_furrow", "surface_drip", "subsurface_drip"])).astype(float).values
    X_se_fao = np.column_stack([np.ones(len(df_se)), dx_s, dx_s * msd_s, dx_s * msdi_s, dx_s * moth_s] + cov_s_cols)
    nm_se_fao = ["intercept", "dECw_surface_flood", "dECw_x_surface_drip", "dECw_x_subsurface_drip", "dECw_x_other_mode"] + cov_s_names
    res_se_fao = _cr1_ols(y_s, X_se_fao, w_s, cl_s, nm_se_fao)

    return {
        "t_pooled_used_dsm": round(t_pooled, 3),
        "cyclic_advantage_robust": False,
        "cyclic_robustness_note": "Cyclic interaction is borderline in additive-covariate CR1 (P = 0.0494), attenuates under origin-constrained slope moderation (P = 0.0690) and when adding scheduling main effects (P = 0.2976), and yields an imprecise adjusted difference CI spanning zero; Pillar 3 uses continuous ECw*90 as baseline and unadjusted cyclic ECw*90 as an upper-bound managed sensitivity.",
        "scheduling_adjusted_cr1": {
            "k": res_sched["k"], "n_clusters": res_sched["n_clusters"],
            "terms": {k: res_sched["terms"][k] for k in ["intercept", "excess_continuous", "excess_x_cyclic", "excess_x_blended"]},
            "adjusted_ecw90_continuous_dsm": round(ecw90_cont_adj, 3),
            "adjusted_ecw90_continuous_ci95": [round(float(np.percentile(b_cont_list, 2.5)), 3), round(float(np.percentile(b_cont_list, 97.5)), 3)] if b_cont_list else None,
            "adjusted_ecw90_cyclic_dsm": round(ecw90_cyc_adj, 3) if math.isfinite(ecw90_cyc_adj) else None,
            "adjusted_ecw90_cyclic_ci95": [round(float(np.percentile(b_cyc_list, 2.5)), 3), round(float(np.percentile(b_cyc_list, 97.5)), 3)] if b_cyc_list else None,
            "adjusted_cyclic_minus_continuous_dsm": round(diff_adj, 3) if math.isfinite(diff_adj) else None,
            "adjusted_cyclic_minus_continuous_ci95": [round(float(np.percentile(diff_list, 2.5)), 3), round(float(np.percentile(diff_list, 97.5)), 3)] if diff_list else None,
            "valid_boot_positive_slope_pct": round(len(diff_list) / n_boot * 100.0, 1),
        },
        "scheduling_origin_slope_sensitivity": {
            "terms": {k: res_sens["terms"][k] for k in ["excess_continuous", "excess_x_cyclic", "excess_x_blended"]},
        },
        "technology_adjusted_cr1": {
            "k": res_tech["k"], "n_clusters": res_tech["n_clusters"],
            "terms": {k: res_tech["terms"][k] for k in ["intercept", "excess_surface_flood", "excess_x_drip_mulched", "excess_x_drip_unmulched", "excess_x_subsurface_drip", "excess_x_other_mode"]},
        },
        "technology_fao48_adjusted_cr1": {
            "k": res_fao["k"], "n_clusters": res_fao["n_clusters"],
            "counts": {k: {"k": int(len(g)), "n_clusters": int(g["family_id"].nunique())} for k, g in df_a.groupby("irrigation_delivery_fao48")},
            "terms": {k: res_fao["terms"][k] for k in ["intercept", "excess_surface_flood", "excess_x_surface_drip", "excess_x_subsurface_drip", "excess_x_other_mode"]},
        },
        "soil_ec_scheduling_adjusted_cr1": {
            "k": res_se_sched["k"], "n_clusters": res_se_sched["n_clusters"],
            "terms": {k: res_se_sched["terms"][k] for k in ["intercept", "dECw_continuous", "dECw_x_cyclic", "dECw_x_blended"]},
        },
        "soil_ec_technology_fao48_adjusted_cr1": {
            "k": res_se_fao["k"], "n_clusters": res_se_fao["n_clusters"],
            "counts": {k: {"k": int(len(g)), "n_clusters": int(g["family_id"].nunique())} for k, g in df_se.groupby("irrigation_delivery_fao48")},
            "terms": {k: res_se_fao["terms"][k] for k in ["intercept", "dECw_surface_flood", "dECw_x_surface_drip", "dECw_x_subsurface_drip", "dECw_x_other_mode"]},
        },
    }


def classify_amendment_coapplied(val: object) -> str:
    """Classify amendment_co_applied per plan: ca_chemical, organic, or none_or_npk (uniform NPK -> none)."""
    s = str(val or "").lower()
    if not s or s in {"none", "nan", "unknown"}:
        return "none_or_npk"
    s_clean = re.sub(r"0\s*(t\s*ha-1|%|mm)\s*(supplementary\s*)?(biochar|zeolite|sheep manure|ca\(no3\)2)", "", s)
    s_clean = s_clean.replace("no straw or biochar", "").replace("no organic fertilization", "").replace("phosphoric acid", "")
    if any(p in s_clean for p in ("marginal mean across cow manure", "averaged across 4 organic", "pooled (none,")):
        return "none_or_npk"
    has_ca = any(k in s_clean for k in ("gypsum", "pyrite", "polyacrylamide", "pam", "zeolite", "ca(no3)2", "calcium nitrate", "sulfur", "sulphur", "acid", "dolomitic limestone"))
    has_org = any(k in s_clean for k in ("manure", "fym", "biochar", "straw", "compost", "organic"))
    if has_ca:
        return "ca_chemical"
    if has_org:
        return "organic"
    return "none_or_npk"


def fit_bridge_moderator_tests(
    yield_all: pd.DataFrame,
    soil_ec: pd.DataFrame,
    t_pooled: float,
) -> Dict[str, Any]:
    """
    Step 1e: Test excess-salinity (or delta-ECw for soil_EC) slope interaction x bridge moderators
    (amendment class, mulching_flag, subsurface/tile/pipe drainage) controlling for crop_group7,
    plus sensitivity specs with moderator main effects and origin-constrained slopes.
    """
    out: Dict[str, Any] = {}
    for ep_name, df_ep, x_mode in [("yield_combined", yield_all, "excess"), ("soil_EC", soil_ec, "delta_ecw")]:
        sub = df_ep.dropna(subset=["lnRR", "ecw_treatment_dsm", "ecw_control_dsm", "w_obs"]).copy()
        sub = sub[sub["ecw_treatment_dsm"] > sub["ecw_control_dsm"]]
        xv = (
            np.maximum(0.0, sub["ecw_treatment_dsm"].values - t_pooled) - np.maximum(0.0, sub["ecw_control_dsm"].values - t_pooled)
            if x_mode == "excess"
            else (sub["ecw_treatment_dsm"].values - sub["ecw_control_dsm"].values)
        )
        cg_dum = pd.get_dummies(sub["crop_group7"], drop_first=True, dtype=float)
        cg_cols = [cg_dum[c].values for c in cg_dum.columns]
        cg_names = list(cg_dum.columns)
        cg_x_cols = [xv * c for c in cg_cols]
        cg_x_names = [f"x_{c}" for c in cg_names]
        y = sub["lnRR"].values
        w = sub["w_obs"].values
        cl = sub["family_id"].values

        ep_res: Dict[str, Any] = {}
        # 1. Amendment class (ca_chemical, organic vs none_or_npk)
        am = sub["amend_cls"].values
        has_ca = int((am == "ca_chemical").sum()) >= 5 and sub.loc[am == "ca_chemical", "family_id"].nunique() >= 3
        has_org = int((am == "organic").sum()) >= 5 and sub.loc[am == "organic", "family_id"].nunique() >= 3
        cols_a = [np.ones(len(sub)), xv]
        nms_a = ["intercept", "slope_none_or_npk"]
        cols_a_me = [np.ones(len(sub)), xv]
        nms_a_me = ["intercept", "slope_none_or_npk"]
        cols_a_orig = [xv]
        nms_a_orig = ["slope_none_or_npk"]
        if has_ca:
            ca_ind = (am == "ca_chemical").astype(float)
            cols_a.append(xv * ca_ind)
            nms_a.append("delta_slope_ca_chemical")
            cols_a_me.extend([ca_ind, xv * ca_ind])
            nms_a_me.extend(["main_ca_chemical", "delta_slope_ca_chemical"])
            cols_a_orig.append(xv * ca_ind)
            nms_a_orig.append("delta_slope_ca_chemical")
        if has_org:
            org_ind = (am == "organic").astype(float)
            cols_a.append(xv * org_ind)
            nms_a.append("delta_slope_organic")
            cols_a_me.extend([org_ind, xv * org_ind])
            nms_a_me.extend(["main_organic", "delta_slope_organic"])
            cols_a_orig.append(xv * org_ind)
            nms_a_orig.append("delta_slope_organic")
        if len(cols_a) > 2:
            r_a = _cr1_ols(y, np.column_stack(cols_a + cg_cols), w, cl, nms_a + cg_names)
            r_a_me = _cr1_ols(y, np.column_stack(cols_a_me + cg_cols), w, cl, nms_a_me + cg_names)
            r_a_orig = _cr1_ols(y, np.column_stack(cols_a_orig + cg_x_cols), w, cl, nms_a_orig + cg_x_names)
            ep_res["amendment_co_applied"] = {
                "counts_by_class": {k: {"k": int(len(g)), "n_clusters": int(g["family_id"].nunique())} for k, g in sub.groupby("amend_cls")},
                "terms": {k: r_a["terms"][k] for k in nms_a},
                "terms_with_main_effect": {k: r_a_me["terms"][k] for k in nms_a_me},
                "terms_origin_slope_only": {k: r_a_orig["terms"][k] for k in nms_a_orig},
            }
        else:
            ep_res["amendment_co_applied"] = {"status": "INSUFFICIENT_MODERATOR_CLUSTERS"}

        # 2. Mulching flag
        mu_v = sub["mulching_flag"].fillna(False).astype(float).values
        r_m = _cr1_ols(y, np.column_stack([np.ones(len(sub)), xv, xv * mu_v] + cg_cols), w, cl, ["intercept", "slope_unmulched", "delta_slope_mulched"] + cg_names)
        r_m_me = _cr1_ols(y, np.column_stack([np.ones(len(sub)), xv, mu_v, xv * mu_v] + cg_cols), w, cl, ["intercept", "slope_unmulched", "main_mulched", "delta_slope_mulched"] + cg_names)
        r_m_orig = _cr1_ols(y, np.column_stack([xv, xv * mu_v] + cg_x_cols), w, cl, ["slope_unmulched", "delta_slope_mulched"] + cg_x_names)
        ep_res["mulching"] = {
            "counts": {"mulched_k": int(mu_v.sum()), "mulched_clusters": int(sub.loc[mu_v == 1, "family_id"].nunique()), "unmulched_k": int((1 - mu_v).sum())},
            "terms": {k: r_m["terms"][k] for k in ["intercept", "slope_unmulched", "delta_slope_mulched"]},
            "terms_with_main_effect": {k: r_m_me["terms"][k] for k in ["intercept", "slope_unmulched", "main_mulched", "delta_slope_mulched"]},
            "terms_origin_slope_only": {k: r_m_orig["terms"][k] for k in ["slope_unmulched", "delta_slope_mulched"]},
        }

        # 3. Subsurface / tile / pipe drainage
        dr_v = sub["subsurface_drain_flag"].astype(float).values
        r_d = _cr1_ols(y, np.column_stack([np.ones(len(sub)), xv, xv * dr_v] + cg_cols), w, cl, ["intercept", "slope_no_subdrain", "delta_slope_subdrain"] + cg_names)
        r_d_me = _cr1_ols(y, np.column_stack([np.ones(len(sub)), xv, dr_v, xv * dr_v] + cg_cols), w, cl, ["intercept", "slope_no_subdrain", "main_subdrain", "delta_slope_subdrain"] + cg_names)
        r_d_orig = _cr1_ols(y, np.column_stack([xv, xv * dr_v] + cg_x_cols), w, cl, ["slope_no_subdrain", "delta_slope_subdrain"] + cg_x_names)
        ep_res["subsurface_drainage"] = {
            "counts": {"subdrain_k": int(dr_v.sum()), "subdrain_clusters": int(sub.loc[dr_v == 1, "family_id"].nunique()), "other_k": int((1 - dr_v).sum())},
            "terms": {k: r_d["terms"][k] for k in ["intercept", "slope_no_subdrain", "delta_slope_subdrain"]},
            "terms_with_main_effect": {k: r_d_me["terms"][k] for k in ["intercept", "slope_no_subdrain", "main_subdrain", "delta_slope_subdrain"]},
            "terms_origin_slope_only": {k: r_d_orig["terms"][k] for k in ["slope_no_subdrain", "delta_slope_subdrain"]},
        }
        out[ep_name] = ep_res
    return out


def fit_environmental_duration_subgroups(
    yield_all: pd.DataFrame,
    soil_ec: pd.DataFrame,
    t_pooled: float,
) -> Dict[str, Any]:
    """
    Deterministic CR1 slope-interaction tests for Climate AI (aridity_cls),
    USDA 3-class soil texture (coarse/medium/fine), and trial duration (1yr/2_3yr/ge4yr)
    controlling for crop_group7 dummies, without using RNG.
    """
    def _norm_tex3(row: pd.Series) -> str | None:
        s = str(row.get("soil_texture_class", "") or "").lower().strip()
        if "multi" in s:
            return None
        cp = row.get("clay_pct")
        if pd.notna(cp):
            cp_f = float(cp)
            if cp_f >= 35.0:
                return "fine"
            if cp_f < 18.0:
                return "coarse"
            return "medium"
        if not s or s in {"nan", "none", "not_reported", "unreported", "unknown", "nr"}:
            return None
        if any(w in s for w in ("clay", "vertisol")) and "sandy clay loam" not in s:
            return "fine"
        if any(w in s for w in ("sand", "loamy sand", "sandy loam")):
            return "coarse"
        if any(w in s for w in ("loam", "silt", "silty", "medium")):
            return "medium"
        return None

    specs = [
        ("climate_ai", "aridity_cls", "semi_arid", ["arid", "humid_subhumid"]),
        ("soil_texture", "tex3", "medium", ["coarse", "fine"]),
        ("duration_pooled", "dur3", "1yr", ["2_3yr", "ge4yr"]),
    ]

    out: Dict[str, Any] = {}
    for ep_name, df_src, x_mode in [("yield_combined", yield_all, "excess"), ("soil_EC", soil_ec, "delta_ecw")]:
        df_ep = df_src.dropna(subset=["lnRR", "ecw_treatment_dsm", "ecw_control_dsm", "w_obs"]).copy()
        df_ep = df_ep[df_ep["ecw_treatment_dsm"] > df_ep["ecw_control_dsm"]].copy()
        df_ep["tex3"] = df_ep.apply(_norm_tex3, axis=1)
        df_ep["dur3"] = np.where(
            df_ep["duration_years"].isna(),
            None,
            np.where(df_ep["duration_years"] <= 1.0, "1yr", np.where(df_ep["duration_years"] <= 3.0, "2_3yr", "ge4yr")),
        )

        ep_res: Dict[str, Any] = {}
        for mod_nm, col_nm, ref_lvl, comp_lvls in specs:
            sub = df_ep[df_ep[col_nm].isin([ref_lvl] + comp_lvls)].copy()
            xv = (
                np.maximum(0.0, sub["ecw_treatment_dsm"].values - t_pooled)
                - np.maximum(0.0, sub["ecw_control_dsm"].values - t_pooled)
                if x_mode == "excess"
                else (sub["ecw_treatment_dsm"].values - sub["ecw_control_dsm"].values)
            )
            cg_dum = pd.get_dummies(sub["crop_group7"], drop_first=True, dtype=float)
            cg_cols = [cg_dum[c].values for c in cg_dum.columns]
            cg_names = list(cg_dum.columns)
            cols = [np.ones(len(sub)), xv] + [xv * (sub[col_nm].values == lv).astype(float) for lv in comp_lvls]
            nms = ["intercept", f"slope_{ref_lvl}"] + [f"delta_{lv}" for lv in comp_lvls]

            res = _cr1_ols(sub["lnRR"].values, np.column_stack(cols + cg_cols), sub["w_obs"].values, sub["family_id"].values, nms + cg_names)
            df_t = max(int(res["n_clusters"]) - 1, 1)
            t_crit = float(stats.t.ppf(0.975, df=df_t))
            counts_by_class = {
                lv: {"k": int((sub[col_nm] == lv).sum()), "n_clusters": int(sub.loc[sub[col_nm] == lv, "family_id"].nunique())}
                for lv in [ref_lvl] + comp_lvls
            }
            contrasts: Dict[str, Any] = {}
            for lv in comp_lvls:
                term = res["terms"][f"delta_{lv}"]
                coef, se_cr, p_val = float(term["coef"]), float(term["se_cr"]), float(term["p_val"])
                contrasts[lv] = {
                    "coef": coef,
                    "se_cr": se_cr,
                    "ci95": [round(coef - t_crit * se_cr, 5), round(coef + t_crit * se_cr, 5)],
                    "p_val": p_val,
                    "k": counts_by_class[lv]["k"],
                    "n_clusters": counts_by_class[lv]["n_clusters"],
                }
            ep_res[mod_nm] = {
                "ref": ref_lvl,
                "k": int(res["k"]),
                "n_clusters": int(res["n_clusters"]),
                "df_t": df_t,
                "counts_by_class": counts_by_class,
                "base_slope": res["terms"][f"slope_{ref_lvl}"],
                "contrasts": contrasts,
            }
        out[ep_name] = ep_res
    return out


def _fit_year_x_depth_core(
    sub: pd.DataFrame,
    rng: np.random.Generator,
    n_boot: int = N_BOOT,
) -> Dict[str, Any]:
    k = int(len(sub))
    n_cl = int(sub["family_id"].nunique())
    if k < 20 or n_cl < 5:
        return {"k": k, "n_clusters": n_cl, "status": "INSUFFICIENT_DATA"}

    dx = sub["delta_ecw"].values
    ly = np.log(sub["study_year_index"].values)
    is_root = (sub["soil_depth_stratum_std"] == "rootzone_0_40_60").astype(float).values
    is_sub = (sub["soil_depth_stratum_std"] == "subsoil_40_100").astype(float).values
    fam_dum = pd.get_dummies(sub["family_id"], drop_first=False, dtype=float)
    X_core = np.column_stack([dx, dx * is_root, dx * is_sub, dx * ly, dx * is_root * ly, dx * is_sub * ly])
    X = np.hstack([X_core, fam_dum.values])
    nm_core = ["dECw_topsoil_yr1", "dECw_x_rootzone", "dECw_x_subsoil", "dECw_x_logYr", "dECw_x_root_x_logYr", "dECw_x_subsoil_x_logYr"]
    names = nm_core + [f"fe_{i}" for i in range(fam_dum.shape[1])]
    res = _cr1_ols(sub["lnRR"].values, X, sub["w_obs"].values, sub["family_id"].values, names)
    beta = res["beta"][:6]
    cov6 = res["cov_cr"][:6, :6]

    def ratio_at_yr(b: np.ndarray, yr: float) -> Tuple[float, float, float]:
        lyv = math.log(yr)
        s_top = b[0] + b[3] * lyv
        s_sub = (b[0] + b[2]) + (b[3] + b[5]) * lyv
        return s_top, s_sub, s_sub / max(s_top, 1e-6)

    t1, s1, r1 = ratio_at_yr(beta, 1.0)
    t2, s2, r2 = ratio_at_yr(beta, 2.0)
    t3, s3, r3 = ratio_at_yr(beta, 3.0)
    sim_b = rng.multivariate_normal(beta, cov6 + 1e-10 * np.eye(6), size=n_boot)
    dr_s = []
    for sb in sim_b:
        st1, _, rr1 = ratio_at_yr(sb, 1.0)
        st3, _, rr3 = ratio_at_yr(sb, 3.0)
        if st1 > 1e-3 and st3 > 1e-3:
            dr_s.append(rr3 - rr1)

    p_int_sub_yr = res["terms"]["dECw_x_subsoil_x_logYr"]["p_val"]
    rises_sig = bool(p_int_sub_yr < 0.05 and res["terms"]["dECw_x_subsoil_x_logYr"]["coef"] > 0 and dr_s and float(np.percentile(dr_s, 2.5)) > 0)

    df_t = max(n_cl - 1, 1)
    t_crit = float(stats.t.ppf(0.975, df=df_t))
    predicted_slopes: Dict[str, Any] = {}
    for yr in (1, 2, 3):
        ly_v = math.log(float(yr))
        yr_dict: Dict[str, Any] = {}
        for d_nm, c_vec in [
            ("topsoil_0_20", np.array([1.0, 0.0, 0.0, ly_v, 0.0, 0.0])),
            ("rootzone_0_40_60", np.array([1.0, 1.0, 0.0, ly_v, ly_v, 0.0])),
            ("subsoil_40_100", np.array([1.0, 0.0, 1.0, ly_v, 0.0, ly_v])),
        ]:
            slp_v = float(c_vec @ beta)
            se_v = math.sqrt(max(float(c_vec @ cov6 @ c_vec), 1e-12))
            yr_dict[d_nm] = {
                "slope": round(slp_v, 5),
                "se_cr": round(se_v, 5),
                "ci95": [round(slp_v - t_crit * se_v, 5), round(slp_v + t_crit * se_v, 5)],
            }
        predicted_slopes[f"year_{yr}"] = yr_dict

    return {
        "k": k, "n_clusters": n_cl,
        "depth_row_counts": {str(dk): int(dv) for dk, dv in sub["soil_depth_stratum_std"].value_counts().to_dict().items()},
        "terms": {nm: res["terms"][nm] for nm in nm_core},
        "predicted_slopes": predicted_slopes,
        "topsoil_slope_yr1": round(t1, 4), "subsoil_slope_yr1": round(s1, 4),
        "topsoil_slope_yr3": round(t3, 4), "subsoil_slope_yr3": round(s3, 4),
        "subsoil_to_topsoil_ratio_yr1": round(r1, 3),
        "subsoil_to_topsoil_ratio_yr2": round(r2, 3),
        "subsoil_to_topsoil_ratio_yr3": round(r3, 3),
        "ratio_yr3_minus_yr1": round(r3 - r1, 3),
        "ratio_yr3_minus_yr1_ci95": [round(float(np.percentile(dr_s, 2.5)), 3), round(float(np.percentile(dr_s, 97.5)), 3)] if dr_s else None,
        "subsoil_to_topsoil_ratio_rises_significantly": rises_sig,
        "recommended_depth_framing": "subsoil_salt_trap" if rises_sig else "persistent_profile_wide_accumulation",
        "status": "OK",
    }


def fit_within_study_year_x_depth(
    soil_ec: pd.DataFrame,
    rng: np.random.Generator,
    n_boot: int = N_BOOT,
) -> Dict[str, Any]:
    """
    Step 1f: Within-study family-FE year x depth analysis restricted to observed multi-depth
    Years 1-3 window (no unsupported Year 5 extrapolation across Year 4-13 gap):
      lnRR ~ dECw + dECw:depth + dECw:log(year) + dECw:depth:log(year) + family_FE
    Reference depth = topsoil_0_20, year = 1 (log(year) = 0).
    """
    sub_all = soil_ec[
        soil_ec["soil_depth_stratum_std"].isin(["topsoil_0_20", "rootzone_0_40_60", "subsoil_40_100"])
        & soil_ec["study_year_index"].notna()
        & (soil_ec["study_year_index"] >= 1.0)
        & (soil_ec["delta_ecw"] > 0)
    ].copy()
    lt_mask = sub_all["study_year_index"] > 3.0
    excl_rows = int(lt_mask.sum())
    excl_fams = int(sub_all.loc[lt_mask, "family_id"].nunique())

    sub = sub_all[sub_all["study_year_index"] <= 3.0].copy()
    yr_cnts = sub.groupby("family_id")["study_year_index"].nunique()
    sub = sub[sub["family_id"].isin(yr_cnts[yr_cnts >= 2].index)].copy()
    main_res = _fit_year_x_depth_core(sub, rng, n_boot=n_boot)
    main_res["observed_year_window"] = "Years 1-3"
    main_res["excluded_longterm_topsoil_only_rows"] = excl_rows
    main_res["excluded_longterm_families"] = excl_fams

    md_cnts = sub.groupby("family_id")["soil_depth_stratum_std"].nunique()
    sub_md = sub[sub["family_id"].isin(md_cnts[md_cnts >= 2].index)].copy()
    main_res["strict_multidepth_sensitivity"] = _fit_year_x_depth_core(
        sub_md, np.random.default_rng(SEED + 7), n_boot=n_boot
    )
    return main_res


def egger_cluster_test_armA(df_sub: pd.DataFrame) -> Dict[str, Any]:
    sub = df_sub.dropna(subset=["lnRR", "w_obs"]).copy()
    sub["se_p"] = 1.0 / np.sqrt(np.maximum(sub["w_obs"].values, 1e-6))
    agg = sub.groupby("family_id", as_index=False).agg(y=("lnRR", "mean"), se=("se_p", "mean"))
    if len(agg) < 6:
        return {"status": "INSUFFICIENT_CLUSTERS", "n_clusters": int(len(agg))}
    prec = 1.0 / np.maximum(agg["se"].values, 1e-4)
    snd = agg["y"].values * prec
    _, intercept, _, p_val, _ = stats.linregress(prec, snd)
    return {"n_clusters": int(len(agg)), "egger_intercept": round(float(intercept), 3), "egger_p": round(float(p_val), 4), "bias_flag": bool(p_val < 0.05)}


def main() -> None:
    if not ARMA_CSV.is_file():
        raise FileNotFoundError(f"Missing Arm A CSV: {ARMA_CSV}")

    rng = np.random.default_rng(SEED)
    df = pd.read_csv(ARMA_CSV, low_memory=False)
    usable = df[df["qc_gate_status"].isin(["PASS_READY", "PASS_NO_VARIANCE"])].copy()

    for col in ["ecw_treatment_dsm", "ecw_control_dsm", "sarw_treatment", "duration_years", "study_year_index", "mean_treatment", "mean_control", "error_treatment", "error_control", "n_replicates", "lat_dd", "lon_dd", "clay_pct"]:
        if col in usable.columns:
            usable[col] = pd.to_numeric(usable[col], errors="coerce")

    usable = usable.dropna(subset=["ecw_treatment_dsm", "ecw_control_dsm", "mean_treatment", "mean_control"]).copy()
    usable = usable[(usable["mean_treatment"] > 0) & (usable["mean_control"] > 0) & (usable["ecw_treatment_dsm"] > usable["ecw_control_dsm"])].copy()
    usable["lnRR"] = np.log(usable["mean_treatment"] / usable["mean_control"])
    usable["delta_ecw"] = usable["ecw_treatment_dsm"] - usable["ecw_control_dsm"]

    usable["lat_dd"] = usable["analysis_lat_dd"]
    usable["lon_dd"] = usable["analysis_lon_dd"]

    usable["lat_c"] = np.floor(usable["lat_dd"] / 0.5) * 0.5 + 0.25
    usable["lon_c"] = np.floor(usable["lon_dd"] / 0.5) * 0.5 + 0.25

    ece90_ref_sites = None
    n_site_cells_matched = 0
    if P1_GRID.is_file():
        g_df = pd.read_parquet(P1_GRID, columns=["lat", "lon", "prc_mm_yr", "et0_mm_yr", "crop_ece90_dsm"])
        g_df["ai"] = g_df["prc_mm_yr"] / np.maximum(g_df["et0_mm_yr"], 1.0)
        usable = usable.merge(g_df[["lat", "lon", "ai", "crop_ece90_dsm"]].drop_duplicates(["lat", "lon"]), left_on=["lat_c", "lon_c"], right_on=["lat", "lon"], how="left")
        site_cells = usable.dropna(subset=["lat_c", "lon_c"])[["lat_c", "lon_c"]].drop_duplicates()
        matched_sites = site_cells.merge(g_df[["lat", "lon", "crop_ece90_dsm"]].drop_duplicates(["lat", "lon"]), left_on=["lat_c", "lon_c"], right_on=["lat", "lon"], how="inner")
        n_site_cells_matched = int(matched_sites["crop_ece90_dsm"].notna().sum())
        ece90_ref_sites = round(float(matched_sites["crop_ece90_dsm"].median()), 4)
    else:
        usable["ai"] = np.nan

    usable["aridity_cls"] = np.where(usable["ai"].isna(), "unknown", np.where(usable["ai"] < 0.20, "arid", np.where(usable["ai"] < 0.50, "semi_arid", "humid_subhumid")))
    usable["exp_type_cls"] = np.where(usable["experiment_type"].astype(str).str.lower() == "field", "field", "non_field")

    metric_aliases = {
        "seed_cotton_yield": "grain_yield", "fruit_yield": "grain_yield", "tuber_yield": "grain_yield",
        "marketable_yield": "grain_yield", "crop_yield": "grain_yield", "lint_yield": "grain_yield",
        "fresh_yield": "grain_yield", "cane_yield": "grain_yield", "WUE": "wue_or_iwue", "IWUE": "wue_or_iwue",
        "soil_ECe": "soil_EC", "soil_EC1_5": "soil_EC",
        "infiltration_rate_steady": "infiltration_rate", "infiltration_rate_initial": "infiltration_rate",
        "infiltration_rate_unspecified": "infiltration_rate",
    }
    usable["metric_std"] = usable["metric_name"].replace(metric_aliases)

    fam_counts = usable.groupby(["family_id", "metric_std"])["lnRR"].transform("count").clip(lower=1)
    n_eff = usable["n_replicates"].fillna(3.0).clip(lower=1.0)
    usable["w_obs"] = n_eff / fam_counts
    et_u = usable["error_type"].fillna("").astype(str).str.upper()
    has_var = (
        (usable["qc_gate_status"] == "PASS_READY")
        & usable["error_treatment"].notna() & usable["error_control"].notna()
        & (usable["error_treatment"] > 0) & (usable["error_control"] > 0)
        & usable["n_replicates"].notna() & (usable["n_replicates"] >= 2)
        & (~et_u.str.contains("LSD|CV|NOT_REPORTED"))
    )
    sd_t = np.where(et_u.str.contains("SE") & (~et_u.str.contains("SD")), usable["error_treatment"] * np.sqrt(n_eff), usable["error_treatment"])
    sd_c = np.where(et_u.str.contains("SE") & (~et_u.str.contains("SD")), usable["error_control"] * np.sqrt(n_eff), usable["error_control"])
    vi_ex = (sd_t ** 2) / (n_eff * (usable["mean_treatment"] ** 2)) + (sd_c ** 2) / (n_eff * (usable["mean_control"] ** 2))
    usable["vi_exact"] = np.where(has_var & np.isfinite(vi_ex) & (vi_ex > 1e-8), np.minimum(vi_ex, 5.0), np.nan)
    usable["has_exact_variance"] = usable["vi_exact"].notna()

    has_lsd = (~usable["has_exact_variance"]) & et_u.str.contains("LSD") & usable["error_treatment"].notna() & (usable["error_treatment"] > 0) & usable["n_replicates"].notna() & (usable["n_replicates"] >= 2)
    t_lsd = np.where(has_lsd, stats.t.ppf(0.975, df=np.maximum(2.0 * (n_eff - 1.0), 2.0)), np.nan)
    sd_lsd = np.where(has_lsd, usable["error_treatment"] * np.sqrt(n_eff) / (math.sqrt(2.0) * t_lsd), np.nan)
    vi_lsd = (sd_lsd ** 2) / (n_eff * (usable["mean_treatment"] ** 2)) + (sd_lsd ** 2) / (n_eff * (usable["mean_control"] ** 2))
    usable["vi_exact_or_lsd"] = np.where(usable["has_exact_variance"], usable["vi_exact"], np.where(has_lsd & np.isfinite(vi_lsd) & (vi_lsd > 1e-8), np.minimum(vi_lsd, 5.0), np.nan))
    usable["has_exact_or_lsd_variance"] = usable["vi_exact_or_lsd"].notna()

    def ecw_bin(v: float) -> str:
        if pd.isna(v): return "unknown"
        if v < 1.0: return "fresh_<1"
        if v < 3.0: return "slightly_brackish_1_3"
        if v < 5.0: return "moderately_brackish_3_5"
        if v < 8.0: return "saline_5_8"
        return "highly_saline_ge8"

    def norm_mode(r: pd.Series) -> str:
        m = str(r.get("irrigation_mode", "")).lower()
        mulch = str(r.get("mulching_flag", "")).lower() == "true"
        lbl = f"{r.get('treatment_label', '')} {r.get('trial_label', '')}".lower()
        if "subsurface" in m or "subsurface drip" in lbl or "sub-surface drip" in lbl or " sdi" in lbl:
            return "subsurface_drip"
        if "drip" in m or ("drip" in lbl and m in {"other", "unreported", "nan", ""}) or "surface di" in lbl:
            is_mulched_drip = (
                ("unmulch" not in m and "mulch" in m)
                or ("unmulch" not in lbl and "no mulch" not in lbl and "mulch/bare" not in lbl and "mulch" in lbl)
                or mulch
            )
            return "drip_mulched" if is_mulched_drip else "drip_unmulched"
        if "sprink" in m or ("sprink" in lbl and m in {"other", "unreported", "nan", ""}):
            return "sprinkler"
        if "flood" in m or "furrow" in m or "border" in m or "basin" in m or "surface" in m:
            return "surface_flood_furrow"
        return "other"

    def norm_drain(d: object) -> str:
        s = str(d or "").lower()
        if any(k in s for k in ("tile", "pipe", "subsurface", "ditch", "shallow_water_table")):
            return "engineered_drain"
        if "lysimeter" in s:
            return "lysimeter_free_drain"
        return "none_or_natural"

    def norm_crop_legacy(r: pd.Series) -> str:
        txt = f"{r.get('crop_group', '')} {r.get('crop_species', '')}".lower()
        if "wheat" in txt or "triticum" in txt: return "Wheat"
        if "maize" in txt or "corn" in txt or "zea " in txt: return "Maize"
        if "rice" in txt or "oryza" in txt: return "Rice"
        if "cotton" in txt or "gossypium" in txt or "fiber" in txt: return "Cotton_Fiber"
        if any(k in txt for k in ("halophyte", "quinoa", "forage", "alfalfa", "medicago", "sorghum", "barley")): return "Forage_Halophyte_SaltTolerant"
        if any(k in txt for k in ("vegetable", "tomato", "cucumber", "eggplant", "pepper", "lettuce", "spinach", "cabbage", "onion", "garlic", "solanum")): return "Vegetable"
        if any(k in txt for k in ("fruit", "grape", "vitis", "citrus", "olive", "olea", "watermelon", "melon", "date", "palm", "apple", "pear", "cherry", "passion", "pomegranate")): return "Fruit_Tree"
        if any(k in txt for k in ("root", "tuber", "potato", "beet", "sugar beet", "cassava", "taro", "carrot")): return "Root_Tuber"
        if any(k in txt for k in ("oilseed", "legume", "soybean", "sunflower", "rape", "canola", "mustard", "brassica juncea", "brassica napus", "peanut", "bean", "pea", "chickpea", "lentil", "sesame")): return "Oilseed_Legume"
        return "Other"

    def norm_crop7(r: pd.Series) -> str:
        txt = f"{r.get('crop_group', '')} {r.get('crop_species', '')}".lower()
        if "wheat" in txt or "triticum" in txt: return "wheat"
        if "cotton" in txt or "gossypium" in txt or "fiber" in txt: return "cotton"
        if "maize" in txt or "corn" in txt or "zea " in txt: return "maize"
        if any(k in txt for k in ("vegetable", "tomato", "cucumber", "eggplant", "pepper", "lettuce", "spinach", "cabbage", "onion", "garlic", "solanum", "lycopersicon", "okra", "abelmoschus", "brassica oleracea")):
            return "vegetables"
        if any(k in txt for k in ("fruit", "grape", "vitis", "citrus", "olive", "olea", "watermelon", "melon", "cucumis", "date", "palm", "apple", "pear", "cherry", "passion", "passiflora", "pomegranate", "strawberr", "banana", "musa")):
            return "fruit"
        if any(k in txt for k in ("oilseed", "legume", "soybean", "sunflower", "helianthus", "rape", "canola", "mustard", "brassica juncea", "brassica napus", "peanut", "bean", "pea", "chickpea", "lentil", "sesame", "guar", "cyamopsis", "vigna", "cowpea", "carthamus", "safflower")):
            return "oilseed_legume"
        if any(k in txt for k in ("cereal", "rice", "oryza", "barley", "hordeum", "sorghum", "millet", "oat", "avena")):
            return "other_cereals"
        return "other"

    def norm_depth(d: object) -> str:
        s = str(d or "").lower()
        if "topsoil" in s or "0_20" in s: return "topsoil_0_20"
        if "root" in s or "0_40" in s or "0_60" in s or "20_40" in s: return "rootzone_0_40_60"
        if "subsoil" in s or "40_100" in s or "gt100" in s: return "subsoil_40_100"
        return "whole_profile_or_unspecified"

    usable["ecw_stratum"] = usable["ecw_treatment_dsm"].map(ecw_bin)
    usable["duration_stratum"] = usable["duration_years"].apply(lambda d: "unknown" if pd.isna(d) else ("short_le_2yr" if d <= 2.0 else "multiyear_ge_3yr"))
    usable["soil_depth_stratum_std"] = usable["soil_depth_stratum"].map(norm_depth)
    usable["irrigation_mode_std"] = usable.apply(norm_mode, axis=1)
    usable["irrigation_delivery_fao48"] = usable["irrigation_mode_std"].replace({
        "drip_mulched": "surface_drip",
        "drip_unmulched": "surface_drip",
    })
    usable["drip_vs_flood"] = usable["irrigation_mode_std"].map(lambda m: "drip_system" if "drip" in m else ("surface_flood_furrow" if m == "surface_flood_furrow" else "other_mode"))
    usable["has_drainage_engineered"] = usable["drainage_regime"].map(norm_drain)
    usable["subsurface_drain_flag"] = usable["drainage_regime"].astype(str).str.lower().apply(lambda s: ("subsurface" in s) or ("tile" in s) or ("pipe" in s))
    usable["crop_group_std"] = usable.apply(norm_crop_legacy, axis=1)
    usable["crop_group7"] = usable.apply(norm_crop7, axis=1)
    usable["amend_cls"] = usable["amendment_co_applied"].map(classify_amendment_coapplied)

    def re_exact_weights(sub_ex: pd.DataFrame, vi_col: str = "vi_exact") -> np.ndarray:
        yv = sub_ex["lnRR"].values
        vv = sub_ex[vi_col].values
        w_fe = 1.0 / vv
        mu_fe = float(np.sum(w_fe * yv) / np.sum(w_fe))
        q_stat = float(np.sum(w_fe * (yv - mu_fe) ** 2))
        c_denom = float(np.sum(w_fe) - np.sum(w_fe ** 2) / np.sum(w_fe))
        tau2 = max(0.0, (q_stat - max(len(yv) - 1, 0)) / max(c_denom, 1e-12))
        fam_cnt = sub_ex.groupby("family_id")["lnRR"].transform("count").values.clip(min=1)
        return 1.0 / ((vv + tau2) * fam_cnt)

    # Legacy strata summary table
    strata_rows: List[Dict[str, Any]] = []
    for metric in ["grain_yield", "shoot_biomass", "wue_or_iwue", "soil_EC", "soil_SAR", "soil_ESP", "leaching_fraction", "ks_saturated", "infiltration_rate"]:
        m_df = usable[usable["metric_std"] == metric]
        if len(m_df) == 0:
            continue
        res_all = wls_cluster_summary(m_df["lnRR"].values, m_df["w_obs"].values, m_df["family_id"].values)
        strata_rows.append({"metric": metric, "group_by": "ALL", "level": "ALL", **res_all})
        ex_df = m_df[m_df["has_exact_variance"]].copy()
        if len(ex_df) >= 3:
            res_ex = wls_cluster_summary(ex_df["lnRR"].values, re_exact_weights(ex_df), ex_df["family_id"].values)
            strata_rows.append({"metric": metric, "group_by": "EXACT_VARIANCE_RE", "level": "PASS_READY", **res_ex})
        for col_grp in ["ecw_stratum", "irrigation_mode_std", "drip_vs_flood", "irrigation_scheduling", "drainage_regime", "has_drainage_engineered", "duration_stratum", "soil_depth_stratum_std", "crop_group_std", "crop_group7"]:
            for lvl, grp in m_df.groupby(col_grp):
                res_g = wls_cluster_summary(grp["lnRR"].values, grp["w_obs"].values, grp["family_id"].values)
                strata_rows.append({"metric": metric, "group_by": col_grp, "level": str(lvl), **res_g})
        if metric == "soil_EC":
            for (dur_s, dep_s), grp in m_df.groupby(["duration_stratum", "soil_depth_stratum_std"]):
                res_dd = wls_cluster_summary(grp["lnRR"].values, grp["w_obs"].values, grp["family_id"].values)
                strata_rows.append({"metric": metric, "group_by": "duration_x_depth", "level": f"{dur_s}::{dep_s}", **res_dd})
            for (dr_s, dur_s), grp in m_df.groupby(["has_drainage_engineered", "duration_stratum"]):
                res_drd = wls_cluster_summary(grp["lnRR"].values, grp["w_obs"].values, grp["family_id"].values)
                strata_rows.append({"metric": metric, "group_by": "drainage_x_duration", "level": f"{dr_s}::{dur_s}", **res_drd})

    OUT_STRATA_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(strata_rows).to_csv(OUT_STRATA_CSV, index=False)

    yield_all = usable[usable["metric_std"].isin(["grain_yield", "shoot_biomass"])].copy()
    grain_only = usable[usable["metric_std"] == "grain_yield"].copy()
    soil_ec = usable[usable["metric_std"] == "soil_EC"].copy()
    wue_df = usable[usable["metric_std"] == "wue_or_iwue"].copy()

    ex_yield = yield_all[yield_all["has_exact_variance"]].copy()
    ex_yield["w_obs"] = re_exact_weights(ex_yield, "vi_exact")
    ex_lsd_yield = yield_all[yield_all["has_exact_or_lsd_variance"]].copy()
    ex_lsd_yield["w_obs"] = re_exact_weights(ex_lsd_yield, "vi_exact_or_lsd")
    ex_grain = grain_only[grain_only["has_exact_variance"]].copy()
    ex_grain["w_obs"] = re_exact_weights(ex_grain, "vi_exact")
    ex_soilec = soil_ec[soil_ec["has_exact_variance"]].copy()
    ex_soilec["w_obs"] = re_exact_weights(ex_soilec, "vi_exact")
    ex_lsd_soilec = soil_ec[soil_ec["has_exact_or_lsd_variance"]].copy()
    ex_lsd_soilec["w_obs"] = re_exact_weights(ex_lsd_soilec, "vi_exact_or_lsd")

    field_yield = yield_all[yield_all["experiment_type"].astype(str).str.lower() == "field"].copy()
    field_soilec = soil_ec[soil_ec["experiment_type"].astype(str).str.lower() == "field"].copy()

    # Legacy dose_response_ecw_star (retained)
    dose_models: Dict[str, Any] = {
        "yield_combined_all": fit_ecw_dose_response_cluster_wls(yield_all, "negative_drop"),
        "yield_combined_field_only": fit_ecw_dose_response_cluster_wls(field_yield, "negative_drop"),
        "yield_combined_exact_variance": fit_ecw_dose_response_cluster_wls(ex_yield, "negative_drop"),
        "yield_combined_exact_plus_lsd_variance": fit_ecw_dose_response_cluster_wls(ex_lsd_yield, "negative_drop"),
        "grain_yield_only": fit_ecw_dose_response_cluster_wls(grain_only, "negative_drop"),
        "grain_yield_exact_variance": fit_ecw_dose_response_cluster_wls(ex_grain, "negative_drop"),
        "wue_or_iwue_all": fit_ecw_dose_response_cluster_wls(wue_df, "negative_drop"),
        "soil_EC_all": fit_ecw_dose_response_cluster_wls(soil_ec, "positive_accumulation"),
        "soil_EC_field_only": fit_ecw_dose_response_cluster_wls(field_soilec, "positive_accumulation"),
        "soil_EC_exact_variance": fit_ecw_dose_response_cluster_wls(ex_soilec, "positive_accumulation"),
        "soil_EC_exact_plus_lsd_variance": fit_ecw_dose_response_cluster_wls(ex_lsd_soilec, "positive_accumulation"),
        "by_scheduling": {}, "by_scheduling_field_only": {}, "by_irrigation_mode": {},
        "by_drip_vs_flood": {}, "by_duration": {}, "by_crop_group": {},
    }
    for sched, grp in yield_all.groupby("irrigation_scheduling"):
        dose_models["by_scheduling"][str(sched)] = fit_ecw_dose_response_cluster_wls(grp, "negative_drop")
    for sched, grp in field_yield.groupby("irrigation_scheduling"):
        dose_models["by_scheduling_field_only"][str(sched)] = fit_ecw_dose_response_cluster_wls(grp, "negative_drop")
    for mode, grp in yield_all.groupby("irrigation_mode_std"):
        dose_models["by_irrigation_mode"][str(mode)] = fit_ecw_dose_response_cluster_wls(grp, "negative_drop")
    for dvf, grp in yield_all.groupby("drip_vs_flood"):
        dose_models["by_drip_vs_flood"][str(dvf)] = fit_ecw_dose_response_cluster_wls(grp, "negative_drop")
    for dur, grp in soil_ec.groupby("duration_stratum"):
        dose_models["by_duration"][f"soil_EC_{dur}"] = fit_ecw_dose_response_cluster_wls(grp, "positive_accumulation")
    for dur, grp in yield_all.groupby("duration_stratum"):
        dose_models["by_duration"][f"yield_{dur}"] = fit_ecw_dose_response_cluster_wls(grp, "negative_drop")
    for cg, grp in yield_all.groupby("crop_group_std"):
        if grp["family_id"].nunique() >= 3:
            dose_models["by_crop_group"][str(cg)] = fit_ecw_dose_response_cluster_wls(grp, "negative_drop")

    # 1a. Absolute ECw Maas-Hoffman piecewise models + cluster bootstrap CIs + logistic sensitivity
    print("[Arm A 1a] Fitting absolute-ECw Maas-Hoffman models + B=1000 cluster bootstrap...")
    multi_yr_yield = yield_all[pd.to_numeric(yield_all["duration_years"], errors="coerce") >= 2.0].copy()
    abs_subsets = {
        "pooled": yield_all,
        "continuous": yield_all[yield_all["irrigation_scheduling"] == "continuous_brackish"],
        "cyclic": yield_all[yield_all["irrigation_scheduling"] == "cyclic_alternating"],
        "blended": yield_all[yield_all["irrigation_scheduling"] == "blended_conjunctive"],
        "field_only": field_yield,
        "multiyear_ge2yr": multi_yr_yield,
        "exact_variance": ex_yield,
        "exact_plus_lsd": ex_lsd_yield,
        "grain_only": grain_only,
        "wue": wue_df,
    }
    dose_abs: Dict[str, Any] = {}
    for s_name, s_df in abs_subsets.items():
        dose_abs[s_name] = fit_maas_hoffman_abs(s_df, rng, n_boot=N_BOOT)
        print(f"  {s_name}: ECw*90={dose_abs[s_name].get('ecw_star_90pct_dsm')} CI={dose_abs[s_name].get('ecw_star_90pct_ci95')}")

    logistic_sens = {
        "pooled_sigmoidal_p_ge_1": fit_logistic_sensitivity(yield_all, p_lower_bound=1.0),
        "continuous_sigmoidal_p_ge_1": fit_logistic_sensitivity(abs_subsets["continuous"], p_lower_bound=1.0),
        "cyclic_sigmoidal_p_ge_1": fit_logistic_sensitivity(abs_subsets["cyclic"], p_lower_bound=1.0),
        "pooled_unconstrained_p_ge_02": fit_logistic_sensitivity(yield_all, p_lower_bound=0.2),
    }

    # 1b. Soil EC absolute-ECw response (overall + separate native-scale ECe vs EC1:5 + multiyear accumulation risk)
    print("[Arm A 1b] Fitting monotonic absolute-ECw soil EC accumulation models (incl. separate ECe vs EC1:5)...")
    if "soil_ec_method_std" not in soil_ec.columns:
        soil_ec["soil_ec_method_std"] = soil_ec["initial_soil_ec_ratio"].astype(str).apply(
            lambda s: "saturated_paste_ECe" if ("ece" in s.lower() or "saturated" in s.lower()) else ("extract_1_5" if "1:5" in s.lower() else "other")
        )
    soilec_ece = soil_ec[soil_ec["soil_ec_method_std"] == "saturated_paste_ECe"].copy()
    soilec_15 = soil_ec[soil_ec["soil_ec_method_std"] == "extract_1_5"].copy()
    soilec_multi = soil_ec[pd.to_numeric(soil_ec["duration_years"], errors="coerce") >= 2.0].copy()

    soil_ec_abs = {
        "all": fit_soil_ec_absolute(soil_ec, rng, n_boot=N_BOOT),
        "field_only": fit_soil_ec_absolute(field_soilec, rng, n_boot=N_BOOT),
        "exact_variance": fit_soil_ec_absolute(ex_soilec, rng, n_boot=N_BOOT),
        "exact_plus_lsd": fit_soil_ec_absolute(ex_lsd_soilec, rng, n_boot=N_BOOT),
        "saturated_paste_ECe_only": fit_soil_ec_absolute(soilec_ece, rng, n_boot=N_BOOT),
        "extract_1_5_only": fit_soil_ec_absolute(soilec_15, rng, n_boot=N_BOOT),
        "multiyear_ge2yr_monitored": fit_soil_ec_absolute(soilec_multi, rng, n_boot=N_BOOT),
    }
    # Multi-year soil EC covariate model (lnRR_soil_EC ~ dECw + irr_mm + rain_mm + engineered_drainage)
    sm_cov = soilec_multi.dropna(subset=["lnRR", "delta_ecw", "w_obs", "irrigation_amount_mm", "seasonal_rainfall_mm"]).copy()
    sm_cov = sm_cov[sm_cov["delta_ecw"] > 0]
    if len(sm_cov) >= 20 and sm_cov["family_id"].nunique() >= 5:
        dx_m = sm_cov["delta_ecw"].values
        irr_m = sm_cov["irrigation_amount_mm"].values / 100.0  # per 100 mm
        rain_m = sm_cov["seasonal_rainfall_mm"].values / 100.0  # per 100 mm
        dr_m = (sm_cov["has_drainage_engineered"] == "engineered_drain").astype(float).values
        X_m = np.column_stack([np.ones(len(sm_cov)), dx_m, irr_m, rain_m, dr_m])
        r_m = _cr1_ols(sm_cov["lnRR"].values, X_m, sm_cov["w_obs"].values, sm_cov["family_id"].values, ["intercept", "delta_ecw_per_dsm", "irr_per_100mm", "rain_per_100mm", "engineered_drain"])
        soil_ec_abs["multiyear_covariate_cr1"] = {
            "k": r_m["k"], "n_clusters": r_m["n_clusters"], "terms": r_m["terms"],
            "note": "Multi-year (>=2 yr) soil salinity CR1 model incorporating backfilled seasonal irrigation, rainfall, and drainage covariates.",
        }
    prepost_p = HARMONIZED / "armA_multiyear_soil_ec_pre_post_pairs_current.csv"
    if prepost_p.is_file():
        pp = pd.read_csv(prepost_p)
        pp["pre"] = pd.to_numeric(pp["soil_ec_pre_val"], errors="coerce")
        pp["post"] = pd.to_numeric(pp["soil_ec_post_val"], errors="coerce")
        pp["ecw"] = pd.to_numeric(pp["ecw_dsm"], errors="coerce")
        pp = pp.dropna(subset=["pre", "post", "ecw"])
        pp = pp[(pp["pre"] > 0) & (pp["post"] > 0)]
        pp["ratio"] = pp["post"] / pp["pre"]
        soil_ec_abs["multiyear_pre_post_monitoring_summary"] = {
            "n_pairs": int(len(pp)),
            "n_families": int(pp["family_id"].nunique()),
            "median_post_to_pre_ratio_lt3_55": round(float(pp.loc[pp["ecw"] < 3.55, "ratio"].median()), 3) if (pp["ecw"] < 3.55).any() else None,
            "median_post_to_pre_ratio_ge3_55": round(float(pp.loc[pp["ecw"] >= 3.55, "ratio"].median()), 3) if (pp["ecw"] >= 3.55).any() else None,
            "share_accumulating_lt3_55": round(float((pp.loc[pp["ecw"] < 3.55, "ratio"] > 1.0).mean()), 3) if (pp["ecw"] < 3.55).any() else None,
            "share_accumulating_ge3_55": round(float((pp.loc[pp["ecw"] >= 3.55, "ratio"] > 1.0).mean()), 3) if (pp["ecw"] >= 3.55).any() else None,
        }

    # 1c. 7 Crop groups vs FAO-29 / SPAM ECe90
    print("[Arm A 1c] Fitting 7 crop groups & FAO-29 comparison...")
    crop7_rows: Dict[str, Any] = {}
    k_emp_vals: List[float] = []
    for cg7_name in ["wheat", "cotton", "maize", "vegetables", "fruit", "oilseed_legume", "other_cereals"]:
        c_df = yield_all[yield_all["crop_group7"] == cg7_name]
        fit_c = fit_maas_hoffman_abs(c_df, rng, n_boot=N_BOOT)
        spam_codes = CROP7_TO_SPAM[cg7_name]
        ece90_vals = [CROP_SALT_PARAMS[sc][0] + 10.0 / CROP_SALT_PARAMS[sc][1] for sc in spam_codes]
        fao_ece90_med = float(np.median(ece90_vals))
        fao_ecw_ref = fao_ece90_med / 1.5
        emp90 = fit_c.get("ecw_star_90pct_dsm")
        k_emp = round(emp90 / fao_ece90_med, 3) if emp90 is not None else None
        if k_emp is not None:
            k_emp_vals.append(k_emp)
        fit_c["fao29_spam_codes"] = spam_codes
        fit_c["fao29_ece90_median_dsm"] = round(fao_ece90_med, 3)
        fit_c["fao29_ecw_ref_dsm"] = round(fao_ecw_ref, 3)
        fit_c["k_emp_ecw90_over_ece90"] = k_emp
        fit_c["k_emp_vs_067_ratio"] = round(k_emp / (1.0 / 1.5), 3) if k_emp is not None else None
        crop7_rows[cg7_name] = fit_c

    crop_vs_fao = {
        "by_crop_group7": crop7_rows,
        "median_k_emp_across_7_groups": round(float(np.median(k_emp_vals)), 3) if k_emp_vals else None,
        "median_k_emp_excl_other_cereals": round(float(np.median([crop7_rows[k]["k_emp_ecw90_over_ece90"] for k in ["wheat", "cotton", "maize", "vegetables", "fruit", "oilseed_legume"] if crop7_rows[k].get("k_emp_ecw90_over_ece90") is not None])), 3),
        "theoretical_fao29_lf15_20_ratio": round(1.0 / 1.5, 3),
    }

    # 1d. Covariate-adjusted scheduling & irrigation technology CR1 regressions
    t_pooled = float(dose_abs["pooled"]["t_dsm"])
    adjusted_mods = fit_adjusted_scheduling_and_tech(yield_all, soil_ec, t_pooled, rng, n_boot=N_BOOT)

    # 1e. Bridge moderator tests (amendments, mulching, subsurface drainage)
    bridge_tests = fit_bridge_moderator_tests(yield_all, soil_ec, t_pooled)

    # 1f. Within-study family-FE year x depth dynamics on soil_EC
    year_x_depth_within = fit_within_study_year_x_depth(soil_ec, rng, n_boot=N_BOOT)

    # Legacy H3 slope tests & LOCO
    interactions = {
        "yield_scheduling_vs_continuous": fit_interaction_slope_cr1(yield_all, "irrigation_scheduling", "continuous_brackish", ["cyclic_alternating", "blended_conjunctive", "growth_stage_switched"]),
        "yield_drip_vs_surface_flood": fit_interaction_slope_cr1(yield_all, "drip_vs_flood", "surface_flood_furrow", ["drip_system"]),
        "soil_EC_scheduling_vs_continuous": fit_interaction_slope_cr1(soil_ec, "irrigation_scheduling", "continuous_brackish", ["cyclic_alternating", "blended_conjunctive", "growth_stage_switched"]),
        "soil_EC_drainage_vs_undrained": fit_interaction_slope_cr1(soil_ec, "has_drainage_engineered", "none_or_natural", ["engineered_drain", "lysimeter_free_drain"]),
    }

    loco_armA: Dict[str, Any] = {}
    for lbl, sub in [("yield_combined", yield_all), ("soil_EC", soil_ec)]:
        ctries = [c for c, g in sub.groupby("country") if str(c) not in {"", "nan", "None", "Unknown"} and g["family_id"].nunique() >= 3]
        if len(ctries) >= 3:
            pcs = []
            for dc in ctries:
                r = wls_cluster_summary(sub[sub["country"] != dc]["lnRR"].values, sub[sub["country"] != dc]["w_obs"].values, sub[sub["country"] != dc]["family_id"].values)
                if "pct_change" in r:
                    pcs.append({"dropped_country": dc, "pct_change": r["pct_change"], "n_clusters": r["n_clusters"]})
            if pcs:
                vals = [x["pct_change"] for x in pcs]
                loco_armA[lbl] = {"n_countries_tested": len(pcs), "min_pct": min(vals), "max_pct": max(vals), "all_same_sign": all(v > 0 for v in vals) or all(v < 0 for v in vals), "details": pcs}

    env_dur_subgroups = fit_environmental_duration_subgroups(yield_all, soil_ec, t_pooled)

    # Independent RNG for non-breaking Phase 1 diagnostics (preserves existing RNG sequence bit-for-bit)
    rng_ext = np.random.default_rng(SEED + 99)
    cont_df = abs_subsets["continuous"]
    cont_excl_ry_gt2_df = cont_df[np.exp(cont_df["lnRR"].values) <= 2.0].copy()
    dose_abs["continuous_excl_ry_gt2"] = fit_maas_hoffman_abs(cont_excl_ry_gt2_df, rng_ext, n_boot=N_BOOT)

    def _band_shares(sub_b: pd.DataFrame) -> Dict[str, Any]:
        n_b = max(len(sub_b), 1)
        hort_mask = sub_b["crop_group7"].isin(["vegetables", "fruit"])
        cer7_mask = sub_b["crop_group7"].isin(["wheat", "other_cereals", "maize"])
        cer_raw_mask = sub_b["crop_group"].astype(str).str.lower().isin(["cereal", "wheat"])
        return {
            "k": int(len(sub_b)),
            "n_clusters": int(sub_b["family_id"].nunique()),
            "horticultural_pct": round(float(hort_mask.mean() * 100.0), 1),
            "cereal_group7_pct": round(float(cer7_mask.mean() * 100.0), 1),
            "cereal_raw_pct": round(float(cer_raw_mask.mean() * 100.0), 1),
            "cotton_pct": round(float((sub_b["crop_group7"] == "cotton").mean() * 100.0), 1),
        }

    crop_composition_by_ecw_band = {
        "continuous_ecw_le_4": _band_shares(cont_df[cont_df["ecw_treatment_dsm"] <= 4.0]),
        "continuous_ecw_4_to_8": _band_shares(cont_df[(cont_df["ecw_treatment_dsm"] > 4.0) & (cont_df["ecw_treatment_dsm"] <= 8.0)]),
        "continuous_ecw_gt_8": _band_shares(cont_df[cont_df["ecw_treatment_dsm"] > 8.0]),
    }

    b_pooled = float(dose_abs["pooled"]["b_per_dsm"])

    def _strict_dose_adj_egger(df_v: pd.DataFrame, vi_col: str) -> Dict[str, Any]:
        s = df_v.dropna(subset=["lnRR", vi_col, "ecw_treatment_dsm", "ecw_control_dsm"]).copy()
        s = s[s[vi_col] > 0].copy()
        ry_t = np.maximum(1.0 - b_pooled * np.maximum(0.0, s["ecw_treatment_dsm"].values - t_pooled), 0.05)
        ry_c = np.maximum(1.0 - b_pooled * np.maximum(0.0, s["ecw_control_dsm"].values - t_pooled), 0.05)
        resid = s["lnRR"].values - np.log(ry_t / ry_c)
        se = np.sqrt(s[vi_col].values)
        z = resid / se
        prec = 1.0 / se
        cl = s["family_id"].astype(str).values
        uniq = np.unique(cl)
        n_cl = len(uniq)
        w = 1.0 / s.groupby("family_id")["lnRR"].transform("count").values.clip(min=1)
        X = np.column_stack([np.ones(len(s)), prec])
        XtWX = X.T @ (w[:, None] * X) + 1e-10 * np.eye(2)
        beta = np.linalg.solve(XtWX, X.T @ (w * z))
        res_e = z - X @ beta
        inv = np.linalg.pinv(XtWX)
        meat = sum(np.outer(X[cl == c].T @ (w[cl == c] * res_e[cl == c]), X[cl == c].T @ (w[cl == c] * res_e[cl == c])) for c in uniq)
        cov = (n_cl / max(n_cl - 1, 1)) * (inv @ meat @ inv)
        se0 = float(np.sqrt(max(cov[0, 0], 1e-12)))
        p0 = float(2.0 * stats.t.sf(abs(beta[0] / se0), df=max(n_cl - 2, 1)))
        return {
            "k": int(len(s)),
            "n_clusters": int(n_cl),
            "egger_intercept": round(float(beta[0]), 3),
            "se_cr": round(se0, 3),
            "egger_p": round(p0, 5),
            "bias_flag": bool(p0 < 0.05),
            "n_prec_gt_90": int((prec > 90.0).sum()),
        }

    strict_exact_yield_egger = _strict_dose_adj_egger(ex_yield, "vi_exact")
    strict_exact_plus_lsd_yield_egger = _strict_dose_adj_egger(ex_lsd_yield, "vi_exact_or_lsd")

    summary = {
        "script": "pillar2_armA_ecw_threshold_model.py",
        "seed": SEED,
        "n_boot": N_BOOT,
        "usable_rows_total": int(len(usable)),
        "usable_studies_total": int(usable["family_id"].nunique()),
        "field_only_rows": int((usable["experiment_type"].astype(str).str.lower() == "field").sum()),
        "field_only_studies": int(usable.loc[usable["experiment_type"].astype(str).str.lower() == "field", "family_id"].nunique()),
        "exact_variance_rows": int(usable["has_exact_variance"].sum()),
        "exact_plus_lsd_variance_rows": int(usable["has_exact_or_lsd_variance"].sum()),
        "ece90_ref_sites": ece90_ref_sites,
        "ece90_ref_sites_n_grid_cells": n_site_cells_matched,
        "countries_represented": {k: int(v) for k, v in usable.groupby("family_id")["country"].first().value_counts().to_dict().items()},
        "metrics_counts": {k: int(v) for k, v in usable["metric_std"].value_counts().to_dict().items()},
        "dose_response_ecw_star": dose_models,
        "dose_response_absolute": dose_abs,
        "crop_composition_by_ecw_band": crop_composition_by_ecw_band,
        "logistic_sensitivity": logistic_sens,
        "soil_EC_absolute": soil_ec_abs,
        "crop_group_vs_fao29": crop_vs_fao,
        "adjusted_scheduling_and_tech": adjusted_mods,
        "bridge_moderator_tests": bridge_tests,
        "environmental_duration_subgroups": env_dur_subgroups,
        "within_study_year_x_depth": year_x_depth_within,
        "h3_interaction_slope_tests": interactions,
        "egger_funnel_checks": {
            "yield_combined": egger_cluster_test_armA(yield_all),
            "soil_EC": egger_cluster_test_armA(soil_ec),
        },
        "strict_exact_variance_yield_egger": strict_exact_yield_egger,
        "strict_exact_plus_lsd_yield_egger": strict_exact_plus_lsd_yield_egger,
        "leave_one_country_out": loco_armA,
        "headline_strata": [r for r in strata_rows if r["group_by"] in {"ALL", "EXACT_VARIANCE_RE", "ecw_stratum", "drip_vs_flood", "irrigation_scheduling", "duration_x_depth", "drainage_x_duration"} and r["k"] >= 3],
    }
    from field_threshold_checks import extend_arm_a
    extend_arm_a(summary)
    with OUT_RESULTS_JSON.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Saved Arm A results -> {OUT_RESULTS_JSON}")


if __name__ == "__main__":
    prepare_outputs()
    main()
