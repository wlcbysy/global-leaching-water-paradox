#!/usr/bin/env python3
"""Family-clustered hydraulic and WUE models from harmonized effect sizes."""

from __future__ import annotations

from config import HARMONIZED, OUT_TABLES, prepare_outputs

import hashlib
import json
import math
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import optimize, stats

MERGED_INPUT = HARMONIZED / "pillar2_armB_merged_current.csv"
AUDIT_7_LEACH_CSV = HARMONIZED / "armB_leaching_7clusters_audit_current.csv"

OUT_RESULTS_JSON = OUT_TABLES / "pillar2_armB_hydraulic_results_current.json"
OUT_SUMMARY_CSV = OUT_TABLES / "pillar2_armB_subgroup_summary_current.csv"

SEED = 20261124
N_BOOT = 1000


def fit_reml_multilevel_1d(
    y: np.ndarray, vi: np.ndarray, clusters: np.ndarray, ctrl_groups: np.ndarray, shared_ctrl_rho: float = 0.5,
) -> Dict[str, Any]:
    ok = np.isfinite(y) & np.isfinite(vi) & (vi > 0)
    y, vi, clusters, ctrl_groups = y[ok], vi[ok], clusters[ok], ctrl_groups[ok]

    k = len(y)
    if k < 3:
        return {"k": int(k), "n_clusters": int(len(set(clusters))), "status": "INSUFFICIENT_K"}

    uniq_cl = np.unique(clusters)
    n_cl = len(uniq_cl)
    if n_cl == 1:
        w = 1.0 / vi
        mu = float(np.sum(w * y) / np.sum(w))
        se = float(math.sqrt(1.0 / np.sum(w)))
        return {
            "k": int(k), "n_clusters": 1, "mu_lnrr": round(mu, 4),
            "se": round(se, 4), "ci95": [round(mu - 1.96 * se, 4), round(mu + 1.96 * se, 4)],
            "pct_change": round((math.exp(mu) - 1.0) * 100.0, 2),
            "p_val": None, "status": "SINGLE_CLUSTER_DESCRIPTIVE",
        }

    cl_blocks: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for c in uniq_cl:
        idx = np.where(clusters == c)[0]
        yi, vii, ci = y[idx], vi[idx], ctrl_groups[idx]
        m = len(idx)
        Vi = np.diag(vii)
        for a in range(m):
            for b in range(a + 1, m):
                if ci[a] == ci[b]:
                    cov_ab = shared_ctrl_rho * math.sqrt(vii[a] * vii[b])
                    Vi[a, b] = cov_ab
                    Vi[b, a] = cov_ab
        cl_blocks.append((yi, Vi, ci))

    def reml_nll(params: np.ndarray) -> float:
        tau2_b = math.exp(params[0])
        tau2_w = math.exp(params[1])
        sum_w1 = sum_wy = logdet_V = yTVinv_y = 0.0
        for yi, Vi, _ in cl_blocks:
            m = len(yi)
            Sigma = Vi + tau2_b * np.ones((m, m)) + tau2_w * np.eye(m) + 1e-9 * np.eye(m)
            try:
                L = np.linalg.cholesky(Sigma)
            except np.linalg.LinAlgError:
                return 1e12
            logdet_V += 2.0 * float(np.sum(np.log(np.diag(L))))
            ones = np.ones((m, 1))
            sol_1 = np.linalg.solve(L.T, np.linalg.solve(L, ones))
            sol_y = np.linalg.solve(L.T, np.linalg.solve(L, yi[:, None]))
            sum_w1 += float(np.sum(sol_1))
            sum_wy += float(np.sum(sol_y))
            yTVinv_y += float(np.dot(yi, sol_y[:, 0]))
        if sum_w1 <= 0:
            return 1e12
        mu_hat = sum_wy / sum_w1
        quad = yTVinv_y - 2.0 * mu_hat * sum_wy + (mu_hat ** 2) * sum_w1
        return 0.5 * (logdet_V + math.log(sum_w1) + quad)

    res = optimize.minimize(reml_nll, x0=np.array([-3.0, -3.5]), bounds=[(-14.0, 2.5), (-14.0, 2.5)], method="L-BFGS-B")
    tau2_b = math.exp(float(res.x[0]))
    tau2_w = math.exp(float(res.x[1]))

    sum_w1 = sum_wy = 0.0
    cl_bread_pieces = []
    for yi, Vi, _ in cl_blocks:
        m = len(yi)
        Sigma = Vi + tau2_b * np.ones((m, m)) + tau2_w * np.eye(m) + 1e-8 * np.eye(m)
        ones = np.ones(m)
        try:
            L = np.linalg.cholesky(Sigma)
            w1 = np.linalg.solve(L.T, np.linalg.solve(L, ones[:, None]))[:, 0]
            wy = np.linalg.solve(L.T, np.linalg.solve(L, yi[:, None]))[:, 0]
            solve_vec = lambda v, _L=L: np.linalg.solve(_L.T, np.linalg.solve(_L, v[:, None]))[:, 0]
        except np.linalg.LinAlgError:
            Sinv = np.linalg.pinv(Sigma, rcond=1e-10)
            w1 = Sinv @ ones
            wy = Sinv @ yi
            solve_vec = lambda v, _Si=Sinv: _Si @ v
        if not (np.all(np.isfinite(w1)) and np.all(np.isfinite(wy))):
            continue
        sum_w1 += float(np.sum(w1))
        sum_wy += float(np.sum(wy))
        cl_bread_pieces.append((ones, yi, solve_vec))

    if sum_w1 <= 0 or not cl_bread_pieces:
        return {"k": int(k), "n_clusters": int(n_cl), "status": "NUMERICAL_DEGENERATE"}

    mu = sum_wy / sum_w1
    se_model = math.sqrt(1.0 / sum_w1)
    meat = 0.0
    for ones, yi, solve_vec in cl_bread_pieces:
        score_i = float(np.dot(ones, solve_vec(yi - mu * ones)))
        if math.isfinite(score_i):
            meat += score_i ** 2
    var_cr = (n_cl / max(n_cl - 1, 1)) * meat / (sum_w1 ** 2)
    se_cr = math.sqrt(max(var_cr, se_model ** 2))

    df_t = max(n_cl - 1, 1)
    t_crit = float(stats.t.ppf(0.975, df=df_t))
    t_stat = mu / max(se_cr, 1e-9)
    p_val = float(2.0 * stats.t.sf(abs(t_stat), df=df_t))

    # 95% Prediction Interval: mu +/- t * sqrt(SE_cr^2 + tau2_b + tau2_w)
    pi_half = t_crit * math.sqrt(max(se_cr ** 2 + tau2_b + tau2_w, 0.0))

    w_fe = 1.0 / vi
    mu_fe = float(np.sum(w_fe * y) / np.sum(w_fe))
    Q = float(np.sum(w_fe * (y - mu_fe) ** 2))
    I2 = max(0.0, (Q - (k - 1)) / Q * 100.0) if Q > 0 else 0.0

    return {
        "k": int(k),
        "n_clusters": int(n_cl),
        "preliminary_low_cluster_flag": bool(n_cl < 10),
        "mu_lnrr": round(mu, 4),
        "se_cr": round(se_cr, 4),
        "se_model": round(se_model, 4),
        "ci95": [round(mu - t_crit * se_cr, 4), round(mu + t_crit * se_cr, 4)],
        "pi95": [round(mu - pi_half, 4), round(mu + pi_half, 4)],
        "pct_change": round((math.exp(mu) - 1.0) * 100.0, 2),
        "pct_ci95": [round((math.exp(mu - t_crit * se_cr) - 1.0) * 100.0, 2), round((math.exp(mu + t_crit * se_cr) - 1.0) * 100.0, 2)],
        "pct_pi95": [round((math.exp(mu - pi_half) - 1.0) * 100.0, 2), round((math.exp(mu + pi_half) - 1.0) * 100.0, 2)],
        "p_val": round(p_val, 6),
        "tau2_between": round(tau2_b, 5),
        "tau2_within": round(tau2_w, 5),
        "I2_pct": round(I2, 1),
        "status": "OK",
    }


def egger_cluster_test(y: np.ndarray, vi: np.ndarray, clusters: np.ndarray) -> Dict[str, Any]:
    df_e = pd.DataFrame({"y": y, "se": np.sqrt(vi), "cl": clusters}).dropna()
    if len(df_e) < 6:
        return {"status": "INSUFFICIENT_K"}
    agg = df_e.groupby("cl", as_index=False).agg(y=("y", "mean"), se=("se", "mean"))
    if len(agg) < 6:
        return {"status": "INSUFFICIENT_CLUSTERS"}
    prec = 1.0 / np.maximum(agg["se"].values, 1e-4)
    snd = agg["y"].values * prec
    _, intercept, _, p_val, _ = stats.linregress(prec, snd)
    return {"n_clusters": int(len(agg)), "egger_intercept": round(float(intercept), 3), "egger_p": round(float(p_val), 4), "bias_flag": bool(p_val < 0.05)}


def egger_cluster_test_exact(sub_k: pd.DataFrame) -> Dict[str, Any]:
    """CR1 cluster-robust Egger regression (z = lnRR/se ~ 1 + 1/se) on exact-variance contrasts."""
    d = sub_k.loc[sub_k["has_exact_variance"] & sub_k["vi_exact"].notna() & (sub_k["vi_exact"] > 0), ["lnRR", "vi_exact", "family_id"]].dropna().copy()
    if len(d) < 6 or d["family_id"].nunique() < 6:
        return {"status": "INSUFFICIENT_CLUSTERS", "k": int(len(d)), "n_clusters": int(d["family_id"].nunique())}
    se = np.sqrt(np.maximum(d["vi_exact"].values, 1e-12))
    z = d["lnRR"].values / se
    prec = 1.0 / se
    cl = d["family_id"].astype(str).values
    uniq = np.unique(cl)
    n_cl = len(uniq)
    fc = d.groupby("family_id")["lnRR"].transform("count").values.clip(min=1)
    w = 1.0 / fc
    X = np.column_stack([np.ones(len(d)), prec])
    XtWX = X.T @ (w[:, None] * X) + 1e-10 * np.eye(2)
    beta = np.linalg.solve(XtWX, X.T @ (w * z))
    res = z - X @ beta
    inv = np.linalg.pinv(XtWX)
    meat = sum(np.outer(X[cl == c].T @ (w[cl == c] * res[cl == c]), X[cl == c].T @ (w[cl == c] * res[cl == c])) for c in uniq)
    cov = (n_cl / max(n_cl - 1, 1)) * (inv @ meat @ inv)
    se0 = float(np.sqrt(max(cov[0, 0], 1e-12)))
    p0 = float(2.0 * stats.t.sf(abs(beta[0] / se0), df=max(n_cl - 2, 1)))
    return {
        "k": int(len(d)),
        "n_clusters": int(n_cl),
        "egger_intercept": round(float(beta[0]), 3),
        "se_cr": round(se0, 3),
        "egger_p": round(p0, 5),
        "bias_flag": bool(p0 < 0.05),
    }


def summarize_ratio(df_sub: pd.DataFrame, col: str, rng: np.random.Generator) -> Dict[str, Any]:
    sub = df_sub.dropna(subset=[col]).copy()
    if sub.empty:
        return {"n_pairs": 0, "n_families": 0}
    fam = sub.groupby("family_id")[col].mean()
    vals = fam.values
    n_f = len(vals)
    med = float(np.median(vals))
    mn = float(np.mean(vals))
    med_b, mn_b = [], []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n_f, size=n_f)
        s = vals[idx]
        med_b.append(math.exp(float(np.median(s))))
        mn_b.append(math.exp(float(np.mean(s))))
    countries = {}
    if "country" in sub.columns:
        fc = sub.groupby("family_id")["country"].first()
        countries = {str(k): int(v) for k, v in fc.value_counts().items()}
    return {
        "n_pairs": int(len(sub)),
        "n_families": int(n_f),
        "n_unique_dois": int(sub["doi"].dropna().nunique()) if "doi" in sub.columns else int(n_f),
        "countries": countries,
        "n_pairs_et_increased": int((sub[col] > 0).sum()),
        "n_families_et_increased": int((vals > 0).sum()),
        "share_families_increased": round(float((vals > 0).mean()), 4),
        "median_ln_ratio": round(med, 5),
        "mean_ln_ratio": round(mn, 5),
        "ratio_median": round(math.exp(med), 4),
        "ratio_median_ci95": [round(float(np.percentile(med_b, 2.5)), 4), round(float(np.percentile(med_b, 97.5)), 4)],
        "ratio_mean": round(math.exp(mn), 4),
        "ratio_mean_ci95": [round(float(np.percentile(mn_b, 2.5)), 4), round(float(np.percentile(mn_b, 97.5)), 4)],
        "implied_saving_frac_median": round(1.0 - math.exp(med), 4),
        "implied_saving_frac_ci95": [round(1.0 - float(np.percentile(med_b, 97.5)), 4), round(1.0 - float(np.percentile(med_b, 2.5)), 4)],
    }


def summarize_denominators():
    df = pd.read_csv(HARMONIZED / "armB_wue_et_64families_audit_current.csv", float_precision="round_trip")
    rng = np.random.default_rng(SEED)
    cat_a = df[df["category"] == "A_direct_reported_ET"].copy()
    cat_b = df[df["category"] == "B_inferred_Y_over_WUE_ET"].copy()
    cat_ab = df[df["category"].isin(["A_direct_reported_ET", "B_inferred_Y_over_WUE_ET"])].copy()
    cat_c = df[df["category"] == "C_IWUE_irrigation_supply"].copy()
    cat_d = df[df["category"] == "D_unclear_or_non_WUE"].copy()

    # Family-level primary category (A > B > C > D)
    prio = {"A_direct_reported_ET": 1, "B_inferred_Y_over_WUE_ET": 2, "C_IWUE_irrigation_supply": 3, "D_unclear_or_non_WUE": 4}
    df["prio"] = df["category"].map(prio)
    fam_primary = df.sort_values("prio").groupby("family_id").first()

    ab_china = cat_ab[cat_ab["country"] == "China"].copy()
    ab_non_china = cat_ab[cat_ab["country"].notna() & (cat_ab["country"] != "China")].copy()
    a_non_china = cat_a[cat_a["country"].notna() & (cat_a["country"] != "China")].copy()
    deficit_confounded_fams = {"DOI_10.3390/su15129602", "10.3390/agronomy14010003"}
    ab_non_deficit = cat_ab[~cat_ab["family_id"].isin(deficit_confounded_fams)].copy()

    summary = {
        "audit_scope": "Complete 100% Arm B WUE denominator audit (all 83 cleaned WUE clusters + 10 previously screened non-WUE exclusions = 93 study records across 91 independent experimental families, clustering 3 Pingluo Ningxia sister DOIs into 1 family).",
        "total_audited_records": int(len(df)),
        "total_studies_audited": int(df["idx"].nunique()),
        "total_families_audited": int(df["family_id"].nunique()),
        "families_by_primary_category": {k: int(v) for k, v in fam_primary["category"].value_counts().to_dict().items()},
        "records_by_category": {k: int(v) for k, v in df["category"].value_counts().to_dict().items()},
        "A_direct_reported_ET": summarize_ratio(cat_a, "ln_et_ratio", rng),
        "B_inferred_Y_over_WUE_ET": summarize_ratio(cat_b, "ln_et_ratio", rng),
        "verified_consumptive_ET_combined_AB": summarize_ratio(cat_ab, "ln_et_ratio", rng),
        "C_IWUE_irrigation_supply_ratio": summarize_ratio(cat_c, "ln_irr_ratio", rng),
        "geographic_and_design_sensitivity": {
            "verified_ET_china": summarize_ratio(ab_china, "ln_et_ratio", rng),
            "verified_ET_non_china": summarize_ratio(ab_non_china, "ln_et_ratio", rng),
            "direct_ET_non_china": summarize_ratio(a_non_china, "ln_et_ratio", rng),
            "verified_ET_excluding_deficit_irrigation_confounded": summarize_ratio(ab_non_deficit, "ln_et_ratio", rng),
        },
        "D_unclear_or_non_WUE_families": int(cat_d["family_id"].nunique()),
        "D_reasons": cat_d.drop_duplicates("family_id")[["idx", "family_id", "country", "exclusion_reason"]].fillna("").to_dict(orient="records"),
    }
    (OUT_TABLES / "armB_wue_et_64families_audit_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def compute_comeasured_et_ratio(df: pd.DataFrame, rng: np.random.Generator) -> Dict[str, Any]:
    """Load audited 64-family WUE/ET denominator stratification (A+B verified consumptive ET vs C IWUE vs D excluded)."""
    # Legacy unverified reference (kept solely for audit transparency)
    fam = (
        df[df["endpoint_std"].isin(["grain_yield", "wue_or_iwue"])]
        .groupby(["family_id", "endpoint_std"])["lnRR"].mean().unstack().dropna()
    )
    d_et = (fam["grain_yield"] - fam["wue_or_iwue"]).values
    legacy_med = round(math.exp(float(np.median(d_et))), 4) if len(d_et) else None

    aud = summarize_denominators()
    comb = aud["verified_consumptive_ET_combined_AB"]
    cat_a = aud["A_direct_reported_ET"]
    cat_b = aud["B_inferred_Y_over_WUE_ET"]
    cat_c = aud["C_IWUE_irrigation_supply_ratio"]
    return {
        "audit_source": "data/harmonized/armB_wue_et_64families_audit_current.csv",
        "n_families": comb["n_families"],
        "n_pairs_verified": comb["n_pairs"],
        "et_ratio_median": comb["ratio_median"],
        "et_ratio_median_ci95": comb["ratio_median_ci95"],
        "et_ratio_mean": comb["ratio_mean"],
        "et_ratio_mean_ci95": comb["ratio_mean_ci95"],
        "implied_et_pct_change_median": round((comb["ratio_median"] - 1.0) * 100.0, 2),
        "implied_et_saving_frac_median": comb["implied_saving_frac_median"],
        "implied_et_saving_frac_ci95": comb["implied_saving_frac_ci95"],
        "share_families_et_increased": comb["share_families_increased"],
        "stratification": {
            "A_direct_reported_ET": cat_a,
            "B_inferred_Y_over_WUE_ET": cat_b,
            "verified_consumptive_ET_combined_AB": comb,
            "C_IWUE_irrigation_supply_ratio": cat_c,
            "D_unclear_or_non_WUE_families": aud["D_unclear_or_non_WUE_families"],
            "families_by_primary_category": aud["families_by_primary_category"],
            "records_by_category": aud["records_by_category"],
        },
        "legacy_unverified_cleaned_reference": {"n_families": int(len(d_et)), "et_ratio_median": legacy_med},
    }


def compute_leaching_efficiency_stratified(df: pd.DataFrame) -> Dict[str, Any]:
    """Stratify the 7 direct leaching-efficiency clusters by physical definition."""
    if not AUDIT_7_LEACH_CSV.is_file():
        return {"status": "MISSING_AUDIT_CSV"}
    aud = pd.read_csv(AUDIT_7_LEACH_CSV)
    le = df[df["endpoint_std"] == "leaching_efficiency_or_fraction"].copy()
    doi_cat = aud.groupby("doi")["physical_category"].first().to_dict()
    doi_dir = aud.groupby("doi")["supports_hydraulic_salt_export_directly"].first().to_dict()
    le["physical_category"] = le["doi_norm"].map(lambda d: doi_cat.get(str(d).lower(), "unknown"))
    le["supports_salt_export"] = le["doi_norm"].map(lambda d: bool(doi_dir.get(str(d).lower(), False)))

    by_cat = {}
    for cat, grp in le.groupby("physical_category"):
        res = fit_reml_multilevel_1d(grp["lnRR"].values, grp["vi_imputed"].values, grp["family_id"].values, grp["control_arm_id"].values)
        by_cat[str(cat)] = {**res, "dois": sorted(grp["doi_norm"].dropna().unique().tolist())}

    direct_sub = le[le["supports_salt_export"]]
    direct_reml = fit_reml_multilevel_1d(direct_sub["lnRR"].values, direct_sub["vi_imputed"].values, direct_sub["family_id"].values, direct_sub["control_arm_id"].values)
    return {
        "n_clusters_total": int(le["family_id"].nunique()),
        "n_contrasts_total": int(len(le)),
        "direct_salt_export_or_storage_only": direct_reml,
        "by_physical_category": by_cat,
    }


def compute_dual_pathway_hyd_vs_ec(df: pd.DataFrame) -> Dict[str, Any]:
    """Dual-pathway check: family-level hydraulic gain (Ks prioritized, fallback infiltration_rate_all) vs soil_EC lnRR."""
    fam = (
        df[df["endpoint_std"].isin(["ks_saturated", "infiltration_rate_all", "soil_EC"])]
        .groupby(["family_id", "soil_sodicity_regime", "endpoint_std"])["lnRR"].mean().unstack().reset_index()
    )
    fam["hyd_lnrr"] = fam["ks_saturated"].fillna(fam["infiltration_rate_all"])
    pair = fam.dropna(subset=["hyd_lnrr", "soil_EC"]).copy()

    def summarize(sub_p: pd.DataFrame) -> Dict[str, Any]:
        if len(sub_p) < 5:
            return {"n_families": int(len(sub_p)), "status": "INSUFFICIENT_FAMILIES"}
        r_sp, p_sp = stats.spearmanr(sub_p["hyd_lnrr"], sub_p["soil_EC"])
        sl, ic, _, p_lin, se_sl = stats.linregress(sub_p["hyd_lnrr"], sub_p["soil_EC"])
        return {
            "n_families": int(len(sub_p)),
            "spearman_r": round(float(r_sp), 4),
            "spearman_p": round(float(p_sp), 5),
            "ols_slope": round(float(sl), 4),
            "ols_slope_se": round(float(se_sl), 4),
            "ols_intercept": round(float(ic), 4),
            "ols_p": round(float(p_lin), 5),
            "mean_soil_EC_lnRR": round(float(sub_p["soil_EC"].mean()), 4),
        }

    return {
        "sign_convention_note": "Negative soil_EC lnRR indicates salinity reduction (mean Arm B soil_EC lnRR = -0.251); a negative slope/correlation implies larger hydraulic gain associates with stronger EC reduction, whereas P > 0.45 indicates no monotonic statistical coupling.",
        "overall": summarize(pair),
        "by_sodicity": {str(reg): summarize(grp) for reg, grp in pair.groupby("soil_sodicity_regime")},
        "family_points": [
            {
                "family_id": str(r["family_id"]),
                "sodicity_regime": str(r["soil_sodicity_regime"]),
                "hyd_lnrr": round(float(r["hyd_lnrr"]), 4),
                "soil_EC_lnrr": round(float(r["soil_EC"]), 4),
                "hyd_source": "ks_saturated" if pd.notna(r.get("ks_saturated")) else "infiltration_rate_all",
            }
            for _, r in pair.iterrows()
        ],
    }


def compute_esp_meta_regression(df: pd.DataFrame) -> Dict[str, Any]:
    """Fit cluster-WLS CR1 meta-regression: Ks lnRR ~ initial_esp_pct + amendment_group + predicted CI band."""
    ks = df[(df["endpoint_std"] == "ks_saturated") & df["initial_esp_pct"].notna()].copy()

    def fit_sub(sub_k: pd.DataFrame, esp_max_grid: float = 92.0) -> Dict[str, Any]:
        sub_k = sub_k.copy()
        fc = sub_k.groupby("family_id")["lnRR"].transform("count").values.clip(min=1)
        wc = 1.0 / (sub_k["vi_imputed"].values * fc)
        ag_dum = pd.get_dummies(sub_k["amendment_group"], drop_first=True, dtype=float)
        X = np.column_stack([np.ones(len(sub_k)), sub_k["initial_esp_pct"].values] + [ag_dum[c].values for c in ag_dum.columns])
        names = ["intercept", "initial_esp_pct"] + list(ag_dum.columns)
        y = sub_k["lnRR"].values
        cl = sub_k["family_id"].values
        uniq = np.unique(cl)
        n_cl = len(uniq)
        XtWX = X.T @ (wc[:, None] * X) + 1e-8 * np.eye(X.shape[1])
        beta = np.linalg.solve(XtWX, X.T @ (wc * y))
        resid = y - X @ beta
        XtWX_inv = np.linalg.pinv(XtWX)
        meat = sum(np.outer(X[cl == c].T @ (wc[cl == c] * resid[cl == c]), X[cl == c].T @ (wc[cl == c] * resid[cl == c])) for c in uniq)
        cov_cr = (n_cl / max(n_cl - 1, 1)) * (XtWX_inv @ meat @ XtWX_inv)
        se_cr = np.sqrt(np.maximum(np.diag(cov_cr), 1e-12))
        df_t = max(n_cl - 1, 1)
        t_crit = float(stats.t.ppf(0.975, df=df_t))
        terms = {
            nm: {"coef": round(float(beta[i]), 5), "se_cr": round(float(se_cr[i]), 5), "p_val": round(float(2.0 * stats.t.sf(abs(beta[i] / se_cr[i]), df=df_t)), 5)}
            for i, nm in enumerate(names)
        }
        # Marginal prediction grid across ESP = 0..92% at sample-weighted amendment_group means
        ag_means = ag_dum.mean().values
        esp_grid = np.linspace(0.0, esp_max_grid, 47)
        Xg = np.column_stack([np.ones_like(esp_grid), esp_grid] + [np.full_like(esp_grid, m) for m in ag_means])
        yhat = Xg @ beta
        se_g = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", Xg, cov_cr, Xg), 1e-12))
        return {
            "k": int(len(sub_k)), "n_clusters": int(n_cl), "terms": terms,
            "pred_grid": [
                {
                    "esp_pct": round(float(esp_grid[i]), 1),
                    "pred_lnrr": round(float(yhat[i]), 4),
                    "ci95_low": round(float(yhat[i] - t_crit * se_g[i]), 4),
                    "ci95_high": round(float(yhat[i] + t_crit * se_g[i]), 4),
                }
                for i in range(len(esp_grid))
            ],
        }

    return {
        "primary_result_key": "physical_domain_esp_le_100",
        "esp_gt100_rows_flagged": int((ks["initial_esp_pct"] > 100.0).sum()),
        "anomaly_note": "Rows with initial_esp_pct > 100% are physically impossible ESP values (one family at 185.5%); physical_domain_esp_le_100 is the primary result and all_rows_incl_esp_gt100_anomaly is a labelled anomaly sensitivity.",
        "physical_domain_esp_le_100": fit_sub(ks[ks["initial_esp_pct"] <= 100.0]),
        "all_rows_incl_esp_gt100_anomaly": fit_sub(ks),
    }


def main() -> None:
    rng = np.random.default_rng(SEED)
    df = pd.read_csv(MERGED_INPUT, low_memory=False, float_precision="round_trip")

    df["endpoint_std"] = df["metric_bucket"].replace({
        "infiltration_rate_steady": "infiltration_rate_all",
        "infiltration_rate_initial": "infiltration_rate_all",
        "infiltration_rate_unspecified": "infiltration_rate_all",
        "infiltration_rate_master": "infiltration_rate_all",
    })
    df["clay_stratum"] = pd.to_numeric(df["clay_pct"], errors="coerce").apply(
        lambda v: "unknown" if pd.isna(v) else ("coarse_lt20" if v < 20.0 else ("medium_20_35" if v < 35.0 else "heavy_clay_ge35"))
    )

    hydraulic_endpoints = ["ks_saturated", "infiltration_rate_all", "bulk_density", "wue_or_iwue", "leaching_efficiency_or_fraction"]
    context_endpoints = ["soil_EC", "grain_yield"]
    results_rows: List[Dict[str, Any]] = []

    for ep in hydraulic_endpoints + context_endpoints:
        sub = df[df["endpoint_std"] == ep]
        if len(sub) < 3:
            continue
        res_all = fit_reml_multilevel_1d(sub["lnRR"].values, sub["vi_imputed"].values, sub["family_id"].values, sub["control_arm_id"].values)
        sub_ex = sub[sub["has_exact_variance"]]
        res_ex = fit_reml_multilevel_1d(sub_ex["lnRR"].values, sub_ex["vi_exact"].values, sub_ex["family_id"].values, sub_ex["control_arm_id"].values)
        eg = egger_cluster_test(sub["lnRR"].values, sub["vi_imputed"].values, sub["family_id"].values)
        results_rows.append({"endpoint": ep, "stratum": "ALL", "model": "REML_harmonized", **res_all, **eg})
        results_rows.append({"endpoint": ep, "stratum": "ALL", "model": "REML_exact_variance_only", **res_ex})

        for reg, rgrp in sub.groupby("soil_sodicity_regime"):
            res_reg = fit_reml_multilevel_1d(rgrp["lnRR"].values, rgrp["vi_imputed"].values, rgrp["family_id"].values, rgrp["control_arm_id"].values)
            results_rows.append({"endpoint": ep, "stratum": f"sodicity:{reg}", "model": "REML_harmonized", **res_reg})

        for wr, wgrp in sub.groupby("water_regime"):
            if wgrp["family_id"].nunique() >= 4:
                res_wr = fit_reml_multilevel_1d(wgrp["lnRR"].values, wgrp["vi_imputed"].values, wgrp["family_id"].values, wgrp["control_arm_id"].values)
                results_rows.append({"endpoint": ep, "stratum": f"water_regime:{wr}", "model": "REML_harmonized", **res_wr})

        for ag, agrp in sub.groupby("amendment_group"):
            if agrp["family_id"].nunique() >= 4:
                res_ag = fit_reml_multilevel_1d(agrp["lnRR"].values, agrp["vi_imputed"].values, agrp["family_id"].values, agrp["control_arm_id"].values)
                results_rows.append({"endpoint": ep, "stratum": f"amendment:{ag}", "model": "REML_harmonized", **res_ag})

        for cs, cgrp in sub[sub["clay_stratum"] != "unknown"].groupby("clay_stratum"):
            if cgrp["family_id"].nunique() >= 3:
                res_cs = fit_reml_multilevel_1d(cgrp["lnRR"].values, cgrp["vi_imputed"].values, cgrp["family_id"].values, cgrp["control_arm_id"].values)
                results_rows.append({"endpoint": ep, "stratum": f"clay:{cs}", "model": "REML_harmonized", **res_cs})

        if ep in {"ks_saturated", "infiltration_rate_all", "bulk_density", "wue_or_iwue"}:
            for (reg, ag), grp in sub.groupby(["soil_sodicity_regime", "amendment_group"]):
                if grp["family_id"].nunique() >= 3:
                    res_2w = fit_reml_multilevel_1d(grp["lnRR"].values, grp["vi_imputed"].values, grp["family_id"].values, grp["control_arm_id"].values)
                    results_rows.append({"endpoint": ep, "stratum": f"sodicity_x_amendment:{reg}::{ag}", "model": "REML_harmonized", **res_2w})
            for (reg, cs), grp in sub[sub["clay_stratum"] != "unknown"].groupby(["soil_sodicity_regime", "clay_stratum"]):
                if grp["family_id"].nunique() >= 3:
                    res_2c = fit_reml_multilevel_1d(grp["lnRR"].values, grp["vi_imputed"].values, grp["family_id"].values, grp["control_arm_id"].values)
                    results_rows.append({"endpoint": ep, "stratum": f"sodicity_x_clay:{reg}::{cs}", "model": "REML_harmonized", **res_2c})

    # Dedicated WUE by sodicity for Pillar 3
    wue_sodic = next((r for r in results_rows if r["endpoint"] == "wue_or_iwue" and r["stratum"] == "sodicity:sodic_or_saline_sodic_high_esp"), {})
    wue_saline = next((r for r in results_rows if r["endpoint"] == "wue_or_iwue" and r["stratum"] == "sodicity:saline_or_unspecified"), {})

    # LOCO robustness
    loco_summary: Dict[str, Any] = {}
    for ep in ["ks_saturated", "infiltration_rate_all", "bulk_density", "wue_or_iwue"]:
        sub = df[(df["endpoint_std"] == ep) & (~df["country"].isin(["Unknown", "Unresolved_Econ", "nan", ""]))]
        countries = [c for c, g in sub.groupby("country") if g["family_id"].nunique() >= 3]
        if len(countries) >= 3:
            pcts = []
            for drop_c in countries:
                keep = sub[sub["country"] != drop_c]
                r_loco = fit_reml_multilevel_1d(keep["lnRR"].values, keep["vi_imputed"].values, keep["family_id"].values, keep["control_arm_id"].values)
                if "pct_change" in r_loco:
                    pcts.append({"dropped_country": drop_c, "pct_change": r_loco["pct_change"], "n_clusters": r_loco["n_clusters"]})
            if pcts:
                vals = [x["pct_change"] for x in pcts]
                loco_summary[ep] = {
                    "n_resolved_countries_tested": len(pcts), "min_pct_change": min(vals),
                    "max_pct_change": max(vals), "all_same_sign": all(v > 0 for v in vals) or all(v < 0 for v in vals), "details": pcts,
                }

    OUT_SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results_rows).to_csv(OUT_SUMMARY_CSV, index=False)

    # Co-measured correlations
    fam_ep = (
        df[df["endpoint_std"].isin(["ks_saturated", "infiltration_rate_all", "bulk_density", "wue_or_iwue", "soil_EC", "grain_yield"])]
        .groupby(["family_id", "endpoint_std"])["lnRR"].mean().unstack()
    )
    co_corrs = {}
    for a, b in [("bulk_density", "soil_EC"), ("infiltration_rate_all", "soil_EC"), ("infiltration_rate_all", "grain_yield"), ("wue_or_iwue", "grain_yield"), ("bulk_density", "grain_yield"), ("ks_saturated", "grain_yield")]:
        if a in fam_ep.columns and b in fam_ep.columns:
            pair = fam_ep[[a, b]].dropna()
            if len(pair) >= 5:
                r_sp, p_sp = stats.spearmanr(pair[a], pair[b])
                co_corrs[f"{a}_vs_{b}"] = {"n_families": int(len(pair)), "spearman_r": round(float(r_sp), 4), "p_val": round(float(p_sp), 5)}

    comeasured_et = compute_comeasured_et_ratio(df, rng)
    leaching_strat = compute_leaching_efficiency_stratified(df)
    dual_pathway_ec = compute_dual_pathway_hyd_vs_ec(df)
    esp_meta_reg = compute_esp_meta_regression(df)

    hyd_set = set(hydraulic_endpoints)
    ctx_set = set(context_endpoints)
    summary_json = {
        "script": "pillar2_armB_hydraulic_model.py",
        "seed": SEED,
        "harmonized_input_sha256": hashlib.sha256(MERGED_INPUT.read_bytes()).hexdigest(),
        "total_merged_rows": int(len(df)),
        "source_corpus_counts": {k: int(v) for k, v in df["source_corpus"].value_counts().to_dict().items()},
        "exact_variance_rows": int(df["has_exact_variance"].sum()),
        "headline_all_endpoints": [r for r in results_rows if r["stratum"] == "ALL" and r["model"] == "REML_harmonized" and r["endpoint"] in hyd_set],
        "headline_exact_variance_sensitivity": [r for r in results_rows if r["stratum"] == "ALL" and r["model"] == "REML_exact_variance_only" and r["endpoint"] in hyd_set],
        "context_endpoints_sibling_overlap": {
            "note": "Contextual yield and soil_EC endpoints from the core reclamation synthesis corpus are reported for denominator auditing and paired diagnostics only (omitted from main hydraulic models).",
            "harmonized": [r for r in results_rows if r["stratum"] == "ALL" and r["model"] == "REML_harmonized" and r["endpoint"] in ctx_set],
            "exact_variance": [r for r in results_rows if r["stratum"] == "ALL" and r["model"] == "REML_exact_variance_only" and r["endpoint"] in ctx_set],
        },
        "wue_by_sodicity": {"sodic": wue_sodic, "saline": wue_saline},
        "comeasured_et_ratio": comeasured_et,
        "leaching_efficiency_stratified": leaching_strat,
        "dual_pathway_hydraulic_vs_ec": dual_pathway_ec,
        "esp_meta_regression": esp_meta_reg,
        "exact_variance_egger_checks": {
            "ks_saturated": egger_cluster_test_exact(df[df["endpoint_std"] == "ks_saturated"]),
            "infiltration_rate_all": egger_cluster_test_exact(df[df["endpoint_std"] == "infiltration_rate_all"]),
        },
        "h2_dual_pathway_sodicity_split": [r for r in results_rows if r["stratum"].startswith("sodicity:") and r["endpoint"] in {"ks_saturated", "infiltration_rate_all", "bulk_density", "wue_or_iwue"}],
        "h2_clay_moderation": [r for r in results_rows if r["stratum"].startswith("clay:") and r["endpoint"] in {"ks_saturated", "infiltration_rate_all", "bulk_density", "wue_or_iwue"}],
        "h2_water_regime_stratification": [r for r in results_rows if r["stratum"].startswith("water_regime:") and r["endpoint"] in {"ks_saturated", "infiltration_rate_all", "bulk_density", "wue_or_iwue"}],
        "h2_two_way_sodicity_x_amendment": [r for r in results_rows if r["stratum"].startswith("sodicity_x_amendment:")],
        "h2_two_way_sodicity_x_clay": [r for r in results_rows if r["stratum"].startswith("sodicity_x_clay:")],
        "leave_one_country_out_robustness": loco_summary,
        "comeasured_family_correlations": co_corrs,
    }
    with OUT_RESULTS_JSON.open("w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2, ensure_ascii=False)
    print(f"Saved Arm B results -> {OUT_RESULTS_JSON}")


if __name__ == "__main__":
    prepare_outputs()
    main()
