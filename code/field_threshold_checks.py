"""Field harvest thresholds, paired native-scale soil monitoring and family-level SAR tests."""

from config import HARMONIZED, OUT_PANELS, table_input, prepare_outputs
from itertools import permutations
import json

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from pillar2_armA_ecw_threshold_model import ARMA_CSV, OUT_RESULTS_JSON, SEED, fit_maas_hoffman_abs

FIG = OUT_PANELS
ED = OUT_PANELS
HARVEST = {"grain_yield", "seed_cotton_yield", "fruit_yield", "tuber_yield", "marketable_yield",
           "crop_yield", "lint_yield", "fresh_yield", "cane_yield"}


def gated_contrasts():
    d = pd.read_csv(ARMA_CSV, low_memory=False)
    ready = d.qc_gate_status.isin(["PASS_READY", "PASS_NO_VARIANCE"])
    d = d[ready & (d.mean_treatment > 0) & (d.mean_control > 0) &
          (d.ecw_treatment_dsm > d.ecw_control_dsm)].copy()
    d["lnRR"] = np.log(d.mean_treatment / d.mean_control)
    return d


def monitoring_checks(thresholds):
    path = HARMONIZED / "armA_multiyear_soil_ec_pre_post_pairs_current.csv"
    d = pd.read_csv(path)
    d = d[d.ecw_dsm.notna() & (d.soil_ec_pre_val > 0) & (d.soil_ec_post_val > 0)].copy()
    d["post_pre_ratio"] = d.soil_ec_post_val / d.soil_ec_pre_val
    d.to_csv(FIG / "Fig2B_monitoring_pairs.csv", index=False)
    keys = ["family_id", "year_index", "soil_depth_cm", "method", "unit"]
    parts, rows = [], []
    for name, cut in thresholds.items():
        d["band"] = np.where(d.ecw_dsm < cut, "lower", "upper")
        matched = d.groupby(keys + ["band"]).post_pre_ratio.median().unstack().dropna().reset_index()
        matched["threshold"] = name
        matched["ecw_cut_dsm"] = cut
        matched["log_high_low"] = np.log(matched.upper / matched.lower)
        parts.append(matched)
        family = matched.groupby("family_id").log_high_low.median()
        positive = family[family != 0]
        from scipy.stats import binomtest
        rows.append({"threshold": name, "ecw_cut_dsm": cut, "matched_strata": len(matched),
                     "paired_families": len(family), "families_higher": int((family > 0).sum()),
                     "family_median_high_low": float(np.exp(family.median())),
                     "paired_sign_p": float(binomtest(int((positive > 0).sum()), len(positive)).pvalue),
                     "bands": {str(k): {"rows": len(g), "families": g.family_id.nunique(),
                                        "median_post_pre": float(g.post_pre_ratio.median()),
                                        "share_rising": float((g.post_pre_ratio > 1).mean())}
                               for k, g in d.groupby("band")}})
    pd.concat(parts).to_csv(FIG / "Fig2B_matched_monitoring.csv", index=False)
    return rows


def family_sar(d):
    d = d[d.sarw_treatment.between(0, 200)].copy()
    d["endpoint"] = d.metric_name.replace({"soil_ECe": "soil_EC", "soil_EC1_5": "soil_EC"})
    d = d[d.endpoint.isin(["soil_SAR", "soil_EC"])]
    d["dose_bin"] = pd.cut(d.sarw_treatment, [-1, 5, 12, 20, 200], labels=False)
    bins = d.groupby(["endpoint", "family_id", "dose_bin"], as_index=False).agg(
        sar=("sarw_treatment", "mean"), response=("lnRR", "mean"), observations=("lnRR", "size"))
    family = bins.groupby(["endpoint", "family_id"], as_index=False).agg(
        sar=("sar", "median"), response=("response", "median"), dose_bins=("sar", "size"),
        observations=("observations", "sum"))
    family.to_csv(ED / "ED_Fig5D_family_estimates.csv", index=False)
    rng = np.random.default_rng(SEED)
    result = {}
    for name, grp in family.groupby("endpoint"):
        x, y = rankdata(grp.sar), rankdata(grp.response)
        x, y = x - x.mean(), y - y.mean()
        denom = np.linalg.norm(x) * np.linalg.norm(y)
        rho = float(x @ y / denom)
        if len(x) <= 8:
            null = np.asarray(list(permutations(y))) @ x / denom
            p = (np.abs(null) >= abs(rho) - 1e-12).mean()
        else:
            null = np.stack([rng.permutation(y) for _ in range(19999)]) @ x / denom
            p = (1 + (np.abs(null) >= abs(rho) - 1e-12).sum()) / (len(null) + 1)
        result[name] = {"n_families": len(grp), "family_bins": int(grp.dose_bins.sum()),
                        "spearman_r": rho, "p": float(p), "permutations": len(null)}
    return result


def extend_arm_a(summary):
    d = gated_contrasts()
    field = d[d.metric_name.isin(HARVEST) & d.experiment_type.str.lower().eq("field") &
              d.irrigation_scheduling.eq("continuous_brackish")].copy()
    field["w_obs"] = field.n_replicates.fillna(3).clip(lower=1) / field.groupby("family_id").family_id.transform("size")
    model = fit_maas_hoffman_abs(field, np.random.default_rng(SEED + 32), curve_grid=np.linspace(0, 16, 161))
    summary["dose_response_absolute"]["field_continuous_harvest"] = model
    summary["monitoring_checks"] = monitoring_checks({"composite": 3.55, "field_harvest": model["ecw_star_90pct_dsm"]})
    summary["family_level_sar"] = family_sar(d)


if __name__ == "__main__":
    prepare_outputs()
    result = json.loads(table_input(OUT_RESULTS_JSON.name).read_text())
    extend_arm_a(result)
    OUT_RESULTS_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print({k:v for k,v in result["dose_response_absolute"]["field_continuous_harvest"].items() if not k.startswith("curve")})
    print(result["monitoring_checks"], result["family_level_sar"])

