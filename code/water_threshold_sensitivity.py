"""Propagate empirically distinct crop thresholds and marginal leaching recovery through basin closure."""

from config import OUT_TABLES, table_input, prepare_outputs
import json
import pandas as pd

import pillar3_safe_windows_and_zoning as zoning

BASE = OUT_TABLES / "pillar3_threshold_and_recovery_current.csv"
FUTURE = OUT_TABLES / "pillar3_future_threshold_sensitivity_current.csv"
RISK = OUT_TABLES / "pillar3_future_salinity_groundwater_cells_current.parquet"


def thresholds():
    arm = json.loads(zoning.P2B_JSON.read_text())
    field = arm["dose_response_absolute"]["field_continuous_harvest"]
    mixed = arm["dose_response_absolute"]["continuous"]
    return {"Composite reference": mixed["ecw_star_90pct_dsm"],
            "Field harvest": field["ecw_star_90pct_dsm"],
            "Field lower CI": field["ecw_star_90pct_ci95"][0], "Field upper CI": field["ecw_star_90pct_ci95"][1],
            "Composite lower CI": mixed["ecw_star_90pct_ci95"][0], "Composite upper CI": mixed["ecw_star_90pct_ci95"][1],
            "Continuous sigmoid": arm["logistic_sensitivity"]["continuous_sigmoidal_p_ge_1"]["ecw_star_90pct_dsm"]}


def outcome(snap):
    return {"withdrawal_relief_km3_yr": snap["net_brackish_fresh_saving_deltaF_km3_yr"],
            "net_capped_km3_yr": snap["basin_net_depletion_relief_km3_yr"],
            "net_et_km3_yr": snap["basin_net_depletion_relief_actual_km3_yr"],
            "rescued_capped": snap["paradox_basins_relieved_to_blue_feasible"],
            "rescued_et": snap["paradox_basins_relieved_actual"],
            "zone3_capped_mha": snap["risk_intersection_breakdown_mha"]["zone3_at_least_one_risk_total_mha"],
            "zone3_et_mha": snap["risk_intersection_actual_et_mha"]["zone3_at_least_one_risk_total_mha"],
            "sal_arid_and_gw_mha": snap["risk_intersection_breakdown_mha"]["sal_arid_and_gw_depleted_mha"]}


def baseline(grid, basins, params):
    limits = thresholds()
    ref = limits["Composite reference"]
    ratios = json.loads(table_input("armB_wue_et_64families_audit_summary.json").read_text())
    direct = ratios["A_direct_reported_ET"]
    et = params["et_ratio_observed"]
    cases = [(name, limit, et, 0.0) for name, limit in limits.items()]
    for name, ratio in [("Direct ET", direct["ratio_median"]), ("Verified ET low", params["et_ratio_observed_ci95"][0]),
                        ("Verified ET high", params["et_ratio_observed_ci95"][1]),
                        ("Direct ET low", direct["ratio_median_ci95"][0]), ("Direct ET high", direct["ratio_median_ci95"][1])]:
        cases.append((name, ref, ratio, 0.0))
    for recovery in (0.25, 0.5, 0.75, 1.0):
        cases.append((f"Marginal recovery {recovery:.0%}", ref, et, recovery))
    rows = []
    for name, value, ratio, recovery in cases:
        _, _, snap = zoning.evaluate_zoning_snapshot(grid, basins, value, params["ece90_ref_sites_used"],
                        params["mu_wue_sodic"], params["mu_wue_saline"], ratio, marginal_leach_recovery=recovery)
        rows.append({"case": name, "ecw_dsm": value, "et_ratio": ratio, "marginal_leach_recovery": recovery, **outcome(snap)})
    pd.DataFrame(rows).to_csv(BASE, index=False)


if __name__ == "__main__":
    prepare_outputs()
    grid = pd.read_parquet(zoning.P1_GRID)
    soils = pd.read_parquet(zoning.PUB_VAL_GRID)
    extra = [c for c in soils if c not in grid]
    grid = grid.merge(soils[["r", "c", *extra]], on=["r", "c"], validate="one_to_one")
    params = json.loads(table_input("pillar3_global_summary_current.json").read_text())["empirical_parameters_used"]
    baseline(grid, pd.read_csv(zoning.P1_BASIN, low_memory=False), params)
    print(f"Wrote {BASE.name}")
