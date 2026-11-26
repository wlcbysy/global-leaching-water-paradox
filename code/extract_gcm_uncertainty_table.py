#!/usr/bin/env python3
"""
Extract individual 5-GCM WaterGAP 2.2e ISIMIP3b x Hassani CMIP6 zoning outcomes
(gfdl-esm4, ipsl-cm6a-lr, mpi-esm1-2-hr, mri-esm2-0, ukesm1-0-ll across ssp126/370/585 x 2050/2080)
for multi-GCM uncertainty quantification using the revised three-tier Pillar 3 model.
"""

from __future__ import annotations

from config import climate_models, OUT_TABLES, table_input, prepare_outputs

import json
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import xarray as xr


import pillar3_safe_windows_and_zoning as p3mod
import water_threshold_sensitivity as sensitivity

OUT_GCM_CSV = OUT_TABLES / "pillar3_gcm_uncertainty_current.csv"


def main() -> None:
    grid_base = pd.read_parquet(p3mod.P1_GRID)
    if p3mod.PUB_VAL_GRID.is_file():
        pub_v = pd.read_parquet(p3mod.PUB_VAL_GRID).drop(columns=["lat", "lon", "aware_basin_idx"], errors="ignore")
        grid_base = grid_base.merge(pub_v, on=["r", "c"], how="left")
    basins_base = pd.read_csv(p3mod.P1_BASIN, low_memory=False)
    p3_sum = json.loads(table_input(p3mod.OUT_SUMMARY.name).read_text(encoding="utf-8"))
    ep = p3_sum["empirical_parameters_used"]
    ecw_star_use = float(ep["ecw_star_baseline_used_dsm"])
    ece90_ref = float(ep["ece90_ref_sites_used"])
    mu_sod = float(ep["mu_wue_sodic"])
    mu_sal = float(ep["mu_wue_saline"])
    et_obs = float(ep["et_ratio_observed"])

    with xr.open_dataset(p3mod.LSM_NC) as dl:
        lsm = np.squeeze(dl["mask"].values) == 1.0

    gcms = climate_models(p3mod.WG_3B_DIR)
    r_idx = grid_base["r"].values
    c_idx = grid_base["c"].values

    sensitivity.baseline(grid_base, basins_base, ep)
    limits = sensitivity.thresholds()
    future_thresholds, risk_cells = [], []
    rows: List[Dict[str, Any]] = []
    for ssp in ["ssp126", "ssp370", "ssp585"]:
        ece_map = {"2050": "hassani_ssp245_2050", "2080": "hassani_ssp245_2080"} if ssp == "ssp126" else {"2050": "hassani_ssp585_2050", "2080": "hassani_ssp585_2080"}
        for h_name, (y0, y1) in [("2050", (2040, 2069)), ("2080", (2070, 2099))]:
            for gcm in gcms:
                gdir = p3mod.WG_3B_DIR / gcm
                f_q = gdir / f"watergap2-2e_{gcm}_w5e5_{ssp}_2015soc-from-histsoc_default_qtot_global_monthly_2015_2100.nc"
                f_p = gdir / f"watergap2-2e_{gcm}_w5e5_{ssp}_2015soc-from-histsoc_default_pirrww_global_monthly_2015_2100.nc"
                f_a = gdir / f"watergap2-2e_{gcm}_w5e5_{ssp}_2015soc-from-histsoc_default_atotuse_global_monthly_2015_2100.nc"
                f_gw = gdir / f"watergap2-2e_{gcm}_w5e5_{ssp}_2015soc-from-histsoc_default_atotusegw_global_monthly_2015_2100.nc"
                f_qr = gdir / f"watergap2-2e_{gcm}_w5e5_{ssp}_2015soc-from-histsoc_default_qr_global_monthly_2015_2100.nc"
                f_gws = gdir / f"watergap2-2e_{gcm}_w5e5_{ssp}_2015soc-from-histsoc_default_groundwstor_global_monthly_2015_2100.nc"
                if not all(f.is_file() for f in [f_q, f_p, f_a, f_gw, f_qr, f_gws]):
                    raise FileNotFoundError(f"Missing WaterGAP ISIMIP3b file for {gcm} {ssp}")

                q_m = p3mod.compute_annual_flux_mm_yr(f_q, "qtot", y0, y1, lsm)[r_idx, c_idx]
                p_m = p3mod.compute_annual_flux_mm_yr(f_p, "pirrww", y0, y1, lsm)[r_idx, c_idx]
                a_m = p3mod.compute_annual_flux_mm_yr(f_a, "atotuse", y0, y1, lsm)[r_idx, c_idx]
                gw_m = p3mod.compute_annual_flux_mm_yr(f_gw, "atotusegw", y0, y1, lsm)[r_idx, c_idx]
                qr_m = p3mod.compute_annual_flux_mm_yr(f_qr, "qr", y0, y1, lsm)[r_idx, c_idx]
                slp_m = p3mod.compute_gws_trend_mm_yr(f_gws, y0, y1, lsm)[r_idx, c_idx]

                gf, b_sub = p3mod.build_gcm_snapshot_grid_and_basins(grid_base, basins_base, q_m, p_m, a_m, gw_m, qr_m, slp_m)
                cells, _, snap = p3mod.evaluate_zoning_snapshot(
                    gf, b_sub, ecw_star_use, ece90_ref, mu_sod, mu_sal, et_obs,
                    ece_fut_col=ece_map[h_name], scenario_tag=f"{ssp}_{h_name}_{gcm}",
                )
                scenario = {"scenario": f"{ssp}_{h_name}", "ssp": ssp, "horizon": h_name, "gcm": gcm}
                risk = cells[["r", "c", "lat", "lon", "salt_irrig_ha", "is_hyper_sal_arid", "gw_still_depleted_post"]].copy()
                risk["saline_arid_and_gw"] = risk.is_hyper_sal_arid & risk.gw_still_depleted_post
                risk_cells.append(risk.assign(**scenario))
                for label, value in limits.items():
                    if label == "Composite reference":
                        alternate = snap
                    else:
                        _, _, alternate = p3mod.evaluate_zoning_snapshot(
                            gf, b_sub, value, ece90_ref, mu_sod, mu_sal, et_obs,
                            ece_fut_col=ece_map[h_name], scenario_tag=f"{ssp}_{h_name}_{gcm}")
                    future_thresholds.append({**scenario, "case": label, "ecw_dsm": value, **sensitivity.outcome(alternate)})
                zp = snap["zone_partitioning"]
                rb = snap.get("risk_intersection_breakdown_mha", {})
                rb_act = snap.get("risk_intersection_actual_et_mha", {})
                rows.append({
                    "scenario": f"{ssp}_{h_name}",
                    "ssp": ssp,
                    "horizon": h_name,
                    "gcm": gcm,
                    "gross_brackish_substitution_km3_yr": snap["gross_brackish_substitution_km3_yr"],
                    "leaching_penalty_P_km3_yr": snap["leaching_penalty_P_km3_yr"],
                    "deltaF_net_brackish_saving_km3_yr": snap["net_brackish_fresh_saving_deltaF_km3_yr"],
                    "hydraulic_recovery_cons_km3_yr": snap["hydraulic_recovery_cons_km3_yr"],
                    "hydraulic_recovery_wd_km3_yr": snap["hydraulic_recovery_wd_km3_yr"],
                    "net_fresh_withdrawal_relief_km3_yr": snap["net_fresh_withdrawal_relief_km3_yr"],
                    "basin_net_depletion_relief_km3_yr": snap["basin_net_depletion_relief_km3_yr"],
                    "basin_net_depletion_relief_actual_km3_yr": snap["basin_net_depletion_relief_actual_km3_yr"],
                    "aware_scarcity_relief_km3_weq": snap["aware_scarcity_relief_km3_weq"],
                    "aware_scarcity_relief_actual_km3_weq": snap["aware_scarcity_relief_actual_km3_weq"],
                    "paradox_exceeded_basins": snap["paradox_exceeded_basins"],
                    "paradox_basins_relieved": snap["paradox_basins_relieved_to_blue_feasible"],
                    "paradox_still_exceeded_post_basins": snap["paradox_still_exceeded_post_basins"],
                    "paradox_basins_relieved_actual": snap["paradox_basins_relieved_actual"],
                    "paradox_newly_exceeded_actual_basins": snap["paradox_newly_exceeded_actual_basins"],
                    "paradox_still_exceeded_actual_basins": snap["paradox_still_exceeded_actual_basins"],
                    "zone1_mha": zp.get("Zone_I_Safe_Brackish_Substitution_Window", {}).get("salt_irrig_mha", 0.0),
                    "zone1_local_potential_gw_constrained_mha": rb.get("zone1_local_potential_gw_constrained_mha", 0.0),
                    "zone2_mha": zp.get("Zone_II_Hydraulic_Recovery_Priority", {}).get("salt_irrig_mha", 0.0),
                    "zone3_mha": zp.get("Zone_III_Water_Salt_Dual_Redline", {}).get("salt_irrig_mha", 0.0),
                    "zone3_bw_only_mha": rb.get("zone3_bw_only_mha", 0.0),
                    "zone3_sal_arid_only_mha": rb.get("zone3_sal_arid_only_mha", 0.0),
                    "zone3_dual_bw_and_sal_arid_mha": rb.get("zone3_dual_bw_and_sal_arid_mha", 0.0),
                    "zone3_actual_et_mha": rb_act.get("zone3_at_least_one_risk_total_mha", 0.0),
                    "zone3_actual_et_bw_only_mha": rb_act.get("zone3_bw_only_mha", 0.0),
                    "zone3_actual_et_dual_mha": rb_act.get("zone3_dual_bw_and_sal_arid_mha", 0.0),
                    "sal_arid_and_gw_depleted_mha": rb.get("sal_arid_and_gw_depleted_mha", 0.0),
                    "salt_debt_post_mt_yr": snap["salt_debt_post_mt_yr"],
                    "salt_debt_post_actual_mt_yr": snap["salt_debt_post_actual_mt_yr"],
                })
                print(f"  {ssp}_{h_name} [{gcm}]: dF={snap['net_brackish_fresh_saving_deltaF_km3_yr']:.1f}, Z3={zp.get('Zone_III_Water_Salt_Dual_Redline', {}).get('salt_irrig_mha', 0.0):.2f} Mha, rel={snap['paradox_basins_relieved_to_blue_feasible']}")

    if len(rows) != 30 or len(future_thresholds) != 210:
        raise ValueError(f"Incomplete climate ensemble: {len(rows)} runs, {len(future_thresholds)} threshold cases")
    df = pd.DataFrame(rows)
    if len(df) != 30 or len(future_thresholds) != 30 * len(limits):
        raise ValueError("Incomplete matched future ensemble")
    pd.DataFrame(future_thresholds).to_csv(sensitivity.FUTURE, index=False)
    pd.concat(risk_cells, ignore_index=True).to_parquet(sensitivity.RISK, index=False)
    df.to_csv(OUT_GCM_CSV, index=False)
    print(f"Saved {len(df)} individual GCM rows -> {OUT_GCM_CSV}")


if __name__ == "__main__":
    prepare_outputs()
    main()
