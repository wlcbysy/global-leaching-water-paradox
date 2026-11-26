#!/usr/bin/env python3
"""
Pillar 3: Supply-Constrained Safe Brackish-Water Substitution, Three-Tier Water Accounting,
Salt-Debt Relief, AOA Applicability Mapping, Country/Hotspot Accounting, and Post-Intervention
Three-Zone Partitioning (Baseline + 2050/2080 5-GCM Ensembles).

Updates for Limits & Salt-Debt Revision:
  - Strictly driven by Arm A absolute Maas-Hoffman ECw*90 (continuous 3.547 baseline; unadjusted cyclic 6.335 sensitivity)
    and Arm B sodicity-stratified WUE (wue_by_sodicity) + comeasured ET ratio (H_obs)
  - Replaces hardcoded 40%/25% shares with quality-blending fraction f_b = clip((ECw*_cell - ECf)/(EC_b - ECf), 0, 0.5)
  - 5-iteration fixed-point solver explicitly deducts brackish-induced leaching penalty P = W1 - W0 (Delta_F = B - P)
  - Three-tier basin accounting: withdrawal (B, P, Delta_F, H_wd), consumption (H_cons, B_gw, Delta_V_leach, return_flow_credit=0),
    and basin net-depletion closure
  - Consumptive AWARE scarcity relief & post-intervention salt debt (u_post, Delta_S)
  - Post-intervention Three-Zone rules (III -> II -> I) using baseline hassani_ssp245_2020
"""

from __future__ import annotations

from config import (
    climate_models,
    EXTERNAL_RAW_DATA_DIR,
    TABLES,
    SOURCE_DATA,
    BASEMAP,
    OUT_TABLES,
    OUT_GRIDS,
    table_input,
    grid_input,
    prepare_outputs,
)

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point
import xarray as xr

P1_GRID = grid_input("pillar1_grid05_baseline_current.parquet")
P1_BASIN = table_input("pillar1_basin_leaching_closure_current.csv")
P2A_JSON = table_input("pillar2_armB_hydraulic_results_current.json")
P2B_JSON = table_input("pillar2_armA_ecw_results_current.json")
PUB_VAL_GRID = grid_input("public_validation_soil_aquifer_grid05_current.parquet")
JAEGER_CSV = EXTERNAL_RAW_DATA_DIR / "Jaegermeyr_2015/Jaegermeyr-2015-supplement_shr.csv"
NE_COUNTRIES_SHP = BASEMAP / "ne_110m_admin_0_countries.zip"
WG_3B_DIR = EXTERNAL_RAW_DATA_DIR / "WaterGAP2_2e/ISIMIP3b_future"
LSM_NC = EXTERNAL_RAW_DATA_DIR / "WaterGAP2_2e/landseamask_water-global.nc"

OUT_GRID = OUT_GRIDS / "pillar3_grid05_zoning_current.parquet"
OUT_FUTURE_CELL_ZONES = OUT_GRIDS / "pillar3_future_cell_zones_ssp585_2050_current.parquet"
OUT_BASIN = OUT_TABLES / "pillar3_basin_zoning_current.csv"
OUT_COUNTRY = OUT_TABLES / "pillar3_country_hotspot_zoning_current.csv"
OUT_FUTURE = OUT_TABLES / "pillar3_future_zoning_scenarios_current.csv"
OUT_SUPPLY_SENS = OUT_TABLES / "pillar3_supply_sensitivity_current.csv"
OUT_SUMMARY = OUT_TABLES / "pillar3_global_summary_current.json"


def ec_to_tds_gl(ec_dsm: np.ndarray) -> np.ndarray:
    return np.where(ec_dsm <= 5.0, 0.64 * ec_dsm, 0.80 * ec_dsm)


def calc_lr_vec(ecw_arr: np.ndarray, c_mm: np.ndarray, peff_mm: np.ndarray, ece_max: np.ndarray) -> np.ndarray:
    ec_aw = np.where((c_mm + peff_mm) > 0, (c_mm * ecw_arr) / np.maximum(c_mm + peff_mm, 1e-6), 0.0)
    denom = 5.0 * ece_max - ec_aw
    unleach = denom <= 0.1 * ec_aw
    return np.where(unleach, 0.80, np.clip(ec_aw / np.maximum(denom, 1e-6), 0.0, 0.80))


def compute_aoa_coverage(grid_df: pd.DataFrame) -> Dict[str, Any]:
    sites = pd.read_csv(SOURCE_DATA / "ED_Fig4A_empirical_sites.csv")
    armA_pts = sites.loc[sites.arm.eq("ArmA_brackish"), ["lat", "lon"]]
    armB_modeled_pts = sites.loc[sites.arm.eq("ArmB_hydraulic"), ["lat", "lon"]]

    raw_cols = ["prc_mm_yr", "et0_mm_yr", "qtot_mm_yr", "crop_ece90_dsm", "wang_ece_2017_dsm", "sg_clay_0_30_pct", "hwsd2_esp_pct"]
    g_sub = grid_df[["r", "c", "salt_affected", "salt_irrig_ha"] + raw_cols].dropna(subset=raw_cols).copy()
    g_sub["ai"] = g_sub["prc_mm_yr"] / np.maximum(g_sub["et0_mm_yr"], 1.0)
    fcols = ["ai", "et0_mm_yr", "qtot_mm_yr", "crop_ece90_dsm", "wang_ece_2017_dsm", "sg_clay_0_30_pct", "hwsd2_esp_pct"]
    target = g_sub[g_sub["salt_affected"]].copy()
    area_tot = float(target["salt_irrig_ha"].sum())

    def eval_pts(pts: pd.DataFrame) -> Dict[str, Any]:
        if pts.empty:
            return {"status": "NO_COORDINATES"}
        p = pts.copy()
        p["r"] = np.clip(np.floor((90.0 - p["lat"].values) / 0.5).astype(int), 0, 359)
        p["c"] = np.clip(np.floor((p["lon"].values + 180.0) / 0.5).astype(int), 0, 719)
        pts_rc = p[["r", "c"]].drop_duplicates()
        train = g_sub.merge(pts_rc, on=["r", "c"], how="inner")
        if len(train) < 8 or len(target) == 0:
            return {"status": "INSUFFICIENT_MATCHED_CELLS", "train_cells": int(len(train))}
        mu = train[fcols].mean().values
        sd = np.maximum(train[fcols].std().values, 1e-6)
        Z_tr = (train[fcols].values - mu) / sd
        Z_tg = (target[fcols].values - mu) / sd
        dists_tr = np.sqrt(((Z_tr[:, None, :] - Z_tr[None, :, :]) ** 2).sum(axis=2))
        np.fill_diagonal(dists_tr, np.inf)
        nn_tr = dists_tr.min(axis=1)
        di_thresh = float(np.percentile(nn_tr, 75) + 1.5 * (np.percentile(nn_tr, 75) - np.percentile(nn_tr, 25)))
        dists_tg = np.sqrt(((Z_tg[:, None, :] - Z_tr[None, :, :]) ** 2).sum(axis=2))
        nn_tg = dists_tg.min(axis=1)
        inside_aoa = nn_tg <= di_thresh
        area_aoa = float(target.loc[inside_aoa, "salt_irrig_ha"].sum())
        return {
            "training_unique_05deg_cells": int(len(train)),
            "target_salt_affected_cells": int(len(target)),
            "inside_aoa_cells": int(inside_aoa.sum()),
            "outside_aoa_cells": int((~inside_aoa).sum()),
            "aoa_di_threshold": round(di_thresh, 3),
            "di_method": "unweighted_standardized_euclidean_tukey_q3_plus_1.5iqr",
            "feature_cols": fcols,
            "cells_inside_aoa_pct": round(float(inside_aoa.mean() * 100.0), 2),
            "salt_irrig_area_inside_aoa_pct": round(area_aoa / max(area_tot, 1.0) * 100.0, 2),
            "salt_irrig_area_outside_aoa_mha": round((area_tot - area_aoa) / 1e6, 2),
        }

    full_evidence = pd.concat([armB_modeled_pts, armA_pts], ignore_index=True).drop_duplicates()
    return {
        # Historical comparator from the original geographic screen.
        "armA_plus_brackish_master": json.loads((TABLES / OUT_SUMMARY.name).read_text())["aoa_applicability_assessment"]["armA_plus_brackish_master"],
        "armA_extracted_only": eval_pts(armA_pts),
        "all_empirical_armA_and_armB_sites": eval_pts(full_evidence),
    }


def evaluate_zoning_snapshot(
    grid: pd.DataFrame,
    basins_base: pd.DataFrame,
    ecw_star_use_dsm: float,
    ece90_ref_sites: float,
    mu_wue_sodic: float,
    mu_wue_saline: float,
    et_ratio_obs: float,
    ece_fut_col: str = "hassani_ssp245_2020",
    scenario_tag: str = "baseline_2020",
    ret_capture_frac: float = 0.35,
    kappa_gw: float = 0.20,
    rho_hyd: float = 0.75,
    ec_ret_mode: str = "drainage_formula",
    count_gw_in_closure: bool = True,
    use_whymap_cap: bool = True,
    whymap_aquifer_prior: bool = False,
    dynqual_aquifer_prior: bool = False,
    marginal_leach_recovery: float = 0.0,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Evaluate supply-constrained brackish blending (5-iter fixed point), three-tier water accounting,
    net consumptive basin closure, post-intervention salt debt, and post-intervention Three-Zone partitioning.
    """
    if not 0 <= marginal_leach_recovery <= 1:
        raise ValueError("Marginal leaching recovery must be within [0, 1].")
    g = grid.copy()
    gw_headroom_km3 = np.maximum(g["qr_km3_yr"].values - g["atotusegw_mining_km3_yr"].values, 0.0)
    has_brackish_gw_local = (g["gw_n_stations"].values >= 1) & (g["gw_frac_brackish_1_5"].fillna(0.0).values >= 0.20)

    basin_idx = basins_base.set_index("aware_basin_idx")[
        [
            "gw_frac_brackish_1_5", "gw_ec_median_dsm", "paradox_status_amb",
            "sbwa_efr60_km3_yr", "qtot_km3_yr", "atotuse_km3_yr", "vleach_amb_km3_yr",
            "blue_water_exceeded_amb", "gw_depletion_active", "atotusegw_mining_km3_yr",
        ]
    ].to_dict(orient="index")

    g["basin_gw_frac_1_5"] = g["aware_basin_idx"].map(lambda b: basin_idx.get(b, {}).get("gw_frac_brackish_1_5", np.nan))
    g["basin_gw_ec_med"] = g["aware_basin_idx"].map(lambda b: basin_idx.get(b, {}).get("gw_ec_median_dsm", np.nan))
    g["basin_paradox_status"] = g["aware_basin_idx"].map(lambda b: basin_idx.get(b, {}).get("paradox_status_amb", "UNKNOWN"))
    g["basin_gw_depletion"] = g["aware_basin_idx"].map(lambda b: bool(basin_idx.get(b, {}).get("gw_depletion_active", False)))

    eff_brack_frac = np.where(
        has_brackish_gw_local,
        g["gw_frac_brackish_1_5"].fillna(0.0).values,
        np.where(g["basin_gw_frac_1_5"].notna().values & (g["basin_gw_frac_1_5"].fillna(0.0).values >= 0.15), g["basin_gw_frac_1_5"].fillna(0.0).values * 0.5, 0.0),
    )
    if whymap_aquifer_prior and "whymap_aquifer_tier" in g.columns:
        hy_ok = g["whymap_aquifer_tier"].isin(["Major_Basin", "Local_Shallow"]).values & g["whymap_hygeo2"].isin([12, 13, 22, 23, 33]).values
        eff_brack_frac = np.where(eff_brack_frac > 0, eff_brack_frac, np.where(hy_ok, 0.15, 0.0))
    if dynqual_aquifer_prior and "whymap_aquifer_tier" in g.columns and "dynqual_ecw_annual_dsm" in g.columns:
        aq_ok = g["whymap_aquifer_tier"].isin(["Major_Basin", "Local_Shallow", "Complex_Hydro"]).values
        dq_brack = (g["dynqual_ecw_annual_dsm"].fillna(0.0).values >= 0.75) | (g["dynqual_ecw_p90_monthly_dsm"].fillna(0.0).values >= 1.50)
        eff_brack_frac = np.where(eff_brack_frac > 0, eff_brack_frac, np.where(aq_ok & dq_brack, 0.25, 0.0))

    whymap_hypersaline = g["whymap_saline_gw_gt5gl"].fillna(False).values if (use_whymap_cap and "whymap_saline_gw_gt5gl" in g.columns) else np.zeros(len(g), dtype=bool)
    gw_brack_supply_km3 = np.where(
        (~g["basin_gw_depletion"].values) & (~whymap_hypersaline) & (eff_brack_frac > 0),
        kappa_gw * gw_headroom_km3 * eff_brack_frac,
        0.0,
    )
    drainage_return_km3 = np.maximum(g["pirrww_km3_yr"].values - g["pirruse_km3_yr"].values, 0.0) * ret_capture_frac

    ECf_g = np.nan_to_num(g["ecw_ambient_dsm"].values, nan=0.70)
    lr0_g = np.nan_to_num(g["lr_ambient"].values, nan=0.0)
    gw_ec_loc = g["gw_ec_median_dsm"].values
    gw_ec_bas = g["basin_gw_ec_med"].values
    EC_gw_g = np.where(
        np.isfinite(gw_ec_loc) & (gw_ec_loc >= 1.0) & (gw_ec_loc <= 5.0),
        gw_ec_loc,
        np.where(np.isfinite(gw_ec_bas) & (gw_ec_bas >= 1.0) & (gw_ec_bas <= 5.0), gw_ec_bas, 3.0),
    )
    if ec_ret_mode == "double_ecf":
        EC_ret_g = 2.0 * ECf_g
    else:
        EC_ret_g = np.minimum(ECf_g / np.maximum(lr0_g, 0.05), 10.0)

    S_tot_g = gw_brack_supply_km3 + drainage_return_km3
    EC_b_g = np.where(S_tot_g > 1e-12, (gw_brack_supply_km3 * EC_gw_g + drainage_return_km3 * EC_ret_g) / np.maximum(S_tot_g, 1e-12), 0.0)

    g["gw_brack_supply_km3_yr"] = gw_brack_supply_km3
    g["drainage_return_km3_yr"] = drainage_return_km3
    g["renewable_brackish_supply_km3_yr"] = S_tot_g
    g["ec_brack_effective_dsm"] = EC_b_g
    g["aridity_index"] = g["prc_mm_yr"].fillna(0.0).values / np.maximum(g["et0_mm_yr"].fillna(1.0).values, 1.0)

    salt = g[g["salt_affected"]].copy()
    lr_nan_count = int(salt["lr_ambient"].isna().sum())

    # 4a. Cell ECw* threshold scaled by crop tolerance
    s_scale = np.clip(salt["crop_ece90_dsm"].fillna(ece90_ref_sites).values / ece90_ref_sites, 0.5, 2.0)
    ecw_star_cell = ecw_star_use_dsm * s_scale
    ECf_s = np.nan_to_num(salt["ecw_ambient_dsm"].values, nan=0.70)
    EC_b_s = salt["ec_brack_effective_dsm"].values
    S_tot_s = salt["renewable_brackish_supply_km3_yr"].values
    S_gw_s = salt["gw_brack_supply_km3_yr"].values

    # 4c. Mass-balance blending limit f_b
    valid_mix = (ECf_s < ecw_star_cell) & (EC_b_s > ECf_s) & (S_tot_s > 0.0)
    f_b = np.where(
        valid_mix,
        np.clip((ecw_star_cell - ECf_s) / np.maximum(EC_b_s - ECf_s, 1e-9), 0.0, 0.5),
        0.0,
    )

    # 4d. 5-iteration fixed-point solver accounting for LR1 amplification
    W0 = np.maximum(salt["pirrww_km3_yr"].values, 0.0)
    C = np.maximum(salt["pirruse_km3_yr"].values, 0.0)
    c_mm = np.maximum(salt["pirruse_mm_yr"].values, 0.0)
    peff_mm = np.maximum(salt["peff_mm_yr"].values, 0.0)
    ece_max = np.where(np.isfinite(salt["crop_ece90_dsm"].values), salt["crop_ece90_dsm"].values, 3.5)
    lr0 = np.nan_to_num(salt["lr_ambient"].values, nan=0.0)
    A0 = C / np.maximum(1.0 - lr0, 0.05)

    f = np.where(W0 > 1e-12, np.minimum(f_b, S_tot_s / np.maximum(W0, 1e-12)), 0.0)
    EC_mix = ECf_s.copy()
    LR1 = lr0.copy()
    A1 = A0.copy()
    W1 = W0.copy()
    B = np.zeros_like(W0)
    for _ in range(5):
        EC_mix = ECf_s + f * (EC_b_s - ECf_s)
        LR1 = calc_lr_vec(EC_mix, c_mm, peff_mm, ece_max)
        A1 = C / np.maximum(1.0 - LR1, 0.05)
        W1 = A1 * np.where(A0 > 1e-12, W0 / np.maximum(A0, 1e-12), 1.0)
        B = np.minimum(f * W1, S_tot_s)
        f = np.where(W1 > 1e-12, B / np.maximum(W1, 1e-12), 0.0)

    EC_mix = ECf_s + f * (EC_b_s - ECf_s)
    LR1 = calc_lr_vec(EC_mix, c_mm, peff_mm, ece_max)
    A1 = C / np.maximum(1.0 - LR1, 0.05)
    W1 = A1 * np.where(A0 > 1e-12, W0 / np.maximum(A0, 1e-12), 1.0)
    B = np.minimum(f * W1, S_tot_s)

    gw_frac_share = np.where(S_tot_s > 1e-12, S_gw_s / np.maximum(S_tot_s, 1e-12), 0.0)
    B_gw = B * gw_frac_share
    B_ret = B * (1.0 - gw_frac_share)
    P = np.maximum(W1 - W0, 0.0)
    Delta_F = B - P
    Delta_V_leach = C * (LR1 / np.maximum(1.0 - LR1, 0.05) - lr0 / np.maximum(1.0 - lr0, 0.05))
    S_in_pre_mt = A0 * ec_to_tds_gl(ECf_s)
    S_in_post_mt = A1 * ec_to_tds_gl(EC_mix)
    Delta_S_mt = S_in_post_mt - S_in_pre_mt

    # Field-boundary vs Basin-boundary salt-import decomposition
    # At field boundary, recycled return flow (1 - gw_frac_share) reapplies salt onto fields
    # but is an internal transfer at the basin boundary (not net new basin salt creation).
    EC_gw_s = np.where(
        np.isfinite(salt["gw_ec_median_dsm"].values) & (salt["gw_ec_median_dsm"].values >= 1.0) & (salt["gw_ec_median_dsm"].values <= 5.0),
        salt["gw_ec_median_dsm"].values,
        np.where(np.isfinite(salt["basin_gw_ec_med"].values) & (salt["basin_gw_ec_med"].values >= 1.0) & (salt["basin_gw_ec_med"].values <= 5.0), salt["basin_gw_ec_med"].values, 3.0),
    )
    EC_ret_s = 2.0 * ECf_s if ec_ret_mode == "double_ecf" else np.minimum(ECf_s / np.maximum(lr0, 0.05), 10.0)
    s_add_gw_raw = f * A1 * gw_frac_share * ec_to_tds_gl(EC_gw_s)
    s_add_ret_raw = f * A1 * (1.0 - gw_frac_share) * ec_to_tds_gl(EC_ret_s)
    s_add_fresh_diff = ((1.0 - f) * A1 - A0) * ec_to_tds_gl(ECf_s)
    # Normalize slight piecewise EC->TDS scaling difference so components sum identically to Delta_S_mt
    raw_sum = s_add_gw_raw + s_add_ret_raw + s_add_fresh_diff
    scale_s = np.divide(Delta_S_mt, raw_sum, out=np.ones_like(raw_sum), where=np.abs(raw_sum) > 1e-12)
    delta_s_field_recycled_gross_mt = s_add_ret_raw * scale_s
    delta_s_basin_gw_gross_mt = s_add_gw_raw * scale_s
    delta_s_fresh_displaced_mt = s_add_fresh_diff * scale_s
    delta_s_field_recycled_mt = delta_s_field_recycled_gross_mt
    delta_s_basin_gw_new_mt = delta_s_basin_gw_gross_mt + delta_s_fresh_displaced_mt

    # 4e. Soil hydraulic recovery (equal-yield conditional vs observed-ET)
    f_sod = np.clip(salt["hwsd2_sodic_frac"].fillna(0.0).values, 0.0, 1.0)
    mu_cell = f_sod * mu_wue_sodic + (1.0 - f_sod) * mu_wue_saline
    H_cons = C * rho_hyd * (1.0 - np.exp(-mu_cell))
    H_wd = np.where(C > 1e-12, H_cons * (W0 / np.maximum(C, 1e-12)), 0.0)
    H_obs = C * (1.0 - et_ratio_obs)

    net_fresh_withdrawal_relief = Delta_F + H_wd
    gw_credit_cell = B_gw if count_gw_in_closure else np.zeros_like(B_gw)
    marginal_cost = (1 - marginal_leach_recovery) * Delta_V_leach
    basin_net_depletion_relief_cell = H_cons + gw_credit_cell - marginal_cost
    basin_net_depletion_relief_actual_cell = H_obs + gw_credit_cell - marginal_cost

    salt["ecw_star_cell_dsm"] = ecw_star_cell
    salt["f_blend_cap"] = f_b
    salt["f_blend_realized"] = f
    salt["ec_mix_dsm"] = EC_mix
    salt["gross_brack_sub_km3_yr"] = B
    salt["brack_sub_from_aquifer_gw_km3_yr"] = B_gw
    salt["brack_sub_from_return_flow_km3_yr"] = B_ret
    salt["leaching_penalty_P_km3_yr"] = P
    salt["net_brackish_fresh_saving_deltaF_km3_yr"] = Delta_F
    salt["delta_vleach_km3_yr"] = Delta_V_leach
    salt["hydraulic_saving_cons_km3_yr"] = H_cons
    salt["hydraulic_saving_wd_km3_yr"] = H_wd
    salt["hydraulic_saving_obs_et_km3_yr"] = H_obs
    salt["net_fresh_withdrawal_relief_km3_yr"] = net_fresh_withdrawal_relief
    salt["basin_net_depletion_relief_km3_yr"] = basin_net_depletion_relief_cell
    salt["basin_net_depletion_relief_actual_km3_yr"] = basin_net_depletion_relief_actual_cell
    salt["scarcity_relief_km3_weq"] = basin_net_depletion_relief_cell * salt["aware_cf_agri"].fillna(1.0).values
    salt["scarcity_relief_actual_km3_weq"] = basin_net_depletion_relief_actual_cell * salt["aware_cf_agri"].fillna(1.0).values
    salt["s_in_pre_mt_yr"] = S_in_pre_mt
    salt["s_in_post_mt_yr"] = S_in_post_mt
    salt["delta_s_mt_yr"] = Delta_S_mt
    salt["delta_s_field_recycled_gross_mt_yr"] = delta_s_field_recycled_gross_mt
    salt["delta_s_basin_gw_gross_mt_yr"] = delta_s_basin_gw_gross_mt
    salt["delta_s_fresh_displaced_mt_yr"] = delta_s_fresh_displaced_mt
    salt["delta_s_field_recycled_mt_yr"] = delta_s_field_recycled_mt
    salt["delta_s_basin_new_mt_yr"] = delta_s_basin_gw_new_mt

    # 4f & 4h. Basin net consumptive closure & post-intervention salt debt
    b_salt = salt[salt["aware_basin_idx"] >= 0].groupby("aware_basin_idx", as_index=False).agg(
        salt_irrig_ha_salt=("salt_irrig_ha", "sum"),
        pirrww_salt_km3_yr=("pirrww_km3_yr", "sum"),
        gross_brack_sub_km3_yr=("gross_brack_sub_km3_yr", "sum"),
        brack_sub_from_aquifer_gw_km3_yr=("brack_sub_from_aquifer_gw_km3_yr", "sum"),
        brack_sub_from_return_flow_km3_yr=("brack_sub_from_return_flow_km3_yr", "sum"),
        leaching_penalty_P_km3_yr=("leaching_penalty_P_km3_yr", "sum"),
        net_brackish_fresh_saving_deltaF_km3_yr=("net_brackish_fresh_saving_deltaF_km3_yr", "sum"),
        delta_vleach_km3_yr=("delta_vleach_km3_yr", "sum"),
        hydraulic_saving_cons_km3_yr=("hydraulic_saving_cons_km3_yr", "sum"),
        hydraulic_saving_wd_km3_yr=("hydraulic_saving_wd_km3_yr", "sum"),
        hydraulic_saving_obs_et_km3_yr=("hydraulic_saving_obs_et_km3_yr", "sum"),
        net_fresh_withdrawal_relief_km3_yr=("net_fresh_withdrawal_relief_km3_yr", "sum"),
        basin_net_depletion_relief_km3_yr=("basin_net_depletion_relief_km3_yr", "sum"),
        basin_net_depletion_relief_actual_km3_yr=("basin_net_depletion_relief_actual_km3_yr", "sum"),
        scarcity_relief_km3_weq=("scarcity_relief_km3_weq", "sum"),
        scarcity_relief_actual_km3_weq=("scarcity_relief_actual_km3_weq", "sum"),
        s_in_pre_mt_yr=("s_in_pre_mt_yr", "sum"),
        s_in_post_mt_yr=("s_in_post_mt_yr", "sum"),
        delta_s_mt_yr=("delta_s_mt_yr", "sum"),
        delta_s_field_recycled_mt_yr=("delta_s_field_recycled_mt_yr", "sum"),
        delta_s_basin_new_mt_yr=("delta_s_basin_new_mt_yr", "sum"),
    )
    basins_out = basins_base.merge(b_salt, on="aware_basin_idx", how="left")
    fill_zero_cols = [
        "gross_brack_sub_km3_yr", "brack_sub_from_aquifer_gw_km3_yr", "brack_sub_from_return_flow_km3_yr",
        "leaching_penalty_P_km3_yr", "net_brackish_fresh_saving_deltaF_km3_yr", "delta_vleach_km3_yr",
        "hydraulic_saving_cons_km3_yr", "hydraulic_saving_wd_km3_yr", "hydraulic_saving_obs_et_km3_yr",
        "net_fresh_withdrawal_relief_km3_yr", "basin_net_depletion_relief_km3_yr",
        "basin_net_depletion_relief_actual_km3_yr", "scarcity_relief_km3_weq", "scarcity_relief_actual_km3_weq",
        "s_in_pre_mt_yr", "s_in_post_mt_yr", "delta_s_mt_yr",
        "delta_s_field_recycled_mt_yr", "delta_s_basin_new_mt_yr",
    ]
    for col_z in fill_zero_cols:
        basins_out[col_z] = basins_out[col_z].fillna(0.0)

    basins_out["return_flow_basin_credit_km3_yr"] = 0.0
    gw_cr_basin = basins_out["brack_sub_from_aquifer_gw_km3_yr"] if count_gw_in_closure else 0.0
    atot_pos = np.maximum(basins_out["atotuse_km3_yr"], 0.0)
    vl_post = basins_out["vleach_amb_km3_yr"] + (1 - marginal_leach_recovery) * basins_out["delta_vleach_km3_yr"]
    non_leach_post_cap = np.maximum(
        atot_pos - basins_out["hydraulic_saving_cons_km3_yr"] - gw_cr_basin,
        0.0,
    )
    non_leach_post_act = np.maximum(
        atot_pos - basins_out["hydraulic_saving_obs_et_km3_yr"] - gw_cr_basin,
        0.0,
    )
    basins_out["post_intervention_net_use_km3_yr"] = non_leach_post_cap + vl_post
    basins_out["post_intervention_net_use_actual_km3_yr"] = non_leach_post_act + vl_post
    basins_out["blue_water_relieved_to_feasible"] = (
        basins_out["blue_water_exceeded_amb"] & (basins_out["post_intervention_net_use_km3_yr"] <= basins_out["sbwa_efr60_km3_yr"])
    )
    basins_out["blue_water_newly_exceeded_post"] = (
        (~basins_out["blue_water_exceeded_amb"]) & (basins_out["post_intervention_net_use_km3_yr"] > basins_out["sbwa_efr60_km3_yr"])
    )
    basins_out["blue_water_still_exceeded_post"] = basins_out["post_intervention_net_use_km3_yr"] > basins_out["sbwa_efr60_km3_yr"]
    basins_out["blue_water_relieved_actual"] = (
        basins_out["blue_water_exceeded_amb"] & (basins_out["post_intervention_net_use_actual_km3_yr"] <= basins_out["sbwa_efr60_km3_yr"])
    )
    basins_out["blue_water_newly_exceeded_actual"] = (
        (~basins_out["blue_water_exceeded_amb"]) & (basins_out["post_intervention_net_use_actual_km3_yr"] > basins_out["sbwa_efr60_km3_yr"])
    )
    basins_out["blue_water_still_exceeded_actual"] = basins_out["post_intervention_net_use_actual_km3_yr"] > basins_out["sbwa_efr60_km3_yr"]
    basins_out["gw_still_depleted_post"] = basins_out["gw_depletion_active"] & (
        basins_out["hydraulic_saving_cons_km3_yr"] < basins_out["atotusegw_mining_km3_yr"]
    )

    # Pre and post unserved leaching fractions (u_pre, u_post, u_post_actual) & salt debt
    u_pre = np.where(
        basins_out["vleach_amb_km3_yr"] > 1e-6,
        1.0 - np.clip((basins_out["sbwa_efr60_km3_yr"] - atot_pos) / basins_out["vleach_amb_km3_yr"], 0.0, 1.0),
        np.where(atot_pos > basins_out["sbwa_efr60_km3_yr"], 1.0, 0.0),
    )
    avail_post = basins_out["sbwa_efr60_km3_yr"] - non_leach_post_cap
    u_post = np.where(
        vl_post > 1e-6,
        1.0 - np.clip(avail_post / vl_post, 0.0, 1.0),
        np.where(non_leach_post_cap > basins_out["sbwa_efr60_km3_yr"], 1.0, 0.0),
    )
    avail_post_act = basins_out["sbwa_efr60_km3_yr"] - non_leach_post_act
    u_post_act = np.where(
        vl_post > 1e-6,
        1.0 - np.clip(avail_post_act / vl_post, 0.0, 1.0),
        np.where(non_leach_post_act > basins_out["sbwa_efr60_km3_yr"], 1.0, 0.0),
    )
    basins_out["u_pre"] = u_pre
    basins_out["u_post"] = u_post
    basins_out["u_post_actual"] = u_post_act
    basins_out["salt_debt_pre_mt_yr"] = u_pre * basins_out["s_in_pre_mt_yr"]
    basins_out["salt_debt_post_mt_yr"] = u_post * basins_out["s_in_post_mt_yr"]
    basins_out["salt_debt_post_actual_mt_yr"] = u_post_act * basins_out["s_in_post_mt_yr"]

    # Map basin post-intervention flags back to salt cells
    b_post_map = basins_out.set_index("aware_basin_idx")[["blue_water_still_exceeded_post", "blue_water_still_exceeded_actual", "gw_still_depleted_post", "u_pre", "u_post"]].to_dict(orient="index")
    salt["blue_water_still_exceeded_post"] = salt["aware_basin_idx"].map(lambda b: bool(b_post_map.get(b, {}).get("blue_water_still_exceeded_post", False)))
    salt["blue_water_still_exceeded_actual"] = salt["aware_basin_idx"].map(lambda b: bool(b_post_map.get(b, {}).get("blue_water_still_exceeded_actual", False)))
    salt["gw_still_depleted_post"] = salt["aware_basin_idx"].map(lambda b: bool(b_post_map.get(b, {}).get("gw_still_depleted_post", False)))
    salt["salt_debt_pre_mt_yr"] = salt["aware_basin_idx"].map(lambda b: float(b_post_map.get(b, {}).get("u_pre", 0.0))) * salt["s_in_pre_mt_yr"]
    salt["salt_debt_post_mt_yr"] = salt["aware_basin_idx"].map(lambda b: float(b_post_map.get(b, {}).get("u_post", 0.0))) * salt["s_in_post_mt_yr"]

    # 4i. Post-intervention Three-Zone classification (exact order: Zone III -> Zone II -> Zone I)
    ece_fut = salt[ece_fut_col].values if ece_fut_col in salt.columns else np.full(len(salt), np.nan)
    is_hyper_sal_arid = np.isfinite(ece_fut) & (ece_fut >= 8.0) & (salt["aridity_index"].values < 0.12)
    is_bw_post = salt["blue_water_still_exceeded_post"].values
    is_gw_post = salt["gw_still_depleted_post"].values
    is_z3 = is_bw_post | is_hyper_sal_arid
    is_whymap_sal = salt["whymap_saline_gw_gt5gl"].fillna(False).values if "whymap_saline_gw_gt5gl" in salt.columns else np.zeros(len(salt), dtype=bool)
    is_sodic_dom = salt["hwsd2_sodic_frac"].fillna(0.0).values >= 0.30
    is_poor_drain = salt["hwsd2_poor_drain_frac"].fillna(0.0).values >= 0.50
    is_low_sub = Delta_F < (0.10 * np.maximum(W0, 1e-6))
    is_z2 = (~is_z3) & (is_sodic_dom | is_poor_drain | is_whymap_sal | is_low_sub)
    salt["is_hyper_sal_arid"] = is_hyper_sal_arid
    salt["management_zone"] = np.where(
        is_z3, "Zone_III_Water_Salt_Dual_Redline",
        np.where(is_z2, "Zone_II_Hydraulic_Recovery_Priority", "Zone_I_Safe_Brackish_Substitution_Window"),
    )
    salt["risk_decomposition_class"] = np.where(
        is_bw_post & is_hyper_sal_arid, "Risk_Dual_BW_and_SalArid",
        np.where(
            is_bw_post & (~is_hyper_sal_arid), "Risk_BW_Only",
            np.where(
                (~is_bw_post) & is_hyper_sal_arid, "Risk_SalArid_Only",
                np.where(
                    is_z2 & is_gw_post, "Zone_II_Hydraulic_GW_Depleted",
                    np.where(
                        is_z2, "Zone_II_Hydraulic_GW_OK",
                        np.where(is_gw_post, "Zone_I_Local_Potential_GW_Constrained", "Zone_I_Safe_Window_GW_Sustainable"),
                    ),
                ),
            ),
        ),
    )

    ha_mha = salt["salt_irrig_ha"].values / 1e6
    risk_breakdown = {
        "zone3_at_least_one_risk_total_mha": round(float(ha_mha[is_z3].sum()), 2),
        "zone3_bw_only_mha": round(float(ha_mha[is_bw_post & (~is_hyper_sal_arid)].sum()), 2),
        "zone3_sal_arid_only_mha": round(float(ha_mha[(~is_bw_post) & is_hyper_sal_arid].sum()), 2),
        "zone3_dual_bw_and_sal_arid_mha": round(float(ha_mha[is_bw_post & is_hyper_sal_arid].sum()), 2),
        "sal_arid_and_gw_depleted_mha": round(float(ha_mha[is_hyper_sal_arid & is_gw_post].sum()), 2),
        "bw_and_gw_depleted_mha": round(float(ha_mha[is_bw_post & is_gw_post].sum()), 2),
        "triple_risk_bw_sal_gw_mha": round(float(ha_mha[is_bw_post & is_hyper_sal_arid & is_gw_post].sum()), 2),
        "zone2_total_mha": round(float(ha_mha[is_z2].sum()), 2),
        "zone2_gw_depleted_mha": round(float(ha_mha[is_z2 & is_gw_post].sum()), 2),
        "zone2_gw_sustainable_mha": round(float(ha_mha[is_z2 & (~is_gw_post)].sum()), 2),
        "zone1_total_mha": round(float(ha_mha[(~is_z3) & (~is_z2)].sum()), 2),
        "zone1_local_potential_gw_constrained_mha": round(float(ha_mha[(~is_z3) & (~is_z2) & is_gw_post].sum()), 2),
        "zone1_safe_window_gw_sustainable_mha": round(float(ha_mha[(~is_z3) & (~is_z2) & (~is_gw_post)].sum()), 2),
    }
    is_bw_act = salt["blue_water_still_exceeded_actual"].values
    risk_breakdown_actual = {
        "zone3_at_least_one_risk_total_mha": round(float(ha_mha[is_bw_act | is_hyper_sal_arid].sum()), 2),
        "zone3_bw_only_mha": round(float(ha_mha[is_bw_act & (~is_hyper_sal_arid)].sum()), 2),
        "zone3_sal_arid_only_mha": round(float(ha_mha[(~is_bw_act) & is_hyper_sal_arid].sum()), 2),
        "zone3_dual_bw_and_sal_arid_mha": round(float(ha_mha[is_bw_act & is_hyper_sal_arid].sum()), 2),
        "sal_arid_and_gw_depleted_mha": round(float(ha_mha[is_hyper_sal_arid & is_gw_post].sum()), 2),
    }

    b_dom_z = salt[salt["aware_basin_idx"] >= 0].groupby("aware_basin_idx")["management_zone"].agg(lambda s: s.value_counts().index[0]).to_dict()
    basins_out["dominant_zone"] = basins_out["aware_basin_idx"].map(lambda b: b_dom_z.get(b, "NONE"))

    zone_summary: Dict[str, Any] = {}
    for z, grp in salt.groupby("management_zone"):
        zone_summary[z] = {
            "cells": int(len(grp)),
            "salt_irrig_mha": round(float(grp["salt_irrig_ha"].sum() / 1e6), 2),
            "gross_brackish_substitution_km3_yr": round(float(grp["gross_brack_sub_km3_yr"].sum()), 2),
            "net_brackish_fresh_saving_deltaF_km3_yr": round(float(grp["net_brackish_fresh_saving_deltaF_km3_yr"].sum()), 2),
            "hydraulic_saving_wd_km3_yr": round(float(grp["hydraulic_saving_wd_km3_yr"].sum()), 2),
            "hydraulic_saving_cons_km3_yr": round(float(grp["hydraulic_saving_cons_km3_yr"].sum()), 2),
            "net_fresh_withdrawal_relief_km3_yr": round(float(grp["net_fresh_withdrawal_relief_km3_yr"].sum()), 2),
        }

    sb = basins_out[basins_out["salt_irrig_ha"] >= 10000.0]
    snap_summary = {
        "scenario": scenario_tag,
        "lr_ambient_nan_cells_on_salt": lr_nan_count,
        "salt_irrig_mha": round(float(salt["salt_irrig_ha"].sum() / 1e6), 2),
        "salt_pirrww_km3_yr": round(float(salt["pirrww_km3_yr"].sum()), 2),
        "salt_pirruse_km3_yr": round(float(salt["pirruse_km3_yr"].sum()), 2),
        "gross_brackish_substitution_km3_yr": round(float(salt["gross_brack_sub_km3_yr"].sum()), 2),
        "brackish_from_aquifer_gw_km3_yr": round(float(salt["brack_sub_from_aquifer_gw_km3_yr"].sum()), 2),
        "brackish_from_return_flow_km3_yr": round(float(salt["brack_sub_from_return_flow_km3_yr"].sum()), 2),
        "leaching_penalty_P_km3_yr": round(float(salt["leaching_penalty_P_km3_yr"].sum()), 2),
        "net_brackish_fresh_saving_deltaF_km3_yr": round(float(salt["net_brackish_fresh_saving_deltaF_km3_yr"].sum()), 2),
        "delta_vleach_km3_yr": round(float(salt["delta_vleach_km3_yr"].sum()), 2),
        "hydraulic_recovery_cons_km3_yr": round(float(salt["hydraulic_saving_cons_km3_yr"].sum()), 2),
        "hydraulic_recovery_wd_km3_yr": round(float(salt["hydraulic_saving_wd_km3_yr"].sum()), 2),
        "hydraulic_observed_et_saving_km3_yr": round(float(salt["hydraulic_saving_obs_et_km3_yr"].sum()), 2),
        "net_fresh_withdrawal_relief_km3_yr": round(float(salt["net_fresh_withdrawal_relief_km3_yr"].sum()), 2),
        "basin_net_depletion_relief_km3_yr": round(float(salt["basin_net_depletion_relief_km3_yr"].sum()), 2),
        "basin_net_depletion_relief_actual_km3_yr": round(float(salt["basin_net_depletion_relief_actual_km3_yr"].sum()), 2),
        "return_flow_basin_credit_km3_yr": 0.0,
        "aware_scarcity_relief_km3_weq": round(float(salt["scarcity_relief_km3_weq"].sum()), 1),
        "aware_scarcity_relief_actual_km3_weq": round(float(salt["scarcity_relief_actual_km3_weq"].sum()), 1),
        "delta_s_salt_import_mt_yr": round(float(salt["delta_s_mt_yr"].sum()), 2),
        "delta_s_field_recycled_gross_mt_yr": round(float(salt["delta_s_field_recycled_gross_mt_yr"].sum()), 2),
        "delta_s_basin_gw_gross_mt_yr": round(float(salt["delta_s_basin_gw_gross_mt_yr"].sum()), 2),
        "delta_s_fresh_displaced_mt_yr": round(float(salt["delta_s_fresh_displaced_mt_yr"].sum()), 2),
        "delta_s_field_recycled_mt_yr": round(float(salt["delta_s_field_recycled_mt_yr"].sum()), 2),
        "delta_s_basin_new_mt_yr": round(float(salt["delta_s_basin_new_mt_yr"].sum()), 2),
        "salt_debt_pre_mt_yr": round(float(sb["salt_debt_pre_mt_yr"].sum()), 2),
        "salt_debt_post_mt_yr": round(float(sb["salt_debt_post_mt_yr"].sum()), 2),
        "salt_debt_post_actual_mt_yr": round(float(sb["salt_debt_post_actual_mt_yr"].sum()), 2),
        "paradox_exceeded_basins": int(sb["blue_water_exceeded_amb"].sum()),
        "paradox_exceeded_mha": round(float(sb.loc[sb["blue_water_exceeded_amb"], "salt_irrig_ha"].sum() / 1e6), 2),
        "paradox_basins_relieved_to_blue_feasible": int(sb["blue_water_relieved_to_feasible"].sum()),
        "paradox_relieved_mha": round(float(sb.loc[sb["blue_water_relieved_to_feasible"], "salt_irrig_ha"].sum() / 1e6), 2),
        "paradox_newly_exceeded_post_basins": int(sb["blue_water_newly_exceeded_post"].sum()),
        "paradox_newly_exceeded_post_mha": round(float(sb.loc[sb["blue_water_newly_exceeded_post"], "salt_irrig_ha"].sum() / 1e6), 2),
        "paradox_still_exceeded_post_basins": int(sb["blue_water_still_exceeded_post"].sum()),
        "paradox_still_exceeded_post_mha": round(float(sb.loc[sb["blue_water_still_exceeded_post"], "salt_irrig_ha"].sum() / 1e6), 2),
        "paradox_basins_relieved_actual": int(sb["blue_water_relieved_actual"].sum()),
        "paradox_relieved_actual_mha": round(float(sb.loc[sb["blue_water_relieved_actual"], "salt_irrig_ha"].sum() / 1e6), 2),
        "paradox_newly_exceeded_actual_basins": int(sb["blue_water_newly_exceeded_actual"].sum()),
        "paradox_newly_exceeded_actual_mha": round(float(sb.loc[sb["blue_water_newly_exceeded_actual"], "salt_irrig_ha"].sum() / 1e6), 2),
        "paradox_still_exceeded_actual_basins": int(sb["blue_water_still_exceeded_actual"].sum()),
        "paradox_still_exceeded_actual_mha": round(float(sb.loc[sb["blue_water_still_exceeded_actual"], "salt_irrig_ha"].sum() / 1e6), 2),
        "gw_still_depleted_post_basins": int(sb["gw_still_depleted_post"].sum()),
        "gw_still_depleted_post_mha": round(float(salt.loc[salt["gw_still_depleted_post"], "salt_irrig_ha"].sum() / 1e6), 2),
        "zone_partitioning": zone_summary,
        "risk_intersection_breakdown_mha": risk_breakdown,
        "risk_intersection_actual_et_mha": risk_breakdown_actual,
    }
    return salt, basins_out, snap_summary


def compute_annual_flux_mm_yr(nc_path: Path, var: str, y0: int, y1: int, lsm: np.ndarray) -> np.ndarray:
    with xr.open_dataset(nc_path) as ds:
        sub = ds[var].sel(time=slice(f"{y0}-01-01", f"{y1}-12-31"))
        dim = sub["time"].dt.days_in_month
        ann = (sub * dim * 86400.0).groupby("time.year").sum("time", skipna=True).mean("year").values
        return np.where(lsm, ann, np.nan)


def compute_gws_trend_mm_yr(nc_path: Path, y0: int, y1: int, lsm: np.ndarray) -> np.ndarray:
    with xr.open_dataset(nc_path) as ds:
        sub = ds["groundwstor"].sel(time=slice(f"{y0}-01-01", f"{y1}-12-31"))
        ann_yr = sub.groupby("time.year").mean("time")
        yrs = ann_yr["year"].values.astype("float64")
        vals = ann_yr.values
        with np.errstate(invalid="ignore"):
            mean_mm = np.nanmean(vals, axis=0)
            t_c = yrs - yrs.mean()
            denom = float(np.sum(t_c ** 2))
            y_c = vals - mean_mm[None, :, :]
            slope = np.nansum(y_c * t_c[:, None, None], axis=0) / denom
        return np.where(lsm, slope, np.nan)


def build_gcm_snapshot_grid_and_basins(
    grid_base: pd.DataFrame,
    basins_base: pd.DataFrame,
    q_m: np.ndarray,
    p_m: np.ndarray,
    a_m: np.ndarray,
    gw_m: np.ndarray,
    qr_m: np.ndarray,
    slp_m: np.ndarray,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    area_km2_vec = grid_base["cell_area_km2"].values
    ret_frac_vec = np.where(
        grid_base["pirrww_km3_yr"].values > 1e-6,
        np.clip(grid_base["pirruse_km3_yr"].values / np.maximum(grid_base["pirrww_km3_yr"].values, 1e-6), 0.15, 0.95),
        0.55,
    )
    b_vleach_map = basins_base.set_index("aware_basin_idx")["vleach_amb_km3_yr"].to_dict()
    b_gwec_map = basins_base.set_index("aware_basin_idx")["gw_ec_median_dsm"].to_dict()

    gf = grid_base.copy()
    gf["qtot_km3_yr"] = np.nan_to_num(q_m, nan=0.0) * 1e-6 * area_km2_vec
    gf["pirrww_km3_yr"] = np.nan_to_num(p_m, nan=0.0) * 1e-6 * area_km2_vec
    gf["pirruse_km3_yr"] = gf["pirrww_km3_yr"].values * ret_frac_vec
    gf["pirruse_mm_yr"] = np.nan_to_num(p_m, nan=0.0) * ret_frac_vec
    gf["atotuse_km3_yr"] = np.nan_to_num(a_m, nan=0.0) * 1e-6 * area_km2_vec
    gf["atotusegw_mining_km3_yr"] = np.maximum(np.nan_to_num(gw_m, nan=0.0), 0.0) * 1e-6 * area_km2_vec
    gf["atotusegw_net_km3_yr"] = np.nan_to_num(gw_m, nan=0.0) * 1e-6 * area_km2_vec
    gf["qr_km3_yr"] = np.nan_to_num(qr_m, nan=0.0) * 1e-6 * area_km2_vec
    gf["gws_trend_km3_yr"] = np.nan_to_num(slp_m, nan=0.0) * 1e-6 * area_km2_vec

    b_sub = gf[gf["aware_basin_idx"] >= 0].groupby("aware_basin_idx", as_index=False).agg(
        basin_area_km2=("cell_area_km2", "sum"),
        salt_irrig_ha=("salt_irrig_ha", "sum"),
        gw_frac_brackish_1_5=("gw_frac_brackish_1_5", "mean"),
        gw_n_stations=("gw_n_stations", "sum"),
        qtot_km3_yr=("qtot_km3_yr", "sum"),
        atotuse_km3_yr=("atotuse_km3_yr", "sum"),
        atotusegw_mining_km3_yr=("atotusegw_mining_km3_yr", "sum"),
        atotusegw_net_km3_yr=("atotusegw_net_km3_yr", "sum"),
        qr_km3_yr=("qr_km3_yr", "sum"),
        gws_trend_km3_yr=("gws_trend_km3_yr", "sum"),
    )
    b_sub["vleach_amb_km3_yr"] = b_sub["aware_basin_idx"].map(lambda b: b_vleach_map.get(b, 0.0))
    b_sub["gw_ec_median_dsm"] = b_sub["aware_basin_idx"].map(lambda b: b_gwec_map.get(b, np.nan))
    b_sub["sbwa_efr60_km3_yr"] = 0.40 * b_sub["qtot_km3_yr"]
    b_sub["gws_trend_mm_yr"] = b_sub["gws_trend_km3_yr"] / (b_sub["basin_area_km2"] * 1e-6)
    b_sub["gw_depletion_active"] = (b_sub["atotusegw_mining_km3_yr"] > 0.05 * b_sub["qr_km3_yr"]) & (
        (b_sub["atotusegw_net_km3_yr"] > b_sub["qr_km3_yr"]) | (b_sub["gws_trend_mm_yr"] < -1.0)
    )
    b_sub["blue_water_exceeded_amb"] = (np.maximum(b_sub["atotuse_km3_yr"], 0.0) + b_sub["vleach_amb_km3_yr"]) > b_sub["sbwa_efr60_km3_yr"]
    b_sub["paradox_status_amb"] = np.where(
        b_sub["blue_water_exceeded_amb"] & b_sub["gw_depletion_active"], "PARADOX_INSOLVENT_BLUE_AND_GW",
        np.where(b_sub["blue_water_exceeded_amb"], "PARADOX_BLUE_WATER_EXCEEDED",
                 np.where(b_sub["gw_depletion_active"], "GW_DEPLETION_ONLY", "HYDROLOGICALLY_FEASIBLE"))
    )
    return gf, b_sub


def evaluate_future_scenarios(
    grid_base: pd.DataFrame,
    basins_base: pd.DataFrame,
    salt_baseline: pd.DataFrame,
    ecw_star_use_dsm: float,
    ece90_ref_sites: float,
    mu_wue_sodic: float,
    mu_wue_saline: float,
    et_ratio_obs: float,
) -> Tuple[List[Dict[str, Any]], float]:
    """Evaluate 5-GCM multi-model medians for ssp126, ssp370, ssp585 at 2050 & 2080, plus SSP5-8.5 2050 cell agreement."""
    if not (WG_3B_DIR.is_dir() and LSM_NC.is_file()):
        if WG_3B_DIR.exists() or LSM_NC.exists():
            raise FileNotFoundError("Both future WaterGAP files and its land mask are required")
        print("Level 1 uses the archived future ensemble. Level 2 recomputes it from monthly forcings.")
        reference = json.loads((TABLES / OUT_SUMMARY.name).read_text())
        return reference["future_zoning_2050_2080"], reference["ssp585_2050_mean_zone3_gcm_agreement_frac"]
    with xr.open_dataset(LSM_NC) as dl:
        lsm = np.squeeze(dl["mask"].values) == 1.0

    gcms = climate_models(WG_3B_DIR)
    r_idx = grid_base["r"].values
    c_idx = grid_base["c"].values

    results: List[Dict[str, Any]] = []
    ssp585_2050_cell_rows: List[pd.DataFrame] = []
    # Baseline rows
    ssp585_2050_cell_rows.append(pd.DataFrame({
        "r": salt_baseline["r"].values,
        "c": salt_baseline["c"].values,
        "gcm": "baseline_2020",
        "zone": salt_baseline["management_zone_baseline"].values,
    }))

    for ssp in ["ssp126", "ssp370", "ssp585"]:
        ece_map = {"2050": "hassani_ssp245_2050", "2080": "hassani_ssp245_2080"} if ssp == "ssp126" else {"2050": "hassani_ssp585_2050", "2080": "hassani_ssp585_2080"}
        for h_name, (y0, y1) in [("2050", (2040, 2069)), ("2080", (2070, 2099))]:
            q_list, p_list, a_list, gw_list, qr_list, slp_list = [], [], [], [], [], []
            gcm_loaded = []
            for gcm in gcms:
                gdir = WG_3B_DIR / gcm
                f_q = gdir / f"watergap2-2e_{gcm}_w5e5_{ssp}_2015soc-from-histsoc_default_qtot_global_monthly_2015_2100.nc"
                f_p = gdir / f"watergap2-2e_{gcm}_w5e5_{ssp}_2015soc-from-histsoc_default_pirrww_global_monthly_2015_2100.nc"
                f_a = gdir / f"watergap2-2e_{gcm}_w5e5_{ssp}_2015soc-from-histsoc_default_atotuse_global_monthly_2015_2100.nc"
                f_gw = gdir / f"watergap2-2e_{gcm}_w5e5_{ssp}_2015soc-from-histsoc_default_atotusegw_global_monthly_2015_2100.nc"
                f_qr = gdir / f"watergap2-2e_{gcm}_w5e5_{ssp}_2015soc-from-histsoc_default_qr_global_monthly_2015_2100.nc"
                f_gws = gdir / f"watergap2-2e_{gcm}_w5e5_{ssp}_2015soc-from-histsoc_default_groundwstor_global_monthly_2015_2100.nc"
                if not all(f.is_file() for f in [f_q, f_p, f_a, f_gw, f_qr, f_gws]):
                    raise FileNotFoundError(f"Missing WaterGAP ISIMIP3b file for {gcm} {ssp}")
                q_arr = compute_annual_flux_mm_yr(f_q, "qtot", y0, y1, lsm)
                p_arr = compute_annual_flux_mm_yr(f_p, "pirrww", y0, y1, lsm)
                a_arr = compute_annual_flux_mm_yr(f_a, "atotuse", y0, y1, lsm)
                gw_arr = compute_annual_flux_mm_yr(f_gw, "atotusegw", y0, y1, lsm)
                qr_arr = compute_annual_flux_mm_yr(f_qr, "qr", y0, y1, lsm)
                slp_arr = compute_gws_trend_mm_yr(f_gws, y0, y1, lsm)
                q_list.append(q_arr)
                p_list.append(p_arr)
                a_list.append(a_arr)
                gw_list.append(gw_arr)
                qr_list.append(qr_arr)
                slp_list.append(slp_arr)
                gcm_loaded.append(gcm)

                if ssp == "ssp585" and h_name == "2050":
                    gf_i, b_i = build_gcm_snapshot_grid_and_basins(
                        grid_base, basins_base,
                        q_arr[r_idx, c_idx], p_arr[r_idx, c_idx], a_arr[r_idx, c_idx],
                        gw_arr[r_idx, c_idx], qr_arr[r_idx, c_idx], slp_arr[r_idx, c_idx],
                    )
                    s_i, _, _ = evaluate_zoning_snapshot(
                        gf_i, b_i, ecw_star_use_dsm, ece90_ref_sites, mu_wue_sodic, mu_wue_saline, et_ratio_obs,
                        ece_fut_col=ece_map[h_name], scenario_tag=f"{ssp}_{h_name}_{gcm}",
                    )
                    ssp585_2050_cell_rows.append(pd.DataFrame({
                        "r": s_i["r"].values, "c": s_i["c"].values, "gcm": gcm, "zone": s_i["management_zone"].values,
                    }))

            with np.errstate(invalid="ignore"):
                q_m = np.nanmedian(np.stack(q_list, axis=0), axis=0)[r_idx, c_idx]
                p_m = np.nanmedian(np.stack(p_list, axis=0), axis=0)[r_idx, c_idx]
                a_m = np.nanmedian(np.stack(a_list, axis=0), axis=0)[r_idx, c_idx]
                gw_m = np.nanmedian(np.stack(gw_list, axis=0), axis=0)[r_idx, c_idx]
                qr_m = np.nanmedian(np.stack(qr_list, axis=0), axis=0)[r_idx, c_idx]
                slp_m = np.nanmedian(np.stack(slp_list, axis=0), axis=0)[r_idx, c_idx]

            gf, b_sub = build_gcm_snapshot_grid_and_basins(grid_base, basins_base, q_m, p_m, a_m, gw_m, qr_m, slp_m)
            _, _, snap = evaluate_zoning_snapshot(
                gf, b_sub, ecw_star_use_dsm, ece90_ref_sites, mu_wue_sodic, mu_wue_saline, et_ratio_obs,
                ece_fut_col=ece_map[h_name], scenario_tag=f"{ssp}_{h_name}",
            )
            snap["ssp"] = ssp
            snap["horizon"] = h_name
            snap["n_gcms"] = len(gcm_loaded)
            results.append(snap)
            print(f"  Future {ssp} {h_name}: dF={snap['net_brackish_fresh_saving_deltaF_km3_yr']}, H_cons={snap['hydraulic_recovery_cons_km3_yr']}, rescued={snap['paradox_basins_relieved_to_blue_feasible']}, Z3={snap['zone_partitioning'].get('Zone_III_Water_Salt_Dual_Redline', {}).get('salt_irrig_mha', 0)} Mha")

    # Compute Zone III GCM agreement fraction across the 5 SSP5-8.5 2050 GCMs
    fcz_df = pd.concat(ssp585_2050_cell_rows, ignore_index=True)
    gcm_only = fcz_df[fcz_df["gcm"] != "baseline_2020"].copy()
    gcm_only["is_z3"] = (gcm_only["zone"] == "Zone_III_Water_Salt_Dual_Redline").astype(float)
    agree = gcm_only.groupby(["r", "c"], as_index=False)["is_z3"].mean().rename(columns={"is_z3": "zone3_gcm_agreement_frac"})
    fcz_df = fcz_df.merge(agree, on=["r", "c"], how="left")
    OUT_FUTURE_CELL_ZONES.parent.mkdir(parents=True, exist_ok=True)
    fcz_df.to_parquet(OUT_FUTURE_CELL_ZONES, index=False)
    mean_agree = float(agree["zone3_gcm_agreement_frac"].mean())
    return results, round(mean_agree, 4)


def main() -> None:
    grid = pd.read_parquet(P1_GRID)
    if PUB_VAL_GRID.is_file():
        pub_v = pd.read_parquet(PUB_VAL_GRID)
        dup_cols = [c for c in pub_v.columns if c in grid.columns and c not in {"r", "c"}]
        pub_v = pub_v.drop(columns=dup_cols, errors="ignore")
        grid = grid.merge(pub_v, on=["r", "c"], how="left")
    basins = pd.read_csv(P1_BASIN, low_memory=False)
    p2a = json.loads(P2A_JSON.read_text(encoding="utf-8"))
    p2b = json.loads(P2B_JSON.read_text(encoding="utf-8"))

    # Strict parameter validation (raise immediately if missing!)
    try:
        ece90_ref_sites = float(p2b["ece90_ref_sites"])
        ecw_star_cont = float(p2b["dose_response_absolute"]["continuous"]["ecw_star_90pct_dsm"])
        ecw_star_cyc = float(p2b["dose_response_absolute"]["cyclic"]["ecw_star_90pct_dsm"])
        mu_wue_sodic = float(p2a["wue_by_sodicity"]["sodic"]["mu_lnrr"])
        mu_wue_saline = float(p2a["wue_by_sodicity"]["saline"]["mu_lnrr"])
        ci_wue_sodic = p2a["wue_by_sodicity"]["sodic"]["ci95"]
        ci_wue_saline = p2a["wue_by_sodicity"]["saline"]["ci95"]
        et_ratio_obs = float(p2a["comeasured_et_ratio"]["et_ratio_median"])
        et_ratio_ci95 = p2a["comeasured_et_ratio"]["et_ratio_median_ci95"]
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"Missing required Arm A/Arm B empirical parameter in JSON: {exc}") from exc

    rho_baseline = 0.75
    # Baseline snapshot (hassani_ssp245_2020 only, continuous ECw*90 = 3.547 dS/m)
    salt, basins_out, base_snap = evaluate_zoning_snapshot(
        grid, basins, ecw_star_cont, ece90_ref_sites, mu_wue_sodic, mu_wue_saline, et_ratio_obs,
        ece_fut_col="hassani_ssp245_2020", scenario_tag="baseline_2020", rho_hyd=rho_baseline,
    )
    salt = salt.rename(columns={"management_zone": "management_zone_baseline"})

    # Propagate H_cons & H_obs 95% CIs
    _, _, snap_h_low = evaluate_zoning_snapshot(
        grid, basins, ecw_star_cont, ece90_ref_sites, float(ci_wue_sodic[0]), float(ci_wue_saline[0]), float(et_ratio_ci95[1]),
        ece_fut_col="hassani_ssp245_2020", scenario_tag="ci_low", rho_hyd=rho_baseline,
    )
    _, _, snap_h_high = evaluate_zoning_snapshot(
        grid, basins, ecw_star_cont, ece90_ref_sites, float(ci_wue_sodic[1]), float(ci_wue_saline[1]), float(et_ratio_ci95[0]),
        ece_fut_col="hassani_ssp245_2020", scenario_tag="ci_high", rho_hyd=rho_baseline,
    )

    z_cols = [
        "r", "c", "ecw_star_cell_dsm", "f_blend_cap", "f_blend_realized", "ec_mix_dsm",
        "gross_brack_sub_km3_yr", "brack_sub_from_aquifer_gw_km3_yr", "brack_sub_from_return_flow_km3_yr",
        "leaching_penalty_P_km3_yr", "net_brackish_fresh_saving_deltaF_km3_yr", "delta_vleach_km3_yr",
        "hydraulic_saving_cons_km3_yr", "hydraulic_saving_wd_km3_yr", "hydraulic_saving_obs_et_km3_yr",
        "net_fresh_withdrawal_relief_km3_yr", "basin_net_depletion_relief_km3_yr",
        "scarcity_relief_km3_weq", "s_in_pre_mt_yr", "s_in_post_mt_yr", "delta_s_mt_yr",
        "salt_debt_pre_mt_yr", "salt_debt_post_mt_yr",
        "blue_water_still_exceeded_post", "blue_water_still_exceeded_actual", "is_hyper_sal_arid", "gw_still_depleted_post",
        "management_zone_baseline", "risk_decomposition_class",
    ]
    grid_out = grid.merge(salt[z_cols], on=["r", "c"], how="left")
    OUT_GRID.parent.mkdir(parents=True, exist_ok=True)
    grid_out.to_parquet(OUT_GRID, index=False)

    # Multi-parameter sensitivity sweep
    sens_specs = [
        ("Baseline_r35_k20_rho75_cont", 0.35, 0.20, 0.75, ecw_star_cont, "drainage_formula", True, True, False, False),
        ("ReturnFlow_r20", 0.20, 0.20, 0.75, ecw_star_cont, "drainage_formula", True, True, False, False),
        ("ReturnFlow_r50", 0.50, 0.20, 0.75, ecw_star_cont, "drainage_formula", True, True, False, False),
        ("ShallowGW_kappa10", 0.35, 0.10, 0.75, ecw_star_cont, "drainage_formula", True, True, False, False),
        ("ShallowGW_kappa40", 0.35, 0.40, 0.75, ecw_star_cont, "drainage_formula", True, True, False, False),
        ("HydraulicRealization_rho50", 0.35, 0.20, 0.50, ecw_star_cont, "drainage_formula", True, True, False, False),
        ("HydraulicRealization_rho100", 0.35, 0.20, 1.00, ecw_star_cont, "drainage_formula", True, True, False, False),
        ("ManagedCyclic_Unadj_ECw634", 0.35, 0.20, 0.75, ecw_star_cyc, "drainage_formula", True, True, False, False),
        ("EC_ret_2xECf", 0.35, 0.20, 0.75, ecw_star_cont, "double_ecf", True, True, False, False),
        ("No_GW_Closure_Credit", 0.35, 0.20, 0.75, ecw_star_cont, "drainage_formula", False, True, False, False),
        ("WHYMAP_Aquifer_Prior", 0.35, 0.20, 0.75, ecw_star_cont, "drainage_formula", True, True, True, False),
        ("DynQual_WHYMAP_Prior", 0.35, 0.20, 0.75, ecw_star_cont, "drainage_formula", True, True, True, True),
    ]
    sens_rows = []
    for tag, rf, kp, rh, ec_th, ec_rm, cg_c, w_cap, w_pri, dq_pri in sens_specs:
        _, _, sn = evaluate_zoning_snapshot(
            grid, basins, ec_th, ece90_ref_sites, mu_wue_sodic, mu_wue_saline, et_ratio_obs,
            ece_fut_col="hassani_ssp245_2020", scenario_tag=tag,
            ret_capture_frac=rf, kappa_gw=kp, rho_hyd=rh, ec_ret_mode=ec_rm,
            count_gw_in_closure=cg_c, use_whymap_cap=w_cap, whymap_aquifer_prior=w_pri, dynqual_aquifer_prior=dq_pri,
        )
        zp = sn["zone_partitioning"]
        sens_rows.append({
            "regime": tag,
            "ret_capture_frac": rf,
            "kappa_gw": kp,
            "rho_hyd": rh,
            "ecw_star_used_dsm": ec_th,
            "ec_ret_mode": ec_rm,
            "count_gw_in_closure": cg_c,
            "whymap_aquifer_prior": w_pri,
            "dynqual_aquifer_prior": dq_pri,
            "B_gross_brackish_km3_yr": sn["gross_brackish_substitution_km3_yr"],
            "P_leaching_penalty_km3_yr": sn["leaching_penalty_P_km3_yr"],
            "deltaF_net_brackish_saving_km3_yr": sn["net_brackish_fresh_saving_deltaF_km3_yr"],
            "H_cons_km3_yr": sn["hydraulic_recovery_cons_km3_yr"],
            "H_wd_km3_yr": sn["hydraulic_recovery_wd_km3_yr"],
            "basin_net_depletion_relief_km3_yr": sn["basin_net_depletion_relief_km3_yr"],
            "basin_net_depletion_relief_actual_km3_yr": sn["basin_net_depletion_relief_actual_km3_yr"],
            "relieved_basins": sn["paradox_basins_relieved_to_blue_feasible"],
            "still_exceeded_basins": sn["paradox_still_exceeded_post_basins"],
            "relieved_basins_actual": sn["paradox_basins_relieved_actual"],
            "still_exceeded_basins_actual": sn["paradox_still_exceeded_actual_basins"],
            "zone1_mha": zp.get("Zone_I_Safe_Brackish_Substitution_Window", {}).get("salt_irrig_mha", 0.0),
            "zone2_mha": zp.get("Zone_II_Hydraulic_Recovery_Priority", {}).get("salt_irrig_mha", 0.0),
            "zone3_mha": zp.get("Zone_III_Water_Salt_Dual_Redline", {}).get("salt_irrig_mha", 0.0),
            "salt_debt_post_mt_yr": sn["salt_debt_post_mt_yr"],
        })
    pd.DataFrame(sens_rows).to_csv(OUT_SUPPLY_SENS, index=False)

    # Soil-mechanistic prescription breakdown across Zone I, II, III
    soil_zone_breakdown = {}
    if "hwsd2_sodic_frac" in salt.columns:
        for z_nm, grp in salt.groupby("management_zone_baseline"):
            s_ha = float(grp["salt_irrig_ha"].sum() / 1e6)
            sod_ha = float(grp.loc[grp["hwsd2_sodic_frac"].fillna(0) >= 0.10, "salt_irrig_ha"].sum() / 1e6)
            hclay_ha = float(grp.loc[grp["sg_clay_0_30_pct"].fillna(0) >= 35.0, "salt_irrig_ha"].sum() / 1e6)
            pdr_ha = float(grp.loc[grp["hwsd2_poor_drain_frac"].fillna(0) >= 0.30, "salt_irrig_ha"].sum() / 1e6)
            wsal_ha = float(grp.loc[grp["whymap_saline_gw_gt5gl"].fillna(False), "salt_irrig_ha"].sum() / 1e6)
            soil_zone_breakdown[z_nm] = {
                "total_mha": round(s_ha, 2),
                "sodic_ca_flocculation_mha": round(sod_ha, 2),
                "non_sodic_mulched_drip_mha": round(max(s_ha - sod_ha, 0.0), 2),
                "heavy_clay_ge35pct_mha": round(hclay_ha, 2),
                "poor_drainage_tile_priority_mha": round(pdr_ha, 2),
                "whymap_saline_gw_gt5gl_mha": round(wsal_ha, 2),
            }

    OUT_BASIN.parent.mkdir(parents=True, exist_ok=True)
    basins_out.to_csv(OUT_BASIN, index=False)

    # Country accounting (includes deltaF, H_wd, H_cons, Zone III share)
    if NE_COUNTRIES_SHP.is_file():
        ne = gpd.read_file(NE_COUNTRIES_SHP)[["NAME", "CONTINENT", "geometry"]]
        pts = gpd.GeoDataFrame(
            {"r": salt["r"], "c": salt["c"]},
            geometry=[Point(xy) for xy in zip(salt["lon"], salt["lat"])],
            crs="EPSG:4326",
        )
        s_cntry = gpd.sjoin(pts, ne, how="left", predicate="within").drop_duplicates(["r", "c"])
        salt_c = salt.merge(s_cntry[["r", "c", "NAME", "CONTINENT"]], on=["r", "c"], how="left")
        salt_c["country"] = salt_c["NAME"].fillna("Other_Coastal_or_Unmatched")
        ctry_rows: List[Dict[str, Any]] = []
        for cn, grp in salt_c.groupby("country"):
            area_mha = float(grp["salt_irrig_ha"].sum() / 1e6)
            if area_mha < 0.05:
                continue
            z3_mha = float(grp.loc[grp["management_zone_baseline"] == "Zone_III_Water_Salt_Dual_Redline", "salt_irrig_ha"].sum() / 1e6)
            ctry_rows.append({
                "country": cn,
                "continent": str(grp["CONTINENT"].dropna().iloc[0]) if grp["CONTINENT"].notna().any() else "Unknown",
                "salt_irrig_mha": round(area_mha, 3),
                "zone1_safe_brackish_mha": round(float(grp.loc[grp["management_zone_baseline"] == "Zone_I_Safe_Brackish_Substitution_Window", "salt_irrig_ha"].sum() / 1e6), 3),
                "zone2_hydraulic_mha": round(float(grp.loc[grp["management_zone_baseline"] == "Zone_II_Hydraulic_Recovery_Priority", "salt_irrig_ha"].sum() / 1e6), 3),
                "zone3_dual_redline_mha": round(z3_mha, 3),
                "zone3_share_pct": round(z3_mha / max(area_mha, 1e-6) * 100.0, 1),
                "gross_brackish_substitution_km3_yr": round(float(grp["gross_brack_sub_km3_yr"].sum()), 2),
                "leaching_penalty_P_km3_yr": round(float(grp["leaching_penalty_P_km3_yr"].sum()), 2),
                "deltaF_net_brackish_saving_km3_yr": round(float(grp["net_brackish_fresh_saving_deltaF_km3_yr"].sum()), 2),
                "hydraulic_saving_wd_km3_yr": round(float(grp["hydraulic_saving_wd_km3_yr"].sum()), 2),
                "hydraulic_saving_cons_km3_yr": round(float(grp["hydraulic_saving_cons_km3_yr"].sum()), 2),
                "net_fresh_withdrawal_relief_km3_yr": round(float(grp["net_fresh_withdrawal_relief_km3_yr"].sum()), 2),
                "aware_scarcity_relief_km3_weq": round(float(grp["scarcity_relief_km3_weq"].sum()), 1),
            })
        ctry_df = pd.DataFrame(ctry_rows).sort_values("net_fresh_withdrawal_relief_km3_yr", ascending=False)
        if JAEGER_CSV.is_file():
            jg = pd.read_csv(JAEGER_CSV)
            jg["country_std"] = jg["Country"].replace({
                "United States": "United States of America",
                "Iran (Islamic Republic of)": "Iran",
                "Viet Nam": "Vietnam",
            })
            ctry_df = ctry_df.merge(
                jg[["country_std", "shr_AEI_surface", "shr_AEI_sprinkler", "shr_AEI_drip"]],
                left_on="country", right_on="country_std", how="left",
            ).drop(columns=["country_std"])
        else:
            cols = ["country", "shr_AEI_surface", "shr_AEI_sprinkler", "shr_AEI_drip"]
            shares = pd.read_csv(TABLES / OUT_COUNTRY.name, usecols=cols)
            ctry_df = ctry_df.merge(shares, on="country", how="left", validate="one_to_one")
        ctry_df.to_csv(OUT_COUNTRY, index=False)
        top_countries = ctry_df.head(12).to_dict(orient="records")
    else:
        top_countries = []

    # Future 2050 & 2080 Three-Zone Shifts across ISIMIP3b 5-GCM ensembles
    fut_snaps, mean_z3_agree = evaluate_future_scenarios(
        grid, basins, salt, ecw_star_cont, ece90_ref_sites, mu_wue_sodic, mu_wue_saline, et_ratio_obs,
    )
    if fut_snaps:
        fut_flat = []
        for fs in fut_snaps:
            zp = fs["zone_partitioning"]
            rb = fs.get("risk_intersection_breakdown_mha", {})
            fut_flat.append({
                "scenario": fs["scenario"],
                "ssp": fs["ssp"],
                "horizon": fs["horizon"],
                "n_gcms": fs["n_gcms"],
                "gross_brackish_substitution_km3_yr": fs["gross_brackish_substitution_km3_yr"],
                "leaching_penalty_P_km3_yr": fs["leaching_penalty_P_km3_yr"],
                "deltaF_net_brackish_saving_km3_yr": fs["net_brackish_fresh_saving_deltaF_km3_yr"],
                "hydraulic_recovery_cons_km3_yr": fs["hydraulic_recovery_cons_km3_yr"],
                "hydraulic_recovery_wd_km3_yr": fs["hydraulic_recovery_wd_km3_yr"],
                "net_fresh_withdrawal_relief_km3_yr": fs["net_fresh_withdrawal_relief_km3_yr"],
                "basin_net_depletion_relief_km3_yr": fs["basin_net_depletion_relief_km3_yr"],
                "aware_scarcity_relief_km3_weq": fs["aware_scarcity_relief_km3_weq"],
                "paradox_exceeded_basins": fs["paradox_exceeded_basins"],
                "paradox_basins_relieved": fs["paradox_basins_relieved_to_blue_feasible"],
                "paradox_still_exceeded_post_basins": fs["paradox_still_exceeded_post_basins"],
                "zone1_mha": zp.get("Zone_I_Safe_Brackish_Substitution_Window", {}).get("salt_irrig_mha", 0.0),
                "zone1_local_potential_gw_constrained_mha": rb.get("zone1_local_potential_gw_constrained_mha", 0.0),
                "zone2_mha": zp.get("Zone_II_Hydraulic_Recovery_Priority", {}).get("salt_irrig_mha", 0.0),
                "zone3_mha": zp.get("Zone_III_Water_Salt_Dual_Redline", {}).get("salt_irrig_mha", 0.0),
                "zone3_bw_only_mha": rb.get("zone3_bw_only_mha", 0.0),
                "zone3_sal_arid_only_mha": rb.get("zone3_sal_arid_only_mha", 0.0),
                "zone3_dual_bw_and_sal_arid_mha": rb.get("zone3_dual_bw_and_sal_arid_mha", 0.0),
                "sal_arid_and_gw_depleted_mha": rb.get("sal_arid_and_gw_depleted_mha", 0.0),
                "salt_debt_post_mt_yr": fs["salt_debt_post_mt_yr"],
            })
        pd.DataFrame(fut_flat).to_csv(OUT_FUTURE, index=False)

    aoa = compute_aoa_coverage(grid)

    summary = {
        "script": "pillar3_safe_windows_and_zoning.py",
        "empirical_parameters_used": {
            "ecw_star_baseline_used_dsm": ecw_star_cont,
            "ecw_star_managed_cyclic_used_dsm": ecw_star_cyc,
            "ecw_star_managed_cyclic_note": "Unadjusted cyclic-subset Maas-Hoffman ECw*90 used strictly as an upper-bound managed sensitivity (adjusted cyclic advantage is non-robust).",
            "ece90_ref_sites_used": ece90_ref_sites,
            "mu_wue_sodic": mu_wue_sodic,
            "mu_wue_saline": mu_wue_saline,
            "rho_realization": rho_baseline,
            "et_ratio_observed": et_ratio_obs,
            "et_ratio_observed_ci95": et_ratio_ci95,
        },
        "water_accounting_three_tier": {
            "withdrawal_tier_km3_yr": {
                "B_gross_brackish_substitution": base_snap["gross_brackish_substitution_km3_yr"],
                "B_gw_shallow_aquifer": base_snap["brackish_from_aquifer_gw_km3_yr"],
                "B_ret_return_flow": base_snap["brackish_from_return_flow_km3_yr"],
                "P_leaching_penalty": base_snap["leaching_penalty_P_km3_yr"],
                "deltaF_net_brackish_fresh_withdrawal_saving": base_snap["net_brackish_fresh_saving_deltaF_km3_yr"],
                "H_wd_conditional_equal_yield_hydraulic_withdrawal_saving": base_snap["hydraulic_recovery_wd_km3_yr"],
                "net_fresh_withdrawal_relief": base_snap["net_fresh_withdrawal_relief_km3_yr"],
            },
            "consumption_tier_km3_yr": {
                "H_cons_conditional_equal_yield_saving": base_snap["hydraulic_recovery_cons_km3_yr"],
                "H_cons_ci95": [snap_h_low["hydraulic_recovery_cons_km3_yr"], snap_h_high["hydraulic_recovery_cons_km3_yr"]],
                "H_obs_observed_et_saving": base_snap["hydraulic_observed_et_saving_km3_yr"],
                "H_obs_ci95": [snap_h_low["hydraulic_observed_et_saving_km3_yr"], snap_h_high["hydraulic_observed_et_saving_km3_yr"]],
                "B_gw_consumptive_credit": base_snap["brackish_from_aquifer_gw_km3_yr"],
                "delta_vleach_field_leaching_increment": base_snap["delta_vleach_km3_yr"],
                "return_flow_basin_credit": 0.0,
            },
            "basin_net_depletion_change_km3_yr": base_snap["basin_net_depletion_relief_km3_yr"],
            "basin_net_depletion_change_ci95": [snap_h_low["basin_net_depletion_relief_km3_yr"], snap_h_high["basin_net_depletion_relief_km3_yr"]],
            "basin_net_depletion_change_actual_et_km3_yr": base_snap["basin_net_depletion_relief_actual_km3_yr"],
            "basin_net_depletion_change_actual_et_ci95": [snap_h_low["basin_net_depletion_relief_actual_km3_yr"], snap_h_high["basin_net_depletion_relief_actual_km3_yr"]],
            "aware_scarcity_relief_km3_weq": base_snap["aware_scarcity_relief_km3_weq"],
            "aware_scarcity_relief_actual_et_km3_weq": base_snap["aware_scarcity_relief_actual_km3_weq"],
        },
        "dual_scenario_basin_closure": {
            "baseline_amb": {
                "exceeded_basins": base_snap["paradox_exceeded_basins"],
                "exceeded_mha": base_snap["paradox_exceeded_mha"],
                "salt_debt_pre_mt_yr": base_snap["salt_debt_pre_mt_yr"],
            },
            "actual_management_observed_et": {
                "et_ratio_used": et_ratio_obs,
                "H_obs_km3_yr": base_snap["hydraulic_observed_et_saving_km3_yr"],
                "H_obs_ci95": [snap_h_low["hydraulic_observed_et_saving_km3_yr"], snap_h_high["hydraulic_observed_et_saving_km3_yr"]],
                "net_basin_depletion_relief_km3_yr": base_snap["basin_net_depletion_relief_actual_km3_yr"],
                "net_basin_depletion_relief_ci95": [snap_h_low["basin_net_depletion_relief_actual_km3_yr"], snap_h_high["basin_net_depletion_relief_actual_km3_yr"]],
                "aware_scarcity_relief_km3_weq": base_snap["aware_scarcity_relief_actual_km3_weq"],
                "relieved_basins": base_snap["paradox_basins_relieved_actual"],
                "relieved_mha": base_snap["paradox_relieved_actual_mha"],
                "newly_exceeded_basins": base_snap["paradox_newly_exceeded_actual_basins"],
                "newly_exceeded_mha": base_snap["paradox_newly_exceeded_actual_mha"],
                "still_exceeded_basins": base_snap["paradox_still_exceeded_actual_basins"],
                "still_exceeded_mha": base_snap["paradox_still_exceeded_actual_mha"],
                "salt_debt_post_mt_yr": base_snap["salt_debt_post_actual_mt_yr"],
            },
            "equal_production_water_capped": {
                "H_cons_km3_yr": base_snap["hydraulic_recovery_cons_km3_yr"],
                "H_cons_ci95": [snap_h_low["hydraulic_recovery_cons_km3_yr"], snap_h_high["hydraulic_recovery_cons_km3_yr"]],
                "net_basin_depletion_relief_km3_yr": base_snap["basin_net_depletion_relief_km3_yr"],
                "net_basin_depletion_relief_ci95": [snap_h_low["basin_net_depletion_relief_km3_yr"], snap_h_high["basin_net_depletion_relief_km3_yr"]],
                "aware_scarcity_relief_km3_weq": base_snap["aware_scarcity_relief_km3_weq"],
                "relieved_basins": base_snap["paradox_basins_relieved_to_blue_feasible"],
                "relieved_mha": base_snap["paradox_relieved_mha"],
                "newly_exceeded_basins": base_snap["paradox_newly_exceeded_post_basins"],
                "newly_exceeded_mha": base_snap["paradox_newly_exceeded_post_mha"],
                "still_exceeded_basins": base_snap["paradox_still_exceeded_post_basins"],
                "still_exceeded_mha": base_snap["paradox_still_exceeded_post_mha"],
                "salt_debt_post_mt_yr": base_snap["salt_debt_post_mt_yr"],
            },
        },
        "global_salt_affected_croplands_savings": {
            "salt_irrig_mha": base_snap["salt_irrig_mha"],
            "salt_pirrww_km3_yr": base_snap["salt_pirrww_km3_yr"],
            "salt_pirruse_km3_yr": base_snap["salt_pirruse_km3_yr"],
            "gross_brackish_substitution_km3_yr": base_snap["gross_brackish_substitution_km3_yr"],
            "brackish_from_aquifer_gw_km3_yr": base_snap["brackish_from_aquifer_gw_km3_yr"],
            "brackish_from_return_flow_km3_yr": base_snap["brackish_from_return_flow_km3_yr"],
            "leaching_penalty_P_km3_yr": base_snap["leaching_penalty_P_km3_yr"],
            "net_brackish_fresh_saving_deltaF_km3_yr": base_snap["net_brackish_fresh_saving_deltaF_km3_yr"],
            "delta_vleach_km3_yr": base_snap["delta_vleach_km3_yr"],
            "hydraulic_recovery_cons_km3_yr": base_snap["hydraulic_recovery_cons_km3_yr"],
            "hydraulic_recovery_wd_km3_yr": base_snap["hydraulic_recovery_wd_km3_yr"],
            "hydraulic_observed_et_saving_km3_yr": base_snap["hydraulic_observed_et_saving_km3_yr"],
            "net_fresh_withdrawal_relief_km3_yr": base_snap["net_fresh_withdrawal_relief_km3_yr"],
            "basin_net_depletion_relief_km3_yr": base_snap["basin_net_depletion_relief_km3_yr"],
            "basin_net_depletion_relief_actual_km3_yr": base_snap["basin_net_depletion_relief_actual_km3_yr"],
            "return_flow_basin_credit_km3_yr": 0.0,
            "aware_scarcity_relief_km3_weq": base_snap["aware_scarcity_relief_km3_weq"],
            "aware_scarcity_relief_actual_km3_weq": base_snap["aware_scarcity_relief_actual_km3_weq"],
            "delta_s_salt_import_mt_yr": base_snap["delta_s_salt_import_mt_yr"],
            "delta_s_field_recycled_gross_mt_yr": base_snap["delta_s_field_recycled_gross_mt_yr"],
            "delta_s_basin_gw_gross_mt_yr": base_snap["delta_s_basin_gw_gross_mt_yr"],
            "delta_s_fresh_displaced_mt_yr": base_snap["delta_s_fresh_displaced_mt_yr"],
            "delta_s_field_recycled_mt_yr": base_snap["delta_s_field_recycled_mt_yr"],
            "delta_s_basin_new_mt_yr": base_snap["delta_s_basin_new_mt_yr"],
            "salt_debt_pre_mt_yr": base_snap["salt_debt_pre_mt_yr"],
            "salt_debt_post_mt_yr": base_snap["salt_debt_post_mt_yr"],
            "salt_debt_post_actual_mt_yr": base_snap["salt_debt_post_actual_mt_yr"],
            "paradox_exceeded_basins_baseline": base_snap["paradox_exceeded_basins"],
            "paradox_basins_relieved_to_blue_feasible": base_snap["paradox_basins_relieved_to_blue_feasible"],
            "paradox_still_exceeded_post_basins": base_snap["paradox_still_exceeded_post_basins"],
            "paradox_basins_relieved_actual": base_snap["paradox_basins_relieved_actual"],
            "paradox_still_exceeded_actual_basins": base_snap["paradox_still_exceeded_actual_basins"],
            "gw_still_depleted_post_basins": base_snap["gw_still_depleted_post_basins"],
            "gw_still_depleted_post_mha": base_snap["gw_still_depleted_post_mha"],
        },
        "supply_sensitivity_sweep": sens_rows,
        "soil_mechanistic_zone_breakdown": soil_zone_breakdown,
        "three_zone_partitioning": base_snap["zone_partitioning"],
        "risk_intersection_breakdown_mha": base_snap["risk_intersection_breakdown_mha"],
        "risk_intersection_actual_et_mha": base_snap["risk_intersection_actual_et_mha"],
        "top12_country_hotspots": top_countries,
        "future_zoning_2050_2080": fut_snaps,
        "ssp585_2050_mean_zone3_gcm_agreement_frac": mean_z3_agree,
        "aoa_applicability_assessment": aoa,
    }
    with OUT_SUMMARY.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Saved Pillar 3 summary -> {OUT_SUMMARY}")


if __name__ == "__main__":
    prepare_outputs()
    main()
