#!/usr/bin/env python3
"""
Pillar 1B: Hydrological Closure Decomposition, Continuous V_leach-ECw Curve,
Salt-Debt Accounting (Mt/yr & indicative rootzone dECe/yr), and DynQual Consistency Check.

Strictly read-only on Pillar 1 outputs (pillar1_grid05_baseline_current.parquet,
pillar1_basin_leaching_closure_current.csv) and DynQual v1.0 annual salinity NetCDF.
"""

from __future__ import annotations

from config import EXTERNAL_RAW_DATA_DIR, TABLES, OUT_TABLES, table_input, grid_input, prepare_outputs

import json
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats
import xarray as xr

P1_GRID = grid_input("pillar1_grid05_baseline_current.parquet")
P1_BASIN = table_input("pillar1_basin_leaching_closure_current.csv")
DYNQUAL_ANN_NC = EXTERNAL_RAW_DATA_DIR / "DynQual_v1/salinity_annualAvg_1980_2019_30min.nc"

OUT_JSON = OUT_TABLES / "pillar1b_salt_debt_closure_sensitivity_current.json"
OUT_BASIN_CSV = OUT_TABLES / "pillar1b_basin_salt_debt_current.csv"


def ec_to_tds_gl(ec_dsm: np.ndarray) -> np.ndarray:
    """FAO-29 conversion: TDS (g/L) = 0.64 * EC (EC <= 5 dS/m) or 0.80 * EC (EC > 5 dS/m)."""
    return np.where(ec_dsm <= 5.0, 0.64 * ec_dsm, 0.80 * ec_dsm)


def compute_vleach_uniform(
    ec_val: float,
    c_mm: np.ndarray,
    peff_mm: np.ndarray,
    ece_max: np.ndarray,
    area_km2: np.ndarray,
    salt_mask: np.ndarray,
) -> tuple[float, float]:
    """Compute global all-irrigated and salt-affected V_leach (km3/yr) using Pillar 1 Rhoades formula."""
    ec_aw = np.where((c_mm + peff_mm) > 0, (c_mm * ec_val) / np.maximum(c_mm + peff_mm, 1e-6), np.nan)
    denom = 5.0 * ece_max - ec_aw
    unleachable = np.isfinite(ec_aw) & np.isfinite(ece_max) & (denom <= 0.1 * ec_aw)
    lr = np.where(
        np.isfinite(ec_aw) & np.isfinite(ece_max),
        np.where(unleachable, 0.80, np.clip(ec_aw / np.maximum(denom, 1e-6), 0.0, 0.80)),
        np.nan,
    )
    v_irr_km3 = c_mm * 1e-6 * area_km2
    vl = np.where(np.isfinite(lr), v_irr_km3 * (lr / (1.0 - lr)), 0.0)
    return float(np.sum(vl)), float(np.sum(vl[salt_mask]))


def main() -> None:
    g = pd.read_parquet(P1_GRID)
    b = pd.read_csv(P1_BASIN, low_memory=False)
    sb = b[b["salt_irrig_ha"] >= 10000.0].copy()

    # 1. Closure decomposition (leaching-flipped basins, inflow-aware sensitivity, salt-area sensitivity)
    atot_pos = np.maximum(sb["atotuse_km3_yr"].values, 0.0)
    flip_amb = int(((atot_pos <= sb["sbwa_efr60_km3_yr"].values) & sb["blue_water_exceeded_amb"].values).sum())
    flip_ec3 = int(((atot_pos <= sb["sbwa_efr60_km3_yr"].values) & sb["blue_water_exceeded_ec3"].values).sum())
    avail_inflow = np.maximum(sb["qtot_km3_yr"].values, sb["basin_max_dis_km3_yr"].values + atot_pos)
    sbwa_inflow = 0.40 * avail_inflow
    exc_inflow = (atot_pos + sb["vleach_amb_km3_yr"].values) > sbwa_inflow
    flip_inflow = int(np.sum(exc_inflow & (atot_pos <= sbwa_inflow)))

    # Salt-affected area sensitivity
    area_any_salt_cell = float(g.loc[g["salt_affected"], "spam_irrig_ha"].sum() / 1e6)
    area_glosis_weighted = float((g.loc[g["salt_affected"], "spam_irrig_ha"] * g.loc[g["salt_affected"], "glosis_sal_frac"].fillna(0.0)).sum() / 1e6)
    strict_mask = g["irrig_active"] & ((g["glosis_sal_frac"].fillna(0.0) >= 0.25) | (g["wang_ece_2017_dsm"].fillna(0.0) >= 4.0))
    area_strict = float(g.loc[strict_mask, "spam_irrig_ha"].sum() / 1e6)

    closure_decomp = {
        "major_salt_basins_ge10kha": int(len(sb)),
        "baseline_blue_water_exceeded_basins": int(sb["blue_water_exceeded_amb"].sum()),
        "leaching_flipped_basins_ambient": flip_amb,
        "leaching_flipped_basins_ec3": flip_ec3,
        "ec3_blue_water_exceeded_basins": int(sb["blue_water_exceeded_ec3"].sum()),
        "inflow_aware_exceeded_basins": int(exc_inflow.sum()),
        "inflow_aware_exceeded_mha": round(float(sb.loc[exc_inflow, "salt_irrig_ha"].sum() / 1e6), 2),
        "inflow_aware_leaching_flipped": flip_inflow,
        "efr_sensitivity_exceeded_basins": {
            "efr_50pct": int(((atot_pos + sb["vleach_amb_km3_yr"].values) > sb["sbwa_efr50_km3_yr"].values).sum()),
            "efr_60pct": int(sb["blue_water_exceeded_amb"].sum()),
            "efr_80pct": int(((atot_pos + sb["vleach_amb_km3_yr"].values) > sb["sbwa_efr80_km3_yr"].values).sum()),
        },
        "salt_area_definitions_mha": {
            "irrigated_cropland_in_salt_affected_cells": round(area_any_salt_cell, 2),
            "strict_threshold_glosis_ge25pct_or_ece_ge4": round(area_strict, 2),
            "glosis_fraction_weighted_salt_cropland": round(area_glosis_weighted, 2),
        },
    }

    # 2. Continuous V_leach vs uniform ECw curve (0.25 to 6.0 dS/m, step 0.25) & ambient Jensen-inequality check
    irr = g[g["irrig_active"]].copy()
    c_mm = irr["pirruse_mm_yr"].values
    peff_mm = irr["peff_mm_yr"].values
    ece_max = np.where(np.isfinite(irr["crop_ece90_dsm"].values), irr["crop_ece90_dsm"].values, 3.5)
    area_km2 = irr["cell_area_km2"].values
    salt_mask = irr["salt_affected"].values

    ec_grid = np.round(np.arange(0.25, 6.01, 0.25), 2)
    curve_rows: List[Dict[str, float]] = []
    for ev in ec_grid:
        v_all, v_salt = compute_vleach_uniform(float(ev), c_mm, peff_mm, ece_max, area_km2, salt_mask)
        curve_rows.append({"ecw_uniform_dsm": float(ev), "vleach_all_irrig_km3_yr": round(v_all, 3), "vleach_salt_cells_km3_yr": round(v_salt, 3)})

    amb_all_v = float(irr["vleach_amb_km3_yr"].sum())
    amb_salt_v = float(irr.loc[salt_mask, "vleach_amb_km3_yr"].sum())
    # Linear interpolation of equivalent uniform ECw matching ambient V_leach
    ec_vals_c = np.array([r["ecw_uniform_dsm"] for r in curve_rows])
    v_all_c = np.array([r["vleach_all_irrig_km3_yr"] for r in curve_rows])
    v_salt_c = np.array([r["vleach_salt_cells_km3_yr"] for r in curve_rows])
    equiv_ec_all = float(np.interp(amb_all_v, v_all_c, ec_vals_c))
    equiv_ec_salt = float(np.interp(amb_salt_v, v_salt_c, ec_vals_c))

    # 3. Salt Debt accounting (Mt/yr & indicative dECe/yr)
    salt_cells = g[g["salt_affected"] & (g["aware_basin_idx"] >= 0)].copy()
    lr0 = np.nan_to_num(salt_cells["lr_ambient"].values, nan=0.0)
    c_km3 = salt_cells["pirruse_km3_yr"].values
    a0_km3 = c_km3 / np.maximum(1.0 - lr0, 0.05)
    tds0_gl = ec_to_tds_gl(salt_cells["ecw_ambient_dsm"].values)
    # 1 km3 * 1 g/L = 1e9 m3 * 1 kg/m3 = 1e9 kg = 1 Mt
    salt_cells["s_in_mt_yr"] = a0_km3 * tds0_gl

    # Basin-level unserved leaching fraction u_b = 1 - clip((SBWA - atotuse_pos) / V_leach, 0, 1)
    sb["avail_for_leach_km3_yr"] = sb["sbwa_efr60_km3_yr"] - atot_pos
    sb["unserved_leach_frac"] = np.where(
        sb["vleach_amb_km3_yr"] > 1e-6,
        1.0 - np.clip(sb["avail_for_leach_km3_yr"] / sb["vleach_amb_km3_yr"], 0.0, 1.0),
        np.where(atot_pos > sb["sbwa_efr60_km3_yr"].values, 1.0, 0.0),
    )
    u_map = sb.set_index("aware_basin_idx")["unserved_leach_frac"].to_dict()
    salt_cells["u_b"] = salt_cells["aware_basin_idx"].map(lambda idx: u_map.get(idx, 0.0))
    salt_cells["salt_debt_mt_yr"] = salt_cells["u_b"] * salt_cells["s_in_mt_yr"]

    b_debt = salt_cells.groupby("aware_basin_idx", as_index=False).agg(
        s_in_salt_cells_mt_yr=("s_in_mt_yr", "sum"),
        salt_debt_mt_yr=("salt_debt_mt_yr", "sum"),
    )
    sb_out = sb.merge(b_debt, on="aware_basin_idx", how="left")
    sb_out["s_in_salt_cells_mt_yr"] = sb_out["s_in_salt_cells_mt_yr"].fillna(0.0)
    sb_out["salt_debt_mt_yr"] = sb_out["salt_debt_mt_yr"].fillna(0.0)
    sb_out["salt_debt_t_ha_yr"] = (sb_out["salt_debt_mt_yr"] * 1e6) / np.maximum(sb_out["salt_irrig_ha"], 1.0)
    # Counterfactual rootzone dECe/yr if leaching curtailed to sustainable limits while irrigation continues (theta_D = 0.45 m baseline; 0.35-0.55 sensitivity)
    sb_out["counterfactual_dece_yr_if_leaching_curtailed_045m"] = (sb_out["salt_debt_t_ha_yr"] * 100.0 / (0.45 * 1000.0)) / 0.64

    # Aquifer-subsidized salt balance: GW_DEPLETION_ONLY (gw depleted but blue water not exceeded)
    gw_sub_mask = sb_out["paradox_status_amb"] == "GW_DEPLETION_ONLY"
    dual_mask = sb_out["paradox_status_amb"] == "PARADOX_INSOLVENT_BLUE_AND_GW"
    bw_only_mask = sb_out["paradox_status_amb"] == "PARADOX_BLUE_WATER_EXCEEDED"
    tot_sin_mt = float(sb_out["s_in_salt_cells_mt_yr"].sum())
    tot_debt_mt = float(sb_out["salt_debt_mt_yr"].sum())
    dual_debt_mt = float(sb_out.loc[dual_mask, "salt_debt_mt_yr"].sum())
    bw_only_debt_mt = float(sb_out.loc[bw_only_mask, "salt_debt_mt_yr"].sum())
    gw_sub_sin_mt = float(sb_out.loc[gw_sub_mask, "s_in_salt_cells_mt_yr"].sum())
    within_sbwa_sin_mt = max(tot_sin_mt - tot_debt_mt - gw_sub_sin_mt, 0.0)

    tot_salt_ha = float(sb_out["salt_irrig_ha"].sum())
    debt_ha_affected = float(sb_out.loc[sb_out["salt_debt_mt_yr"] > 0, "salt_irrig_ha"].sum())
    mean_t_ha_affected = (tot_debt_mt * 1e6) / max(debt_ha_affected, 1.0)

    # 4. DynQual v1.0 1991-2019 annual salinity OLS trend consistency check (spike-screened <= 5,000 mg/L)
    if DYNQUAL_ANN_NC.is_file():
        with xr.open_dataset(DYNQUAL_ANN_NC) as ds:
            sub_dq = ds["tds"].sel(time=slice("1991-01-01", "2019-12-31"))
            vals = sub_dq.values.astype("float64")
            if float(ds.lat.values[0]) < 0:
                vals = vals[:, ::-1, :]
            yrs = np.arange(1991, 2020, dtype=float)
            vals = np.where((vals > 0.0) & (vals <= 5000.0) & np.isfinite(vals), vals, np.nan)
            n_ok = np.sum(np.isfinite(vals), axis=0)
            t_b = np.where(np.isfinite(vals), yrs[:, None, None], np.nan)
            with np.errstate(invalid="ignore"):
                t_mu = np.nanmean(t_b, axis=0)
                y_mu = np.nanmean(vals, axis=0)
                num = np.nansum((vals - y_mu[None, :, :]) * (t_b - t_mu[None, :, :]), axis=0)
                den = np.nansum((t_b - t_mu[None, :, :]) ** 2, axis=0)
                slp = np.where((n_ok >= 20) & (den > 0.0), num / np.maximum(den, 1e-12), np.nan)

        salt_cells["dynqual_tds_slope_mgl_yr"] = slp[salt_cells["r"].values, salt_cells["c"].values]
        b_dq = salt_cells.groupby("aware_basin_idx", as_index=False).agg(
            dynqual_tds_slope_median_mgl_yr=("dynqual_tds_slope_mgl_yr", "median")
        )
    else:
        # Level 1 uses the released basin-median trend as a covariate.
        b_dq = pd.read_csv(TABLES / OUT_BASIN_CSV.name, usecols=[
            "aware_basin_idx", "dynqual_tds_slope_median_mgl_yr"])
    sb_out = sb_out.merge(b_dq, on="aware_basin_idx", how="left")
    valid_dq = sb_out.dropna(subset=["dynqual_tds_slope_median_mgl_yr"]).copy()
    debt_pos = valid_dq[valid_dq["salt_debt_mt_yr"] > 0]
    debt_zero = valid_dq[valid_dq["salt_debt_mt_yr"] <= 0]
    mw_u2, mw_p2 = stats.mannwhitneyu(debt_pos["dynqual_tds_slope_median_mgl_yr"], debt_zero["dynqual_tds_slope_median_mgl_yr"], alternative="two-sided")
    mw_u, mw_p = stats.mannwhitneyu(debt_pos["dynqual_tds_slope_median_mgl_yr"], debt_zero["dynqual_tds_slope_median_mgl_yr"], alternative="greater")
    sp_r, sp_p = stats.spearmanr(valid_dq["salt_debt_t_ha_yr"], valid_dq["dynqual_tds_slope_median_mgl_yr"])

    OUT_BASIN_CSV.parent.mkdir(parents=True, exist_ok=True)
    sb_out.to_csv(OUT_BASIN_CSV, index=False)

    summary = {
        "script": "pillar1b_salt_debt_and_closure_sensitivity.py",
        "closure_decomposition": closure_decomp,
        "continuous_vleach_ecw_curve": curve_rows,
        "ambient_vleach_jensen_inequality": {
            "ambient_vleach_all_irrig_km3_yr": round(amb_all_v, 2),
            "ambient_vleach_salt_cells_km3_yr": round(amb_salt_v, 2),
            "equivalent_uniform_ecw_all_irrig_dsm": round(equiv_ec_all, 3),
            "equivalent_uniform_ecw_salt_cells_dsm": round(equiv_ec_salt, 3),
            "explanation": "Ambient irrigation ECw is right-skewed across SW/GW mixes and LR(ECw) is strictly convex, placing ambient V_leach above uniform freshwater benchmarks.",
        },
        "salt_debt_accounting": {
            "salt_debt_definition": "salt export whose leaching volume can only be supplied beyond SBWA (EFR 60%); i.e., unsustainably funded salt-export obligation",
            "total_irrigation_salt_input_salt_cells_mt_yr": round(tot_sin_mt, 2),
            "total_salt_debt_mt_yr": round(tot_debt_mt, 2),
            "unfunded_share_of_salt_input": round(tot_debt_mt / max(tot_sin_mt, 1e-9), 4),
            "salt_debt_by_status_mt_yr": {
                "dual_insolvent": round(dual_debt_mt, 2),
                "blue_water_exceeded_only": round(bw_only_debt_mt, 2),
                "gw_subsidized_salt_input": round(gw_sub_sin_mt, 2),
                "within_sbwa_salt_input": round(within_sbwa_sin_mt, 2),
            },
            "basins_with_positive_salt_debt": int((sb_out["salt_debt_mt_yr"] > 0).sum()),
            "cropland_in_salt_debt_basins_mha": round(debt_ha_affected / 1e6, 2),
            "mean_salt_debt_over_all_major_basin_salt_mha_t_ha_yr": round((tot_debt_mt * 1e6) / max(tot_salt_ha, 1.0), 3),
            "mean_salt_debt_over_affected_basins_t_ha_yr": round(mean_t_ha_affected, 3),
            "counterfactual_dece_yr_if_leaching_curtailed_thetaD_045m": round((mean_t_ha_affected * 100.0 / (0.45 * 1000.0)) / 0.64, 3),
            "counterfactual_dece_yr_if_leaching_curtailed_sensitivity_035_to_055m": [
                round((mean_t_ha_affected * 100.0 / (0.55 * 1000.0)) / 0.64, 3),
                round((mean_t_ha_affected * 100.0 / (0.35 * 1000.0)) / 0.64, 3),
            ],
            "counterfactual_dece_note": "Counterfactual annual ECe equivalent if leaching were curtailed to sustainable SBWA limits while irrigation continues; not an observed accumulation rate.",
            "aquifer_subsidized_salt_balance_gw_only_basins": int(gw_sub_mask.sum()),
            "aquifer_subsidized_salt_input_mt_yr": round(gw_sub_sin_mt, 2),
            "aquifer_subsidized_cropland_mha": round(float(sb_out.loc[gw_sub_mask, "salt_irrig_ha"].sum() / 1e6), 2),
        },
        "dynqual_1991_2019_trend_consistency": {
            "note": "Spike-screened (<= 5,000 mg/L) historical DynQual v1.0 surface-water salinity trend consistency check (not independent validation, as DynQual gap-fills ambient ECw where river stations are absent).",
            "basins_with_valid_trend": int(len(valid_dq)),
            "salt_debt_positive_n": int(len(debt_pos)),
            "salt_debt_zero_n": int(len(debt_zero)),
            "median_tds_slope_debt_pos_mgl_yr": round(float(debt_pos["dynqual_tds_slope_median_mgl_yr"].median()), 3),
            "median_tds_slope_debt_zero_mgl_yr": round(float(debt_zero["dynqual_tds_slope_median_mgl_yr"].median()), 3),
            "mannwhitney_twosided_u": round(float(mw_u2), 1),
            "mannwhitney_twosided_p": float(mw_p2),
            "mannwhitney_greater_u": round(float(mw_u), 1),
            "mannwhitney_greater_p": float(mw_p),
            "spearman_salt_debt_intensity_vs_tds_slope_rho": round(float(sp_r), 4),
            "spearman_salt_debt_intensity_vs_tds_slope_r": round(float(sp_r), 4),
            "spearman_p": float(sp_p),
        },
    }
    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Saved Pillar 1B results -> {OUT_JSON}")


if __name__ == "__main__":
    prepare_outputs()
    main()
