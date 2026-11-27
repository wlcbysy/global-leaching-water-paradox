#!/usr/bin/env python3
"""
Redesigned Nature Water Main Figures 4-5 & Extended Data Figures 1-6 + Source Data CSVs:
  - Fig4_Water_Salt_Accounting_Zones (a-d):
      a) Global Three-Zone map (Zone I blue, Zone II #8e44ad, Zone III vermilion) + persistent GW depletion dots
      b) Three-tier water accounting waterfall (Withdrawal -> Consumption -> Basin depletion) with sensitivity & CI bars
      c) Named 12-basin pre -> post stress ratios (Demand / SBWA)
      d) Top-8 countries net fresh withdrawal relief (Delta_F + conditional H_wd hatched) + Zone III share
  - Fig5_Limits_Salt_Future (a-d):
      a) SSP5-8.5 2050 Zone III GCM-agreement fraction map (colorbar below)
      b) Global salt import & unfunded salt-export obligation pre vs post intervention
      c) Future Zone III Mha across SSPs & horizons (individual GCM points + median bars + baseline dashed)
      d) Future rescued & still-exceeded basin counts across SSPs & horizons (single axis)
  - ED_Fig1_PRISMA_Funnel_Bias (Arm B updated to 7,139 rows)
  - ED_Fig2_Hydrological_Validation (GRACE, Jasechko 1,693 aquifers, 4-GHM, EFR/Inflow, Salt area, Jensen V_leach-ECw curve)
  - ED_Fig3_Empirical_Sites_AOA (Site overlay + AOA map + PCA biplot)
  - ED_Fig4_ArmA_Sensitivity (9-subset ECw*90 forest, MH vs Logistic, other_cereals, SARw coupling, LOCO)
  - ED_Fig5_ArmB_Clay_LOCO (Clay texture moderation + LOCO bounds)
  - ED_Fig6_Pillar3_Sensitivity_Tornado (12-regime Delta_F, H_cons, and rescued-basin sensitivity)
"""

from __future__ import annotations

from config import (
    HARMONIZED,
    SOURCE_DATA,
    OUT_TABLES,
    OUT_PANELS,
    OUT_FIGURES,
    table_input,
    grid_input,
    panel_input,
    prepare_outputs,
)

import argparse
from decimal import Decimal, ROUND_HALF_UP
import json
import math

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely.geometry import box
from sklearn.decomposition import PCA

from fig_style import (
    NE_BORDERS_50M,
    NE_LAKES_50M,
    NE_LAND_50M,
    NW_FIG1_PALETTE,
    PALETTE,
    PARADOX_COLORS,
    ROBIN_CRS,
    add_fig_panel_label,
    add_panel_label,
    fmt_p,
    plot_robinson_basemap,
    save_pub_figure,
    setup_mpl_style,
)

FIG_DIR = OUT_FIGURES
SUP_DIR = OUT_FIGURES
SRC_DIR = OUT_PANELS
SUP_SRC_DIR = OUT_PANELS

ZONE2_PURPLE = "#8e44ad"


def build_fig4() -> None:
    p3 = json.loads((table_input("pillar3_global_summary_current.json")).read_text(encoding="utf-8"))
    grid = pd.read_parquet(grid_input("pillar3_grid05_zoning_current.parquet"))
    basins = pd.read_csv(table_input("pillar3_basin_zoning_current.csv"), low_memory=False)
    ctry = pd.read_csv(table_input("pillar3_country_hotspot_zoning_current.csv"))
    sens = pd.read_csv(table_input("pillar3_supply_sensitivity_current.csv"))

    fig = plt.figure(figsize=(180 / 25.4, 154 / 25.4))

    # =========================================================================
    # Panel a: Full-Width Rectangular 0.5-degree Robinson Three-Zone Map +
    #          Embedded South Pacific Dual-Scenario Area Comparison Inset
    # =========================================================================
    ax_a = fig.add_axes([0.014, 0.542, 0.972, 0.426])
    ax_a.set_facecolor("#F2F6FA")
    xlim_lon = (-162.0, 174.0)
    ylim_lat = (-49.0, 64.0)
    clip_poly = box(-170.0, -55.0, 180.0, 70.0)
    land_wgs = gpd.clip(gpd.read_file(NE_LAND_50M), clip_poly)
    land_rob = land_wgs.to_crs(ROBIN_CRS)
    tr_a = Transformer.from_crs("EPSG:4326", ROBIN_CRS, always_xy=True)

    grat_lons = np.linspace(-165.0, 175.0, 180)
    for glat in (-30.0, 0.0, 30.0, 60.0):
        gx, gy = tr_a.transform(grat_lons, np.full_like(grat_lons, glat))
        ax_a.plot(gx, gy, ls="--", color="#DFE7EE", lw=0.38, zorder=0.4)
    grat_lats = np.linspace(-52.0, 66.0, 100)
    for glon in (-120.0, -60.0, 0.0, 60.0, 120.0):
        gx, gy = tr_a.transform(np.full_like(grat_lats, glon), grat_lats)
        ax_a.plot(gx, gy, ls=":", color="#DFE7EE", lw=0.35, zorder=0.4)

    land_rob.plot(ax=ax_a, facecolor="#ECEEE9", edgecolor="#9CA49E", linewidth=0.30, zorder=1.0)
    if NE_BORDERS_50M.is_file():
        borders = gpd.clip(gpd.read_file(NE_BORDERS_50M), clip_poly).to_crs(ROBIN_CRS)
        borders.plot(ax=ax_a, color="#D5D8D2", linewidth=0.20, zorder=1.4)
    if NE_LAKES_50M.is_file():
        lakes = gpd.clip(gpd.read_file(NE_LAKES_50M), clip_poly).to_crs(ROBIN_CRS)
        lakes.plot(ax=ax_a, facecolor="#F2F6FA", edgecolor="#B8C5CE", linewidth=0.18, zorder=1.6)

    x_min, _ = tr_a.transform(xlim_lon[0], 0.0)
    x_max, _ = tr_a.transform(xlim_lon[1], 0.0)
    _, y_min = tr_a.transform(0.0, ylim_lat[0])
    _, y_max = tr_a.transform(0.0, ylim_lat[1])
    ax_a.set_xlim(x_min, x_max)
    ax_a.set_ylim(y_min, y_max)
    ax_a.set_aspect("equal", adjustable="box")
    ax_a.set_anchor("S")
    ax_a.set_xticks([])
    ax_a.set_yticks([])
    for sp in ("top", "right", "bottom", "left"):
        ax_a.spines[sp].set_visible(True)
        ax_a.spines[sp].set_color("#BCC7D1")
        ax_a.spines[sp].set_linewidth(0.60)

    salt = grid[grid["salt_affected"] & grid["risk_decomposition_class"].notna()].copy()
    rb = p3["risk_intersection_breakdown_mha"]
    rb_act = p3.get("risk_intersection_actual_et_mha", {})

    zone_palette = {
        "Z1_SAFE": "#2B6E8C",
        "Z1_GWC": "#8EC0D8",
        "Z2_HYD": "#7D4E9F",
        "Z3_BW": "#E09F24",
        "Z3_SA": "#D95F28",
        "Z3_DUAL": "#8C1515",
    }
    class_to_code = {
        "Zone_I_Safe_Window_GW_Sustainable": "Z1_SAFE",
        "Zone_I_Local_Potential_GW_Constrained": "Z1_GWC",
        "Zone_II_Hydraulic_GW_OK": "Z2_HYD",
        "Zone_II_Hydraulic_GW_Depleted": "Z2_HYD",
        "Risk_BW_Only": "Z3_BW",
        "Risk_SalArid_Only": "Z3_SA",
        "Risk_Dual_BW_and_SalArid": "Z3_DUAL",
    }
    salt["z_code"] = salt["risk_decomposition_class"].map(class_to_code)

    cell_geoms = [
        box(lon - 0.25, lat - 0.25, lon + 0.25, lat + 0.25)
        for lon, lat in zip(salt["lon"], salt["lat"])
    ]
    cell_gdf = gpd.GeoDataFrame(salt[["z_code"]], geometry=cell_geoms, crs="EPSG:4326")
    cell_dis = cell_gdf.dissolve(by="z_code")
    cell_clip = gpd.clip(cell_dis, land_wgs).to_crs(ROBIN_CRS)
    for code in ["Z1_GWC", "Z1_SAFE", "Z2_HYD", "Z3_BW", "Z3_SA", "Z3_DUAL"]:
        if code in cell_clip.index:
            cell_clip.loc[[code]].plot(
                ax=ax_a, facecolor=zone_palette[code], edgecolor="none", alpha=0.95, zorder=3.0
            )

    gw_z12 = salt[
        salt["management_zone_baseline"].isin([
            "Zone_I_Safe_Brackish_Substitution_Window",
            "Zone_II_Hydraulic_Recovery_Priority",
        ]) & salt["gw_still_depleted_post"].fillna(False)
    ]
    if not gw_z12.empty:
        gw_geoms = [box(lon - 0.25, lat - 0.25, lon + 0.25, lat + 0.25) for lon, lat in zip(gw_z12["lon"], gw_z12["lat"])]
        gw_gdf = gpd.GeoDataFrame(geometry=gw_geoms, crs="EPSG:4326").dissolve()
        gw_clip = gpd.clip(gw_gdf, land_wgs).to_crs(ROBIN_CRS)
        with mpl.rc_context({"hatch.linewidth": 0.30}):
            gw_clip.plot(ax=ax_a, facecolor="none", edgecolor="#1F2933", hatch="////", linewidth=0.16, alpha=0.72, zorder=4.2)

    # Lock Panel a position & embed dual-scenario stacked bar in open South Pacific pocket
    fig.canvas.draw()
    pos_a = ax_a.get_position()
    add_fig_panel_label(fig, 0.010, min(pos_a.y1 + 0.024, 0.992), "a")

    inset_card = mpl.patches.FancyBboxPatch(
        (0.008, 0.018), 0.220, 0.547,
        boxstyle="round,pad=0.0,rounding_size=0.008",
        transform=ax_a.transAxes,
        facecolor="#FFFFFF", edgecolor="#C4CFD8", linewidth=0.52, alpha=0.92, zorder=5.5,
    )
    ax_a.add_patch(inset_card)

    ax_bar = fig.add_axes([
        pos_a.x0 + 0.056 * pos_a.width,
        pos_a.y0 + 0.102 * pos_a.height,
        0.166 * pos_a.width,
        0.398 * pos_a.height,
    ], zorder=7)
    ax_bar.set_facecolor("none")

    z3_act_mask = salt["blue_water_still_exceeded_actual"].fillna(False).astype(bool) | salt["is_hyper_sal_arid"].fillna(False).astype(bool)
    gw_post_mask = salt["gw_still_depleted_post"].fillna(False).astype(bool)
    rem_act = salt[~z3_act_mask]
    rem_gw = gw_post_mask.loc[rem_act.index]
    act_z2_mha = float(rem_act.loc[rem_act["management_zone_baseline"] == "Zone_II_Hydraulic_Recovery_Priority", "salt_irrig_ha"].sum() / 1e6)
    act_z1_gwc_mha = float(rem_act.loc[(rem_act["management_zone_baseline"] == "Zone_I_Safe_Brackish_Substitution_Window") & rem_gw, "salt_irrig_ha"].sum() / 1e6)
    act_z1_safe_mha = float(rem_act.loc[(rem_act["management_zone_baseline"] == "Zone_I_Safe_Brackish_Substitution_Window") & (~rem_gw), "salt_irrig_ha"].sum() / 1e6)
    act_z1_tot = act_z1_gwc_mha + act_z1_safe_mha

    act_dual_mha = float(rb_act.get("zone3_dual_bw_and_sal_arid_mha", 8.22))
    act_sa_mha = float(rb_act.get("zone3_sal_arid_only_mha", 6.18))
    act_bw_mha = float(rb_act.get("zone3_bw_only_mha", 21.69))
    act_z3_tot = act_dual_mha + act_sa_mha + act_bw_mha
    cap_z3_tot = float(rb["zone3_at_least_one_risk_total_mha"])
    tot_mha = cap_z3_tot + float(rb["zone2_total_mha"]) + float(rb["zone1_total_mha"])
    act_tot_mha = act_z3_tot + act_z2_mha + act_z1_tot
    print(
        f"[AUDIT Fig4] Area totals (Mha): cap={tot_mha:.2f} (Z1={rb['zone1_total_mha']:.2f}, Z2={rb['zone2_total_mha']:.2f}, Z3={cap_z3_tot:.2f}) | "
        f"actual={act_tot_mha:.2f} (Z1={act_z1_tot:.2f}, Z2={act_z2_mha:.2f}, Z3={act_z3_tot:.2f})"
    )

    bar_specs = [
        ("Z3_DUAL", rb["zone3_dual_bw_and_sal_arid_mha"], act_dual_mha),
        ("Z3_SA", rb["zone3_sal_arid_only_mha"], act_sa_mha),
        ("Z3_BW", rb["zone3_bw_only_mha"], act_bw_mha),
        ("Z2_HYD", rb["zone2_total_mha"], act_z2_mha),
        ("Z1_GWC", rb["zone1_local_potential_gw_constrained_mha"], act_z1_gwc_mha),
        ("Z1_SAFE", rb["zone1_safe_window_gw_sustainable_mha"], act_z1_safe_mha),
    ]
    xb_pos = np.array([0.0, 0.92])
    bots = np.zeros(2)
    for code, v_cap, v_act in bar_specs:
        vals = np.array([v_cap, v_act])
        ax_bar.bar(xb_pos, vals, bottom=bots, width=0.54, color=zone_palette[code], edgecolor="white", linewidth=0.40, alpha=0.95, zorder=3)
        bots += vals

    # Subtle alluvial transition bands connecting Zone III & Zone II/I boundaries
    cap_z2_top = cap_z3_tot + float(rb["zone2_total_mha"])
    act_z2_top = act_z3_tot + act_z2_mha
    ax_bar.fill_between([0.27, 0.65], [0.0, 0.0], [cap_z3_tot, act_z3_tot], facecolor="#8C1515", alpha=0.12, edgecolor="none", zorder=2)
    ax_bar.fill_between([0.27, 0.65], [cap_z3_tot, act_z3_tot], [cap_z2_top, act_z2_top], facecolor="#7D4E9F", alpha=0.11, edgecolor="none", zorder=2)
    ax_bar.plot([0.27, 0.65], [cap_z2_top, act_z2_top], ls=":", color="#633982", lw=0.60, alpha=0.75, zorder=4.2)

    # Outer right-side Zone brackets
    brk_cfg = [
        (0.5, act_z3_tot - 0.5, "#8C1515", f"Zone III\n{cap_z3_tot:.1f}→{act_z3_tot:.1f}"),
        (act_z3_tot + 0.5, act_z3_tot + act_z2_mha - 0.5, "#633982", f"Zone II\n{rb['zone2_total_mha']:.1f}→{act_z2_mha:.1f}"),
        (act_z3_tot + act_z2_mha + 0.5, tot_mha - 0.5, "#1D4E66", f"Zone I\n{rb['zone1_total_mha']:.1f}→{act_z1_tot:.1f}"),
    ]
    for y0_b, y1_b, col_b, lbl_b in brk_cfg:
        ax_bar.plot([1.24, 1.31, 1.31, 1.24], [y0_b, y0_b, y1_b, y1_b], color=col_b, lw=0.70, clip_on=False, zorder=5)
        ax_bar.text(1.36, 0.5 * (y0_b + y1_b), lbl_b, ha="left", va="center", fontsize=4.55, fontweight="bold", color=col_b)

    # Dashed step indicator highlighting Zone III redline expansion (+7.0 Mha)
    ax_bar.plot([0.27, 0.65], [cap_z3_tot, act_z3_tot], ls="--", color="#8C1515", lw=0.80, zorder=4.5)
    ax_bar.text(0.46, 100.5, f"Zone III: +{act_z3_tot - cap_z3_tot:.1f} Mha", ha="center", va="bottom", fontsize=4.85, fontweight="bold", color="#8C1515")

    ax_bar.set_xticks(xb_pos)
    ax_bar.set_xticklabels(["Equal-prod.\ncap", "Actual\nobs. ET"], fontsize=4.65)
    ax_bar.tick_params(axis="x", length=2.0, pad=1.5)
    ax_bar.set_xlim(-0.44, 2.48)
    ax_bar.spines["bottom"].set_bounds(-0.44, 1.24)
    ax_bar.set_ylim(0, tot_mha)
    ax_bar.set_yticks([0, 25, 50, 75, 97.12])
    ax_bar.set_yticklabels(["0", "25", "50", "75", "97"], fontsize=4.7)
    ax_bar.tick_params(axis="y", length=2.0, pad=1.5)
    ax_bar.set_ylabel("Area (Mha)", fontsize=5.2, labelpad=1.5)

    # Two centered, logically grouped frameless legend rows below Panel a
    leg_row1 = [
        Patch(facecolor=zone_palette["Z1_SAFE"], edgecolor="none", label=f"Zone I: Safe, GW-sustainable ({rb['zone1_safe_window_gw_sustainable_mha']:.2f} Mha)"),
        Patch(facecolor=zone_palette["Z1_GWC"], edgecolor="none", label=f"Zone I: GW-constrained ({rb['zone1_local_potential_gw_constrained_mha']:.2f} Mha)"),
        Patch(facecolor=zone_palette["Z2_HYD"], edgecolor="none", label=f"Zone II: Hydraulic priority ({rb['zone2_total_mha']:.2f} Mha)"),
        Patch(facecolor="#FFFFFF", edgecolor="#1F2933", hatch="////", linewidth=0.40, label="Persistent GW depletion (Zones I–II)"),
    ]
    leg_row2 = [
        Patch(facecolor=zone_palette["Z3_BW"], edgecolor="none", label=f"Zone III: Blue-water only ({rb['zone3_bw_only_mha']:.2f} Mha)"),
        Patch(facecolor=zone_palette["Z3_SA"], edgecolor="none", label=f"Zone III: Salinity–aridity only ({rb['zone3_sal_arid_only_mha']:.2f} Mha)"),
        Patch(facecolor=zone_palette["Z3_DUAL"], edgecolor="none", label=f"Zone III: Dual blue-water + salinity–aridity ({rb['zone3_dual_bw_and_sal_arid_mha']:.2f} Mha)"),
    ]
    fig.legend(
        handles=leg_row1,
        loc="upper center",
        bbox_to_anchor=(0.50, pos_a.y0 - 0.012),
        ncol=4,
        frameon=False,
        fontsize=5.2,
        handlelength=1.0,
        handletextpad=0.32,
        columnspacing=0.72,
    )
    fig.legend(
        handles=leg_row2,
        loc="upper center",
        bbox_to_anchor=(0.50, pos_a.y0 - 0.039),
        ncol=3,
        frameon=False,
        fontsize=5.2,
        handlelength=1.0,
        handletextpad=0.32,
        columnspacing=0.95,
    )

    # =========================================================================
    # Panel b: Split Two-Scale Water Accounting (Withdrawal Relief vs Basin Depletion)
    # =========================================================================
    wat = p3["water_accounting_three_tier"]
    wd = wat["withdrawal_tier_km3_yr"]
    cn = wat["consumption_tier_km3_yr"]
    dual_c = p3["dual_scenario_basin_closure"]
    act_c = dual_c["actual_management_observed_et"]
    cap_c = dual_c["equal_production_water_capped"]

    B = wd["B_gross_brackish_substitution"]
    P = wd["P_leaching_penalty"]
    dF = wd["deltaF_net_brackish_fresh_withdrawal_saving"]
    Hwd = wd["H_wd_conditional_equal_yield_hydraulic_withdrawal_saving"]
    dF_lo, dF_hi = float(sens["deltaF_net_brackish_saving_km3_yr"].min()), float(sens["deltaF_net_brackish_saving_km3_yr"].max())

    B_gw = cn["B_gw_consumptive_credit"]
    H_cons = cn["H_cons_conditional_equal_yield_saving"]
    H_lo, H_hi = cn["H_cons_ci95"]
    H_obs = cn["H_obs_observed_et_saving"]
    H_obs_lo, H_obs_hi = cn["H_obs_ci95"]
    dVl = cn["delta_vleach_field_leaching_increment"]
    net_cap = cap_c["net_basin_depletion_relief_km3_yr"]
    net_cap_lo, net_cap_hi = cap_c["net_basin_depletion_relief_ci95"]
    net_act = act_c["net_basin_depletion_relief_km3_yr"]
    net_act_lo, net_act_hi = act_c["net_basin_depletion_relief_ci95"]

    add_fig_panel_label(fig, 0.010, 0.458, "b")

    checks = pd.read_csv(table_input("pillar3_threshold_and_recovery_current.csv")).set_index("case")
    field = checks.loc["Field harvest"]
    ax_b1 = fig.add_axes([0.068, 0.084, 0.116, 0.336])
    ax_b1.grid(axis="y", color="#EEF1F4", lw=0.45, zorder=0.3)
    ax_b1.bar([0, 1], [dF, field.withdrawal_relief_km3_yr], color=["#2B6E8C", "#C96A1B"], width=0.56, alpha=0.92, zorder=2.5)
    ax_b1.errorbar(0, dF, yerr=[[dF - dF_lo], [dF_hi - dF]], color="#1B3D4F", lw=1.15, capsize=2.8, capthick=0.9, zorder=4.0)
    dx = [checks.loc[q, "withdrawal_relief_km3_yr"] for q in ["Field lower CI", "Field upper CI"]]
    ax_b1.errorbar(1, field.withdrawal_relief_km3_yr, yerr=[[field.withdrawal_relief_km3_yr - min(dx)], [max(dx) - field.withdrawal_relief_km3_yr]], color="#6E380B", lw=1.15, capsize=2.8, capthick=0.9, zorder=4.0)
    ax_b1.text(0, 74.0, f"{dF:.1f}", ha="center", va="center", fontsize=5.8, fontweight="bold", color="white", zorder=5.0)
    ax_b1.text(1, 74.0, f"{field.withdrawal_relief_km3_yr:.1f}", ha="center", va="center", fontsize=5.8, fontweight="bold", color="white", zorder=5.0)
    ax_b1.text(0, dF_hi + 7.0, "12 supply\nregimes", ha="center", va="bottom", fontsize=4.85, color="#4A5A68", zorder=5.0)
    ax_b1.text(1, max(dx) + 7.0, r"95% $\mathrm{EC_w}$" + "\nCI", ha="center", va="bottom", fontsize=4.85, color="#4A5A68", zorder=5.0)
    ax_b1.set(xticks=[0, 1], xticklabels=["Mixed", "Harvest"], ylim=(0, 285), yticks=[0, 100, 200])
    ax_b1.set_ylabel(r"Withdrawal relief (km$^3$ yr$^{-1}$)", fontsize=6.6, labelpad=2.0)
    ax_b1.tick_params(labelsize=5.7)
    ax_b1.text(0.5, 1.03, "Withdrawal relief", transform=ax_b1.transAxes, ha="center", va="bottom", fontsize=6.0, fontweight="bold", color="#222222")

    import matplotlib.patheffects as pe
    ax_b2 = fig.add_axes([0.264, 0.084, 0.188, 0.336])
    ax_b2.set_xlim(-50.0, 108.0)
    ax_b2.set_ylim(-0.55, 2.72)
    ax_b2.fill_between([-50.0, 0.0], -0.55, 2.72, facecolor="#FCECE8", alpha=0.45, zorder=0.1)
    ax_b2.fill_between([0.0, 108.0], -0.55, 2.72, facecolor="#EBF6F4", alpha=0.52, zorder=0.1)
    ax_b2.grid(axis="x", color="#EEF1F4", lw=0.45, zorder=0.2)
    ax_b2.axvline(0.0, color="#333333", ls="--", lw=0.95, zorder=1.5)
    ax_b2.text(0.04, 0.96, r"$\leftarrow$ Depletion", transform=ax_b2.transAxes, ha="left", va="top", fontsize=5.2, fontweight="bold", color="#A61C28", zorder=5.0)
    ax_b2.text(0.96, 0.96, r"Relief $\rightarrow$", transform=ax_b2.transAxes, ha="right", va="top", fontsize=5.2, fontweight="bold", color="#1B7867", zorder=5.0)
    net_values = [
        (net_cap, net_cap_lo, net_cap_hi, "#1B7867"),
        (net_act, net_act_lo, net_act_hi, "#A61C28"),
        (checks.loc["Direct ET", "net_et_km3_yr"], checks.loc["Direct ET low", "net_et_km3_yr"], checks.loc["Direct ET high", "net_et_km3_yr"], "#476971"),
    ]
    for y, (v, lo, hi, col) in zip([2, 1, 0], net_values):
        lo_s, hi_s = min(lo, hi), max(lo, hi)
        ax_b2.plot([lo_s, hi_s], [y, y], lw=2.6, color=col, solid_capstyle="round", zorder=3.0)
        ax_b2.scatter(v, y, color=col, s=34, marker="D", edgecolors="white", linewidths=0.55, zorder=4.0)
        v_str = f"{v:+.1f}".replace("-", "−")
        ax_b2.text(
            v, y + 0.17, v_str, ha="center", va="bottom", fontsize=5.9, color=col, fontweight="bold", zorder=5.0,
            path_effects=[pe.withStroke(linewidth=2.2, foreground="white")],
        )
        lo_str = f"{lo_s:+.1f}".replace("-", "−")
        hi_str = f"{hi_s:+.1f}".replace("-", "−")
        x_ci_txt = v if y == 2 else 14.0
        ha_ci = "center" if y == 2 else "left"
        ax_b2.text(x_ci_txt, y - 0.17, f"[{lo_str}, {hi_str}]", ha=ha_ci, va="top", fontsize=4.9, color="#555555", zorder=5.0)
    ax_b2.set(yticks=[0, 1, 2], yticklabels=["Direct ET", "Verified ET", "Capped"], xticks=[-40, 0, 40, 80])
    ax_b2.set_xticklabels(["−40", "0", "+40", "+80"], fontsize=5.7)
    ax_b2.set_xlabel(r"Net relief (km$^3$ yr$^{-1}$)", fontsize=6.6, labelpad=2.0)
    ax_b2.tick_params(axis="y", labelsize=5.7)
    ax_b2.text(0.5, 1.03, "Basin depletion closure", transform=ax_b2.transAxes, ha="center", va="bottom", fontsize=6.0, fontweight="bold", color="#222222")
    checks.reset_index().to_csv(SRC_DIR/"Fig4B_threshold_and_ET.csv",index=False)

    # =========================================================================
    # Panel c: Named 12-Basin Pre vs Actual-ET vs Equal-Prod Stress Ratios
    # =========================================================================
    add_fig_panel_label(fig, 0.474, 0.458, "c")
    ax_c = fig.add_axes([0.574, 0.084, 0.178, 0.336])
    sb = basins[basins["salt_irrig_ha"] >= 10000.0].copy()
    sb["ratio_pre"] = (np.maximum(sb["atotuse_km3_yr"], 0.0) + sb["vleach_amb_km3_yr"]) / np.maximum(sb["sbwa_efr60_km3_yr"], 1e-6)
    sb["ratio_post_actual"] = sb["post_intervention_net_use_actual_km3_yr"] / np.maximum(sb["sbwa_efr60_km3_yr"], 1e-6)
    sb["ratio_post_cap"] = sb["post_intervention_net_use_km3_yr"] / np.maximum(sb["sbwa_efr60_km3_yr"], 1e-6)
    named_bids = [
        (6562, "Upper Indus (PK–IN)"), (5705, "Haihe–NCP (CN)"), (6889, "Lower Indus (PK)"), (5560, "Tarim (CN)"),
        (5651, "Karakum (TM–IR)"), (4998, "Junggar (CN)"), (6261, "Imperial–Salton (US)"), (7461, "Krishna (IN)"),
        (5965, "San Joaquin (US)"), (5513, "West Liaohe (CN)"), (6181, "Central Iran (IR)"), (6376, "Lower Nile (EG–SD)"),
    ]
    brows = []
    for bid, nm in named_bids:
        r = sb[sb["aware_basin_idx"] == bid]
        if not r.empty:
            gw_base = bool(r["gw_depletion_active"].iloc[0])
            gw_dep = bool(r["gw_still_depleted_post"].iloc[0])
            suffix = "†" if gw_dep else ("‡" if gw_base else "")
            brows.append({
                "bid": bid, "name": f"{nm}{suffix}",
                "pre": float(r["ratio_pre"].iloc[0]),
                "post_actual": float(r["ratio_post_actual"].iloc[0]),
                "post_cap": float(r["ratio_post_cap"].iloc[0]),
                "relieved_cap": bool(r["blue_water_relieved_to_feasible"].iloc[0]),
                "relieved_actual": bool(r["blue_water_relieved_actual"].iloc[0]),
                "gw_depletion_active": gw_base,
                "gw_still_depleted_post": gw_dep,
            })
    bdf_raw = pd.DataFrame(brows)
    rescued_df = bdf_raw[bdf_raw["relieved_cap"]].sort_values("post_cap", ascending=False).reset_index(drop=True)
    insolv_df = bdf_raw[~bdf_raw["relieved_cap"]].sort_values("pre", ascending=False).reset_index(drop=True)
    rescued_df["y_pos"] = np.arange(len(rescued_df)) + 8.0
    insolv_df["y_pos"] = np.arange(len(insolv_df)) + 0.2
    bdf = pd.concat([insolv_df, rescued_df], ignore_index=True)

    ax_c.grid(axis="x", color="#EEF1F4", lw=0.45, zorder=0.2)
    ax_c.axvspan(0.74, 1.0, color="#E8F6F3", alpha=0.72, zorder=0.3)
    for idx_r, (_, row_r) in enumerate(bdf.iterrows()):
        if idx_r % 2 == 1:
            yr_c = float(row_r["y_pos"])
            ax_c.fill_between([1.0, 2.95], yr_c - 0.42, yr_c + 0.42, facecolor="#F7F9FB", edgecolor="none", zorder=0.25)
    ax_c.axvline(1.0, ls="--", color="#222222", lw=0.85, zorder=2)
    ax_c.axhline(7.25, ls="--", color="#B8C0C8", lw=0.65, zorder=1.5)

    for _, row in bdf.iterrows():
        yi = float(row["y_pos"])
        col = "#1B7867" if row["relieved_cap"] else "#C84B31"
        ax_c.plot([row["post_cap"], row["pre"]], [yi, yi], "-", color=col, lw=1.45, alpha=0.88, zorder=3)
        if row["post_actual"] > row["pre"]:
            ax_c.plot([row["pre"], row["post_actual"]], [yi, yi], "-", color="#E09F24", lw=1.45, alpha=0.90, zorder=3.2)
        ax_c.scatter([row["pre"]], [yi], facecolor="white", edgecolor="#333333", s=20, lw=0.85, zorder=4)
        ax_c.scatter([row["post_actual"]], [yi], color="#E09F24", marker="D", s=14, edgecolors="#222222", lw=0.35, zorder=5)
        ax_c.scatter([row["post_cap"]], [yi], color=col, s=26, edgecolors="white", linewidths=0.40, zorder=6)

    ax_c.text(2.90, 12.65, "Cap-rescued", ha="right", va="bottom", fontsize=5.2, fontweight="bold", color="#1B7867")
    ax_c.text(2.90, 6.75, "Still insolvent", ha="right", va="top", fontsize=5.2, fontweight="bold", color="#C84B31")

    ax_c.scatter([], [], facecolor="white", edgecolor="#333333", s=17, lw=0.85, label="Pre (amb.)")
    ax_c.scatter([], [], color="#E09F24", marker="D", s=14, edgecolor="#222222", lw=0.35, label="Actual ET (0/80)")
    ax_c.scatter([], [], color="#1B7867", s=20, edgecolor="white", lw=0.35, label=f"Cap rescued ({cap_c['relieved_basins']}/80)")
    ax_c.scatter([], [], color="#C84B31", s=20, edgecolor="white", lw=0.35, label="Cap insolvent")

    ax_c.set_yticks(bdf["y_pos"].values)
    ax_c.set_yticklabels(bdf["name"], fontsize=5.3)
    ax_c.set_xlim(0.74, 2.95)
    ax_c.set_ylim(-0.45, 13.05)
    ax_c.set_xticks([1.0, 1.5, 2.0, 2.5])
    ax_c.tick_params(axis="x", labelsize=5.7)
    ax_c.set_xlabel("Demand / SBWA stress ratio", fontsize=6.6, labelpad=2.0)
    ax_c.legend(loc="upper right", bbox_to_anchor=(1.00, 0.915), frameon=True, facecolor="white", edgecolor="none", framealpha=0.85, fontsize=5.2, handletextpad=0.25, labelspacing=0.26, borderpad=0.20)

    # =========================================================================
    # Panel d: Top-8 Countries Withdrawal Relief + Zone III Redline Share
    # =========================================================================
    add_fig_panel_label(fig, 0.772, 0.458, "d")
    ax_d = fig.add_axes([0.822, 0.084, 0.164, 0.336])
    top8 = ctry[ctry["country"] != "Other_Coastal_or_Unmatched"].sort_values("net_fresh_withdrawal_relief_km3_yr", ascending=False).head(8).iloc[::-1].copy()
    top8["short_name"] = top8["country"].replace({"United States of America": "USA", "Australia": "Australia*"})
    yd = np.arange(len(top8))
    ax_d.grid(axis="x", color="#EEF1F4", lw=0.45, zorder=0.3)
    for yi in yd:
        if yi % 2 == 1:
            ax_d.axhspan(yi - 0.42, yi + 0.42, facecolor="#F7F9FB", zorder=0.2)
    ax_d.barh(yd, top8["deltaF_net_brackish_saving_km3_yr"], height=0.58, color="#2B6E8C", edgecolor="white", lw=0.45, label=r"Brackish $\Delta\mathit{F}$", zorder=3)
    with mpl.rc_context({"hatch.linewidth": 0.35}):
        ax_d.barh(yd, top8["hydraulic_saving_wd_km3_yr"], left=top8["deltaF_net_brackish_saving_km3_yr"], height=0.58, color="#EDB440", hatch="////", edgecolor="#8A5A10", lw=0.40, label=r"WUE $\mathit{H}_{\mathrm{wd}}^*$ (cap*)", zorder=3)
    for yi, (_, row) in enumerate(top8.iterrows()):
        tot = row["net_fresh_withdrawal_relief_km3_yr"]
        z3p = row["zone3_share_pct"]
        is_hi = z3p >= 50.0
        txt = f"{tot:.1f} ({z3p:.0f}% Z3)" if is_hi else f"{tot:.1f} ({z3p:.0f}%)"
        ax_d.text(
            tot + 1.4, yi, txt,
            va="center", ha="left", fontsize=5.2,
            fontweight="bold" if is_hi else "normal",
            color="#A61C28" if is_hi else "#222222",
        )
    ax_d.set_yticks(yd)
    ax_d.set_yticklabels(top8["short_name"], fontsize=5.7)
    ax_d.set_xlim(0, 128)
    ax_d.set_ylim(-0.78, len(top8) - 0.25)
    ax_d.set_xticks([0, 30, 60, 90, 120])
    ax_d.tick_params(axis="x", labelsize=5.7)
    ax_d.set_xlabel(r"Withdrawal relief (km$^3$ yr$^{-1}$)", fontsize=6.6, labelpad=2.0)
    d_handles = [
        Patch(facecolor="#2B6E8C", edgecolor="white", linewidth=0.45, label=r"Brackish $\Delta\mathit{F}$"),
        Patch(facecolor="#EDB440", edgecolor="#8A5A10", hatch="////", linewidth=0.40, label=r"WUE $\mathit{H}_{\mathrm{wd}}^*$ (cap*)"),
    ]
    ax_d.legend(handles=d_handles, loc="lower right", bbox_to_anchor=(1.01, 0.165), frameon=False, fontsize=4.95, handlelength=0.95, handletextpad=0.30, labelspacing=0.25)
    ax_d.text(0.99, 0.085, "Red: ≥50% Zone III", transform=ax_d.transAxes, ha="right", va="bottom", fontsize=5.0, fontweight="bold", color="#A61C28")
    ax_d.text(0.99, 0.015, "*GMIA incl. pasture", transform=ax_d.transAxes, ha="right", va="bottom", fontsize=4.85, color="#444444")

    SRC_DIR.mkdir(parents=True, exist_ok=True)
    salt[["r", "c", "lat", "lon", "salt_irrig_ha", "management_zone_baseline", "risk_decomposition_class", "blue_water_still_exceeded_post", "blue_water_still_exceeded_actual", "is_hyper_sal_arid", "gw_still_depleted_post"]].to_csv(SRC_DIR / "Fig4A_three_zone_map.csv", index=False)
    pd.DataFrame([
        {"tier": "withdrawal", "item": "B_gross", "km3_yr": B, "lo": np.nan, "hi": np.nan},
        {"tier": "withdrawal", "item": "P_leaching_penalty", "km3_yr": P, "lo": np.nan, "hi": np.nan},
        {"tier": "withdrawal", "item": "deltaF_net_brackish", "km3_yr": dF, "lo": dF_lo, "hi": dF_hi},
        {"tier": "withdrawal", "item": "H_wd_conditional", "km3_yr": Hwd, "lo": np.nan, "hi": np.nan},
        {"tier": "consumption", "item": "return_flow_credit", "km3_yr": 0.0, "lo": np.nan, "hi": np.nan},
        {"tier": "consumption", "item": "B_gw_credit", "km3_yr": B_gw, "lo": np.nan, "hi": np.nan},
        {"tier": "consumption", "item": "H_cons_conditional", "km3_yr": H_cons, "lo": H_lo, "hi": H_hi},
        {"tier": "consumption", "item": "H_obs_et_ratio", "km3_yr": H_obs, "lo": H_obs_lo, "hi": H_obs_hi},
        {"tier": "consumption", "item": "delta_V_leach", "km3_yr": dVl, "lo": np.nan, "hi": np.nan},
        {"tier": "basin", "item": "net_depletion_relief_equal_prod", "km3_yr": net_cap, "lo": net_cap_lo, "hi": net_cap_hi},
        {"tier": "basin", "item": "net_depletion_relief_actual_et", "km3_yr": net_act, "lo": net_act_lo, "hi": net_act_hi},
    ]).to_csv(SRC_DIR / "Fig4B_three_tier_waterfall.csv", index=False)
    bdf.to_csv(SRC_DIR / "Fig4C_basin_rescue_ratios.csv", index=False)
    top8.to_csv(SRC_DIR / "Fig4D_country_withdrawal_relief.csv", index=False)

    with mpl.rc_context({"hatch.linewidth": 0.35}):
        save_pub_figure(fig, FIG_DIR / "Fig4_Water_Salt_Accounting_Zones.pdf", FIG_DIR / "Fig4_Water_Salt_Accounting_Zones.png", "Fig4")


def build_fig5() -> None:
    p3 = json.loads((table_input("pillar3_global_summary_current.json")).read_text(encoding="utf-8"))
    fut = pd.read_csv(table_input("pillar3_future_zoning_scenarios_current.csv"))
    gcm_df = pd.read_csv(table_input("pillar3_gcm_uncertainty_current.csv"))
    grid = pd.read_parquet(grid_input("pillar3_grid05_zoning_current.parquet"))
    fcz = pd.read_parquet(grid_input("pillar3_future_cell_zones_ssp585_2050_current.parquet"))
    obs_l = json.loads((table_input("observed_district_basin_water_salt_ledger_current.json")).read_text(encoding="utf-8"))["global_two_boundary_salt_ledger"]

    fcz_base = fcz[fcz["gcm"] == "baseline_2020"].merge(
        grid[["r", "c", "lat", "lon", "salt_irrig_ha"]], on=["r", "c"], how="left"
    )
    fcz_base["agree_n"] = np.rint(fcz_base["zone3_gcm_agreement_frac"].values * 5.0).astype(int)
    fcz_base["base_z3"] = fcz_base["zone"] == "Zone_III_Water_Salt_Dual_Redline"

    # Five exhaustive Persistence x Emergence classes (summing strictly to 97.12 Mha)
    fcz_base["risk_tier"] = np.where(
        (~fcz_base["base_z3"]) & (fcz_base["agree_n"] >= 3), "EMERGING",
        np.where(
            fcz_base["base_z3"] & (fcz_base["agree_n"] >= 4), "LOCKED",
            np.where(
                fcz_base["base_z3"] & (fcz_base["agree_n"] == 3), "PERSISTENT_3",
                np.where(fcz_base["agree_n"].isin([1, 2]), "MARGINAL", "OUTSIDE"),
            ),
        ),
    )

    def _assign_region(lon: float, lat: float) -> str:
        if 60.0 < lon <= 95.0 and 5.0 <= lat <= 36.0:
            return "South Asia"
        if -15.0 <= lon <= 60.0 and 12.0 <= lat <= 45.0:
            return "MENA & Med."
        if 95.0 < lon <= 135.0 and lat >= 18.0:
            return "East Asia"
        if 45.0 <= lon <= 95.0 and lat > 36.0:
            return "Central Asia"
        return "Americas & rest"

    fcz_all = fcz.merge(grid[["r", "c", "lat", "lon", "salt_irrig_ha"]], on=["r", "c"], how="left")
    fcz_all["region"] = [_assign_region(float(x), float(y)) for x, y in zip(fcz_all["lon"], fcz_all["lat"])]
    fcz_all["is_z3"] = fcz_all["zone"] == "Zone_III_Water_Salt_Dual_Redline"
    fcz_base["region"] = [_assign_region(float(x), float(y)) for x, y in zip(fcz_base["lon"], fcz_base["lat"])]

    fig = plt.figure(figsize=(178 / 25.4, 132 / 25.4))

    # Shared 5-GCM styles across Panel a (right), Panel c, and Panel d
    gcm_styles = [
        ("gfdl-esm4", "GFDL-ESM4", "#D55E00", "o"),
        ("ipsl-cm6a-lr", "IPSL-CM6A-LR", "#8C1515", "s"),
        ("mpi-esm1-2-hr", "MPI-ESM1-2-HR", "#2B6E8C", "^"),
        ("mri-esm2-0", "MRI-ESM2-0", "#7D4E9F", "D"),
        ("ukesm1-0-ll", "UKESM1-0-LL", "#1B7867", "v"),
    ]

    # =========================================================================
    # Panel a: Centered Regional Robinson Map + Western US Arabian Sea Inset +
    #          Joint Salinity–Aridity–GW Depletion Hotspot Map & Trajectories
    # =========================================================================
    ROBIN_FOCUS = "+proj=robin +lon_0=58 +datum=WGS84 +units=m +no_defs"
    tr_f = Transformer.from_crs("EPSG:4326", ROBIN_FOCUS, always_xy=True)

    ax_a = fig.add_axes([0.012, 0.622, 0.606, 0.335])
    ax_a.set_facecolor("#F2F6FA")

    focus_buf = box(-40.0, -8.0, 165.0, 66.0)
    land_wgs_focus = gpd.clip(gpd.read_file(NE_LAND_50M), focus_buf)
    land_focus = land_wgs_focus.to_crs(ROBIN_FOCUS)

    # Subtle graticules before land
    grat_lons = np.linspace(-30.0, 155.0, 120)
    for glat in (20.0, 30.0, 40.0):
        gx, gy = tr_f.transform(grat_lons, np.full_like(grat_lons, glat))
        ax_a.plot(gx, gy, ls="--", color="#DFE7EE", lw=0.36, zorder=0.4)
    grat_lats = np.linspace(-2.0, 60.0, 80)
    for glon in (0.0, 30.0, 60.0, 90.0, 120.0):
        gx, gy = tr_f.transform(np.full_like(grat_lats, glon), grat_lats)
        ax_a.plot(gx, gy, ls=":", color="#DFE7EE", lw=0.36, zorder=0.4)

    land_focus.plot(ax=ax_a, facecolor="#ECEEE9", edgecolor="#9CA49E", linewidth=0.30, zorder=1.0)
    if NE_BORDERS_50M.is_file():
        borders_focus = gpd.clip(gpd.read_file(NE_BORDERS_50M), focus_buf).to_crs(ROBIN_FOCUS)
        borders_focus.plot(ax=ax_a, color="#D5D8D2", linewidth=0.20, zorder=1.4)
    if NE_LAKES_50M.is_file():
        lakes_focus = gpd.clip(gpd.read_file(NE_LAKES_50M), focus_buf).to_crs(ROBIN_FOCUS)
        lakes_focus.plot(ax=ax_a, facecolor="#F2F6FA", edgecolor="#B8C5CE", linewidth=0.20, zorder=1.6)

    x_min, _ = tr_f.transform(-12.0, 28.0)
    x_max, _ = tr_f.transform(128.0, 28.0)
    _, y_min = tr_f.transform(58.0, 6.0)
    _, y_max = tr_f.transform(58.0, 52.0)
    ax_a.set_xlim(x_min, x_max)
    ax_a.set_ylim(y_min, y_max)
    ax_a.set_aspect("equal", adjustable="box")
    ax_a.set_anchor("S")
    ax_a.set_xticks([])
    ax_a.set_yticks([])
    for sp in ("top", "right", "bottom", "left"):
        ax_a.spines[sp].set_visible(True)
        ax_a.spines[sp].set_color("#BCC7D1")
        ax_a.spines[sp].set_linewidth(0.58)

    tier_palette = {
        "OUTSIDE": "#CBD6DF",
        "MARGINAL": "#E8C478",
        "PERSISTENT_3": "#D95F28",
        "LOCKED": "#7A111B",
        "EMERGING": "#007C80",
    }
    tier_mha = {
        t: float(fcz_base.loc[fcz_base["risk_tier"] == t, "salt_irrig_ha"].sum() / 1e6)
        for t in ["LOCKED", "PERSISTENT_3", "EMERGING", "MARGINAL", "OUTSIDE"]
    }
    print(
        f"[AUDIT Fig5] Persistence x Emergence (Mha, sum={sum(tier_mha.values()):.2f}): "
        + ", ".join(f"{k}={v:.2f}" for k, v in tier_mha.items())
    )

    cell_geoms = [
        box(lon - 0.25, lat - 0.25, lon + 0.25, lat + 0.25)
        for lon, lat in zip(fcz_base["lon"], fcz_base["lat"])
    ]
    cell_gdf = gpd.GeoDataFrame(fcz_base[["risk_tier"]], geometry=cell_geoms, crs="EPSG:4326")
    cell_dis = cell_gdf.dissolve(by="risk_tier")
    cell_clip_main = gpd.clip(cell_dis, land_wgs_focus).to_crs(ROBIN_FOCUS)
    for tier_nm in ["OUTSIDE", "MARGINAL", "PERSISTENT_3", "LOCKED", "EMERGING"]:
        if tier_nm in cell_clip_main.index:
            cell_clip_main.loc[[tier_nm]].plot(
                ax=ax_a,
                facecolor=tier_palette[tier_nm],
                edgecolor="none",
                alpha=0.96,
                zorder=3.2 if tier_nm == "EMERGING" else 3.0,
            )

    for reg_lbl, r_lon, r_lat in [
        ("MENA & Med.", 17.5, 34.5),
        ("Central Asia", 67.0, 47.8),
        ("South Asia", 87.5, 15.5),
        ("East Asia", 123.5, 29.5),
    ]:
        lx, ly = tr_f.transform(r_lon, r_lat)
        ax_a.text(
            lx, ly, reg_lbl,
            fontsize=5.2, style="italic", color="#4A5A68",
            ha="center", va="center", zorder=5.5,
        )

    # Western US Inset neatly placed in the zero-salt Arabian Sea
    fig.canvas.draw()
    pos_a = ax_a.get_position()
    add_fig_panel_label(fig, 0.010, min(pos_a.y1 + 0.025, 0.985), "a")

    ROBIN_US = "+proj=robin +lon_0=-112 +datum=WGS84 +units=m +no_defs"
    tr_u = Transformer.from_crs("EPSG:4326", ROBIN_US, always_xy=True)
    ax_a_us = fig.add_axes([
        pos_a.x0 + 0.426 * pos_a.width,
        pos_a.y0 + 0.014,
        0.148 * pos_a.width,
        0.278 * pos_a.height,
    ])
    ax_a_us.set_facecolor("#F2F6FA")
    us_buf = box(-138.0, 12.0, -86.0, 54.0)
    land_us_wgs = gpd.clip(gpd.read_file(NE_LAND_50M), us_buf)
    land_us_rob = land_us_wgs.to_crs(ROBIN_US)

    us_grat_lons = np.linspace(-132.0, -92.0, 60)
    for glat in (30.0, 40.0):
        uxg, uyg = tr_u.transform(us_grat_lons, np.full_like(us_grat_lons, glat))
        ax_a_us.plot(uxg, uyg, ls="--", color="#DFE7EE", lw=0.35, zorder=0.4)
    land_us_rob.plot(ax=ax_a_us, facecolor="#ECEEE9", edgecolor="#9CA49E", linewidth=0.28, zorder=1.0)
    if NE_BORDERS_50M.is_file():
        borders_us = gpd.clip(gpd.read_file(NE_BORDERS_50M), us_buf).to_crs(ROBIN_US)
        borders_us.plot(ax=ax_a_us, color="#D5D8D2", linewidth=0.18, zorder=1.4)
    if NE_LAKES_50M.is_file():
        lakes_us = gpd.clip(gpd.read_file(NE_LAKES_50M), us_buf).to_crs(ROBIN_US)
        lakes_us.plot(ax=ax_a_us, facecolor="#F2F6FA", edgecolor="#B8C5CE", linewidth=0.18, zorder=1.6)

    cell_clip_us = gpd.clip(cell_dis, land_us_wgs).to_crs(ROBIN_US)
    for tier_nm in ["OUTSIDE", "MARGINAL", "PERSISTENT_3", "LOCKED", "EMERGING"]:
        if tier_nm in cell_clip_us.index:
            cell_clip_us.loc[[tier_nm]].plot(
                ax=ax_a_us,
                facecolor=tier_palette[tier_nm],
                edgecolor="none",
                alpha=0.96,
                zorder=3.2 if tier_nm == "EMERGING" else 3.0,
            )

    ux_min, _ = tr_u.transform(-124.5, 33.0)
    ux_max, _ = tr_u.transform(-100.5, 33.0)
    _, uy_min = tr_u.transform(-112.0, 22.5)
    _, uy_max = tr_u.transform(-112.0, 44.0)
    ax_a_us.set_xlim(ux_min, ux_max)
    ax_a_us.set_ylim(uy_min, uy_max)
    ax_a_us.set_aspect("equal", adjustable="box")
    ax_a_us.set_xticks([])
    ax_a_us.set_yticks([])
    for sp in ("top", "right", "bottom", "left"):
        ax_a_us.spines[sp].set_visible(True)
        ax_a_us.spines[sp].set_color("#5E6D7A")
        ax_a_us.spines[sp].set_linewidth(0.70)
    ax_a_us.text(0.06, 0.94, "W. US", transform=ax_a_us.transAxes, ha="left", va="top", fontsize=5.2, fontweight="bold", color="#222222", zorder=6)

    # Flush right: Two-tier joint Salinity–Aridity–GW Depletion panel
    reg_z3 = (
        fcz_all[fcz_all["is_z3"]]
        .groupby(["region", "gcm"], as_index=False)["salt_irrig_ha"]
        .sum()
    )
    reg_z3["mha"] = reg_z3["salt_irrig_ha"] / 1e6
    reg_piv = reg_z3.pivot(index="region", columns="gcm", values="mha").fillna(0.0)
    risk_path = OUT_TABLES / "pillar3_future_salinity_groundwater_cells_current.parquet"
    if risk_path.is_file():
        risk = pd.read_parquet(risk_path)
        combo = risk[risk.scenario.eq("ssp585_2050")].groupby(["r","c","lon","lat"],as_index=False).agg(
                        agreement=("saline_arid_and_gw","sum"), area_ha=("salt_irrig_ha","first"))
        indicators = risk[risk.scenario.eq("ssp585_2050")].pivot(index=["r","c"],columns="gcm",values="saline_arid_and_gw")
        combo.merge(indicators,on=["r","c"],validate="one_to_one").to_csv(SRC_DIR/"Fig5A_groundwater_salinity_map.csv",index=False)

    else:
        combo = pd.read_csv(SOURCE_DATA / "Fig5A_groundwater_salinity_map.csv")
        combo.to_csv(SRC_DIR / "Fig5A_groundwater_salinity_map.csv", index=False)

    # Single-line subtitle aligned horizontally with panel label "a"
    fig.text(
        0.642, min(pos_a.y1 + 0.010, 0.975),
        "Salinity, aridity & GW depletion (capped)",
        ha="left", va="bottom", fontsize=5.7, fontweight="bold", color="#222222",
    )

    # Upper sub-axes: Zoomed framed Robinson hotspot map (captures 100% of all 232 exposed cells)
    ROBIN_RIGHT = "+proj=robin +lon_0=50 +datum=WGS84 +units=m +no_defs"
    tr_r = Transformer.from_crs("EPSG:4326", ROBIN_RIGHT, always_xy=True)
    map_h = 0.168
    ax_a_map = fig.add_axes([0.642, pos_a.y1 - map_h, 0.344, map_h])
    ax_a_map.set_facecolor("#F2F6FA")

    right_buf = box(-22.0, 0.0, 128.0, 58.0)
    land_wgs_right = gpd.clip(gpd.read_file(NE_LAND_50M), right_buf)
    land_right = land_wgs_right.to_crs(ROBIN_RIGHT)

    r_grat_lons = np.linspace(-15.0, 120.0, 90)
    for glat in (20.0, 30.0, 40.0):
        gx, gy = tr_r.transform(r_grat_lons, np.full_like(r_grat_lons, glat))
        ax_a_map.plot(gx, gy, ls="--", color="#DFE7EE", lw=0.32, zorder=0.4)
    r_grat_lats = np.linspace(6.0, 52.0, 60)
    for glon in (0.0, 30.0, 60.0, 90.0):
        gx, gy = tr_r.transform(np.full_like(r_grat_lats, glon), r_grat_lats)
        ax_a_map.plot(gx, gy, ls=":", color="#DFE7EE", lw=0.32, zorder=0.4)

    land_right.plot(ax=ax_a_map, facecolor="#ECEEE9", edgecolor="#9CA49E", linewidth=0.25, zorder=1.0)
    if NE_BORDERS_50M.is_file():
        borders_right = gpd.clip(gpd.read_file(NE_BORDERS_50M), right_buf).to_crs(ROBIN_RIGHT)
        borders_right.plot(ax=ax_a_map, color="#D5D8D2", linewidth=0.16, zorder=1.4)
    if NE_LAKES_50M.is_file():
        lakes_right = gpd.clip(gpd.read_file(NE_LAKES_50M), right_buf).to_crs(ROBIN_RIGHT)
        lakes_right.plot(ax=ax_a_map, facecolor="#F2F6FA", edgecolor="#B8C5CE", linewidth=0.16, zorder=1.6)

    # Background context: all salt-affected irrigated cells in soft grey
    salt_clip_right = gpd.clip(cell_dis, land_wgs_right).to_crs(ROBIN_RIGHT)
    salt_clip_right.plot(ax=ax_a_map, facecolor="#D7DFE5", edgecolor="none", alpha=0.80, zorder=2.4)

    # Exposed hotspot cells dissolved & land-clipped into 3 agreement tiers
    pos_combo = combo[combo["agreement"] >= 1].copy()
    pos_combo["r_tier"] = np.where(
        pos_combo["agreement"] == 5, "UNAN",
        np.where(pos_combo["agreement"].isin([3, 4]), "CONS", "LOW"),
    )
    r_geoms = [box(lon - 0.25, lat - 0.25, lon + 0.25, lat + 0.25) for lon, lat in zip(pos_combo["lon"], pos_combo["lat"])]
    r_gdf = gpd.GeoDataFrame(pos_combo[["r_tier"]], geometry=r_geoms, crs="EPSG:4326")
    r_dis = r_gdf.dissolve(by="r_tier")
    r_clip = gpd.clip(r_dis, land_wgs_right).to_crs(ROBIN_RIGHT)
    r_palette = {"LOW": "#CFA8C6", "CONS": "#A34886", "UNAN": "#4A1456"}
    for r_tier, zo in [("LOW", 3.0), ("CONS", 3.2), ("UNAN", 3.4)]:
        if r_tier in r_clip.index:
            r_clip.loc[[r_tier]].plot(ax=ax_a_map, facecolor=r_palette[r_tier], edgecolor="none", alpha=0.96, zorder=zo)

    rx_min, _ = tr_r.transform(-9.0, 30.0)
    rx_max, _ = tr_r.transform(111.0, 30.0)
    _, ry_min = tr_r.transform(50.0, 12.0)
    _, ry_max = tr_r.transform(50.0, 48.0)
    ax_a_map.set_xlim(rx_min, rx_max)
    ax_a_map.set_ylim(ry_min, ry_max)
    ax_a_map.set_aspect("equal", adjustable="box")
    ax_a_map.set_anchor("N")
    ax_a_map.set_xticks([])
    ax_a_map.set_yticks([])
    for sp in ("top", "right", "bottom", "left"):
        ax_a_map.spines[sp].set_visible(True)
        ax_a_map.spines[sp].set_color("#BCC7D1")
        ax_a_map.spines[sp].set_linewidth(0.58)

    ax_a_map.text(0.025, 0.06, "2050, SSP5-8.5", transform=ax_a_map.transAxes, ha="left", va="bottom", fontsize=4.7, color="#4A5A68", zorder=6)
    for lbl, rlon, rlat in [("MENA", 14.0, 38.0), ("Indus", 73.0, 19.0), ("Hetao", 104.5, 36.8)]:
        lx, ly = tr_r.transform(rlon, rlat)
        ax_a_map.text(lx, ly, lbl, fontsize=4.5, style="italic", color="#4A5A68", ha="center", va="center", zorder=6)

    fig.canvas.draw()
    pos_map = ax_a_map.get_position()

    mha_5 = float(pos_combo.loc[pos_combo["r_tier"] == "UNAN", "area_ha"].sum() / 1e6)
    mha_34 = float(pos_combo.loc[pos_combo["r_tier"] == "CONS", "area_ha"].sum() / 1e6)
    r_leg_handles = [
        Patch(facecolor=r_palette["UNAN"], edgecolor="none", label=f"5/5 ({mha_5:.2f} Mha)"),
        Patch(facecolor=r_palette["CONS"], edgecolor="none", label=f"3–4/5 ({mha_34:.2f} Mha)"),
        Patch(facecolor=r_palette["LOW"], edgecolor="none", label="1–2/5 (<0.01 Mha)"),
    ]
    fig.legend(
        handles=r_leg_handles,
        loc="upper center",
        bbox_to_anchor=(0.814, pos_map.y0 - 0.004),
        ncol=3,
        frameon=False,
        fontsize=4.7,
        handlelength=0.85,
        handletextpad=0.25,
        columnspacing=0.45,
    )

    # Lower sub-axes: Matched 30-run trajectory strip across Baseline + 6 scenarios
    ts_y0 = pos_a.y0 + 0.004
    ts_h = max(pos_map.y0 - ts_y0 - 0.044, 0.085)
    ax_a_ts = fig.add_axes([0.684, ts_y0, 0.302, ts_h])
    sc_order_r = ["ssp126_2050", "ssp126_2080", "ssp370_2050", "ssp370_2080", "ssp585_2050", "ssp585_2080"]
    xt_r = np.array([0.0, 0.90, 2.22, 3.12, 4.44, 5.34])
    x_jit_r = [-0.14, -0.07, 0.0, 0.07, 0.14]

    base_joint_mha = float(
        grid.loc[
            grid["is_hyper_sal_arid"].fillna(False).astype(bool)
            & grid["gw_still_depleted_post"].fillna(False).astype(bool),
            "salt_irrig_ha",
        ].sum() / 1e6
    )
    if risk_path.is_file():
        risk_by_sc_gcm = risk[risk["saline_arid_and_gw"]].groupby(["scenario", "gcm"])["salt_irrig_ha"].sum() / 1e6
    else:
        risk_by_sc_gcm = pd.read_csv(table_input("pillar3_gcm_uncertainty_current.csv")).set_index(
            ["scenario", "gcm"])["sal_arid_and_gw_depleted_mha"]

    ax_a_ts.axvspan(1.62, 3.72, color="#F4F6F9", alpha=0.95, zorder=0.2)
    ax_a_ts.grid(axis="y", color="#EEF1F4", lw=0.45, zorder=0.3)
    ax_a_ts.axhline(base_joint_mha, ls="--", color="#4A5A68", lw=0.75, zorder=2.2)
    ax_a_ts.text(5.78, base_joint_mha + 0.07, f"Base: {base_joint_mha:.2f}", ha="right", va="bottom", fontsize=4.6, color="#4A5A68", zorder=4.5)

    for xi, sc in zip(xt_r, sc_order_r):
        vals = [float(risk_by_sc_gcm.get((sc, g_id), 0.0)) for g_id, _, _, _ in gcm_styles]
        v_lo, v_hi, v_med = min(vals), max(vals), float(np.median(vals))
        ax_a_ts.bar(xi, v_hi - v_lo, bottom=v_lo, width=0.44, color="#E8D8E6", edgecolor="#B284AA", linewidth=0.45, alpha=0.90, zorder=1.5)
        ax_a_ts.plot([xi - 0.24, xi + 0.24], [v_med, v_med], color="#4A1456", lw=1.7, solid_capstyle="round", zorder=4.2)
        for (_, _, col, mkr), val, jit in zip(gcm_styles, vals, x_jit_r):
            ax_a_ts.scatter([xi + jit], [val], color=col, marker=mkr, s=11, edgecolors="white", linewidths=0.30, zorder=5.0)
        if sc == "ssp585_2050":
            ax_a_ts.text(xi, v_med + 0.14, f"{v_med:.2f}", ha="center", va="bottom", fontsize=4.8, fontweight="bold", color="#4A1456", zorder=6.0)

    ax_a_ts.set_xlim(-0.54, 5.88)
    ax_a_ts.set_ylim(3.45, 5.75)
    ax_a_ts.set_yticks([3.5, 4.5, 5.5])
    ax_a_ts.tick_params(axis="y", labelsize=5.0, pad=1.5)
    ax_a_ts.set_ylabel("Joint risk\n(Mha)", fontsize=5.2, labelpad=1.5)
    ax_a_ts.set_xticks(xt_r)
    ax_a_ts.set_xticklabels(["'50", "'80", "'50", "'80", "'50", "'80"], fontsize=4.8)
    ax_a_ts.tick_params(axis="x", pad=1.5)
    for grp_x, grp_lbl in [(0.45, "SSP1-2.6"), (2.67, "SSP3-7.0"), (4.89, "SSP5-8.5")]:
        ax_a_ts.text(grp_x, -0.32, grp_lbl, transform=ax_a_ts.get_xaxis_transform(), ha="center", va="top", fontsize=4.8, fontweight="bold", color="#222222")

    # Left map 5-item legend in two compact rows strictly beneath ax_a
    map_leg_row1 = [
        Patch(facecolor=tier_palette["LOCKED"], edgecolor="none", label=f"Locked 4–5/5 ({tier_mha['LOCKED']:.1f} Mha)"),
        Patch(facecolor=tier_palette["PERSISTENT_3"], edgecolor="none", label=f"Baseline, 3/5 ({tier_mha['PERSISTENT_3']:.1f} Mha)"),
        Patch(facecolor=tier_palette["EMERGING"], edgecolor="none", label=f"Emerging ≥3/5 ({tier_mha['EMERGING']:.1f} Mha)"),
    ]
    map_leg_row2 = [
        Patch(facecolor=tier_palette["MARGINAL"], edgecolor="none", label=f"Marginal 1–2/5 ({tier_mha['MARGINAL']:.1f} Mha)"),
        Patch(facecolor=tier_palette["OUTSIDE"], edgecolor="none", label=f"Outside ({tier_mha['OUTSIDE']:.1f} Mha)"),
    ]
    fig.legend(
        handles=map_leg_row1,
        loc="upper center",
        bbox_to_anchor=(0.315, pos_a.y0 - 0.010),
        ncol=3,
        frameon=False,
        fontsize=5.0,
        handlelength=1.0,
        handletextpad=0.28,
        columnspacing=0.65,
    )
    fig.legend(
        handles=map_leg_row2,
        loc="upper center",
        bbox_to_anchor=(0.315, pos_a.y0 - 0.036),
        ncol=2,
        frameon=False,
        fontsize=5.0,
        handlelength=1.0,
        handletextpad=0.28,
        columnspacing=0.80,
    )

    # =========================================================================
    # Panel b: Two-Boundary Global Salt Ledger & Dual-Scenario Unfunded Debt
    # =========================================================================
    add_fig_panel_label(fig, 0.010, 0.546, "b")
    ax_b = fig.add_axes([0.056, 0.130, 0.284, 0.390])

    sv = p3["global_salt_affected_croplands_savings"]
    s_in_pre_all = float(obs_l["field_boundary_baseline_salt_input_all_salt_cells_mt_yr"])
    dS = float(sv["delta_s_salt_import_mt_yr"])
    dS_field = float(sv["delta_s_field_recycled_gross_mt_yr"])
    dS_fresh_disp = float(sv["delta_s_fresh_displaced_mt_yr"])
    dS_gw = float(sv["delta_s_basin_gw_gross_mt_yr"])
    dS_basin_net = float(sv["delta_s_basin_new_mt_yr"])
    s_post_fresh_gw = s_in_pre_all + dS_fresh_disp + dS_gw
    sd_pre = float(sv["salt_debt_pre_mt_yr"])
    sd_post_cap = float(sv["salt_debt_post_mt_yr"])
    sd_post_act = float(sv["salt_debt_post_actual_mt_yr"])
    basins_p3 = pd.read_csv(table_input("pillar3_basin_zoning_current.csv"), low_memory=False)
    sb_p3 = basins_p3[basins_p3["salt_irrig_ha"] >= 10000.0].copy()
    sb_net_in = np.maximum(sb_p3["s_in_pre_mt_yr"] + sb_p3["delta_s_basin_new_mt_yr"], 0.0)
    sd_basin_cap = float((sb_p3["u_post"] * sb_net_in).sum())
    sd_basin_act = float((sb_p3["u_post_actual"] * sb_net_in).sum())

    ax_b.axvspan(1.62, 2.86, color="#EFF8F5", alpha=0.85, zorder=0.2)
    ax_b.axvspan(2.86, 5.82, color="#FDF3F3", alpha=0.72, zorder=0.2)
    ax_b.grid(axis="y", color="#EEF1F4", lw=0.45, zorder=0.3)
    ax_b.axhline(0.0, color="#222222", lw=0.75, zorder=2.0)
    ax_b.axvline(1.62, ls="--", color="#9AA3AC", lw=0.65, zorder=2.0)
    ax_b.axvline(2.86, ls="--", color="#9AA3AC", lw=0.65, zorder=2.0)

    xb = np.array([0.0, 0.98, 2.24, 3.46, 4.34, 5.22])
    ax_b.bar(0.0, s_in_pre_all, width=0.60, color="#6FA8DC", alpha=0.92, zorder=3)
    ax_b.bar(0.98, s_post_fresh_gw, width=0.60, color="#6FA8DC", alpha=0.92, label="Freshwater + GW", zorder=3)
    ax_b.bar(0.98, dS_field, bottom=s_post_fresh_gw, width=0.60, color="#E09F24", alpha=0.92, label="Recycled return flow", zorder=3)
    ax_b.bar(2.24, dS_basin_net, width=0.60, color="#1B7867", alpha=0.92, label="Basin-boundary net", zorder=3)
    ax_b.bar(3.46, sd_pre, width=0.60, color="#E08E6D", alpha=0.90, zorder=3)
    ax_b.bar(4.34, sd_basin_cap, width=0.60, color="#1B7867", alpha=0.92, zorder=3)
    ax_b.bar(4.34, sd_post_cap - sd_basin_cap, bottom=sd_basin_cap, width=0.60, color="#C84B31", alpha=0.92, zorder=3)
    ax_b.bar(5.22, sd_basin_act, width=0.60, color="#1B7867", alpha=0.92, zorder=3)
    ax_b.bar(5.22, sd_post_act - sd_basin_act, bottom=sd_basin_act, width=0.60, color="#7A111B", alpha=0.95, zorder=3)

    ax_b.text(0.0, s_in_pre_all + 14, f"{s_in_pre_all:.0f}", ha="center", va="bottom", fontsize=5.2, fontweight="bold")
    ax_b.text(0.98, s_in_pre_all + dS + 14, f"{s_in_pre_all + dS:.0f}\n(+{dS:.0f})", ha="center", va="bottom", fontsize=5.2, fontweight="bold")
    ax_b.text(0.98, s_post_fresh_gw + 0.5 * dS_field, f"+{dS_field:.0f}", ha="center", va="center", fontsize=5.2, fontweight="bold", color="#111111", zorder=5)
    ax_b.text(2.24, dS_basin_net - 12, f"{dS_basin_net:+.0f}", ha="center", va="top", fontsize=5.2, fontweight="bold", color="#1B7867")
    ax_b.text(2.24, 22, "Recycled\n= 0", ha="center", va="bottom", fontsize=5.2, color="#555555")
    ax_b.text(3.46, sd_pre + 14, f"{sd_pre:.0f}", ha="center", va="bottom", fontsize=5.2, fontweight="bold", color="#B85D3B")
    ax_b.text(4.34, 0.5 * sd_basin_cap, f"{sd_basin_cap:.0f}", ha="center", va="center", fontsize=4.8, fontweight="bold", color="white", zorder=5)
    ax_b.text(4.34, sd_post_cap + 14, f"{sd_post_cap:.0f}", ha="center", va="bottom", fontsize=5.2, fontweight="bold", color="#C84B31")
    ax_b.text(5.22, 0.5 * sd_basin_act, f"{sd_basin_act:.0f}", ha="center", va="center", fontsize=4.8, fontweight="bold", color="white", zorder=5)
    ax_b.text(5.22, sd_post_act + 14, f"{sd_post_act:.0f}", ha="center", va="bottom", fontsize=5.2, fontweight="bold", color="#7A111B")

    ax_b.text(0.49, 1100, "Field\n(97.1 Mha)", ha="center", va="top", fontsize=5.2, fontweight="bold", color="#2B6E8C")
    ax_b.text(2.24, 1100, "Basin", ha="center", va="top", fontsize=5.2, fontweight="bold", color="#1B7867")
    ax_b.text(4.34, 1100, "Unfunded debt\n(376 basins)", ha="center", va="top", fontsize=5.2, fontweight="bold", color="#7A111B")

    ax_b.set_xticks(xb)
    ax_b.set_xticklabels(["Imp.\npre", "Imp.\npost", "Net\nshift", "Debt\npre", "Debt\ncap*", "Debt\nact."], fontsize=5.2)
    ax_b.set_xlim(-0.55, 5.82)
    ax_b.set_ylim(-115, 1130)
    ax_b.set_yticks([0, 200, 400, 600, 800, 1000])
    ax_b.tick_params(axis="y", labelsize=5.6)
    ax_b.set_ylabel(r"Annual salt mass flux (Mt yr$^{-1}$)", fontsize=6.3, labelpad=2.0)
    ax_b.legend(loc="upper right", bbox_to_anchor=(1.00, 0.80), frameon=False, fontsize=5.2, handlelength=1.0, handletextpad=0.35, labelspacing=0.25)

    # =========================================================================
    # Shared Bottom 5-GCM Ensemble Legend for Panels c, d, and a (right)
    # =========================================================================
    sc_order = ["ssp126_2050", "ssp126_2080", "ssp370_2050", "ssp370_2080", "ssp585_2050", "ssp585_2080"]
    xc = np.array([0.0, 0.96, 2.34, 3.30, 4.68, 5.64])
    x_jit = [-0.05, -0.025, 0.0, 0.025, 0.05]

    ens_leg_gcm_handles = [
        Line2D([0], [0], marker=mkr, color="none", markerfacecolor=col, markeredgecolor="white", markeredgewidth=0.45, markersize=4.4, label=lbl)
        for _, lbl, col, mkr in gcm_styles
    ]
    ens_leg_med_handles = [
        Line2D([0], [0], color="#111111", lw=1.9, label="Capped median"),
        Line2D([0], [0], color="#A61C28", lw=1.9, label="Actual-ET median"),
    ]
    fig.legend(
        handles=ens_leg_gcm_handles,
        loc="lower center",
        bbox_to_anchor=(0.680, 0.028),
        ncol=5,
        frameon=False,
        fontsize=5.2,
        handlelength=1.0,
        handletextpad=0.28,
        columnspacing=0.95,
    )
    fig.legend(
        handles=ens_leg_med_handles,
        loc="lower center",
        bbox_to_anchor=(0.680, 0.004),
        ncol=2,
        frameon=False,
        fontsize=5.2,
        handlelength=1.4,
        handletextpad=0.35,
        columnspacing=1.50,
    )

    # =========================================================================
    # Panel c: Dual-Governance Zone III Future Trajectories (Capped vs Actual ET)
    # =========================================================================
    add_fig_panel_label(fig, 0.356, 0.546, "c")
    ax_c = fig.add_axes([0.406, 0.130, 0.270, 0.390])
    future_checks=pd.read_csv(table_input("pillar3_future_threshold_sensitivity_current.csv"))
    future_checks.to_csv(SRC_DIR/"Fig5C_future_thresholds.csv",index=False)
    gcm_df["paired_zone3_difference_mha"]=gcm_df.zone3_actual_et_mha-gcm_df.zone3_mha
    true_med_rows=[]
    for xi,sc in zip(xc,sc_order):
        g=gcm_df[gcm_df.scenario==sc]
        harvest=future_checks[future_checks.scenario.eq(sc)&future_checks["case"].eq("Field harvest")]
        for (gcm,_,col,mkr),offset in zip(gcm_styles,[-.18,-.09,0,.09,.18]):
            val=g.loc[g.gcm==gcm,"paired_zone3_difference_mha"].iloc[0]
            ax_c.scatter(xi+offset,val,c=col,marker=mkr,s=21,edgecolors="white",linewidth=.35,zorder=3)
        med=g.paired_zone3_difference_mha.median()
        ax_c.plot([xi-.25,xi+.25],[med,med],color="#222222",lw=1.5)
        hm=(harvest.zone3_et_mha-harvest.zone3_capped_mha).median()
        ax_c.scatter(xi,hm,s=37,marker="D",facecolors="none",edgecolors="#222222",linewidth=.8,zorder=4)
        true_med_rows.append(dict(scenario=sc,zone3_mha_5gcm_output_median=g.zone3_mha.median(),
                zone3_actual_et_mha_5gcm_output_median=g.zone3_actual_et_mha.median(),
                paired_diff_mha=med,field_harvest_paired_diff_mha=hm,
                sal_arid_and_gw_mha=g.sal_arid_and_gw_depleted_mha.median(),
                exceeded_basins_5gcm_output_median=g.paradox_still_exceeded_post_basins.median(),
                rescued_basins_5gcm_output_median=g.paradox_basins_relieved.median(),
                exceeded_actual_basins_5gcm_output_median=g.paradox_still_exceeded_actual_basins.median(),
                rescued_actual_basins_5gcm_output_median=g.paradox_basins_relieved_actual.median()))
    ax_c.axhline(0,ls=":",color="#555555",lw=.8)
    ax_c.set(xlim=(-.55,6.2),ylim=(-.8,18.5),xticks=xc,xticklabels=["’50","’80"]*3,
             ylabel="Additional Zone III area (Mha)")
    ax_c.tick_params(labelsize=5.7)
    for x,label in zip([.48,2.82,5.16],["SSP1-2.6","SSP3-7.0","SSP5-8.5"]):
        ax_c.text(x,-.14,label,transform=ax_c.get_xaxis_transform(),ha="center",fontsize=5.4)
    ax_c.legend(handles=[Line2D([],[],color="#222222",label="Mixed threshold median"),
                         Line2D([],[],marker="D",color="#222222",ls="",mfc="none",label="Field harvest median")],
                loc="upper left",frameon=False,fontsize=5.3)
    # =========================================================================
    # Panel d: Dual-Governance Exceeded (Top) & Rescued Basins (Bottom)
    # =========================================================================
    add_fig_panel_label(fig, 0.690, 0.546, "d")
    ax_d1 = fig.add_axes([0.738, 0.322, 0.248, 0.198])
    ax_d2 = fig.add_axes([0.738, 0.130, 0.248, 0.162])

    base_exc_cap = int(sv["paradox_still_exceeded_post_basins"])
    base_exc_act = int(sv["paradox_still_exceeded_actual_basins"])
    base_rel_cap = int(sv["paradox_basins_relieved_to_blue_feasible"])

    # d1: Still exceeded (Capped vs Actual ET)
    ax_d1.axvspan(1.76, 3.88, color="#F4F6F9", alpha=0.95, zorder=0.2)
    ax_d1.grid(axis="y", color="#EEF1F4", lw=0.45, zorder=0.3)
    ax_d1.plot([-0.58, 6.02], [base_exc_cap, base_exc_cap], ls="--", color="#222222", lw=0.80, zorder=2.2)
    ax_d1.plot([-0.58, 6.02], [base_exc_act, base_exc_act], ls=":", color="#A61C28", lw=0.90, zorder=2.2)
    for xi, sc in zip(xc, sc_order):
        sub_g = gcm_df[gcm_df["scenario"] == sc]
        c_lo, c_hi, c_med = float(sub_g["paradox_still_exceeded_post_basins"].min()), float(sub_g["paradox_still_exceeded_post_basins"].max()), float(sub_g["paradox_still_exceeded_post_basins"].median())
        x_c0 = xi - 0.19
        ax_d1.bar(x_c0, c_hi - c_lo, bottom=c_lo, width=0.28, color="#D6E2EC", edgecolor="#8FA8BE", linewidth=0.45, alpha=0.88, zorder=1.5)
        ax_d1.plot([x_c0 - 0.16, x_c0 + 0.16], [c_med, c_med], color="#111111", lw=1.8, solid_capstyle="round", zorder=4.2)
        for (g_id, _, col, mkr), jit in zip(gcm_styles, x_jit):
            val = float(sub_g.loc[sub_g["gcm"] == g_id, "paradox_still_exceeded_post_basins"].iloc[0])
            ax_d1.scatter([x_c0 + jit], [val], color=col, marker=mkr, s=13, edgecolors="white", linewidths=0.35, zorder=5.0)

        a_lo, a_hi, a_med = float(sub_g["paradox_still_exceeded_actual_basins"].min()), float(sub_g["paradox_still_exceeded_actual_basins"].max()), float(sub_g["paradox_still_exceeded_actual_basins"].median())
        x_a0 = xi + 0.19
        ax_d1.bar(x_a0, a_hi - a_lo, bottom=a_lo, width=0.28, color="#F8D7DA", edgecolor="#C96A73", linewidth=0.45, alpha=0.88, zorder=1.6)
        ax_d1.plot([x_a0 - 0.16, x_a0 + 0.16], [a_med, a_med], color="#A61C28", lw=1.8, solid_capstyle="round", zorder=4.3)
        for (g_id, _, col, mkr), jit in zip(gcm_styles, x_jit):
            val = float(sub_g.loc[sub_g["gcm"] == g_id, "paradox_still_exceeded_actual_basins"].iloc[0])
            ax_d1.scatter([x_a0 + jit], [val], color=col, marker=mkr, s=13, edgecolors="white", linewidths=0.35, zorder=5.0)

    ax_d1.text(0.02, 0.96, "Still exceeded", transform=ax_d1.transAxes, ha="left", va="top", fontsize=5.2, fontweight="bold", color="#33414D")
    ax_d1.text(6.08, base_exc_cap, f"Cap: {base_exc_cap}", ha="left", va="center", fontsize=5.2, color="#222222")
    ax_d1.text(6.08, base_exc_act, f"Act.: {base_exc_act}", ha="left", va="center", fontsize=5.2, fontweight="bold", color="#A61C28")
    ax_d1.set_xlim(-0.58, 7.15)
    ax_d1.set_ylim(48, 108)
    ax_d1.set_yticks([50, 75, 100])
    ax_d1.tick_params(axis="y", labelsize=5.4)
    ax_d1.set_xticks(xc)
    ax_d1.set_xticklabels([])
    ax_d1.tick_params(axis="x", length=0)

    # d2: Rescued basins (Capped vs Actual ET 0 median)
    ax_d2.axvspan(1.76, 3.88, color="#F4F6F9", alpha=0.95, zorder=0.2)
    ax_d2.grid(axis="y", color="#EEF1F4", lw=0.45, zorder=0.3)
    ax_d2.plot([-0.58, 6.02], [base_rel_cap, base_rel_cap], ls="--", color="#1B7867", lw=0.85, zorder=2.2)
    for xi, sc in zip(xc, sc_order):
        sub_g = gcm_df[gcm_df["scenario"] == sc]
        lo_v, hi_v, med_v = float(sub_g["paradox_basins_relieved"].min()), float(sub_g["paradox_basins_relieved"].max()), float(sub_g["paradox_basins_relieved"].median())
        ax_d2.bar(xi, hi_v - lo_v, bottom=lo_v, width=0.36, color="#D6E2EC", edgecolor="#8FA8BE", linewidth=0.45, alpha=0.88, zorder=1.5)
        ax_d2.plot([xi - 0.22, xi + 0.22], [med_v, med_v], color="#111111", lw=1.9, solid_capstyle="round", zorder=4.2)
        for (g_id, _, col, mkr), jit in zip(gcm_styles, [-0.10, -0.05, 0.0, 0.05, 0.10]):
            val = float(sub_g.loc[sub_g["gcm"] == g_id, "paradox_basins_relieved"].iloc[0])
            ax_d2.scatter([xi + jit], [val], color=col, marker=mkr, s=16, edgecolors="white", linewidths=0.38, zorder=5.0)

    ax_d2.text(0.02, 0.96, "Rescued (≤ SBWA)", transform=ax_d2.transAxes, ha="left", va="top", fontsize=5.2, fontweight="bold", color="#1B7867")
    ax_d2.text(6.08, base_rel_cap, f"Base: {base_rel_cap}", ha="left", va="center", fontsize=5.2, fontweight="bold", color="#1B7867")
    act_rel_med = int(round(float(gcm_df["paradox_basins_relieved_actual"].median())))
    act_rel_max = int(gcm_df["paradox_basins_relieved_actual"].max())
    act_rel_lbl = "Actual ET: 0 (all 30 runs)" if act_rel_max == 0 else f"Actual ET med.={act_rel_med} (max {act_rel_max})"
    ax_d2.text(0.98, 0.05, act_rel_lbl, transform=ax_d2.transAxes, ha="right", va="bottom", fontsize=5.2, fontweight="bold", color="#A61C28")
    ax_d2.set_xlim(-0.58, 7.15)
    ax_d2.set_ylim(-0.5, 24.0)
    ax_d2.set_yticks([0, 10, 20])
    ax_d2.tick_params(axis="y", labelsize=5.4)
    ax_d2.set_xticks(xc)
    ax_d2.set_xticklabels(["'50", "'80", "'50", "'80", "'50", "'80"], fontsize=5.35)
    for grp_x, grp_lbl in [(0.48, "SSP1-2.6"), (2.82, "SSP3-7.0"), (5.16, "SSP5-8.5")]:
        ax_d2.text(grp_x, -6.5, grp_lbl, ha="center", va="top", fontsize=5.3, fontweight="bold", color="#222222")
    fig.text(0.700, 0.324, "Major basins (n)", rotation=90, va="center", ha="center", fontsize=6.3)

    fut_out = fut.merge(pd.DataFrame(true_med_rows), on="scenario", how="left")
    fcz_base[["r", "c", "lat", "lon", "salt_irrig_ha", "zone", "zone3_gcm_agreement_frac", "agree_n", "region", "risk_tier"]].to_csv(SRC_DIR / "Fig5A_ssp585_2050_zone3_agreement.csv", index=False)
    reg_piv.to_csv(SRC_DIR / "Fig5A_regional_z3_2020_vs_2050_gcms.csv")
    pd.DataFrame([
        {"metric": "salt_import_pre_all_salt_cells", "mt_yr": s_in_pre_all},
        {"metric": "salt_import_post_all_salt_cells", "mt_yr": s_in_pre_all + dS},
        {"metric": "delta_s_field_recycled_gross", "mt_yr": dS_field},
        {"metric": "delta_s_fresh_displaced", "mt_yr": dS_fresh_disp},
        {"metric": "delta_s_basin_gw_gross", "mt_yr": dS_gw},
        {"metric": "delta_s_basin_new_net", "mt_yr": dS_basin_net},
        {"metric": "unfunded_salt_debt_pre_376_basins", "mt_yr": sd_pre},
        {"metric": "unfunded_salt_debt_post_equal_prod_376_basins", "mt_yr": sd_post_cap},
        {"metric": "unfunded_salt_debt_post_actual_et_376_basins", "mt_yr": sd_post_act},
        {"metric": "unfunded_salt_debt_post_basin_boundary_equal_prod_376_basins", "mt_yr": round(sd_basin_cap, 2)},
        {"metric": "unfunded_salt_debt_post_basin_boundary_actual_et_376_basins", "mt_yr": round(sd_basin_act, 2)},
    ]).to_csv(SRC_DIR / "Fig5B_global_salt_fluxes.csv", index=False)
    fut_out.to_csv(SRC_DIR / "Fig5CD_future_scenarios_medians.csv", index=False)
    gcm_df.to_csv(SRC_DIR / "Fig5CD_future_gcm_ensemble.csv", index=False)

    save_pub_figure(fig, FIG_DIR / "Fig5_Limits_Salt_Future.pdf", FIG_DIR / "Fig5_Limits_Salt_Future.png", "Fig5")


def _cr1_egger_z(lnrr: np.ndarray, se: np.ndarray, fam: np.ndarray) -> tuple[float, float, int]:
    """CR1 cluster-robust Egger test (z = lnRR/se ~ 1 + 1/se) weighted by 1/n_family."""
    from scipy import stats as sp_stats
    s = pd.DataFrame({"y": lnrr, "se": se, "fam": fam.astype(str)}).dropna()
    s = s[s["se"] > 1e-6].copy()
    z = s["y"].values / s["se"].values
    prec = 1.0 / s["se"].values
    cl = s["fam"].values
    uniq = np.unique(cl)
    n_cl = len(uniq)
    fc = s.groupby("fam")["y"].transform("count").values.clip(min=1)
    w = 1.0 / fc
    X = np.column_stack([np.ones(len(s)), prec])
    XtWX = X.T @ (w[:, None] * X) + 1e-10 * np.eye(2)
    beta = np.linalg.solve(XtWX, X.T @ (w * z))
    res = z - X @ beta
    inv = np.linalg.pinv(XtWX)
    meat = sum(np.outer(X[cl == c].T @ (w[cl == c] * res[cl == c]), X[cl == c].T @ (w[cl == c] * res[cl == c])) for c in uniq)
    cov = (n_cl / max(n_cl - 1, 1)) * (inv @ meat @ inv)
    se0 = float(np.sqrt(max(cov[0, 0], 1e-12)))
    p0 = float(2.0 * sp_stats.t.sf(abs(beta[0] / se0), df=max(n_cl - 2, 1)))
    return float(beta[0]), p0, int(n_cl)


def _fmt_hu(val: float, dec: int = 1, signed: bool = False) -> str:
    q = Decimal(1).scaleb(-dec)
    d = Decimal(str(float(val))).quantize(q, rounding=ROUND_HALF_UP)
    s = format(d, "f")
    if signed and d > 0:
        s = "+" + s
    return s.replace("-", "−")


def build_ed_fig1() -> None:
    p2a = json.loads((table_input("pillar2_armB_hydraulic_results_current.json")).read_text(encoding="utf-8"))
    p2b = json.loads((table_input("pillar2_armA_ecw_results_current.json")).read_text(encoding="utf-8"))
    arma = pd.read_csv(HARMONIZED / "armA_extracted_current.csv", low_memory=False)
    armb = pd.read_csv(grid_input("pillar2_armB_merged_current.csv"), low_memory=False)

    fig = plt.figure(figsize=(179 / 25.4, 118 / 25.4))

    # Panel a: Conserved Contrast Corpus Waterfall (all bars in k contrasts)
    add_fig_panel_label(fig, 0.010, 0.980, "a")
    ax_a = fig.add_axes([0.305, 0.578, 0.675, 0.375])
    screening = pd.read_csv(SOURCE_DATA / "ED_Fig3A_prisma_counts.csv")
    stages = screening["stage"].tolist()
    source_counts = armb["source_corpus"].value_counts()
    backfill = int(source_counts["targeted_hydraulic_backfill"])
    counts = [len(arma), p2b["usable_rows_total"], len(armb) - backfill, backfill, len(armb)]
    cols = [PALETTE["sky"], PALETTE["blue"], "#4A7C59", PALETTE["amber"], PALETTE["teal"]]
    ya = np.arange(len(stages))[::-1]
    ax_a.grid(axis="x", color="#EEF1F4", lw=0.45, zorder=0.3)
    ax_a.barh(ya, counts, height=0.58, color=cols, alpha=0.92, zorder=2.5)
    for yi, cnt in zip(ya, counts):
        ax_a.text(cnt + 110, yi, f"{cnt:,}", va="center", ha="left", fontsize=6.2, fontweight="bold", color="#222222")
    ax_a.set_yticks(ya)
    ax_a.set_yticklabels(stages, fontsize=5.6)
    ax_a.set_xlim(0, 8150)
    ax_a.tick_params(axis="x", labelsize=5.6)
    ax_a.set_xlabel(r"Effect-size contrast count ($\mathit{k}$)", fontsize=6.2, labelpad=2.0)

    # Panel b: Arm A Dose-Adjusted Strict Exact-Variance Yield Residual Funnel (k=298, 30 cl.)
    add_fig_panel_label(fig, 0.010, 0.495, "b")
    ax_b = fig.add_axes([0.076, 0.095, 0.415, 0.375])
    ready = arma[arma["qc_gate_status"] == "PASS_READY"].copy()
    yld_m = ["grain_yield", "seed_cotton_yield", "fruit_yield", "tuber_yield", "marketable_yield", "crop_yield", "lint_yield", "fresh_yield", "cane_yield", "shoot_biomass"]
    r_y = ready[ready["metric_name"].isin(yld_m)].copy()
    for col_num in ["mean_treatment", "mean_control", "error_treatment", "error_control", "n_replicates", "ecw_treatment_dsm", "ecw_control_dsm"]:
        r_y[col_num] = pd.to_numeric(r_y[col_num], errors="coerce")
    r_y = r_y[(r_y["mean_treatment"] > 0) & (r_y["mean_control"] > 0) & (r_y["ecw_treatment_dsm"] > r_y["ecw_control_dsm"])].copy()
    et_u = r_y["error_type"].fillna("").astype(str).str.upper()
    has_var = (
        r_y["error_treatment"].notna() & r_y["error_control"].notna()
        & (r_y["error_treatment"] > 0) & (r_y["error_control"] > 0)
        & r_y["n_replicates"].notna() & (r_y["n_replicates"] >= 2)
        & (~et_u.str.contains("LSD|CV|NOT_REPORTED"))
    )
    r_y = r_y[has_var].copy()
    et_u_f = r_y["error_type"].fillna("").astype(str).str.upper()
    n_eff_b = r_y["n_replicates"].values
    sd_t = np.where(et_u_f.str.contains("SE") & (~et_u_f.str.contains("SD")), r_y["error_treatment"].values * np.sqrt(n_eff_b), r_y["error_treatment"].values)
    sd_c = np.where(et_u_f.str.contains("SE") & (~et_u_f.str.contains("SD")), r_y["error_control"].values * np.sqrt(n_eff_b), r_y["error_control"].values)
    vi_ex = (sd_t ** 2) / (n_eff_b * (r_y["mean_treatment"].values ** 2)) + (sd_c ** 2) / (n_eff_b * (r_y["mean_control"].values ** 2))
    valid_v = np.isfinite(vi_ex) & (vi_ex > 1e-8)
    r_y = r_y[valid_v].copy()
    r_y["se"] = np.sqrt(np.minimum(vi_ex[valid_v], 5.0))
    r_y["lnRR"] = np.log(r_y["mean_treatment"].values / r_y["mean_control"].values)
    tp_a, bp_a = float(p2b["dose_response_absolute"]["pooled"]["t_dsm"]), float(p2b["dose_response_absolute"]["pooled"]["b_per_dsm"])
    ry_t = np.maximum(1.0 - bp_a * np.maximum(0.0, r_y["ecw_treatment_dsm"].values - tp_a), 0.05)
    ry_c = np.maximum(1.0 - bp_a * np.maximum(0.0, r_y["ecw_control_dsm"].values - tp_a), 0.05)
    r_y["resid_lnRR"] = r_y["lnRR"].values - np.log(ry_t / ry_c)
    eg_int_a, eg_p_a, n_cl_a = _cr1_egger_z(r_y["resid_lnRR"].values, r_y["se"].values, r_y["family_id"].values)

    ax_b.grid(axis="both", color="#EEF1F4", lw=0.45, zorder=0.3)
    prec_a = 1.0 / r_y["se"].values
    y_cap_b = 90.0
    in_b = prec_a <= y_cap_b
    ax_b.axhline(y_cap_b, ls="--", color="#AAAAAA", lw=0.60, zorder=1.2)
    ax_b.scatter(r_y.loc[in_b, "resid_lnRR"], prec_a[in_b], s=10, color=PALETTE["blue"], alpha=0.55, edgecolors="none", zorder=2.5)
    if (~in_b).any():
        ax_b.scatter(r_y.loc[~in_b, "resid_lnRR"], np.full((~in_b).sum(), y_cap_b + 5.5), marker="^", s=14, color=PALETTE["blue"], alpha=0.85, edgecolors="white", linewidths=0.3, zorder=3.0)
    ax_b.axvline(0.0, ls="--", color=PALETTE["red"], lw=1.0, zorder=3.5)
    se_grid = np.linspace(0.0112, 0.90, 120)
    ax_b.plot(-1.96 * se_grid, 1.0 / se_grid, ":", color="#666666", lw=0.8, zorder=3.0)
    ax_b.plot(+1.96 * se_grid, 1.0 / se_grid, ":", color="#666666", lw=0.8, zorder=3.0)
    n_hi_b = int((~in_b).sum())
    cap_note_b = f"; {n_hi_b} at >{int(y_cap_b)}" if n_hi_b > 0 else ""
    ax_b.text(
        0.98, 0.96,
        rf"Arm A yield ($\mathit{{k}}$ = {len(r_y)}, {n_cl_a} cl.)" + "\n" + rf"Egger {fmt_p(eg_p_a)} · ▲ {n_hi_b} obs > {int(y_cap_b)}",
        transform=ax_b.transAxes, ha="right", va="top", fontsize=5.2, color="#222222",
        bbox={"boxstyle": "square,pad=0.12", "facecolor": "white", "edgecolor": "none", "alpha": 0.88}, zorder=5.0,
    )
    ax_b.set_xlim(-2.0, 1.95)
    ax_b.set_ylim(0, y_cap_b + 14.0)
    ax_b.set_yticks([0, 20, 40, 60, 80, 90])
    ax_b.tick_params(labelsize=5.6)
    ax_b.set_xlabel(r"Yield residual ($\Delta\mathrm{lnRR}$ vs Maas–Hoffman)", fontsize=6.0, labelpad=2.0)
    ax_b.set_ylabel("Precision (1/SE)", fontsize=6.0, labelpad=2.0)

    # Panel c: Arm B Endpoint-Centered Exact-Variance Residual Funnel (k=205)
    add_fig_panel_label(fig, 0.515, 0.495, "c")
    ax_c = fig.add_axes([0.568, 0.095, 0.415, 0.375])
    hyd_b = armb[armb["metric_bucket"].isin(["ks_saturated", "infiltration_rate_steady", "infiltration_rate_initial", "infiltration_rate_unspecified", "infiltration_rate_master"]) & (armb["has_exact_variance"] == True)].copy()
    hyd_b["se"] = np.sqrt(pd.to_numeric(hyd_b["vi_exact"], errors="coerce"))
    hyd_b = hyd_b[hyd_b["se"] > 1e-6].copy()
    h_map_b = {r["endpoint"]: r for r in p2a["headline_all_endpoints"]}
    mu_ks_b = float(h_map_b["ks_saturated"]["mu_lnrr"])
    mu_inf_b = float(h_map_b["infiltration_rate_all"]["mu_lnrr"])
    is_ks = (hyd_b["metric_bucket"] == "ks_saturated").values
    hyd_b["resid_lnRR"] = hyd_b["lnRR"].values - np.where(is_ks, mu_ks_b, mu_inf_b)

    ax_c.grid(axis="both", color="#EEF1F4", lw=0.45, zorder=0.3)
    prec_c = 1.0 / hyd_b["se"].values
    y_cap_c = 62.0
    in_c = prec_c <= y_cap_c
    ax_c.axhline(y_cap_c, ls="--", color="#AAAAAA", lw=0.60, zorder=1.2)
    for mask_ep, lbl_ep, col_ep in [(is_ks, r"$\mathit{K_s}$", "#D55E00"), (~is_ks, "Infiltration", PALETTE["teal"])]:
        m_in = mask_ep & in_c
        m_hi = mask_ep & (~in_c)
        ax_c.scatter(hyd_b.loc[m_in, "resid_lnRR"], prec_c[m_in], s=10.5, color=col_ep, alpha=0.58, edgecolors="none", label=rf"{lbl_ep} ($\mathit{{k}}$ = {int(mask_ep.sum())})", zorder=2.5)
        if m_hi.any():
            ax_c.scatter(hyd_b.loc[m_hi, "resid_lnRR"], np.full(int(m_hi.sum()), y_cap_c + 4.8), marker="^", s=13, color=col_ep, alpha=0.85, edgecolors="white", linewidths=0.3, zorder=3.0)
    ax_c.axvline(0.0, ls="--", color=PALETTE["red"], lw=1.0, zorder=3.5)
    se_b = np.linspace(0.0165, 1.35, 120)
    ax_c.plot(-1.96 * se_b, 1.0 / se_b, ":", color="#666666", lw=0.8, zorder=3.0)
    ax_c.plot(+1.96 * se_b, 1.0 / se_b, ":", color="#666666", lw=0.8, zorder=3.0)
    eg_b = p2a["exact_variance_egger_checks"]
    n_hi_c = int((~in_c).sum())
    ax_c.text(
        0.98, 0.96,
        rf"Arm B hydraulics ($\mathit{{k}}$ = {len(hyd_b)})"
        + "\n"
        + rf"Egger {fmt_p(eg_b['ks_saturated']['egger_p'])} / {fmt_p(eg_b['infiltration_rate_all']['egger_p'])} · ▲ {n_hi_c} > {int(y_cap_c)}",
        transform=ax_c.transAxes, ha="right", va="top", fontsize=5.2, color="#222222",
        bbox={"boxstyle": "square,pad=0.12", "facecolor": "white", "edgecolor": "none", "alpha": 0.88}, zorder=5.0,
    )
    ax_c.legend(loc="upper right", bbox_to_anchor=(0.98, 0.72), frameon=False, fontsize=5.2, handletextpad=0.25)
    ax_c.set_xlim(-3.2, 3.35)
    ax_c.set_ylim(0, y_cap_c + 12.0)
    ax_c.set_yticks([0, 15, 30, 45, 62])
    ax_c.tick_params(labelsize=5.6)
    ax_c.set_xlabel(r"Hydraulic residual ($\Delta\mathrm{lnRR}$ vs REML $\mu$)", fontsize=6.0, labelpad=2.0)
    ax_c.set_ylabel("Precision (1/SE)", fontsize=6.0, labelpad=2.0)

    SUP_SRC_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"stage": stages, "count": counts}).to_csv(SUP_SRC_DIR / "ED_Fig3A_prisma_counts.csv", index=False)
    r_y[["family_id", "metric_name", "lnRR", "resid_lnRR", "se"]].to_csv(SUP_SRC_DIR / "ED_Fig3B_arma_yield_funnel.csv", index=False)
    hyd_b[["family_id", "metric_bucket", "lnRR", "resid_lnRR", "se"]].to_csv(SUP_SRC_DIR / "ED_Fig3C_armb_hydraulic_funnel.csv", index=False)
    save_pub_figure(fig, SUP_DIR / "ED_Fig3_PRISMA_Funnel_Bias.pdf", SUP_DIR / "ED_Fig3_PRISMA_Funnel_Bias.png", "ED_Fig3")


def build_ed_fig2() -> None:
    hv = json.loads((table_input("hydrological_independent_validation_current.json")).read_text(encoding="utf-8"))
    p1b = json.loads((table_input("pillar1b_salt_debt_closure_sensitivity_current.json")).read_text(encoding="utf-8"))
    bv = pd.read_csv(table_input("basin_grace_jasechko_validation_current.csv"), low_memory=False)
    ghm = pd.read_csv(table_input("pillar1_multighm_structural_validation_current.csv"))

    fig = plt.figure(figsize=(179 / 25.4, 122 / 25.4))
    gs = gridspec.GridSpec(2, 3, figure=fig, left=0.072, right=0.988, top=0.945, bottom=0.095, hspace=0.35, wspace=0.28)

    # Panel a: Continuous V_leach vs Uniform ECw Curve & Jensen Inequality (first cited in Results)
    ax_a = fig.add_subplot(gs[0, 0])
    add_panel_label(ax_a, "a", x=-0.24, y=1.06)
    curve_df = pd.DataFrame(p1b["continuous_vleach_ecw_curve"])
    jen = p1b["ambient_vleach_jensen_inequality"]
    ax_a.plot(curve_df["ecw_uniform_dsm"], curve_df["vleach_all_irrig_km3_yr"], "-", color=NW_FIG1_PALETTE["all_irrig"], lw=1.5, label="All irrigated (290.6 Mha)", zorder=4)
    ax_a.plot(curve_df["ecw_uniform_dsm"], curve_df["vleach_salt_cells_km3_yr"], "-", color=NW_FIG1_PALETTE["salt_cells"], lw=1.5, label="Salt-affected (97.1 Mha)", zorder=4)
    ax_a.fill_between(curve_df["ecw_uniform_dsm"], 0.0, curve_df["vleach_salt_cells_km3_yr"], color=NW_FIG1_PALETTE["salt_cells"], alpha=0.06, zorder=1)

    amb_all = jen["ambient_vleach_all_irrig_km3_yr"]
    amb_salt = jen["ambient_vleach_salt_cells_km3_yr"]
    eq_all = jen["equivalent_uniform_ecw_all_irrig_dsm"]
    eq_salt = jen["equivalent_uniform_ecw_salt_cells_dsm"]

    ax_a.plot([0.15, eq_all, eq_all], [amb_all, amb_all, 0.0], ls="--", color=NW_FIG1_PALETTE["all_irrig"], lw=0.75, alpha=0.85, zorder=2)
    ax_a.plot([0.15, eq_salt, eq_salt], [amb_salt, amb_salt, 0.0], ls=":", color=NW_FIG1_PALETTE["salt_cells"], lw=0.90, alpha=0.90, zorder=2)
    ax_a.scatter([eq_all], [amb_all], color=NW_FIG1_PALETTE["all_irrig"], s=30, edgecolor="white", lw=0.8, zorder=6)
    ax_a.scatter([eq_salt], [amb_salt], color=NW_FIG1_PALETTE["salt_cells"], s=30, edgecolor="white", lw=0.8, zorder=6)

    g1_p = grid_input("pillar1_grid05_baseline_current.parquet")
    if g1_p.is_file():
        g1_df = pd.read_parquet(g1_p, columns=["irrig_active", "salt_affected", "ecw_ambient_dsm", "pirruse_km3_yr"])
        g1_act = g1_df[g1_df["irrig_active"] & g1_df["ecw_ambient_dsm"].notna()]
        g1_slt = g1_df[g1_df["salt_affected"] & g1_df["ecw_ambient_dsm"].notna()]
        mean_all = float(np.average(g1_act["ecw_ambient_dsm"], weights=g1_act["pirruse_km3_yr"]))
        mean_salt = float(np.average(g1_slt["ecw_ambient_dsm"], weights=g1_slt["pirruse_km3_yr"]))
        y_m_all = float(np.interp(mean_all, curve_df["ecw_uniform_dsm"], curve_df["vleach_all_irrig_km3_yr"]))
        y_m_salt = float(np.interp(mean_salt, curve_df["ecw_uniform_dsm"], curve_df["vleach_salt_cells_km3_yr"]))
        ax_a.scatter([mean_all], [y_m_all], marker="D", facecolor="white", edgecolor=NW_FIG1_PALETTE["all_irrig"], s=18, lw=0.9, zorder=6.2)
        ax_a.scatter([mean_salt], [y_m_salt], marker="D", facecolor="white", edgecolor=NW_FIG1_PALETTE["salt_cells"], s=18, lw=0.9, zorder=6.2)
        ax_a.text(0.04, 0.77, rf"$\diamond$ Mean $\mathrm{{EC_w}}$: {mean_all:.2f}, {mean_salt:.2f}", transform=ax_a.transAxes, ha="left", va="top", fontsize=5.2, color="#444444")

    ax_a.annotate(
        f"Amb. {amb_all:.1f} (eq. {eq_all:.2f})",
        xy=(eq_all, amb_all), xytext=(0.40, 56.0),
        fontsize=5.2, color=NW_FIG1_PALETTE["all_irrig"],
        arrowprops={"arrowstyle": "->", "color": NW_FIG1_PALETTE["all_irrig"], "lw": 0.55, "shrinkA": 2, "shrinkB": 3},
        zorder=7,
    )
    ax_a.annotate(
        f"Amb. {amb_salt:.1f} (eq. {eq_salt:.2f})",
        xy=(eq_salt, amb_salt), xytext=(2.40, 8.5),
        fontsize=5.2, color=NW_FIG1_PALETTE["salt_cells"],
        arrowprops={"arrowstyle": "->", "color": NW_FIG1_PALETTE["salt_cells"], "lw": 0.55, "shrinkA": 2, "shrinkB": 3},
        zorder=7,
    )
    ax_a.set_xlim(0.15, 6.10)
    ax_a.set_ylim(0.0, 134.0)
    ax_a.set_xlabel(r"Uniform $\mathrm{EC_w}$ (dS m$^{-1}$)")
    ax_a.set_ylabel(r"Leaching demand $\mathit{V}_{\mathrm{leach}}$ (km$^3$ yr$^{-1}$)")
    ax_a.legend(loc="upper left", frameon=False, fontsize=5.2, handlelength=1.15)

    # Panel b: Salt-Affected Cropland Definitions (Mha)
    ax_b = fig.add_subplot(gs[0, 1])
    add_panel_label(ax_b, "b", x=-0.20, y=1.06)
    cd = p1b["closure_decomposition"]
    sa = cd["salt_area_definitions_mha"]
    s_lbls = ["Salt-affected\ncells (all)", "Strict\n(≥25%/≥4)", "GLOSIS frac.\nweighted"]
    s_vals = [sa["irrigated_cropland_in_salt_affected_cells"], sa["strict_threshold_glosis_ge25pct_or_ece_ge4"], sa["glosis_fraction_weighted_salt_cropland"]]
    xb = np.arange(len(s_lbls))
    ax_b.bar(xb, s_vals, width=0.52, color=[PARADOX_COLORS["HYDROLOGICALLY_FEASIBLE"], PARADOX_COLORS["GW_DEPLETION_ONLY"], PARADOX_COLORS["PARADOX_BLUE_WATER_EXCEEDED"]], alpha=0.90)
    for xi, v in zip(xb, s_vals):
        ax_b.text(xi, v + 2.0, f"{v:.2f}", ha="center", va="bottom", fontsize=5.7, fontweight="bold")
    ax_b.set_xticks(xb)
    ax_b.set_xticklabels(s_lbls, fontsize=5.4)
    ax_b.set_ylim(0, 118)
    ax_b.set_ylabel("Salt-affected cropland (Mha)")

    # Panel c: EFR, ECw=3, and Inflow-Aware Closure Sensitivity
    ax_c = fig.add_subplot(gs[0, 2])
    add_panel_label(ax_c, "c", x=-0.20, y=1.06)
    efr_d = cd["efr_sensitivity_exceeded_basins"]
    c_lbls = ["EFR\n50%", "Inflow\n60%", "Base\n60%", r"$\mathrm{EC_w}$=3" + "\n60%", "EFR\n80%"]
    c_lbls_full = ["EFR 50% (Amb.)", "Inflow-aware (EFR 60%)", "EFR 60% (Baseline)", "EFR 60% (ECw=3)", "EFR 80% (Amb.)"]
    c_vals = [efr_d["efr_50pct"], cd["inflow_aware_exceeded_basins"], efr_d["efr_60pct"], cd["ec3_blue_water_exceeded_basins"], efr_d["efr_80pct"]]
    c_cols = [PARADOX_COLORS["HYDROLOGICALLY_FEASIBLE"], "#5FA8D3", PARADOX_COLORS["GW_DEPLETION_ONLY"], PARADOX_COLORS["PARADOX_BLUE_WATER_EXCEEDED"], PARADOX_COLORS["PARADOX_INSOLVENT_BLUE_AND_GW"]]
    xc = np.arange(len(c_lbls))
    ax_c.bar(xc, c_vals, width=0.56, color=c_cols, alpha=0.90)
    for xi, v in zip(xc, c_vals):
        ax_c.text(xi, v + 3.5, f"{v}", ha="center", va="bottom", fontsize=5.6, fontweight="bold")
    ax_c.set_xticks(xc)
    ax_c.set_xticklabels(c_lbls, fontsize=5.3)
    ax_c.set_ylim(0, 206)
    ax_c.set_ylabel(r"Exceeded basins ($\mathit{n}$ / 376)")
    ax_c.text(
        0.03, 0.96,
        f"Flipped: {cd['leaching_flipped_basins_ambient']} amb.,\n" + rf"{cd['inflow_aware_leaching_flipped']} inflow, {cd['leaching_flipped_basins_ec3']} ($\mathrm{{EC_w}}$=3)",
        transform=ax_c.transAxes, va="top", fontsize=5.2, color="#333333",
    )

    # Panel d: ISIMIP3a 4-GHM Exceeded Basins
    ax_d = fig.add_subplot(gs[1, 0])
    add_panel_label(ax_d, "d", x=-0.20, y=1.06)
    xd = np.arange(len(ghm))
    ax_d.bar(xd - 0.18, ghm["paradox_exceeded_efr60_native_n"], width=0.35, color=PARADOX_COLORS["HYDROLOGICALLY_FEASIBLE"], label="Native demand")
    ax_d.bar(xd + 0.18, ghm["paradox_exceeded_efr60_wg_demand_n"], width=0.35, color=PARADOX_COLORS["PARADOX_INSOLVENT_BLUE_AND_GW"], label="WaterGAP demand")
    for xi, n1, n2 in zip(xd, ghm["paradox_exceeded_efr60_native_n"], ghm["paradox_exceeded_efr60_wg_demand_n"]):
        ax_d.text(xi - 0.18, n1 + 3, f"{int(n1)}", ha="center", va="bottom", fontsize=5.2)
        ax_d.text(xi + 0.18, n2 + 3, f"{int(n2)}", ha="center", va="bottom", fontsize=5.2)
    ax_d.set_xticks(xd)
    ax_d.set_xticklabels(["WaterGAP", "H08", "LPJmL5", "CWatM"], fontsize=5.4)
    ax_d.set_ylim(0, 232)
    ax_d.set_ylabel(r"Exceeded basins ($\mathit{n}$ / 376)")
    ax_d.legend(loc="upper right", frameon=False, fontsize=5.2)

    # Panel e: GRACE TWSA Boxplot (376 basins)
    ax_e = fig.add_subplot(gs[1, 1])
    add_panel_label(ax_e, "e", x=-0.20, y=1.06)
    g_dep = bv.loc[bv["gw_depletion_active"], "grace_mean_twsa_mm_yr"].dropna().values
    g_non = bv.loc[~bv["gw_depletion_active"], "grace_mean_twsa_mm_yr"].dropna().values
    rng_a = np.random.default_rng(20261124)
    ax_e.scatter(rng_a.uniform(-0.14, 0.14, size=len(g_non)), g_non, color=PARADOX_COLORS["HYDROLOGICALLY_FEASIBLE"], s=4.0, alpha=0.28, linewidths=0, zorder=1)
    ax_e.scatter(1.0 + rng_a.uniform(-0.14, 0.14, size=len(g_dep)), g_dep, color=PARADOX_COLORS["PARADOX_INSOLVENT_BLUE_AND_GW"], s=4.0, alpha=0.28, linewidths=0, zorder=1)
    bp_e = ax_e.boxplot([g_non, g_dep], positions=[0, 1], widths=0.45, patch_artist=True, showfliers=False, medianprops={"color": "#111111", "lw": 1.3}, boxprops={"alpha": 0.78}, zorder=3)
    bp_e["boxes"][0].set_facecolor(PARADOX_COLORS["HYDROLOGICALLY_FEASIBLE"])
    bp_e["boxes"][1].set_facecolor(PARADOX_COLORS["PARADOX_INSOLVENT_BLUE_AND_GW"])
    ax_e.axhline(0, ls=":", color="#666666", lw=0.7)
    ax_e.set_xticks([0, 1])
    ax_e.set_xticklabels([rf"No GW dep." + "\n" + rf"($\mathit{{n}}$ = {len(g_non)})", rf"GW dep." + "\n" + rf"($\mathit{{n}}$ = {len(g_dep)})"], fontsize=5.5)
    ax_e.set_ylabel(r"GRACE TWSA trend (mm yr$^{-1}$)")
    gr = hv["grace_csr_validation_376_basins"]
    med_non_s = f"{gr['non_depletion_basins_median_grace_mm_yr']:.2f}".replace("-", "−")
    med_dep_s = f"{gr['gw_depletion_basins_median_grace_mm_yr']:.2f}".replace("-", "−")
    ax_e.text(0.04, 0.04, rf"Med: {med_non_s} vs {med_dep_s}" + "\n" + rf"{fmt_p(gr['mannwhitney_less_p'])}, $\mathit{{\rho}}$ = {gr['spearman_r_watergap_gws_vs_grace_twsa']:+.2f}", transform=ax_e.transAxes, fontsize=5.2, color="#222222", bbox={"boxstyle": "square,pad=0.10", "facecolor": "white", "edgecolor": "none", "alpha": 0.85}, zorder=5.0)

    # Panel f: Jasechko 1,693-Aquifer DTW Decline (180 matched basins)
    ax_f = fig.add_subplot(gs[1, 2])
    add_panel_label(ax_f, "f", x=-0.20, y=1.06)
    j_sub = bv.dropna(subset=["jasechko_aq_median_dtw_m_yr"]).copy()
    j_dep = j_sub.loc[j_sub["gw_depletion_active"], "jasechko_aq_median_dtw_m_yr"].values * 100.0
    j_non = j_sub.loc[~j_sub["gw_depletion_active"], "jasechko_aq_median_dtw_m_yr"].values * 100.0
    rng_b = np.random.default_rng(20261125)
    ax_f.scatter(rng_b.uniform(-0.14, 0.14, size=len(j_non)), j_non, color=PARADOX_COLORS["HYDROLOGICALLY_FEASIBLE"], s=4.5, alpha=0.30, linewidths=0, zorder=1)
    ax_f.scatter(1.0 + rng_b.uniform(-0.14, 0.14, size=len(j_dep)), j_dep, color=PARADOX_COLORS["GW_DEPLETION_ONLY"], s=4.5, alpha=0.30, linewidths=0, zorder=1)
    bp_f = ax_f.boxplot([j_non, j_dep], positions=[0, 1], widths=0.45, patch_artist=True, showfliers=False, medianprops={"color": "#111111", "lw": 1.3}, boxprops={"alpha": 0.78}, zorder=3)
    bp_f["boxes"][0].set_facecolor(PARADOX_COLORS["HYDROLOGICALLY_FEASIBLE"])
    bp_f["boxes"][1].set_facecolor(PARADOX_COLORS["GW_DEPLETION_ONLY"])
    ax_f.axhline(0, ls=":", color="#666666", lw=0.7)
    ax_f.set_xticks([0, 1])
    ax_f.set_xticklabels([rf"No GW dep." + "\n" + rf"($\mathit{{n}}$ = {len(j_non)})", rf"GW dep." + "\n" + rf"($\mathit{{n}}$ = {len(j_dep)})"], fontsize=5.5)
    ax_f.set_ylabel(r"Water-table decline (cm yr$^{-1}$)")
    ax_f.set_ylim(-36, 192)
    ja = hv["jasechko_2024_1693_aquifer_systems_validation"]
    ax_f.text(0.04, 0.96, f"Med: +{ja['non_depletion_basins_median_dtw_m_yr']*100:.1f} vs +{ja['gw_depletion_basins_median_dtw_m_yr']*100:.1f}\n{fmt_p(ja['mannwhitney_greater_p'])}, $\\mathit{{\\rho}}$ = {ja['spearman_r_vs_neg_watergap_gws']:+.2f}", transform=ax_f.transAxes, va="top", fontsize=5.2, color="#222222", bbox={"boxstyle": "square,pad=0.10", "facecolor": "white", "edgecolor": "none", "alpha": 0.85}, zorder=5.0)

    curve_df.to_csv(SUP_SRC_DIR / "ED_Fig2A_vleach_ecw_curve.csv", index=False)
    pd.DataFrame({"definition": s_lbls, "area_mha": s_vals}).to_csv(SUP_SRC_DIR / "ED_Fig2B_salt_area_defs.csv", index=False)
    pd.DataFrame({"regime": c_lbls_full, "exceeded_basins": c_vals}).to_csv(SUP_SRC_DIR / "ED_Fig2C_closure_sensitivity.csv", index=False)
    ghm.to_csv(SUP_SRC_DIR / "ED_Fig2D_multighm.csv", index=False)
    bv[["aware_basin_idx", "NAME", "gw_depletion_active", "grace_mean_twsa_mm_yr", "jasechko_aq_median_dtw_m_yr"]].to_csv(SUP_SRC_DIR / "ED_Fig2EF_grace_jasechko.csv", index=False)
    save_pub_figure(fig, SUP_DIR / "ED_Fig2_Hydrological_Validation.pdf", SUP_DIR / "ED_Fig2_Hydrological_Validation.png", "ED_Fig2")


def build_ed_fig3() -> None:
    p2b = json.loads((table_input("pillar2_armA_ecw_results_current.json")).read_text(encoding="utf-8"))
    grid = pd.read_parquet(grid_input("pillar3_grid05_zoning_current.parquet"))
    sites = pd.read_csv(SOURCE_DATA / "ED_Fig4A_empirical_sites.csv")
    armA_pts = sites.loc[sites.arm.eq("ArmA_brackish"), ["lat", "lon"]].rename(columns={"lat": "lat_dd", "lon": "lon_dd"})
    armb_pts = sites.loc[sites.arm.eq("ArmB_hydraulic"), ["lat", "lon"]].rename(columns={"lat": "lat_final", "lon": "lon_final"})

    full_pts = pd.concat([armA_pts.rename(columns={"lat_dd": "lat", "lon_dd": "lon"})[["lat", "lon"]], armb_pts.rename(columns={"lat_final": "lat", "lon_final": "lon"})[["lat", "lon"]]], ignore_index=True).drop_duplicates()
    full_pts["r"] = np.clip(np.floor((90.0 - full_pts["lat"].values) / 0.5).astype(int), 0, 359)
    full_pts["c"] = np.clip(np.floor((full_pts["lon"].values + 180.0) / 0.5).astype(int), 0, 719)
    pts_rc = full_pts[["r", "c"]].drop_duplicates()

    raw_cols = ["prc_mm_yr", "et0_mm_yr", "qtot_mm_yr", "crop_ece90_dsm", "wang_ece_2017_dsm", "sg_clay_0_30_pct", "hwsd2_esp_pct"]
    g_sub = grid[["r", "c", "lat", "lon", "salt_affected", "salt_irrig_ha", "management_zone_baseline"] + raw_cols].dropna(subset=raw_cols).copy()
    g_sub["ai"] = g_sub["prc_mm_yr"] / np.maximum(g_sub["et0_mm_yr"], 1.0)
    fcols = ["ai", "et0_mm_yr", "qtot_mm_yr", "crop_ece90_dsm", "wang_ece_2017_dsm", "sg_clay_0_30_pct", "hwsd2_esp_pct"]
    target = g_sub[g_sub["salt_affected"]].copy()
    train = g_sub.merge(pts_rc, on=["r", "c"], how="inner")
    mu, sd = train[fcols].mean().values, np.maximum(train[fcols].std().values, 1e-6)
    Z_tr, Z_tg = (train[fcols].values - mu) / sd, (target[fcols].values - mu) / sd
    dists_tr = np.sqrt(((Z_tr[:, None, :] - Z_tr[None, :, :]) ** 2).sum(axis=2))
    np.fill_diagonal(dists_tr, np.inf)
    nn_tr = dists_tr.min(axis=1)
    di_thresh = float(np.percentile(nn_tr, 75) + 1.5 * (np.percentile(nn_tr, 75) - np.percentile(nn_tr, 25)))
    target["nn_di"] = np.sqrt(((Z_tg[:, None, :] - Z_tr[None, :, :]) ** 2).sum(axis=2)).min(axis=1)
    target["inside_aoa"] = target["nn_di"] <= di_thresh
    pca = PCA(n_components=2).fit(Z_tg)
    pc_tg, pc_tr = pca.transform(Z_tg), pca.transform(Z_tr)
    target["pc1"], target["pc2"] = pc_tg[:, 0], pc_tg[:, 1]

    studies = pd.read_csv(HARMONIZED / "armA_extracted_current.csv", usecols=[
        "family_id", "analysis_lat_dd", "analysis_lon_dd", "qc_gate_status"])
    eligible = studies.qc_gate_status.isin(["PASS_READY", "PASS_NO_VARIANCE"])
    n_arma_geo_studies = studies.loc[eligible].dropna(subset=["analysis_lat_dd", "analysis_lon_dd"]).family_id.nunique()

    fig = plt.figure(figsize=(179 / 25.4, 126 / 25.4))

    # Panel a: Framed 0.5-degree Robinson Empirical Sites Map
    add_fig_panel_label(fig, 0.010, 0.985, "a")
    ax_a = fig.add_axes([0.032, 0.548, 0.950, 0.422])
    land_wgs_a = plot_robinson_basemap(
        ax_a,
        xlim_lon=(-128.0, 152.0),
        ylim_lat=(-44.0, 60.0),
        graticule_lats=(-30, 0, 30, 60),
        curved_frame=False,
    )
    salt = grid[grid["salt_affected"]].copy()
    salt_geoms = [box(lon - 0.25, lat - 0.25, lon + 0.25, lat + 0.25) for lon, lat in zip(salt["lon"], salt["lat"])]
    salt_gdf = gpd.GeoDataFrame(geometry=salt_geoms, crs="EPSG:4326")
    salt_dis = salt_gdf.dissolve()
    salt_clip = gpd.clip(salt_dis, land_wgs_a).to_crs(ROBIN_CRS)
    salt_clip.plot(ax=ax_a, facecolor="#DFC5A8", edgecolor="none", alpha=0.88, zorder=2.5)

    tr_rob = Transformer.from_crs("EPSG:4326", ROBIN_CRS, always_xy=True)
    if not armb_pts.empty:
        bx_m, by_m = tr_rob.transform(armb_pts["lon_final"].values, armb_pts["lat_final"].values)
        ax_a.scatter(bx_m, by_m, c="#009E73", s=12, marker="^", edgecolors="white", linewidths=0.25, alpha=0.88, zorder=3.2)
    ax_m, ay_m = tr_rob.transform(armA_pts["lon_dd"].values, armA_pts["lat_dd"].values)
    ax_a.scatter(ax_m, ay_m, c="#D55E00", s=16, marker="o", edgecolors="white", linewidths=0.35, alpha=0.92, zorder=4.0)

    fig.canvas.draw()
    pos_a = ax_a.get_position()
    leg_a_handles = [
        Patch(facecolor="#DFC5A8", edgecolor="none", label="Salt-affected cropland (97.1 Mha)"),
        Line2D([0], [0], marker="^", color="none", markerfacecolor="#009E73", markeredgecolor="white", markeredgewidth=0.3, markersize=4.8, label=f"Arm B sites ({len(armb_pts)})"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#D55E00", markeredgecolor="white", markeredgewidth=0.35, markersize=4.8, label=f"Arm A trials ({n_arma_geo_studies} studies, {len(armA_pts)} sites)"),
    ]
    fig.legend(handles=leg_a_handles, loc="upper center", bbox_to_anchor=(0.50, pos_a.y0 - 0.010), ncol=3, frameon=False, fontsize=5.35, handlelength=1.1, handletextpad=0.30, columnspacing=1.0)

    # Panel b: Framed 0.5-degree Robinson AOA Map + Clean South Pacific Outside-AOA Card Inset
    add_fig_panel_label(fig, 0.010, 0.478, "b")
    ax_b = fig.add_axes([0.014, 0.098, 0.585, 0.362])
    land_wgs_b = plot_robinson_basemap(
        ax_b,
        xlim_lon=(-162.0, 165.0),
        ylim_lat=(-46.0, 62.0),
        graticule_lats=(-30, 0, 30, 60),
        curved_frame=False,
    )
    in_pts, out_pts = target[target["inside_aoa"]], target[~target["inside_aoa"]]
    in_pct = float(in_pts["salt_irrig_ha"].sum() / max(target["salt_irrig_ha"].sum(), 1e-6) * 100.0)
    out_mha = float(out_pts["salt_irrig_ha"].sum() / 1e6)
    aoa_geoms = [box(lon - 0.25, lat - 0.25, lon + 0.25, lat + 0.25) for lon, lat in zip(target["lon"], target["lat"])]
    aoa_cls = np.where(target["inside_aoa"].values, "IN", "OUT")
    aoa_gdf = gpd.GeoDataFrame({"aoa_cls": aoa_cls}, geometry=aoa_geoms, crs="EPSG:4326")
    aoa_dis = aoa_gdf.dissolve(by="aoa_cls")
    aoa_clip = gpd.clip(aoa_dis, land_wgs_b).to_crs(ROBIN_CRS)
    if "IN" in aoa_clip.index:
        aoa_clip.loc[["IN"]].plot(ax=ax_b, facecolor="#1B8A6B", edgecolor="none", alpha=0.90, zorder=3.0)
    if "OUT" in aoa_clip.index:
        aoa_clip.loc[["OUT"]].plot(ax=ax_b, facecolor="#9B3B8A", edgecolor="none", alpha=0.95, zorder=3.4)

    # Inset: Outside-AOA share across Zones I-III (framed white card inside open South Pacific ocean)
    z_order = [
        ("Zone_I_Safe_Brackish_Substitution_Window", "Zone I"),
        ("Zone_II_Hydraulic_Recovery_Priority", "Zone II"),
        ("Zone_III_Water_Salt_Dual_Redline", "Zone III"),
    ]
    z_out_pcts = []
    for z_k, _ in z_order:
        zs = target[target["management_zone_baseline"] == z_k]
        tot_z = float(zs["salt_irrig_ha"].sum())
        out_z = float(zs.loc[~zs["inside_aoa"], "salt_irrig_ha"].sum())
        z_out_pcts.append(out_z / max(tot_z, 1e-6) * 100.0)
    fig.canvas.draw()
    pos_b = ax_b.get_position()
    card_b = mpl.patches.FancyBboxPatch(
        (pos_b.x0 + 0.014 * pos_b.width, pos_b.y0 + 0.032 * pos_b.height),
        0.278 * pos_b.width, 0.460 * pos_b.height,
        boxstyle="round,pad=0.0,rounding_size=0.008",
        transform=fig.transFigure, facecolor="white", edgecolor="#C4CCD4",
        linewidth=0.55, alpha=0.96, zorder=5.0,
    )
    fig.add_artist(card_b)
    ax_b_ins = fig.add_axes([
        pos_b.x0 + 0.088 * pos_b.width,
        pos_b.y0 + 0.145 * pos_b.height,
        0.190 * pos_b.width,
        0.300 * pos_b.height,
    ], zorder=6.0)
    yi_ins = np.arange(len(z_order))[::-1]
    ax_b_ins.barh(yi_ins, z_out_pcts, height=0.58, color=["#2B6E8C", "#7D4E9F", "#D95F28"], alpha=0.92)
    for yv_i, pv_i in zip(yi_ins, z_out_pcts):
        ax_b_ins.text(pv_i + 0.35, yv_i, f"{pv_i:.1f}%", va="center", ha="left", fontsize=5.1, color="#222222")
    ax_b_ins.set_yticks(yi_ins)
    ax_b_ins.set_yticklabels([z[1] for z in z_order], fontsize=5.1)
    ax_b_ins.set_xlim(0, max(z_out_pcts) * 1.62)
    ax_b_ins.set_xticks([0, 4, 8])
    ax_b_ins.tick_params(axis="x", labelsize=5.0, pad=1.0)
    ax_b_ins.tick_params(axis="y", pad=1.5)
    ax_b_ins.set_xlabel("Outside AOA (%)", fontsize=5.1, labelpad=1.2)
    ax_b_ins.patch.set_alpha(0.0)

    leg_b_handles = [
        Patch(facecolor="#1B8A6B", edgecolor="none", label=f"Inside AOA ({in_pct:.1f}%; {len(in_pts):,} cells)"),
        Patch(facecolor="#9B3B8A", edgecolor="none", label=f"Outside AOA ({out_mha:.2f} Mha; DI > {di_thresh:.2f})"),
    ]
    fig.legend(handles=leg_b_handles, loc="upper center", bbox_to_anchor=(pos_b.x0 + 0.5 * pos_b.width, pos_b.y0 - 0.014), ncol=2, frameon=False, fontsize=5.3, handlelength=1.0, handletextpad=0.30, columnspacing=0.90)

    # Panel c: 2D PCA Feature Space (7 non-collinear climate-crop-soil covariates)
    add_fig_panel_label(fig, 0.614, 0.478, "c")
    ax_c = fig.add_axes([0.668, 0.098, 0.318, 0.362])
    ax_c.grid(axis="both", color="#EEF1F4", lw=0.45, zorder=0.3)
    ax_c.scatter(in_pts["pc1"], in_pts["pc2"], color="#1B8A6B", s=5, alpha=0.25, edgecolors="none", label=rf"Inside AOA ($\mathit{{n}}$ = {len(in_pts):,})", zorder=1.5)
    ax_c.scatter(out_pts["pc1"], out_pts["pc2"], color="#9B3B8A", s=9, alpha=0.68, edgecolors="none", label=rf"Outside AOA ($\mathit{{n}}$ = {len(out_pts)})", zorder=2.5)
    ax_c.scatter(pc_tr[:, 0], pc_tr[:, 1], color="#111111", marker="^", s=6.5, alpha=0.52, edgecolors="none", label=rf"Empirical ({len(train)} cells)", zorder=3.5)
    arrow_scale = 5.2
    lbl_specs = [
        ("AI", (0.28, 0.25)),
        (r"ET$_0$", (-0.52, 0.12)),
        (r"$q_{\mathrm{tot}}$", (0.35, 0.28)),
        (r"EC$_{e90}$", (-1.05, -0.05)),
        (r"Soil EC$_e$", (-0.82, -0.35)),
        ("Clay", (0.18, 0.22)),
        ("ESP", (-0.05, 0.68)),
    ]
    for idx_f, (nm_f, (ox, oy)) in enumerate(lbl_specs):
        vx, vy = pca.components_[0, idx_f] * arrow_scale, pca.components_[1, idx_f] * arrow_scale
        ax_c.annotate("", xy=(vx, vy), xytext=(0, 0), arrowprops={"arrowstyle": "->", "color": PALETTE["red"], "lw": 0.95}, zorder=5)
        ax_c.text(
            vx * 1.15 + ox, vy * 1.15 + oy, nm_f,
            fontsize=5.2, color="#8C2D04", fontweight="bold", ha="center", va="center",
            bbox={"boxstyle": "round,pad=0.10", "facecolor": "white", "alpha": 0.90, "edgecolor": "none"},
            zorder=6,
        )
    ve = pca.explained_variance_ratio_ * 100
    ax_c.axhline(0, ls=":", color="#888888", lw=0.6)
    ax_c.axvline(0, ls=":", color="#888888", lw=0.6)
    ax_c.set_xlim(-4.5, 6.5)
    ax_c.set_ylim(-3.8, 5.2)
    ax_c.tick_params(labelsize=5.4)
    ax_c.set_xlabel(f"PC1 ({ve[0]:.1f}% variance)", fontsize=5.9, labelpad=2.0)
    ax_c.set_ylabel(f"PC2 ({ve[1]:.1f}% variance)", fontsize=5.9, labelpad=2.0)
    ax_c.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="none", framealpha=0.88, fontsize=5.15, handletextpad=0.22, labelspacing=0.22)

    sites_out = pd.concat([
        armA_pts[["lat_dd", "lon_dd"]].rename(columns={"lat_dd": "lat", "lon_dd": "lon"}).assign(arm="ArmA_brackish"),
        armb_pts[["lat_final", "lon_final"]].rename(columns={"lat_final": "lat", "lon_final": "lon"}).assign(arm="ArmB_hydraulic"),
    ], ignore_index=True)
    sites_out[["arm", "lat", "lon"]].to_csv(SUP_SRC_DIR / "ED_Fig4A_empirical_sites.csv", index=False)
    target[["r", "c", "lat", "lon", "salt_irrig_ha", "management_zone_baseline", "nn_di", "inside_aoa", "pc1", "pc2"]].to_csv(SUP_SRC_DIR / "ED_Fig4BC_aoa_pca.csv", index=False)
    train.assign(pc1=pc_tr[:,0],pc2=pc_tr[:,1])[["r","c","lat","lon","pc1","pc2"]+fcols].to_csv(
        SUP_SRC_DIR / "ED_Fig4C_empirical_pca.csv", index=False)
    pd.DataFrame(dict(variable=fcols,pc1_loading=pca.components_[0],pc2_loading=pca.components_[1],
                      empirical_mean=mu,empirical_sd=sd)).to_csv(SUP_SRC_DIR/"ED_Fig4C_loadings.csv", index=False)
    save_pub_figure(fig, SUP_DIR / "ED_Fig4_Empirical_Sites_AOA.pdf", SUP_DIR / "ED_Fig4_Empirical_Sites_AOA.png", "ED_Fig4")


def build_ed_fig4() -> None:
    p2b = json.loads((table_input("pillar2_armA_ecw_results_current.json")).read_text(encoding="utf-8"))
    arma = pd.read_csv(HARMONIZED / "armA_extracted_current.csv", low_memory=False)
    u = arma[arma["qc_gate_status"].isin(["PASS_READY", "PASS_NO_VARIANCE"])].copy()
    u["lnRR"] = np.log(pd.to_numeric(u["mean_treatment"]) / pd.to_numeric(u["mean_control"]))
    u["sarw_t"] = pd.to_numeric(u["sarw_treatment"], errors="coerce")

    fig = plt.figure(figsize=(179 / 25.4, 130 / 25.4))

    # =========================================================================
    # Panel a: 12-Subset Absolute ECw,90* Forest (incl. Primary Field Harvest)
    # =========================================================================
    add_fig_panel_label(fig, 0.010, 0.975, "a")
    ax_a = fig.add_axes([0.135, 0.558, 0.218, 0.388])
    da = p2b["dose_response_absolute"]
    subs = [
        ("field_continuous_harvest", "Field harvest (cont.)"),
        ("continuous", "Mixed ref. (cont.)"),
        ("continuous_excl_ry_gt2", "Cont. (excl. RY > 2)"),
        ("pooled", "Pooled"),
        ("multiyear_ge2yr", "Multi-year (≥2 yr)"),
        ("field_only", "Field only"),
        ("exact_variance", "Exact var."),
        ("exact_plus_lsd", "Exact + LSD"),
        ("grain_only", "Harvest (all sched.)"),
        ("wue", "WUE / IWUE (−10%)"),
        ("blended", "Blended"),
        ("cyclic", "Cyclic (unadj.)"),
    ]
    ys = np.arange(len(subs))[::-1]
    cont_d = da["field_continuous_harvest"]
    c_e90, (c_lo, c_hi) = cont_d["ecw_star_90pct_dsm"], cont_d["ecw_star_90pct_ci95"]
    ax_a.axvspan(c_lo, c_hi, color="#F8D7DA", alpha=0.55, zorder=0.5)
    ax_a.axvline(c_e90, ls="--", color="#8C1515", lw=0.85, zorder=1.5)
    ax_a.grid(axis="x", color="#EEF1F4", lw=0.45, zorder=0.3)

    s_rows = []
    cap_x = 11.5
    for yi, (sk, lbl) in zip(ys, subs):
        d = da[sk]
        e90 = float(d["ecw_star_90pct_dsm"])
        lo, hi = float(d["ecw_star_90pct_ci95"][0]), float(d["ecw_star_90pct_ci95"][1])
        is_trunc = hi > cap_x
        hi_plot = min(hi, cap_x)
        if sk == "field_continuous_harvest":
            col, mkr, mfc = "#8C1515", "D", "#8C1515"
        elif sk == "cyclic":
            col, mkr, mfc = "#2B6E8C", "o", "white"
        else:
            col, mkr, mfc = "#2B6E8C", "o", "#2B6E8C"
        ax_a.plot([lo, hi_plot], [yi, yi], color=col, lw=1.1, zorder=3.0)
        ax_a.plot([lo, lo], [yi - 0.14, yi + 0.14], color=col, lw=0.95, zorder=3.0)
        e90_str = format(Decimal(str(e90)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP), "f")
        if is_trunc:
            ax_a.scatter([cap_x], [yi], marker=">", color=col, s=12, zorder=3.5)
            ax_a.text(cap_x + 0.35, yi, f"{e90_str} (→{hi:.1f})", va="center", ha="left", fontsize=5.1, color=col)
        else:
            ax_a.plot([hi_plot, hi_plot], [yi - 0.14, yi + 0.14], color=col, lw=0.95, zorder=3.0)
            ax_a.text(hi_plot + 0.25, yi, e90_str, va="center", ha="left", fontsize=5.1, fontweight="bold" if sk == "field_continuous_harvest" else "normal", color=col)
        ax_a.scatter([e90], [yi], marker=mkr, facecolor=mfc, edgecolor=col, s=18 if sk == "field_continuous_harvest" else 16, linewidths=1.0, zorder=4.5)
        s_rows.append({"subset": sk, "label": lbl, "ecw90": e90, "ci_lo": lo, "ci_hi": hi, "k": d["k"], "n_clusters": d["n_clusters"]})

    ax_a.set_yticks(ys)
    ax_a.set_yticklabels([x[1] for x in subs], fontsize=5.15)
    ax_a.set_xlim(1.5, 16.0)
    ax_a.set_ylim(-0.65, len(subs) - 0.35)
    ax_a.set_xticks([2, 4, 6, 8, 10, 12])
    ax_a.tick_params(axis="x", labelsize=5.3)
    ax_a.set_xlabel(r"$\mathrm{EC}_{\mathrm{w},90}^*$ (dS m$^{-1}$)", fontsize=5.9, labelpad=2.0)

    # =========================================================================
    # Panel b: Maas-Hoffman vs Sigmoidal Logistic (p >= 1) Curves
    # =========================================================================
    add_fig_panel_label(fig, 0.366, 0.975, "b")
    ax_b = fig.add_axes([0.422, 0.558, 0.218, 0.388])
    ec_g = np.linspace(0.0, 15.0, 200)
    tc, bc = float(da["continuous"]["t_dsm"]), float(da["continuous"]["b_per_dsm"])
    tp, bp = float(da["pooled"]["t_dsm"]), float(da["pooled"]["b_per_dsm"])
    lg_c = p2b["logistic_sensitivity"]["continuous_sigmoidal_p_ge_1"]
    lg_p = p2b["logistic_sensitivity"]["pooled_sigmoidal_p_ge_1"]

    ry_mh_c = np.maximum(1.0 - bc * np.maximum(0.0, ec_g - tc), 0.05)
    ry_lg_c = 1.0 / (1.0 + (np.maximum(ec_g, 1e-4) / float(lg_c["ec50_dsm"])) ** float(lg_c["p_shape"]))
    ry_mh_p = np.maximum(1.0 - bp * np.maximum(0.0, ec_g - tp), 0.05)
    ry_lg_p = 1.0 / (1.0 + (np.maximum(ec_g, 1e-4) / float(lg_p["ec50_dsm"])) ** float(lg_p["p_shape"]))

    ax_b.grid(axis="both", color="#EEF1F4", lw=0.45, zorder=0.3)
    ax_b.axhline(0.90, ls=":", color="#666666", lw=0.75, zorder=1.5)
    ax_b.plot(ec_g, ry_mh_c, "-", color="#8C1515", lw=1.45, label=f"Cont. MH ({float(da['continuous']['ecw_star_90pct_dsm']) + 1e-6:.2f})", zorder=3.5)
    ax_b.plot(ec_g, ry_lg_c, "--", color="#8C1515", lw=1.30, label=f"Cont. Log. ({float(lg_c['ecw_star_90pct_dsm']) + 1e-6:.2f})", zorder=3.4)
    ax_b.plot(ec_g, ry_mh_p, "-", color="#2B6E8C", lw=1.45, label=f"Pooled MH ({float(da['pooled']['ecw_star_90pct_dsm']) + 1e-6:.2f})", zorder=3.3)
    ax_b.plot(ec_g, ry_lg_p, "--", color="#2B6E8C", lw=1.30, label=f"Pooled Log. ({float(lg_p['ecw_star_90pct_dsm']) + 1e-6:.2f})", zorder=3.2)

    for xv_90, col_90 in [
        (float(da["continuous"]["ecw_star_90pct_dsm"]), "#8C1515"),
        (float(lg_c["ecw_star_90pct_dsm"]), "#8C1515"),
        (float(da["pooled"]["ecw_star_90pct_dsm"]), "#2B6E8C"),
        (float(lg_p["ecw_star_90pct_dsm"]), "#2B6E8C"),
    ]:
        ax_b.scatter([xv_90], [0.90], color=col_90, s=16, edgecolors="white", linewidths=0.55, zorder=5.0)

    raw_bins_df = pd.read_csv(panel_input("Fig2A_yield_absolute_ecw.csv"))
    ax_b.errorbar(
        raw_bins_df["mean_ecw"],
        raw_bins_df["raw_mean_ry"],
        yerr=[
            raw_bins_df["raw_mean_ry"] - raw_bins_df["raw_ci_lo"],
            raw_bins_df["raw_ci_hi"] - raw_bins_df["raw_mean_ry"],
        ],
        fmt="o",
        color="#8C1515",
        mfc="white",
        mec="#8C1515",
        ecolor="#8C1515",
        markersize=3.2,
        lw=0.85,
        capsize=1.6,
        alpha=0.90,
        label="Raw field bins",
        zorder=4.2,
    )

    ax_b.set_xlim(0.0, 15.2)
    ax_b.set_ylim(0.50, 1.05)
    ax_b.set_xticks([0, 3, 6, 9, 12, 15])
    ax_b.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax_b.tick_params(labelsize=5.3)
    ax_b.set_xlabel(r"Irrigation $\mathrm{EC_w}$ (dS m$^{-1}$)", fontsize=5.9, labelpad=2.0)
    ax_b.set_ylabel("Relative production (T/C)", fontsize=5.9, labelpad=2.0)
    ax_b.legend(loc="upper right", bbox_to_anchor=(0.99, 0.99), frameon=False, fontsize=5.0, handlelength=1.15, handletextpad=0.28, labelspacing=0.20)

    # =========================================================================
    # Panel c: All-7-Crop k_emp (Including Small-Cluster Other Cereals)
    # =========================================================================
    add_fig_panel_label(fig, 0.652, 0.975, "c")
    ax_c = fig.add_axes([0.750, 0.558, 0.234, 0.388])
    cg7 = p2b["crop_group_vs_fao29"]["by_crop_group7"]
    cg_meta = [
        ("other_cereals", f"Other cereals ({cg7['other_cereals']['n_clusters']})*"),
        ("fruit", f"Fruit crops ({cg7['fruit']['n_clusters']})"),
        ("vegetables", f"Vegetables ({cg7['vegetables']['n_clusters']})"),
        ("maize", f"Maize ({cg7['maize']['n_clusters']})"),
        ("wheat", f"Wheat ({cg7['wheat']['n_clusters']})"),
        ("oilseed_legume", f"Oilseed & leg. ({cg7['oilseed_legume']['n_clusters']})"),
        ("cotton", f"Cotton ({cg7['cotton']['n_clusters']})"),
    ]
    c7_rows = []
    for key, lbl in cg_meta:
        cd = cg7[key]
        ece90 = float(cd["fao29_ece90_median_dsm"])
        kemp = float(cd["k_emp_ecw90_over_ece90"])
        klo = float(cd["ecw_star_90pct_ci95"][0]) / ece90
        khi = float(cd["ecw_star_90pct_ci95"][1]) / ece90
        c7_rows.append({
            "crop_group": key, "label": lbl, "n_clusters": cd["n_clusters"], "k": cd["k"],
            "ecw90": cd["ecw_star_90pct_dsm"], "fao_ece90": ece90, "k_emp": kemp, "k_lo": klo, "k_hi": khi,
        })
    c7_rows = sorted(c7_rows, key=lambda x: x["k_emp"])
    yc = np.arange(len(c7_rows))
    med6_k = float(p2b["crop_group_vs_fao29"]["median_k_emp_excl_other_cereals"])
    fao_k = float(p2b["crop_group_vs_fao29"]["theoretical_fao29_lf15_20_ratio"])

    ax_c.grid(axis="x", color="#EEF1F4", lw=0.45, zorder=0.3)
    ax_c.axvline(fao_k, ls="--", color="#D95F28", lw=0.85, zorder=1.6)
    ax_c.axvline(med6_k, ls=":", color="#1D3557", lw=0.95, zorder=1.6)
    for yi, cr in zip(yc, c7_rows):
        is_oc = cr["crop_group"] == "other_cereals"
        col = "#D95F28" if is_oc else "#2B6E8C"
        mfc = "white" if is_oc else "#2B6E8C"
        ax_c.plot([cr["k_lo"], cr["k_hi"]], [yi, yi], color=col, lw=1.1, zorder=3.0)
        ax_c.scatter([cr["k_emp"]], [yi], marker="o", facecolor=mfc, edgecolor=col, s=19, linewidths=1.05, zorder=4.2)
        tag = f"{cr['k_emp']:.2f} ({cr['ecw90']:.1f})" if is_oc else f"{cr['k_emp']:.2f}"
        ax_c.text(cr["k_hi"] + 0.06, yi, tag, va="center", ha="left", fontsize=5.2, fontweight="bold" if is_oc else "normal", color=col)

    c_leg = [
        Line2D([0], [0], ls="--", color="#D95F28", lw=0.95, label=f"FAO-29 ({fao_k:.2f})"),
        Line2D([0], [0], ls=":", color="#1D3557", lw=1.05, label=f"6-crop med. ({med6_k:.2f})"),
    ]
    ax_c.legend(handles=c_leg, loc="lower right", bbox_to_anchor=(0.99, 0.02), frameon=False, fontsize=5.2, handlelength=1.2, handletextpad=0.30, labelspacing=0.24)
    ax_c.set_yticks(yc)
    ax_c.set_yticklabels([r["label"] for r in c7_rows], fontsize=5.25)
    ax_c.set_xlim(0.0, 3.65)
    ax_c.set_ylim(-0.6, len(c7_rows) - 0.35)
    ax_c.set_xticks([0, 1, 2, 3])
    ax_c.tick_params(axis="x", labelsize=5.4)
    ax_c.set_xlabel(r"$\mathit{k}_{\mathrm{emp}} = \mathrm{EC}_{\mathrm{w},90}^* / \mathrm{EC}_{\mathrm{e}90}$", fontsize=6.0, labelpad=2.0)

    # =========================================================================
    # Panel d: Cluster-Collapsed Irrigation Water SARw -> Soil SAR & Soil EC Coupling
    # =========================================================================
    add_fig_panel_label(fig, 0.010, 0.486, "d")
    ax_d = fig.add_axes([0.070, 0.090, 0.252, 0.370])
    ax_d.grid(axis="y", color="#EEF1F4", lw=0.45, zorder=0.3)
    ax_d.axhline(0.0, ls=":", color="#888888", lw=0.70, zorder=1.2)
    sar_rows = []
    for ep, nm_ep, col, mkr_sym, ls_str in [("soil_SAR", "Soil SAR", "#D55E00", "o", "-"), ("soil_EC", "Soil EC", "#2B6E8C", "s", "--")]:
        sub = u[(u["metric_name"] == ep) & u["sarw_t"].notna() & u["lnRR"].notna()].copy()
        sub["sb"] = pd.cut(sub["sarw_t"], bins=[-1, 5, 12, 20, 200], labels=[0, 1, 2, 3])
        cl_bin = sub.groupby(["family_id", "sb"], observed=True).agg(m=("lnRR", "mean"), sar=("sarw_t", "mean"), n_c=("lnRR", "size")).reset_index()
        family_test = p2b["family_level_sar"][ep]
        rs, ps = family_test["spearman_r"], family_test["p"]
        n_fams_ep = int(sub["family_id"].nunique())
        grp = cl_bin.groupby("sb", observed=True).agg(m=("m", "mean"), se=("m", "sem"), n_fam=("m", "size"), n_contr=("n_c", "sum")).reset_index()
        grp["bi"] = grp["sb"].astype(int)
        ax_d.plot(grp["bi"], grp["m"], ls=ls_str, color=col, lw=1.25, label=rf"{nm_ep} ($\mathit{{\rho}}$={rs:+.2f}, {fmt_p(ps)}, {n_fams_ep}f)", zorder=3.2)
        multi = grp[grp["n_fam"] >= 2]
        single = grp[grp["n_fam"] == 1]
        if not multi.empty:
            ax_d.errorbar(multi["bi"], multi["m"], yerr=multi["se"], fmt="none", ecolor=col, lw=1.05, capsize=2.0, zorder=3.5)
            ax_d.scatter(multi["bi"], multi["m"], marker=mkr_sym, facecolor=col, edgecolor=col, s=16, zorder=4.0)
        if not single.empty:
            ax_d.scatter(single["bi"], single["m"], marker=mkr_sym, facecolor="white", edgecolor=col, s=16, lw=1.0, zorder=4.1)
        for _, rr in grp.iterrows():
            sar_rows.append({"endpoint": ep, "sarw_bin": int(rr["bi"]), "mean_lnRR": rr["m"], "se_cluster": rr["se"] if pd.notna(rr["se"]) else np.nan, "n_families": int(rr["n_fam"]), "n_contrasts": int(rr["n_contr"])})
    ax_d.set_xticks([0, 1, 2, 3])
    ax_d.set_xticklabels(["≤5", "5–12", "12–20", ">20"], fontsize=5.5)
    ax_d.tick_params(axis="y", labelsize=5.3)
    ax_d.set_xlim(-0.25, 3.25)
    ax_d.set_ylim(-0.18, 2.75)
    ax_d.set_xlabel(r"Irrigation $\mathrm{SAR_w}$ ($\mathrm{mmol_c}$ L$^{-1}$)$^{1/2}$", fontsize=5.8, labelpad=2.0)
    ax_d.set_ylabel("Soil SAR or EC change (lnRR)", fontsize=5.9, labelpad=2.0)
    ax_d.legend(loc="upper left", bbox_to_anchor=(0.01, 0.99), frameon=False, fontsize=5.15, handlelength=1.1, handletextpad=0.25, labelspacing=0.22)

    # =========================================================================
    # Panel e: Arm A 12-Country LOCO Yield Stability
    # =========================================================================
    add_fig_panel_label(fig, 0.340, 0.486, "e")
    ax_e = fig.add_axes([0.432, 0.090, 0.205, 0.370])
    loco_y = p2b["leave_one_country_out"]["yield_combined"]
    loco_ec = p2b["leave_one_country_out"]["soil_EC"]
    det = pd.DataFrame(loco_y["details"]).sort_values("pct_change")
    ye = np.arange(len(det))
    med_loco = float(det["pct_change"].median())
    ax_e.axvspan(float(loco_y["min_pct"]), float(loco_y["max_pct"]), color="#DDE8F1", alpha=0.72, zorder=0.5)
    ax_e.axvline(med_loco, ls="--", color="#2B6E8C", lw=0.85, zorder=1.5)
    ax_e.grid(axis="x", color="#EEF1F4", lw=0.45, zorder=0.3)
    for yi, pv in zip(ye, det["pct_change"]):
        ax_e.plot([med_loco, pv], [yi, yi], color="#7A9CB6", lw=0.95, zorder=2.5)
    ax_e.scatter(det["pct_change"], ye, color="#2B6E8C", s=16, edgecolors="white", linewidths=0.35, zorder=4.0)
    ax_e.text(
        0.04, 0.97,
        f"Yield: {_fmt_hu(loco_y['min_pct'])}% to {_fmt_hu(loco_y['max_pct'])}%\nSoil EC (8 ctry): +{_fmt_hu(loco_ec['min_pct'], 0)}% to +{_fmt_hu(loco_ec['max_pct'], 0)}%",
        transform=ax_e.transAxes, ha="left", va="top", fontsize=5.1, color="#1D3557",
        bbox={"boxstyle": "square,pad=0.10", "facecolor": "white", "edgecolor": "none", "alpha": 0.90},
        zorder=6.0,
    )
    ax_e.set_yticks(ye)
    ax_e.set_yticklabels([f"excl. {c}" for c in det["dropped_country"]], fontsize=5.15)
    ax_e.set_xlim(-21.2, -17.2)
    ax_e.set_ylim(-0.65, len(det) + 1.85)
    ax_e.set_xticks([-21, -20, -19, -18, -17])
    ax_e.tick_params(axis="x", labelsize=5.3)
    ax_e.set_xlabel("LOCO yield effect (%)", fontsize=5.9, labelpad=2.0)

    # =========================================================================
    # Panel f: Environmental & Duration CR1 Subgroups (Climate AI, Texture, Duration)
    # =========================================================================
    add_fig_panel_label(fig, 0.652, 0.486, "f")
    ax_f = fig.add_axes([0.750, 0.090, 0.234, 0.370])
    ed_sg = p2b["environmental_duration_subgroups"]
    sg_specs = [
        ("climate_ai", "arid", "Arid (<0.20)"),
        ("climate_ai", "humid_subhumid", "Sub-humid+ (≥0.50)"),
        ("soil_texture", "coarse", "Coarse (sandy)"),
        ("soil_texture", "fine", "Fine (clay)"),
        ("duration_pooled", "2_3yr", "2–3 yr"),
        ("duration_pooled", "ge4yr", "≥4 yr (pooled)"),
    ]
    yf_pos = [5.15, 4.10, 2.65, 1.60, 0.15, -0.90]
    ax_f.axhspan(3.62, 5.88, color="#F6F8FA", alpha=0.85, zorder=0.1)
    ax_f.axhspan(-1.42, 0.88, color="#F6F8FA", alpha=0.85, zorder=0.1)
    ax_f.axhline(3.40, color="#DDE2E8", lw=0.55, zorder=0.5)
    ax_f.axhline(1.00, color="#DDE2E8", lw=0.55, zorder=0.5)
    ax_f.axvline(0.0, ls="--", color="#222222", lw=0.80, zorder=1.6)
    ax_f.grid(axis="x", color="#EEF1F4", lw=0.45, zorder=0.3)

    ax_f.text(-0.140, 5.82, "Climate AI vs semi-arid", ha="left", va="top", fontsize=5.2, fontweight="bold", color="#33414D")
    ax_f.text(-0.140, 3.32, "Soil texture vs medium", ha="left", va="top", fontsize=5.2, fontweight="bold", color="#33414D")
    ax_f.text(-0.140, 0.84, "Duration vs 1 yr", ha="left", va="top", fontsize=5.2, fontweight="bold", color="#33414D")

    sg_rows = []
    for yi, (mod_nm, lvl, lbl) in zip(yf_pos, sg_specs):
        yd = ed_sg["yield_combined"][mod_nm]["contrasts"][lvl]
        yv, ylo, yhi, yp = float(yd["coef"]), float(yd["ci95"][0]), float(yd["ci95"][1]), float(yd["p_val"])
        y_sig = yp < 0.05
        ax_f.plot([ylo, yhi], [yi + 0.14, yi + 0.14], color="#2B6E8C", lw=1.1, zorder=3.0)
        ax_f.scatter([yv], [yi + 0.14], marker="o", facecolor="#2B6E8C" if y_sig else "white", edgecolor="#2B6E8C", s=16, linewidths=1.0, zorder=4.2)

        sd = ed_sg["soil_EC"][mod_nm]["contrasts"][lvl]
        sv_c, slo, shi, sp = float(sd["coef"]), float(sd["ci95"][0]), float(sd["ci95"][1]), float(sd["p_val"])
        s_sig = sp < 0.05
        ax_f.plot([slo, shi], [yi - 0.14, yi - 0.14], color="#D55E00", lw=1.1, zorder=3.0)
        ax_f.scatter([sv_c], [yi - 0.14], marker="s", facecolor="#D55E00" if s_sig else "white", edgecolor="#D55E00", s=15, linewidths=1.0, zorder=4.2)
        if s_sig:
            ax_f.text(
                shi + (0.011 if sv_c < 0 else 0.006),
                yi - 0.14,
                rf"$\mathit{{P}}$={sp:.3f}",
                ha="left",
                va="center",
                fontsize=5.2,
                fontweight="bold",
                color="#B84A00",
            )

        sg_rows.append({"moderator": mod_nm, "level": lvl, "label": lbl, "endpoint": "yield_combined", "coef": yv, "ci_lo": ylo, "ci_hi": yhi, "p_val": yp, "k": yd["k"], "n_clusters": yd["n_clusters"]})
        sg_rows.append({"moderator": mod_nm, "level": lvl, "label": lbl, "endpoint": "soil_EC", "coef": sv_c, "ci_lo": slo, "ci_hi": shi, "p_val": sp, "k": sd["k"], "n_clusters": sd["n_clusters"]})

    f_leg = [
        Line2D([0], [0], marker="o", color="#2B6E8C", markerfacecolor="#2B6E8C", markersize=3.8, lw=1.0, label="Yield"),
        Line2D([0], [0], marker="s", color="#D55E00", markerfacecolor="#D55E00", markersize=3.6, lw=1.0, label="Soil EC"),
        Line2D([0], [0], marker="o", color="#666666", markerfacecolor="white", markersize=3.6, lw=0.0, label=r"Open: $\mathit{P}$≥0.05"),
    ]
    leg_f = ax_f.legend(handles=f_leg, loc="lower center", bbox_to_anchor=(0.50, 0.008), ncol=3, frameon=True, facecolor="white", edgecolor="none", framealpha=0.95, borderpad=0.18, fontsize=5.1, handlelength=0.75, handletextpad=0.18, columnspacing=0.40)
    leg_f.set_zorder(6.0)
    ax_f.set_yticks(yf_pos)
    ax_f.set_yticklabels([x[2] for x in sg_specs], fontsize=5.2)
    ax_f.set_xlim(-0.145, 0.195)
    ax_f.set_ylim(-2.22, 6.00)
    ax_f.set_xticks([-0.10, -0.05, 0.00, 0.05, 0.10, 0.15])
    ax_f.tick_params(axis="x", labelsize=5.2)
    ax_f.set_xlabel(r"Salinity-slope difference (lnRR per dS m$^{-1}$)", fontsize=5.8, labelpad=2.0)

    pd.DataFrame(s_rows).to_csv(SUP_SRC_DIR / "ED_Fig5A_arma_subsets.csv", index=False)
    pd.DataFrame({
        "ecw": ec_g,
        "ry_mh_continuous": ry_mh_c,
        "ry_logistic_continuous": ry_lg_c,
        "ry_mh_pooled": ry_mh_p,
        "ry_logistic_pooled": ry_lg_p,
    }).to_csv(SUP_SRC_DIR / "ED_Fig5B_mh_vs_logistic.csv", index=False)
    raw_bins_df[["bin", "mean_ecw", "raw_mean_ry", "raw_ci_lo", "raw_ci_hi", "k", "n_clusters", "series"]].to_csv(
        SUP_SRC_DIR / "ED_Fig5B_raw_binned_means.csv", index=False
    )
    pd.DataFrame(c7_rows).to_csv(SUP_SRC_DIR / "ED_Fig5C_crop7_kemp.csv", index=False)
    pd.DataFrame(sar_rows).to_csv(SUP_SRC_DIR / "ED_Fig5D_sarw_coupling.csv", index=False)
    det.to_csv(SUP_SRC_DIR / "ED_Fig5E_arma_loco.csv", index=False)
    pd.DataFrame(sg_rows).to_csv(SUP_SRC_DIR / "ED_Fig5F_env_duration_subgroups.csv", index=False)
    save_pub_figure(fig, SUP_DIR / "ED_Fig5_ArmA_Sensitivity.pdf", SUP_DIR / "ED_Fig5_ArmA_Sensitivity.png", "ED_Fig5")


def build_ed_fig5() -> None:
    p2a = json.loads((table_input("pillar2_armB_hydraulic_results_current.json")).read_text(encoding="utf-8"))
    merged = pd.read_csv(grid_input("pillar2_armB_merged_current.csv"), low_memory=False)
    merged["endpoint_std"] = merged["metric_bucket"].replace({
        "infiltration_rate_steady": "infiltration_rate_all",
        "infiltration_rate_initial": "infiltration_rate_all",
        "infiltration_rate_unspecified": "infiltration_rate_all",
        "infiltration_rate_master": "infiltration_rate_all",
    })
    loco = p2a["leave_one_country_out_robustness"]
    wr_rows = p2a["h2_water_regime_stratification"]

    fig = plt.figure(figsize=(179 / 25.4, 140 / 25.4))

    # Panel a: Arm B Country LOCO Bounds
    add_fig_panel_label(fig, 0.010, 0.975, "a")
    ax_a = fig.add_axes([0.095, 0.615, 0.218, 0.335])
    loco_items = [
        (r"$\mathit{K_s}$ sat.", loco["ks_saturated"], PALETTE["red"]),
        ("Infiltration", loco["infiltration_rate_all"], PALETTE["teal"]),
        ("WUE / IWUE", loco["wue_or_iwue"], PALETTE["blue"]),
        ("Bulk density", loco["bulk_density"], PALETTE["amber"]),
    ]
    ya = np.arange(len(loco_items))[::-1]
    ax_a.axvline(0.0, ls="--", color="#555555", lw=0.8)
    lrows = []
    for yi, (nm, d, col) in zip(ya, loco_items):
        vals = [x["pct_change"] for x in d["details"]]
        ax_a.plot([min(vals), max(vals)], [yi, yi], "-", color=col, lw=2.8, alpha=0.72)
        ax_a.scatter(vals, [yi] * len(vals), color="#111111", s=14, zorder=4)
        mid_x = (min(vals) + max(vals)) / 2.0
        tx, ha_v = (10.0, "left") if min(vals) < 40.0 else (mid_x, "center")
        rng_txt = f"[{_fmt_hu(d['min_pct_change'], 1, True)}, {_fmt_hu(d['max_pct_change'], 1, True)}%] ({len(vals)}c)"
        ax_a.text(tx, yi + 0.22, rng_txt, ha=ha_v, va="bottom", fontsize=5.15)
        lrows.append({"endpoint": {"wue_or_iwue": "WUE / IWUE", "bulk_density": "Bulk density"}.get(d.get("endpoint", ""), str(nm).replace("$", "").replace("\\mathit{", "").replace("}", "")), "min_pct": d["min_pct_change"], "max_pct": d["max_pct_change"], "n_countries": len(vals)})
    ax_a.set_yticks(ya)
    ax_a.set_yticklabels([x[0] for x in loco_items], fontsize=5.5)
    ax_a.set_xlim(-22, 200)
    ax_a.set_ylim(-0.6, 3.7)
    ax_a.tick_params(axis="x", labelsize=5.3)
    ax_a.set_xlabel("Leave-one-country-out change (%)", fontsize=5.8, labelpad=2.0)

    # Panel b: Continuous ESP <= 100% Meta-Regression (17 Study-Cluster Bubbles)
    add_fig_panel_label(fig, 0.332, 0.975, "b")
    ax_b = fig.add_axes([0.388, 0.615, 0.262, 0.335])
    esp_reg = p2a["esp_meta_regression"]["physical_domain_esp_le_100"]
    pg = pd.DataFrame(esp_reg["pred_grid"])
    ks_pts = merged[(merged["endpoint_std"] == "ks_saturated") & merged["initial_esp_pct"].notna() & (merged["initial_esp_pct"] <= 100.0) & merged["lnRR"].notna()].copy()
    ag_dum = pd.get_dummies(ks_pts["amendment_group"], drop_first=True, dtype=float)
    ag_terms = esp_reg["terms"]
    ag_cols = [c for c in ag_dum.columns if c in ag_terms]
    beta_ag = np.array([float(ag_terms[c]["coef"]) for c in ag_cols])
    ag_mat = ag_dum[ag_cols].values
    ag_mean = ag_mat.mean(axis=0)
    ks_pts["lnRR_adj"] = ks_pts["lnRR"].values - (ag_mat @ beta_ag) + float(ag_mean @ beta_ag)
    ax_b.grid(axis="y", color="#EEF1F4", lw=0.45, zorder=0.4)
    ax_b.scatter(ks_pts["initial_esp_pct"], ks_pts["lnRR_adj"], color="#9BA8B4", alpha=0.24, s=7.0, linewidths=0, zorder=1.6)
    cl_c = ks_pts.groupby("family_id", as_index=False).agg(esp_m=("initial_esp_pct", "mean"), lnrr_m=("lnRR_adj", "mean"), k_cl=("lnRR_adj", "count"))
    col_cl_c = np.where(cl_c["esp_m"].values >= 15.0, "#C96A1B", "#2B6E8C")
    s_cl_c = 16.0 + 7.5 * np.sqrt(cl_c["k_cl"].values)
    ax_b.scatter(cl_c["esp_m"], cl_c["lnrr_m"], color=col_cl_c, s=s_cl_c, alpha=0.88, edgecolors="white", linewidths=0.65, zorder=4.2)
    ax_b.fill_between(pg["esp_pct"], pg["ci95_low"], pg["ci95_high"], color="#A61C28", alpha=0.14, zorder=3.0)
    ax_b.plot(pg["esp_pct"], pg["pred_lnrr"], "--", color="#A61C28", lw=1.55, zorder=4.6)
    ax_b.axhline(0.0, ls=":", color="#777777", lw=0.7)
    ax_b.axvline(15.0, ls="--", color="#666666", lw=0.75)
    ax_b.text(16.8, 4.48, "ESP = 15%", ha="left", va="top", fontsize=5.3, color="#555555")
    t_esp = esp_reg["terms"]["initial_esp_pct"]
    ax_b.text(0.97, 0.04, rf"{esp_reg['n_clusters']} clusters ($\mathit{{k}}$ = {esp_reg['k']})" + "\n" + rf"Slope = {t_esp['coef']:+.4f}/% ({fmt_p(t_esp['p_val'])})", transform=ax_b.transAxes, ha="right", va="bottom", fontsize=5.15, color="#222222")
    esp_leg = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#C96A1B", markeredgecolor="white", markeredgewidth=0.6, markersize=4.8, label="Sodic (ESP ≥ 15%)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#2B6E8C", markeredgecolor="white", markeredgewidth=0.6, markersize=4.8, label="Non-sodic (< 15%)"),
    ]
    ax_b.legend(handles=esp_leg, loc="upper right", bbox_to_anchor=(0.99, 0.99), frameon=False, fontsize=5.15, handletextpad=0.22, labelspacing=0.20)
    ax_b.set_xlim(0, 98)
    ax_b.set_ylim(-1.65, 4.75)
    ax_b.tick_params(labelsize=5.3)
    ax_b.set_xlabel("Baseline soil ESP (%)", fontsize=5.8, labelpad=2.0)
    ax_b.set_ylabel(r"Adjusted $\mathit{K_s}$ response (lnRR)", fontsize=5.8, labelpad=2.0)

    # Panel c: Water-Regime Harmonized REML Stratification (Saline Trial vs Brackish vs Upland)
    add_fig_panel_label(fig, 0.668, 0.975, "c")
    ax_c = fig.add_axes([0.758, 0.615, 0.226, 0.335])
    wr_map = {(r["endpoint"], r["stratum"]): r for r in wr_rows}
    ep_groups = [
        ("wue_or_iwue", "WUE / IWUE", ["water_regime:brackish_irrigated", "water_regime:irrigated_upland", "water_regime:saline_soil_trial"]),
        ("infiltration_rate_all", "Infiltration", ["water_regime:brackish_irrigated", "water_regime:irrigated_upland", "water_regime:saline_soil_trial"]),
        ("bulk_density", "Bulk density", ["water_regime:brackish_irrigated", "water_regime:irrigated_upland"]),
        ("ks_saturated", r"$\mathit{K_s}$ sat.", ["water_regime:saline_soil_trial"]),
    ]
    reg_style = {
        "water_regime:brackish_irrigated": ("Brackish irrig.", "#2B6E8C", "o"),
        "water_regime:irrigated_upland": ("Irrig. upland", "#009E73", "s"),
        "water_regime:saline_soil_trial": ("Saline trial", "#D55E00", "D"),
    }
    ax_c.axvline(0.0, ls="--", color="#333333", lw=0.80, zorder=1.5)
    ax_c.grid(axis="x", color="#EEF1F4", lw=0.45, zorder=0.3)

    y_pos = 0.0
    ep_centers, ep_labels = [], []
    ax_c.axhspan(0.55, 2.35, color="#F6F8FA", alpha=0.85, zorder=0.1)
    ax_c.axhspan(4.45, 5.25, color="#F6F8FA", alpha=0.85, zorder=0.1)
    # Build top-to-bottom coordinates
    y_cursor = 8.4
    for idx_e, (ep_k, ep_lbl, strata_k) in enumerate(ep_groups):
        ys_blk = [y_cursor - j * 0.62 for j in range(len(strata_k))]
        ep_centers.append(float(np.mean(ys_blk)))
        ep_labels.append(ep_lbl)
        for yv_s, sk_s in zip(ys_blk, strata_k):
            d_s = wr_map[(ep_k, sk_s)]
            _, col_s, mkr_s = reg_style[sk_s]
            mu_v, lo_v, hi_v = float(d_s["mu_lnrr"]), float(d_s["ci95"][0]), float(d_s["ci95"][1])
            pct_v, p_v = float(d_s["pct_change"]), float(d_s["p_val"])
            sig_s = p_v < 0.05
            ax_c.plot([lo_v, hi_v], [yv_s, yv_s], color=col_s, lw=1.15, zorder=3.0)
            ax_c.scatter([mu_v], [yv_s], marker=mkr_s, facecolor=col_s if sig_s else "white", edgecolor=col_s, s=16, linewidths=1.0, zorder=4.2)
            lbl_pct = f"{_fmt_hu(pct_v, 1, True)}%" + ("" if sig_s else " (n.s.)")
            tx_c = max(hi_v + 0.04, 0.065) if hi_v < 0 else hi_v + 0.04
            ax_c.text(tx_c, yv_s, lbl_pct, va="center", ha="left", fontsize=5.15, color=col_s if sig_s else "#555555",
                      bbox={"boxstyle": "square,pad=0.05", "facecolor": "white", "edgecolor": "none", "alpha": 0.90}, zorder=6.0)
        if idx_e < len(ep_groups) - 1:
            sep_y = ys_blk[-1] - 0.48
            ax_c.axhline(sep_y, color="#E2E6EB", lw=0.55, zorder=0.5)
        y_cursor = ys_blk[-1] - 0.96

    c_leg = [
        Line2D([0], [0], marker=mkr_s, color=col_s, markerfacecolor=col_s, markersize=4.0, lw=1.0, label=nm_s)
        for _, (nm_s, col_s, mkr_s) in reg_style.items()
    ]
    ax_c.legend(handles=c_leg, loc="upper right", bbox_to_anchor=(0.99, 0.99), frameon=True, facecolor="white", edgecolor="none", framealpha=0.90, fontsize=5.15, handlelength=0.90, handletextpad=0.22, labelspacing=0.18)
    ax_c.set_yticks(ep_centers)
    ax_c.set_yticklabels(ep_labels, fontsize=5.5)
    ax_c.set_xlim(-0.32, 2.15)
    ax_c.set_ylim(y_cursor + 0.35, 9.05)
    ax_c.set_xticks([math.log(1.0 + p / 100.0) for p in [0, 50, 150, 350]])
    ax_c.set_xticklabels(["0%", "+50%", "+150%", "+350%"], fontsize=5.3)
    ax_c.set_xlabel("REML effect (%, lnRR scale)", fontsize=5.8, labelpad=2.0)

    pd.DataFrame(lrows).to_csv(SUP_SRC_DIR / "ED_Fig6A_armb_loco.csv", index=False)
    pg.to_csv(SUP_SRC_DIR / "ED_Fig6B_armb_esp.csv", index=False)
    ks_pts[["family_id", "amendment_group", "initial_esp_pct", "lnRR", "lnRR_adj"]].to_csv(
        SUP_SRC_DIR / "ED_Fig6B_observations.csv", index=False)
    pd.DataFrame([dict(endpoint=endpoint, **record) for endpoint, result in loco.items()
                  for record in result["details"]]).to_csv(SUP_SRC_DIR / "ED_Fig6A_country_estimates.csv", index=False)
    pd.DataFrame(wr_rows).to_csv(SUP_SRC_DIR / "ED_Fig6C_armb_water_regime.csv", index=False)
    from generate_main_figures_1_3 import plot_amendment_strata
    add_fig_panel_label(fig, 0.010, 0.515, "d")
    axes = [fig.add_axes([x, y, 0.365, 0.168]) for y in (0.312, 0.068) for x in (0.125, 0.618)]
    strata = pd.read_csv(table_input("pillar2_armB_subgroup_summary_current.csv"))
    plot_amendment_strata(strata, axes).to_csv(SUP_SRC_DIR/"ED_Fig6D_sodicity_amendment_matrix.csv",index=False)
    save_pub_figure(fig, SUP_DIR / "ED_Fig6_ArmB_Robustness_WaterRegime.pdf", SUP_DIR / "ED_Fig6_ArmB_Robustness_WaterRegime.png", "ED_Fig6")


def build_ed_fig6() -> None:
    sens = pd.read_csv(table_input("pillar3_supply_sensitivity_current.csv"))
    p3 = json.loads((table_input("pillar3_global_summary_current.json")).read_text(encoding="utf-8"))
    n_exc_base = int(p3["global_salt_affected_croplands_savings"]["paradox_exceeded_basins_baseline"])
    fig = plt.figure(figsize=(179 / 25.4, 136 / 25.4))

    s_rev = sens.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(s_rev))
    lbls = [
        "DynQual+WHYMAP prior",
        "WHYMAP prior",
        "No GW credit",
        r"$\mathrm{EC_{ret}} = 2\,\mathrm{EC_f}$",
        "Managed cyclic (6.34)",
        r"Hyd. $\mathit{\rho}$ = 1.00",
        r"Hyd. $\mathit{\rho}$ = 0.50",
        r"GW $\mathit{\kappa}$ = 0.40",
        r"GW $\mathit{\kappa}$ = 0.10",
        r"Return flow $\mathit{r}$ = 0.50",
        r"Return flow $\mathit{r}$ = 0.20",
        r"Baseline ($\mathit{r}$ = 0.35, $\mathit{\rho}$ = 0.75)",
    ]

    panels_spec = [
        ("deltaF_net_brackish_saving_km3_yr", r"Net brackish $\Delta F$ (km$^3$ yr$^{-1}$)", "#2B6E8C", "#C9D8E4", "a", [0.220, 0.505, 0.225, 0.440], 0.010, 2.5, 0.05),
        ("H_cons_km3_yr", r"Capped $\mathit{H}_{\mathrm{cons}}^*$ (km$^3$ yr$^{-1}$)", "#D99118", "#EFE0C1", "b", [0.482, 0.505, 0.225, 0.440], 0.464, 1.5, 0.05),
        ("relieved_basins", rf"Rescued basins ($\mathit{{n}}$ / {n_exc_base})", "#1B7867", "#C5DFD8", "c", [0.744, 0.505, 0.240, 0.440], 0.726, 0.35, 0.5),
    ]
    for idx_p, (col_nm, xlab, col_acc, col_mute, letter, rect, lx, pad_v, tol) in enumerate(panels_spec):
        add_fig_panel_label(fig, lx, 0.975, letter)
        ax = fig.add_axes(rect)
        base_v = float(sens.iloc[0][col_nm])
        ax.grid(axis="x", color="#EEF1F4", lw=0.45, zorder=0.3)
        ax.axvline(base_v, ls="--", color="#222222", lw=0.80, zorder=2.2)

        vals = s_rev[col_nm].values
        bar_cols = [col_acc if abs(float(v) - base_v) > tol else col_mute for v in vals]
        bars = ax.barh(y, vals, height=0.62, color=bar_cols, alpha=0.92, zorder=3.0)
        bars[-1].set_edgecolor("#222222")
        bars[-1].set_linewidth(0.65)

        for yi, v in zip(y, vals):
            fv = float(v)
            is_diff = abs(fv - base_v) > tol or yi == len(vals) - 1
            if idx_p == 2 and 0.0 < base_v - fv <= 1.5:
                ax.text(
                    fv - 0.30, yi,
                    f"{int(v)}",
                    va="center", ha="right", fontsize=5.2,
                    fontweight="bold", color="#FFFFFF",
                    zorder=6.0,
                )
            else:
                x_txt = max(fv, base_v) + pad_v if (idx_p < 2 and fv < base_v < fv + pad_v * 3.2) else fv + pad_v
                ax.text(
                    x_txt, yi,
                    _fmt_hu(fv, 1) if idx_p < 2 else f"{int(v)}",
                    va="center", ha="left", fontsize=5.2,
                    fontweight="bold" if is_diff else "normal",
                    color="#111111" if is_diff else "#555555",
                    bbox={"boxstyle": "square,pad=0.05", "facecolor": "white", "edgecolor": "none", "alpha": 0.90},
                    zorder=6.0,
                )
        ax.set_yticks(y)
        ax.set_yticklabels(lbls if idx_p == 0 else [], fontsize=5.3)
        ax.set_xlim(0.0, max(vals) * 1.26)
        ax.set_ylim(-1.60, len(s_rev) - 0.35)
        ax.tick_params(axis="x", labelsize=5.3)
        ax.set_xlabel(xlab, fontsize=5.8, labelpad=2.0)
        if idx_p == 2:
            ax.text(0.98, 0.02, "Actual ET: 0 rescued (all 12)", transform=ax.transAxes, ha="right", va="bottom", fontsize=5.15, fontweight="bold", color="#A61C28", bbox={"boxstyle": "square,pad=0.10", "facecolor": "white", "edgecolor": "none", "alpha": 0.92}, zorder=6.5)

    ck = pd.read_csv(table_input("pillar3_threshold_and_recovery_current.csv"))
    ck.to_csv(SUP_SRC_DIR / "ED_Fig7_threshold_recovery.csv", index=False)
    add_fig_panel_label(fig, 0.010, 0.420, "d")
    add_fig_panel_label(fig, 0.512, 0.420, "e")

    ax_t = fig.add_axes([0.095, 0.088, 0.395, 0.312])
    crop = ck.iloc[:7].sort_values("ecw_dsm")
    ax_t.grid(axis="both", color="#EEF1F4", lw=0.45, zorder=0.3)
    ax_t.plot(crop["ecw_dsm"], crop["withdrawal_relief_km3_yr"], color="#527689", lw=1.25, zorder=2.5)
    is_fh = crop["case"] == "Field harvest"
    ax_t.scatter(crop.loc[~is_fh, "ecw_dsm"], crop.loc[~is_fh, "withdrawal_relief_km3_yr"], color="#2B6E8C", s=20, edgecolor="white", lw=0.45, zorder=3.5)
    ax_t.scatter(crop.loc[is_fh, "ecw_dsm"], crop.loc[is_fh, "withdrawal_relief_km3_yr"], color="#D95F28", marker="D", s=24, edgecolor="white", lw=0.5, zorder=4.2)
    sig_r = crop[crop["case"] == "Continuous sigmoid"].iloc[0]
    fh_r = crop[is_fh].iloc[0]
    hi_r = crop[crop["case"] == "Field upper CI"].iloc[0]
    ax_t.annotate(f"Sigmoid ({_fmt_hu(sig_r['ecw_dsm'], 2)})", xy=(sig_r["ecw_dsm"], sig_r["withdrawal_relief_km3_yr"]), xytext=(2.48, 154.0), fontsize=5.2, color="#333333", arrowprops={"arrowstyle": "->", "color": "#666666", "lw": 0.5, "shrinkA": 1, "shrinkB": 3}, zorder=5.0)
    ax_t.annotate(f"Field harvest ({_fmt_hu(fh_r['ecw_dsm'], 2)}; ref. 3.55)", xy=(fh_r["ecw_dsm"], fh_r["withdrawal_relief_km3_yr"]), xytext=(2.20, 180.2), fontsize=5.2, fontweight="bold", color="#B84A00", arrowprops={"arrowstyle": "->", "color": "#D95F28", "lw": 0.55, "shrinkA": 1, "shrinkB": 3}, zorder=5.0)
    ax_t.annotate(f"95% CI upper ({_fmt_hu(hi_r['ecw_dsm'], 2)})", xy=(hi_r["ecw_dsm"], hi_r["withdrawal_relief_km3_yr"]), xytext=(3.60, 171.8), fontsize=5.1, color="#555555", arrowprops={"arrowstyle": "->", "color": "#666666", "lw": 0.5, "shrinkA": 1, "shrinkB": 3}, zorder=5.0)
    ax_t.set_xlim(2.05, 4.48)
    ax_t.set_ylim(150.5, 188.5)
    ax_t.tick_params(labelsize=5.3)
    ax_t.set_xlabel(r"Production threshold $\mathrm{EC}_{\mathrm{w},90}^*$ (dS m$^{-1}$)", fontsize=5.8, labelpad=2.0)
    ax_t.set_ylabel(r"Net brackish $\Delta F$ (km$^3$ yr$^{-1}$)", fontsize=5.8, labelpad=2.0)

    ax_r = fig.add_axes([0.592, 0.088, 0.392, 0.312])
    rec = ck[ck["case"].eq("Composite reference") | ck["case"].str.startswith("Marginal recovery")]
    ax_r.grid(axis="both", color="#EEF1F4", lw=0.45, zorder=0.3)
    ax_r.axhline(0.0, ls="--", color="#555555", lw=0.75, zorder=1.5)
    ax_r.plot(100.0 * rec["marginal_leach_recovery"], rec["net_et_km3_yr"], "o-", color="#A61C28", lw=1.25, ms=3.8, markeredgecolor="white", markeredgewidth=0.4, zorder=3.0)
    cross = float(-rec.iloc[0]["net_et_km3_yr"] / (rec.iloc[-1]["net_et_km3_yr"] - rec.iloc[0]["net_et_km3_yr"]) * 100.0)
    ax_r.plot(cross, 0.0, "D", mfc="white", mec="#A61C28", ms=5.0, mew=1.05, zorder=4.5)
    y_min_r = min(float(rec["net_et_km3_yr"].min()) - 2.2, -14.0)
    y_max_r = max(float(rec["net_et_km3_yr"].max()) + 2.8, 4.2)
    xy_txt = (max(cross - 48.0, 6.0), 1.8) if cross > 65.0 else (cross + 10.0, -3.2)
    ax_r.annotate(f"Break-even: {cross:.1f}%", xy=(cross, 0.0), xytext=xy_txt, fontsize=5.3, fontweight="bold", color="#A61C28", arrowprops={"arrowstyle": "->", "color": "#A61C28", "lw": 0.55, "shrinkA": 1, "shrinkB": 3}, zorder=5.0)
    ax_r.set_xlim(-4.0, 104.0)
    ax_r.set_ylim(y_min_r, y_max_r)
    ax_r.tick_params(labelsize=5.3)
    ax_r.set_xlabel("Same-basin leaching recovery (%)", fontsize=5.8, labelpad=2.0)
    ax_r.set_ylabel(r"Verified-ET net relief (km$^3$ yr$^{-1}$)", fontsize=5.8, labelpad=2.0)

    sens.to_csv(SUP_SRC_DIR / "ED_Fig7_pillar3_sensitivity.csv", index=False)
    save_pub_figure(fig, SUP_DIR / "ED_Fig7_Pillar3_Sensitivity_Tornado.pdf", SUP_DIR / "ED_Fig7_Pillar3_Sensitivity_Tornado.png", "ED_Fig7")


def build_ed_fig7() -> None:
    ledger = json.loads((table_input("observed_district_basin_water_salt_ledger_current.json")).read_text(encoding="utf-8"))
    p3 = json.loads((table_input("pillar3_global_summary_current.json")).read_text(encoding="utf-8"))
    g3 = pd.read_parquet(grid_input("pillar3_grid05_zoning_current.parquet"), columns=["salt_affected", "salt_irrig_ha", "hwsd2_gypsum_pct", "management_zone_baseline"])

    fig = plt.figure(figsize=(179 / 25.4, 120 / 25.4))

    # Panel a: La Violada District Observed Salt Budget & Geogenic Gypsum Dissolution (1982-2008)
    add_fig_panel_label(fig, 0.010, 0.975, "a")
    ax_a = fig.add_axes([0.078, 0.568, 0.405, 0.378])
    lv_recs = [r for r in ledger["district_and_basin_records"] if r["case_id"].startswith("LaViolada_")]
    lv_periods = ["1982–1984\n(Flood)", "1995–1998\n(Flood)", "2006–2008\n(Modernized)"]
    f_in = [float(r["field_salt_input_kt_yr"]) for r in lv_recs]
    b_in = [float(r["basin_external_salt_input_kt_yr"]) for r in lv_recs]
    b_out = [float(r["basin_outlet_salt_export_kt_yr"]) for r in lv_recs]
    bias_pcts = [float(r["ec_to_tds_fao29_underestimate_pct"]) for r in lv_recs]
    xa = np.arange(len(lv_periods))
    w_a = 0.26
    ax_a.grid(axis="y", color="#EEF1F4", lw=0.45, zorder=0.3)
    ax_a.bar(xa - w_a, f_in, width=w_a, color="#2B6E8C", label="Field input", zorder=2.5)
    ax_a.bar(xa, b_in, width=w_a, color="#5FA8D3", label="Basin external input", zorder=2.5)
    ax_a.bar(xa + w_a, b_out, width=w_a, color="#D55E00", label="Outlet export (82% gypsum)", zorder=2.5)
    for xi, fi, bi, bo, bpct in zip(xa, f_in, b_in, b_out, bias_pcts):
        ax_a.text(xi - w_a, fi + 1.5, _fmt_hu(fi, 1), ha="center", va="bottom", fontsize=5.15, color="#2B6E8C")
        ax_a.text(xi, bi + 1.5, _fmt_hu(bi, 1), ha="center", va="bottom", fontsize=5.15, color="#2B6E8C")
        ax_a.text(xi + w_a, bo + 1.5, _fmt_hu(bo, 1), ha="center", va="bottom", fontsize=5.15, fontweight="bold", color="#B84A00")
        ax_a.text(xi + w_a, bo * 0.50, f"{_fmt_hu(bpct, 0)}%", ha="center", va="center", fontsize=5.15, color="white", fontweight="bold", zorder=4.0)
    ax_a.set_xticks(xa)
    ax_a.set_xticklabels(lv_periods, fontsize=5.3)
    ax_a.set_ylim(0, 114)
    ax_a.tick_params(axis="y", labelsize=5.3)
    ax_a.set_ylabel(r"La Violada salt flux (kt yr$^{-1}$)", fontsize=5.9, labelpad=2.0)
    ax_a.legend(loc="upper right", frameon=False, fontsize=5.15, handlelength=1.05, labelspacing=0.22)
    ax_a.text(0.98, 0.65, "1,045 → 550 mm yr$^{-1}$\nExport −56.6%\nBar %: 640×EC bias", transform=ax_a.transAxes, ha="right", va="top", fontsize=5.05, color="#333333", bbox={"boxstyle": "square,pad=0.08", "facecolor": "white", "edgecolor": "none", "alpha": 0.88})

    # Panel b: Hetao Irrigation District Seasonal Partitioning & Internal Salt Sink (1990-2013)
    add_fig_panel_label(fig, 0.512, 0.975, "b")
    ax_b = fig.add_axes([0.576, 0.568, 0.408, 0.378])
    het = [r for r in ledger["district_and_basin_records"] if r["case_id"].startswith("Hetao_")][0]
    het_in = float(het["field_salt_input_kt_yr"]) / 1000.0
    het_out = float(het["basin_outlet_salt_export_kt_yr"]) / 1000.0
    het_int = float(het["internal_transfer_or_storage_kt_yr"]) / 1000.0
    het_mod = float(het["model_matched_sin_pre_kt_yr"]) / 1000.0
    het_grow = het_out * 0.358
    het_aut = het_out * 0.554
    het_other = het_out * 0.088
    ax_b.grid(axis="y", color="#EEF1F4", lw=0.45, zorder=0.3)
    xb_pos = np.array([0, 1, 2])
    ax_b.bar([0], [het_in], width=0.52, color="#2B6E8C", zorder=2.5)
    ax_b.bar([1], [het_grow], width=0.52, color="#009E73", label="Outlet: growing (35.8%)", zorder=2.5)
    ax_b.bar([1], [het_aut], bottom=[het_grow], width=0.52, color="#D55E00", label="Outlet: autumn leach (55.4%)", zorder=2.5)
    ax_b.bar([1], [het_other], bottom=[het_grow + het_aut], width=0.52, color="#999999", label="Outlet: non-irrig. (8.8%)", zorder=2.5)
    ax_b.bar([1], [het_int], bottom=[het_out], width=0.52, color="#E8D5B7", edgecolor="#8C6D46", lw=0.65, label="Internal GW/saline sink (66.3%)", zorder=2.5)
    ax_b.bar([2], [het_mod], width=0.52, color="#4A7C59", alpha=0.88, zorder=2.5)
    ax_b.text(0, het_in + 0.08, f"{het_in:.2f}", ha="center", va="bottom", fontsize=5.2, fontweight="bold", color="#2B6E8C")
    ax_b.text(1, het_out + het_int + 0.08, f"0.95 out +\n1.86 sink", ha="center", va="bottom", fontsize=5.15, fontweight="bold", color="#5C4428")
    ax_b.text(2, het_mod + 0.08, f"{het_mod:.2f}\n(modeled)", ha="center", va="bottom", fontsize=5.15, fontweight="bold", color="#3A6347")
    ax_b.set_xticks(xb_pos)
    ax_b.set_xticklabels(["Observed\ndiversion", "Observed partition\n(outlet + sink)", "Modeled\n" + r"$S_{\mathrm{in,pre}}$"], fontsize=5.25)
    ax_b.set_ylim(0, 4.70)
    ax_b.tick_params(axis="y", labelsize=5.3)
    ax_b.set_ylabel(r"Hetao salt budget (Mt yr$^{-1}$)", fontsize=5.9, labelpad=2.0)
    ax_b.legend(loc="upper left", frameon=False, fontsize=5.15, handlelength=1.0, labelspacing=0.18)

    # Panel c: UCRB Lees Ferry (09380000) 1991-2019 Annual River Salt Flux vs Modeled S_in,pre
    add_fig_panel_label(fig, 0.010, 0.478, "c")
    ax_c = fig.add_axes([0.078, 0.090, 0.405, 0.362])
    lf_df = pd.DataFrame(ledger["usgs_ucrb_benchmark"]["lees_ferry_annual_series"])
    lf_stats = ledger["usgs_ucrb_benchmark"]["lees_ferry_outlet_09380000"]
    ucrb_rec = [r for r in ledger["district_and_basin_records"] if r["case_id"].startswith("USGS_")][0]
    ucrb_mod = float(ucrb_rec["model_matched_sin_pre_kt_yr"]) / 1000.0
    lf_mean, lf_sd = float(lf_stats["mean_flux_mt_yr"]), float(lf_stats["sd_flux_mt_yr"])
    ax_c.grid(axis="both", color="#EEF1F4", lw=0.45, zorder=0.3)
    ax_c.axhspan(lf_mean - lf_sd, lf_mean + lf_sd, color="#DDE8F1", alpha=0.65, zorder=0.8)
    ax_c.axhline(lf_mean, ls="--", color="#2B6E8C", lw=0.95, label=rf"Observed mean ({lf_mean:.2f} ± {lf_sd:.2f} Mt yr$^{{-1}}$)", zorder=2.0)
    ax_c.axhline(ucrb_mod, ls=":", color="#D55E00", lw=1.25, label=rf"Modeled cropland $S_{{\mathrm{{in,pre}}}}$ ({ucrb_mod:.2f} Mt yr$^{{-1}}$)", zorder=2.2)
    ax_c.plot(lf_df["Year"], lf_df["flux_mt"], "-o", color="#1D3557", lw=1.15, markersize=2.6, zorder=3.5)
    z_fit = np.polyfit(lf_df["Year"], lf_df["flux_mt"], 1)
    ax_c.plot(lf_df["Year"], np.polyval(z_fit, lf_df["Year"]), "-", color="#8C1515", lw=0.85, alpha=0.85, zorder=3.0)
    ax_c.text(0.98, 0.96, rf"Slope {_fmt_hu(lf_stats['trend_mt_yr2'], 3)} Mt yr$^{{-2}}$ ({fmt_p(lf_stats['trend_p'])})" + "\n97–98% tributary baseflow", transform=ax_c.transAxes, ha="right", va="top", fontsize=5.1, color="#333333", bbox={"boxstyle": "square,pad=0.08", "facecolor": "white", "edgecolor": "none", "alpha": 0.88})
    ax_c.legend(loc="lower left", frameon=False, fontsize=5.15, handlelength=1.15, labelspacing=0.18)
    ax_c.set_xlim(1990.2, 2019.8)
    ax_c.set_ylim(2.4, 9.6)
    ax_c.tick_params(labelsize=5.3)
    ax_c.set_xlabel("Year", fontsize=5.9, labelpad=2.0)
    ax_c.set_ylabel(r"Lees Ferry salt flux (Mt yr$^{-1}$)", fontsize=5.9, labelpad=2.0)

    # Panel d: Global Zones I-III Soil-Engineering & Gypsum Prescriptions (Mha)
    add_fig_panel_label(fig, 0.512, 0.478, "d")
    ax_d = fig.add_axes([0.576, 0.090, 0.408, 0.362])
    smz = p3["soil_mechanistic_zone_breakdown"]
    s_g3 = g3[g3["salt_affected"]].copy()
    z_keys = [
        ("Zone_I_Safe_Brackish_Substitution_Window", "Zone I\n(47.98 Mha)"),
        ("Zone_II_Hydraulic_Recovery_Priority", "Zone II\n(20.07 Mha)"),
        ("Zone_III_Water_Salt_Dual_Redline", "Zone III\n(29.07 Mha)"),
    ]
    presc_rows = []
    for zk, zlbl in z_keys:
        sd_z = smz[zk]
        gyp5 = float(s_g3.loc[(s_g3["management_zone_baseline"] == zk) & (s_g3["hwsd2_gypsum_pct"] >= 5.0), "salt_irrig_ha"].sum() / 1e6)
        presc_rows.append({
            "zone": zk,
            "zone_label": zlbl,
            "sodic_ca_flocculation_mha": float(sd_z["sodic_ca_flocculation_mha"]),
            "non_sodic_mulched_drip_mha": float(sd_z["non_sodic_mulched_drip_mha"]),
            "poor_drainage_tile_priority_mha": float(sd_z["poor_drainage_tile_priority_mha"]),
            "heavy_clay_ge35pct_mha": float(sd_z["heavy_clay_ge35pct_mha"]),
            "whymap_saline_gw_gt5gl_mha": float(sd_z["whymap_saline_gw_gt5gl_mha"]),
            "hwsd2_gypsum_ge5pct_mha": round(gyp5, 2),
        })
    p_df = pd.DataFrame(presc_rows)
    xd_pos = np.arange(len(p_df))
    p_specs = [
        ("sodic_ca_flocculation_mha", r"Sodic ($\mathrm{Ca}^{2+}$ priority)", "#C96A1B"),
        ("non_sodic_mulched_drip_mha", "Non-sodic mulched drip", "#1B7867"),
        ("poor_drainage_tile_priority_mha", "Poor drainage (tile)", "#2B6E8C"),
        ("heavy_clay_ge35pct_mha", "Heavy clay (≥35%)", "#6A4C93"),
        ("hwsd2_gypsum_ge5pct_mha", "Soil gypsum ≥5%", "#D4A017"),
        ("whymap_saline_gw_gt5gl_mha", r"Saline GW >5 g L$^{-1}$", "#A61C28"),
    ]
    w_d = 0.132
    offsets_d = np.linspace(-2.5 * w_d, +2.5 * w_d, len(p_specs))
    ax_d.grid(axis="y", color="#EEF1F4", lw=0.45, zorder=0.3)
    for (col_d, lbl_d, clr_d), off_d in zip(p_specs, offsets_d):
        vals_d = p_df[col_d].values
        ax_d.bar(xd_pos + off_d, vals_d, width=w_d, color=clr_d, alpha=0.92, label=lbl_d, zorder=2.5)
        for xi_d, vd in zip(xd_pos, vals_d):
            if vd > 0.05:
                ax_d.text(xi_d + off_d, vd + 0.45, f"{vd:.1f}", ha="center", va="bottom", fontsize=5.15, color="#222222", rotation=90)
    ax_d.set_xticks(xd_pos)
    ax_d.set_xticklabels(p_df["zone_label"], fontsize=5.3)
    ax_d.set_ylim(0, 45.5)
    ax_d.tick_params(axis="y", labelsize=5.3)
    ax_d.set_ylabel("Salt-affected cropland (Mha)", fontsize=5.9, labelpad=2.0)
    ax_d.legend(loc="upper right", ncol=2, frameon=False, fontsize=5.15, handlelength=0.9, handletextpad=0.22, columnspacing=0.60)

    pd.DataFrame(lv_recs).to_csv(SUP_SRC_DIR / "ED_Fig8A_la_violada.csv", index=False)
    pd.DataFrame([{
        "case_id": het["case_id"],
        "diversion_salt_kt_yr": het["field_salt_input_kt_yr"],
        "outlet_total_kt_yr": het["basin_outlet_salt_export_kt_yr"],
        "outlet_growing_kt_yr": round(het_grow * 1000.0, 1),
        "outlet_autumn_leaching_kt_yr": round(het_aut * 1000.0, 1),
        "outlet_other_kt_yr": round(het_other * 1000.0, 1),
        "internal_sink_kt_yr": het["internal_transfer_or_storage_kt_yr"],
        "modeled_sin_pre_kt_yr": het["model_matched_sin_pre_kt_yr"],
    }]).to_csv(SUP_SRC_DIR / "ED_Fig8B_hetao_seasonal.csv", index=False)
    lf_df.to_csv(SUP_SRC_DIR / "ED_Fig8C_ucrb_lees_ferry.csv", index=False)
    p_df.to_csv(SUP_SRC_DIR / "ED_Fig8D_zone_prescriptions.csv", index=False)
    save_pub_figure(fig, SUP_DIR / "ED_Fig8_Observed_Budgets_Soil_Prescriptions.pdf", SUP_DIR / "ED_Fig8_Observed_Budgets_Soil_Prescriptions.png", "ED_Fig8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Nature Water Main Figures 4-5 & ED Figures 1-7.")
    parser.add_argument("--only", choices=["fig4", "fig5", "ed_fig1", "ed_fig2", "ed_fig3", "ed_fig4", "ed_fig5", "ed_fig6", "ed_fig7", "ed_batch", "all"], default="all")
    args = parser.parse_args()

    setup_mpl_style()
    print(f"Generating revised Fig 4-5 & ED Fig 1-7 (target={args.only})...")
    if args.only == "fig4":
        build_fig4()
    elif args.only == "fig5":
        build_fig5()
    elif args.only == "ed_fig1":
        build_ed_fig1()
    elif args.only == "ed_fig2":
        build_ed_fig2()
    elif args.only == "ed_fig3":
        build_ed_fig3()
    elif args.only == "ed_fig4":
        build_ed_fig4()
    elif args.only == "ed_fig5":
        build_ed_fig5()
    elif args.only == "ed_fig6":
        build_ed_fig6()
    elif args.only == "ed_fig7":
        build_ed_fig7()
    elif args.only == "ed_batch":
        build_ed_fig1()
        build_ed_fig2()
        build_ed_fig3()
        build_ed_fig5()
        build_ed_fig6()
        build_ed_fig7()
    else:
        build_fig4()
        build_fig5()
        build_ed_fig1()
        build_ed_fig2()
        build_ed_fig3()
        build_ed_fig4()
        build_ed_fig5()
        build_ed_fig6()
        build_ed_fig7()
    print("Done.")


if __name__ == "__main__":
    prepare_outputs()
    main()
