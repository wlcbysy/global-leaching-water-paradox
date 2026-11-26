#!/usr/bin/env python3
"""
Redesigned Nature Water Main Figures 1-3 & Supplementary Fig. 1 + Source Data CSVs:
  - FigS1_Conceptual_Framework_Three_Tier (Supplementary Fig. 1):
      3-column horizontal schematic of leaching-water paradox, intervention levers, and three-tier accounting
  - Fig1_Leaching_Paradox_Salt_Debt (a-d):
      a) Global curved 0.5-degree Robinson basin-closure globe (376 major basins) + continuous 0.5 deg latitudinal area ribbon + 100% proportion bar
      b) Log-log basin demand+leaching vs SBWA (with top SBWA KDE, all 4 leaching-flipped vertical stems, and 6 extreme-arid edge triangles)
      c) Top-12 basins by unfunded salt-export obligation ("salt debt", Mt yr-1) + ghost total salt input + embedded 830.3 Mt yr-1 salt-budget bar
      d) Horizontal DynQual v1.0 1991-2019 TDS trend rainclouds on symlog scale (debt > 0 vs debt = 0)
  - Fig2_ECw_Thresholds_ArmA (a-e):
      a) Geometric-mean relative yield vs irrigation ECw (continuous 3.55 [3.02-4.27] vs unadjusted cyclic 6.33)
      b) Power-law soil EC % change vs irrigation ECw (beta=0.40, +50% at 2.18 [1.94-2.48] dS m-1, standardized to ECc=0.80)
      c) 6 crop-group dumbbell: empirical ECw*90 +/- 95% CI vs FAO-29 ECe90/1.5 (other cereals in ED4)
      d) Moderator forest of slope delta +/- 95% CI across yield & soil EC with directional cues
      e) Predicted within-study soil EC slope by depth across observed Years 1, 2, 3 ("persistent profile-wide accumulation")
  - Fig3_Hydraulic_Recovery_ArmB (a-e):
      a) Hydraulic & water-efficiency REML forest on lnRR scale with % ticks, 95% CI & 95% PI, exact-var diamonds, I2
      b) 8x5 Sodicity x Amendment heatmap on lnRR scale (Ks, Infil, WUE, BD; P >= 0.05 faded alpha=0.35; missing grey)
      c) ESP meta-regression line (ESP <= 100 physical domain, P = 0.126 n.s.) with 95% CI band & observed Ks points
      d) Dual-pathway scatter: family hydraulic lnRR vs soil_EC lnRR by sodicity (P > 0.50)
      e) Co-measured family yield lnRR vs WUE lnRR (n=64) with 1:1 equal-ET line (ET ratio 0.994 [0.983-1.007])
"""

from __future__ import annotations

from config import HARMONIZED, OUT_PANELS, OUT_FIGURES, table_input, grid_input, prepare_outputs

import argparse
import json
import math

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from patsy import dmatrix
from pyproj import Transformer
from scipy.ndimage import gaussian_filter1d
from scipy import stats
from scipy.stats import gaussian_kde
from shapely.geometry import box
import statsmodels.formula.api as smf

from fig_style import (
    NW_FIG1_PALETTE,
    PARADOX_COLORS,
    ROBIN_CRS,
    add_fig_panel_label,
    fmt_p,
    plot_robinson_basemap,
    save_pub_figure,
    setup_mpl_style,
)

FIG_DIR = OUT_FIGURES
SUP_DIR = OUT_FIGURES
SRC_DIR = OUT_PANELS

BASIN_NAME_MAP = {
    6562: "Upper Indus (PK–IN)",
    5705: "Haihe–NCP (CN)",
    6889: "Lower Indus (PK)",
    5560: "Tarim (CN)",
    5651: "Karakum (TM–IR)",
    4998: "Junggar (CN)",
    6412: "E. Nile Delta (EG)",
    7461: "Krishna (IN)",
    5965: "San Joaquin (US)",
    5513: "West Liaohe (CN)",
    6181: "Central Iran (IR)",
    6376: "Lower Nile (EG–SD)",
    6261: "Imperial–Salton (US)",
    6409: "W. Nile Delta (EG)",
    6347: "Zayandeh–Isfahan (IR)",
    5870: "Atrak (IR)",
    6410: "Mid Nile Delta (EG)",
}


def build_figS1_schematic() -> None:
    """Render Supplementary Fig. 1: standalone conceptual framework of the leaching-water paradox and three-tier accounting."""
    p1 = json.loads((table_input("pillar1_global_summary_current.json")).read_text(encoding="utf-8"))
    p1b = json.loads((table_input("pillar1b_salt_debt_closure_sensitivity_current.json")).read_text(encoding="utf-8"))
    p2a = json.loads((table_input("pillar2_armB_hydraulic_results_current.json")).read_text(encoding="utf-8"))
    p2b = json.loads((table_input("pillar2_armA_ecw_results_current.json")).read_text(encoding="utf-8"))
    p3 = json.loads((table_input("pillar3_global_summary_current.json")).read_text(encoding="utf-8"))

    cd = p1b["closure_decomposition"]
    sd_acc = p1b["salt_debt_accounting"]
    vl_amb = float(p1["physical_leaching_demand_V_leach_km3_yr"]["ambient_observed_water_ec"])
    vl_salt = float(p1b["ambient_vleach_jensen_inequality"]["ambient_vleach_salt_cells_km3_yr"])
    s_in_pre = float(sd_acc["total_irrigation_salt_input_salt_cells_mt_yr"])
    debt_mt = float(sd_acc["total_salt_debt_mt_yr"])
    debt_pct = float(sd_acc["unfunded_share_of_salt_input"]) * 100.0
    n_exc = int(cd["baseline_blue_water_exceeded_basins"])
    n_maj = int(cd["major_salt_basins_ge10kha"])

    ecw_star = float(p2b["dose_response_absolute"]["continuous"]["ecw_star_90pct_dsm"])
    ks_row = next(r for r in p2a["headline_all_endpoints"] if r["endpoint"] == "ks_saturated")
    wue_row = next(r for r in p2a["headline_all_endpoints"] if r["endpoint"] == "wue_or_iwue")
    ks_pct = float(ks_row["pct_change"])
    wue_pct = float(wue_row["pct_change"])

    wat = p3["water_accounting_three_tier"]
    wd = wat["withdrawal_tier_km3_yr"]
    cn = wat["consumption_tier_km3_yr"]
    gs = p3["global_salt_affected_croplands_savings"]
    d_f = float(wd["deltaF_net_brackish_fresh_withdrawal_saving"])
    p_pen = float(wd["P_leaching_penalty"])
    h_wd = float(wd["H_wd_conditional_equal_yield_hydraulic_withdrawal_saving"])
    h_cons = float(cn["H_cons_conditional_equal_yield_saving"])
    b_gw = float(cn["B_gw_consumptive_credit"])
    d_vl = float(cn["delta_vleach_field_leaching_increment"])
    net_dep = float(wat["basin_net_depletion_change_km3_yr"])
    net_act = float(wat["basin_net_depletion_change_actual_et_km3_yr"])
    d_s = float(gs["delta_s_salt_import_mt_yr"])
    n_rescued = int(gs["paradox_basins_relieved_to_blue_feasible"])
    n_res_act = int(gs.get("paradox_basins_relieved_actual", 0))
    n_unrescued = int(gs["paradox_still_exceeded_post_basins"])

    fig = plt.figure(figsize=(181 / 25.4, 58 / 25.4))
    ax = fig.add_axes([0.010, 0.04, 0.980, 0.92])
    ax.set_xlim(0.0, 10.2)
    ax.set_ylim(0.0, 3.45)
    ax.axis("off")

    def draw_accent_card(x: float, y: float, w: float, h: float, title: str, body: str, accent: str, fill: str = "#FFFFFF", body_fs: float = 5.45) -> None:
        outer = mpl.patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.0,rounding_size=0.07",
            facecolor=fill, edgecolor="#D6D9DD", linewidth=0.55, zorder=3,
        )
        ax.add_patch(outer)
        stripe = mpl.patches.FancyBboxPatch(
            (x + 0.012, y + 0.05), 0.055, h - 0.10,
            boxstyle="round,pad=0.0,rounding_size=0.02",
            facecolor=accent, edgecolor="none", zorder=4,
        )
        ax.add_patch(stripe)
        ax.text(x + 0.12, y + h * 0.65, title, ha="left", va="center", fontsize=6.4, fontweight="bold", color="#111111", zorder=5)
        ax.text(x + 0.12, y + h * 0.29, body, ha="left", va="center", fontsize=body_fs, color="#333333", zorder=5)

    def draw_funnel_tier(xc: float, y: float, w: float, h: float, title: str, body: str, fill: str, edge: str, lw: float = 0.55) -> None:
        box_t = mpl.patches.FancyBboxPatch(
            (xc - w / 2, y), w, h,
            boxstyle="round,pad=0.0,rounding_size=0.07",
            facecolor=fill, edgecolor=edge, linewidth=lw, zorder=3,
        )
        ax.add_patch(box_t)
        ax.text(xc, y + h * 0.65, title, ha="center", va="center", fontsize=6.4, fontweight="bold", color="#111111", zorder=5)
        ax.text(xc, y + h * 0.29, body, ha="center", va="center", fontsize=5.35, color="#222222", zorder=5)

    def draw_arrow(x1: float, y1: float, x2: float, y2: float, col: str = "#555555", lw: float = 0.95) -> None:
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops={"arrowstyle": "->,head_width=0.14,head_length=0.19", "color": col, "lw": lw},
            zorder=2,
        )

    # Column 1: Self-reinforcing leaching-water paradox (baseline 1991-2019 cycle)
    ax.text(0.0, 3.24, "Leaching-water paradox (baseline)", ha="left", va="center", fontsize=7.0, fontweight="bold", color="#7A2424")
    draw_accent_card(0.00, 1.76, 1.94, 1.14, "Freshwater scarcity", f"{n_exc}/{n_maj} basins > SBWA", PARADOX_COLORS["PARADOX_INSOLVENT_BLUE_AND_GW"], "#FDF6F6")
    draw_accent_card(2.42, 1.76, 1.94, 1.14, "Brackish & return reuse", r"Applied $\mathrm{EC_w}\uparrow$", PARADOX_COLORS["GW_DEPLETION_ONLY"], "#FEFAF2")
    draw_accent_card(2.42, 0.12, 1.94, 1.14, r"Baseline LR$\uparrow$ & salt load", rf"$\mathit{{V}}_{{\mathrm{{leach}}}}$={vl_salt:.1f} ({vl_amb:.1f} all) km$^3$ yr$^{{-1}}$, {s_in_pre:.0f} Mt", PARADOX_COLORS["PARADOX_BLUE_WATER_EXCEEDED"], "#FDF7F5", body_fs=5.2)
    draw_accent_card(0.00, 0.12, 1.94, 1.14, "Modeled salt debt", rf"{debt_mt:.1f} Mt yr$^{{-1}}$ ({debt_pct:.1f}% input)", PARADOX_COLORS["PARADOX_INSOLVENT_BLUE_AND_GW"], "#FDF6F6")

    draw_arrow(1.96, 2.33, 2.39, 2.33, PARADOX_COLORS["PARADOX_INSOLVENT_BLUE_AND_GW"], lw=1.05)
    draw_arrow(3.39, 1.72, 3.39, 1.30, PARADOX_COLORS["GW_DEPLETION_ONLY"], lw=1.05)
    draw_arrow(2.39, 0.69, 1.96, 0.69, PARADOX_COLORS["PARADOX_BLUE_WATER_EXCEEDED"], lw=1.05)
    draw_arrow(0.97, 1.30, 0.97, 1.72, PARADOX_COLORS["PARADOX_INSOLVENT_BLUE_AND_GW"], lw=1.05)
    ax.text(2.18, 1.51, "Leaching\nparadox", ha="center", va="center", fontsize=6.0, fontweight="bold", color="#8C2323")

    # Bridging arrow from paradox to levers
    draw_arrow(4.39, 1.51, 4.60, 1.51, "#666666", lw=1.0)

    # Column 2: Three intervention levers (post-intervention feedbacks)
    ax.text(4.63, 3.24, "Intervention levers", ha="left", va="center", fontsize=7.0, fontweight="bold", color="#222222")
    draw_accent_card(4.63, 2.06, 2.58, 0.84, "1. Blending reference", rf"$\mathrm{{EC_w^*}}_{{90}}$={ecw_star:.2f} dS m$^{{-1}}$ ($\mathit{{P}}$={p_pen + 1e-9:.1f} km$^3$, $\Delta\mathit{{S}}$=+{d_s:.0f} Mt)", NW_FIG1_PALETTE["lever_blend"], "#F4F8FC", body_fs=5.2)
    draw_accent_card(4.63, 1.09, 2.58, 0.84, "2. Soil hydraulics", rf"$\mathit{{K_s}}$ +{ks_pct:.0f}%, WUE +{wue_pct:.1f}% ($\mathit{{H}}_{{\mathrm{{cons}}}}^*$)", NW_FIG1_PALETTE["lever_hyd"], "#F2FAF7", body_fs=5.45)
    draw_accent_card(4.63, 0.12, 2.58, 0.84, "3. Demand caps & transition", f"Enforce ET cap · shift {n_unrescued} exceeded", NW_FIG1_PALETTE["lever_cap"], "#F7F5FA", body_fs=5.4)

    # Column 3: Narrowing 3-tier accounting funnel
    ax.text(7.48, 3.24, "Three-tier accounting", ha="left", va="center", fontsize=7.0, fontweight="bold", color="#222222")
    xc_f = 8.82
    draw_arrow(7.23, 2.48, 7.45, 2.48, NW_FIG1_PALETTE["lever_blend"], lw=0.9)
    draw_arrow(7.23, 1.51, 7.55, 1.51, NW_FIG1_PALETTE["lever_hyd"], lw=0.9)
    draw_arrow(7.23, 0.54, 7.52, 0.54, NW_FIG1_PALETTE["lever_cap"], lw=0.9)

    draw_funnel_tier(xc_f, 2.06, 2.74, 0.84, "Withdrawal", rf"$\Delta\mathit{{F}}$ = {d_f:.1f}, $\mathit{{H}}_{{\mathrm{{wd}}}}^*$ = {h_wd + 1e-6:.1f} km$^3$ yr$^{{-1}}$", "#EBF2F8", "#8CAED0")
    draw_funnel_tier(xc_f, 1.09, 2.54, 0.84, "Consumption (Ret = 0)", rf"$\mathit{{H}}_{{\mathrm{{cons}}}}^*$={h_cons:.1f}, $\mathit{{B}}_{{\mathrm{{gw}}}}$={b_gw:.1f}, −$\Delta\mathit{{V}}_l$=−{d_vl:.1f} km$^3$ yr$^{{-1}}$", "#EAF4F1", "#7FBAA6")
    net_act_str = f"−{abs(net_act):.1f}" if net_act < 0 else f"+{net_act:.1f}"
    draw_funnel_tier(xc_f, 0.12, 2.60, 0.84, "Basin closure (cap vs act.)", rf"Cap: +{net_dep:.1f} ({n_rescued}/{n_exc}) · Act: {net_act_str} ({n_res_act}/{n_exc})", "#FBECEB", PARADOX_COLORS["PARADOX_INSOLVENT_BLUE_AND_GW"], lw=0.85)
    draw_arrow(xc_f, 2.05, xc_f, 1.94, "#777777", lw=0.8)
    draw_arrow(xc_f, 1.08, xc_f, 0.97, "#777777", lw=0.8)

    summary = [
        ("Major basins", n_maj, "count"), ("Exceeded basins", n_exc, "count"),
        ("Leaching on salt cells", vl_salt, "km3/yr"), ("Leaching on all cells", vl_amb, "km3/yr"),
        ("Salt imports in major basins", s_in_pre, "Mt/yr"), ("Modeled salt debt", debt_mt, "Mt/yr"),
        ("Mixed-production ECw90", ecw_star, "dS/m"), ("Withdrawal penalty", p_pen, "km3/yr"),
        ("Increase in salt applied to fields", d_s, "Mt/yr"), ("Hydraulic conductivity change", ks_pct, "%"),
        ("Water use efficiency change", wue_pct, "%"), ("Net freshwater withdrawal relief", d_f, "km3/yr"),
        ("Hydraulic withdrawal relief under equal production", h_wd, "km3/yr"),
        ("Equal-production consumptive relief", h_cons, "km3/yr"), ("Groundwater credit", b_gw, "km3/yr"),
        ("Incremental leaching", d_vl, "km3/yr"), ("Net relief under caps", net_dep, "km3/yr"),
        ("Net relief under verified ET", net_act, "km3/yr"), ("Basins rescued under caps", n_rescued, "count"),
        ("Basins rescued under verified ET", n_res_act, "count")]
    pd.DataFrame(summary,columns=["quantity","value","unit"]).to_csv(OUT_PANELS/"ED_Fig1_accounting.csv",index=False)
    save_pub_figure(fig, SUP_DIR / "ED_Fig1_Conceptual_Framework_Three_Tier.pdf", SUP_DIR / "ED_Fig1_Conceptual_Framework_Three_Tier.png", "ED_Fig1")


def build_fig1() -> None:
    import matplotlib.patheffects as pe

    grid = pd.read_parquet(grid_input("pillar1_grid05_baseline_current.parquet"))
    basins = pd.read_csv(table_input("pillar1_basin_leaching_closure_current.csv"), low_memory=False)
    p1 = json.loads((table_input("pillar1_global_summary_current.json")).read_text(encoding="utf-8"))
    p1b = json.loads((table_input("pillar1b_salt_debt_closure_sensitivity_current.json")).read_text(encoding="utf-8"))
    sb_debt = pd.read_csv(table_input("pillar1b_basin_salt_debt_current.csv"), low_memory=False)

    sb = basins[basins["salt_irrig_ha"] >= 10000.0].copy()
    fig = plt.figure(figsize=(183 / 25.4, 150 / 25.4))

    X_LEFT_LABEL = 0.010
    X_RIGHT_LABEL = 0.504
    X_RIGHT_EDGE = 0.982
    X_COL2 = 0.634
    W_COL2 = X_RIGHT_EDGE - X_COL2
    SLATE_ZERO = "#5B7387"

    # -------------------------------------------------------------------------
    # Panel a: Frameless Curved 0.5-degree Robinson Globe + Continuous 0.5° Latitudinal Ribbon
    # -------------------------------------------------------------------------
    ax_a = fig.add_axes([0.016, 0.622, 0.788, 0.362])
    add_fig_panel_label(fig, X_LEFT_LABEL, 0.990, "a")
    land_wgs = plot_robinson_basemap(
        ax_a,
        xlim_lon=(-138.0, 168.0),
        ylim_lat=(-48.0, 64.0),
        graticule_lats=(-30, 0, 30, 60),
        curved_frame=True,
    )

    b_status = sb.set_index("aware_basin_idx")["paradox_status_amb"].to_dict()
    salt_cells = grid[grid["salt_affected"] & (grid["salt_irrig_ha"] > 0) & (grid["vleach_amb_km3_yr"].fillna(0.0) >= 0)].copy()
    salt_cells["b_stat"] = salt_cells["aware_basin_idx"].map(lambda b: b_status.get(b, "MINOR_BASIN"))

    # Land-clipped 0.5-deg dissolved salt-cropland patches: 376 major basins in status colors (96.15 Mha)
    # and minor basins (<10,000 ha; 0.97 Mha) explicitly separated in neutral grey (never silently masked as Feasible)
    order = ["HYDROLOGICALLY_FEASIBLE", "GW_DEPLETION_ONLY", "PARADOX_BLUE_WATER_EXCEEDED", "PARADOX_INSOLVENT_BLUE_AND_GW"]
    plot_order = ["MINOR_BASIN"] + order
    map_colors = {**PARADOX_COLORS, "MINOR_BASIN": "#A3AEB8"}
    cell_geoms = [box(lon - 0.25, lat - 0.25, lon + 0.25, lat + 0.25) for lon, lat in zip(salt_cells["lon"], salt_cells["lat"])]
    cell_gdf = gpd.GeoDataFrame(salt_cells[["b_stat"]], geometry=cell_geoms, crs="EPSG:4326")
    cell_dis = cell_gdf.dissolve(by="b_stat")
    cell_clip = gpd.clip(cell_dis, land_wgs).to_crs(ROBIN_CRS)
    for st in plot_order:
        if st in cell_clip.index:
            z_ord = 2.6 if st == "MINOR_BASIN" else 3.0
            cell_clip.loc[[st]].plot(ax=ax_a, facecolor=map_colors[st], edgecolor="none", alpha=0.96, zorder=z_ord)

    # Leaching-flipped basin centroid rings (based on non-negative consumptive use)
    sb["atotuse_pos"] = np.maximum(sb["atotuse_km3_yr"], 0.0)
    sb["tot_demand"] = sb["atotuse_pos"] + sb["vleach_amb_km3_yr"]
    sb["leach_flipped"] = (sb["atotuse_pos"] <= sb["sbwa_efr60_km3_yr"]) & sb["blue_water_exceeded_amb"]
    fl_all = sb[sb["leach_flipped"]].copy()
    tr_rob = Transformer.from_crs("EPSG:4326", ROBIN_CRS, always_xy=True)
    fl_x, fl_y = tr_rob.transform(fl_all["lon_centroid"].values, fl_all["lat_centroid"].values)
    ax_a.scatter(fl_x, fl_y, facecolor="none", edgecolor="white", linewidths=1.65, s=32, zorder=4.8)
    ax_a.scatter(fl_x, fl_y, facecolor="none", edgecolor="#111111", linewidths=0.95, s=22, zorder=5.0)
    bx_2, by_2 = tr_rob.transform(35.4, 28.5)
    ax_a.text(
        bx_2, by_2, "×2",
        fontsize=5.4, fontweight="bold", color="#111111",
        ha="left", va="center", zorder=5.5,
        path_effects=[pe.withStroke(linewidth=1.6, foreground="white")],
    )

    # Freeze ax_a geometry and pixel-lock smooth continuous latitudinal ribbon alongside it (exact 96.15 Mha major-basin scope)
    fig.canvas.draw()
    pos_a = ax_a.get_position()
    ax_lat = fig.add_axes([pos_a.x1 + 0.028, pos_a.y0, X_RIGHT_EDGE - (pos_a.x1 + 0.028), pos_a.height])
    ax_lat.set_ylim(ax_a.get_ylim())

    salt_maj = salt_cells[salt_cells["b_stat"].isin(order)].copy()
    lats_grid = np.arange(-47.75, 64.0, 0.5)
    stack_order = ["PARADOX_INSOLVENT_BLUE_AND_GW", "PARADOX_BLUE_WATER_EXCEEDED", "GW_DEPLETION_ONLY", "HYDROLOGICALLY_FEASIBLE"]
    mat_lat = np.zeros((len(lats_grid), len(stack_order)))
    for j, st in enumerate(stack_order):
        sub_l = salt_maj[salt_maj["b_stat"] == st]
        grp_l = sub_l.groupby("lat")["salt_irrig_ha"].sum() / 1e6
        for i, lv in enumerate(lats_grid):
            mat_lat[i, j] = float(grp_l.get(lv, 0.0)) * 2.0  # Mha per degree latitude
    mat_smooth = np.column_stack([gaussian_filter1d(mat_lat[:, j], sigma=1.6, mode="constant") for j in range(len(stack_order))])
    _, y_rob_lat = tr_rob.transform(np.zeros_like(lats_grid), lats_grid)

    _, y_25n = tr_rob.transform(0.0, 25.0)
    _, y_45n = tr_rob.transform(0.0, 45.0)
    _, y_60n = tr_rob.transform(0.0, 60.0)
    ax_lat.axhspan(y_25n, y_45n, facecolor="#F5F3EF", alpha=0.92, zorder=0.5)
    ax_lat.axhline(y_25n, ls="--", color="#CFCAC0", lw=0.50, zorder=1.2)
    ax_lat.axhline(y_45n, ls="--", color="#CFCAC0", lw=0.50, zorder=1.2)

    cum_prev = np.zeros_like(y_rob_lat)
    for j, st in enumerate(stack_order):
        cum_curr = cum_prev + mat_smooth[:, j]
        ax_lat.fill_betweenx(y_rob_lat, cum_prev, cum_curr, facecolor=PARADOX_COLORS[st], edgecolor="none", alpha=0.88, zorder=3.0)
        if j < len(stack_order) - 1:
            ax_lat.plot(cum_curr, y_rob_lat, color="white", lw=0.45, zorder=3.4)
        cum_prev = cum_curr
    ax_lat.plot(cum_prev, y_rob_lat, color="#33414D", lw=0.65, zorder=4.0)
    ax_lat.plot([8.60, 8.95, 8.95, 8.60], [y_25n, y_25n, y_45n, y_45n], color="#666666", lw=0.65, zorder=4.2)

    for lat in (-30, 0, 30, 60):
        _, yg = tr_rob.transform(0.0, float(lat))
        ax_lat.axhline(yg, ls="--", color="#DCE5EC", lw=0.45, zorder=1.0)
    y_ticks_lat = [tr_rob.transform(0.0, float(lat))[1] for lat in (-30, 0, 30, 60)]
    ax_lat.set_yticks(y_ticks_lat)
    ax_lat.set_yticklabels(["30°S", "0°", "30°N", "60°N"], fontsize=6.1)
    ax_lat.set_xlim(0.0, 9.6)
    ax_lat.set_xticks([0, 4, 8])
    ax_lat.tick_params(axis="x", labelsize=6.1)
    ax_lat.tick_params(axis="y", length=2.0, pad=1.5)
    ax_lat.set_xlabel(r"Cropland area (Mha deg$^{-1}$)", fontsize=6.8, labelpad=1.8)
    pct_25_45n = salt_maj.loc[(salt_maj["lat"] >= 25.0) & (salt_maj["lat"] < 45.0), "salt_irrig_ha"].sum() / salt_maj["salt_irrig_ha"].sum() * 100.0
    y_note_lat = y_45n + (y_60n - y_45n) * 0.45
    ax_lat.text(4.8, y_note_lat, f"25°–45°N\n({pct_25_45n:.0f}%)", ha="center", va="center", fontsize=5.65, color="#333333")

    # Embedded 100%-stacked global proportion bar directly below globe
    st_cnt = p1["salt_affected_basins_ge_10kha"]["paradox_status_ambient_counts"]
    st_mha = p1["salt_affected_basins_ge_10kha"]["salt_irrig_mha_by_status_ambient"]
    tot_maj_mha = sum(float(st_mha.get(s, 0.0)) for s in order)
    minor_mha = float(salt_cells.loc[salt_cells["b_stat"] == "MINOR_BASIN", "salt_irrig_ha"].sum() / 1e6)
    ax_comp = fig.add_axes([pos_a.x0 + 0.088, pos_a.y0 - 0.024, pos_a.width - 0.110, 0.017])
    ax_comp.set_xlim(0.0, 100.0)
    ax_comp.set_ylim(0.0, 1.0)
    ax_comp.set_xticks([])
    ax_comp.set_yticks([])
    ax_comp.axis("off")
    x_acc = 0.0
    for st in order:
        pct_s = float(st_mha.get(st, 0.0)) / tot_maj_mha * 100.0
        ax_comp.add_patch(mpl.patches.Rectangle((x_acc, 0.0), pct_s, 1.0, facecolor=PARADOX_COLORS[st], edgecolor="white", linewidth=0.6))
        ax_comp.text(x_acc + pct_s / 2.0, 0.48, f"{pct_s:.1f}%", ha="center", va="center", fontsize=5.4, fontweight="bold", color="white")
        x_acc += pct_s
    ax_comp.text(-1.5, 0.48, f"{tot_maj_mha:.2f} Mha", ha="right", va="center", fontsize=5.7, color="#444444", fontweight="bold")

    # Shared borderless 2-row severity-ordered legend below composition bar
    labels_map = {
        "HYDROLOGICALLY_FEASIBLE": f"Feasible ({st_cnt.get('HYDROLOGICALLY_FEASIBLE', 0)}; {st_mha.get('HYDROLOGICALLY_FEASIBLE', 0):.1f} Mha)",
        "GW_DEPLETION_ONLY": f"GW depletion only ({st_cnt.get('GW_DEPLETION_ONLY', 0)}; {st_mha.get('GW_DEPLETION_ONLY', 0):.1f} Mha)",
        "PARADOX_BLUE_WATER_EXCEEDED": f"Blue exceeded ({st_cnt.get('PARADOX_BLUE_WATER_EXCEEDED', 0)}; {st_mha.get('PARADOX_BLUE_WATER_EXCEEDED', 0):.1f} Mha)",
        "PARADOX_INSOLVENT_BLUE_AND_GW": f"Dual blue + GW ({st_cnt.get('PARADOX_INSOLVENT_BLUE_AND_GW', 0)}; {st_mha.get('PARADOX_INSOLVENT_BLUE_AND_GW', 0):.1f} Mha)",
    }
    leg_row1 = [
        mpl.patches.Patch(facecolor=PARADOX_COLORS["HYDROLOGICALLY_FEASIBLE"], edgecolor="none", label=labels_map["HYDROLOGICALLY_FEASIBLE"]),
        mpl.patches.Patch(facecolor=PARADOX_COLORS["GW_DEPLETION_ONLY"], edgecolor="none", label=labels_map["GW_DEPLETION_ONLY"]),
        mpl.patches.Patch(facecolor=PARADOX_COLORS["PARADOX_BLUE_WATER_EXCEEDED"], edgecolor="none", label=labels_map["PARADOX_BLUE_WATER_EXCEEDED"]),
        mpl.patches.Patch(facecolor=PARADOX_COLORS["PARADOX_INSOLVENT_BLUE_AND_GW"], edgecolor="none", label=labels_map["PARADOX_INSOLVENT_BLUE_AND_GW"]),
    ]
    leg_row2 = [
        mpl.lines.Line2D([0], [0], marker="o", color="none", markerfacecolor="none", markeredgecolor="#111111", markeredgewidth=0.95, markersize=4.8, label=rf"Flipped by leaching ($\mathit{{n}}$ = {len(fl_all)})"),
        mpl.patches.Patch(facecolor="#A3AEB8", edgecolor="none", label=f"Minor basins <10 kha ({minor_mha:.2f} Mha)"),
    ]
    fig.legend(
        handles=leg_row1,
        loc="upper center",
        bbox_to_anchor=(pos_a.x0 + pos_a.width * 0.49, pos_a.y0 - 0.031),
        ncol=4,
        frameon=False,
        fontsize=5.65,
        handlelength=0.90,
        handleheight=0.80,
        handletextpad=0.32,
        columnspacing=0.75,
        borderaxespad=0.0,
    )
    fig.legend(
        handles=leg_row2,
        loc="upper center",
        bbox_to_anchor=(pos_a.x0 + pos_a.width * 0.49, pos_a.y0 - 0.054),
        ncol=2,
        frameon=False,
        fontsize=5.65,
        handlelength=0.90,
        handleheight=0.80,
        handletextpad=0.32,
        columnspacing=1.20,
        borderaxespad=0.0,
    )

    # -------------------------------------------------------------------------
    # Panel b: Large Square Log-Log Basin Closure Scatter + Top SBWA KDE
    # -------------------------------------------------------------------------
    add_fig_panel_label(fig, X_LEFT_LABEL, 0.536, "b")
    ax_b_top = fig.add_axes([0.082, 0.486, 0.414, 0.042])
    lims_b = [0.004, 480.0]
    xs_b = np.linspace(np.log10(lims_b[0]), np.log10(lims_b[1]), 320)
    ins_log = np.log10(sb.loc[sb["blue_water_exceeded_amb"], "sbwa_efr60_km3_yr"].clip(lower=1e-4).values)
    oth_log = np.log10(sb.loc[~sb["blue_water_exceeded_amb"], "sbwa_efr60_km3_yr"].clip(lower=1e-4).values)
    kde_ins = gaussian_kde(ins_log)(xs_b)
    kde_oth = gaussian_kde(oth_log)(xs_b)
    within_col = SLATE_ZERO
    ax_b_top.fill_between(10 ** xs_b, 0.0, kde_oth, facecolor=within_col, alpha=0.22, edgecolor="none")
    ax_b_top.plot(10 ** xs_b, kde_oth, color=within_col, lw=0.90)
    ax_b_top.fill_between(10 ** xs_b, 0.0, kde_ins, facecolor=PARADOX_COLORS["PARADOX_INSOLVENT_BLUE_AND_GW"], alpha=0.26, edgecolor="none")
    ax_b_top.plot(10 ** xs_b, kde_ins, color=PARADOX_COLORS["PARADOX_INSOLVENT_BLUE_AND_GW"], lw=0.95)
    ax_b_top.set_xscale("log")
    ax_b_top.set_xlim(*lims_b)
    ax_b_top.set_ylim(0.0, max(float(kde_ins.max()), float(kde_oth.max())) * 1.15)
    ax_b_top.set_xticks([])
    ax_b_top.set_yticks([])
    for sp in ("top", "left", "right"):
        ax_b_top.spines[sp].set_visible(False)
    ax_b_top.spines["bottom"].set_linewidth(0.45)
    ax_b_top.spines["bottom"].set_color("#CCCCCC")
    ax_b_top.text(0.79, 0.72, "Exceeded", transform=ax_b_top.transAxes, ha="right", va="center", fontsize=5.8, fontweight="bold", color=PARADOX_COLORS["PARADOX_INSOLVENT_BLUE_AND_GW"])
    ax_b_top.text(0.99, 0.72, "Within limit", transform=ax_b_top.transAxes, ha="right", va="center", fontsize=5.8, color=within_col)

    ax_b = fig.add_axes([0.082, 0.068, 0.414, 0.416])
    ax_b.grid(True, which="major", color="#EEF0F2", lw=0.45, zorder=0.5)
    sb_plot = sb[(sb["sbwa_efr60_km3_yr"] >= lims_b[0]) & (sb["tot_demand"] >= lims_b[0])].copy()
    ax_b.fill_between(lims_b, lims_b, lims_b[1], color="#FBF0EE", alpha=0.75, zorder=1)
    ax_b.plot([lims_b[0], lims_b[1] / 2.0], [lims_b[0] * 2.0, lims_b[1]], ls=":", color="#C4C8CE", lw=0.70, zorder=1.5)
    ax_b.text(55.0, 140.0, "2×", fontsize=5.8, color="#777777", rotation=43.5, ha="center", va="center", zorder=2)
    ax_b.plot(lims_b, lims_b, "--", color="#222222", lw=1.0, zorder=2)
    ax_b.text(0.055, 0.032, "1:1", fontsize=5.6, color="#444444", rotation=43.5, ha="center", va="center", zorder=2)

    for st in order:
        sub = sb_plot[(sb_plot["paradox_status_amb"] == st) & (~sb_plot["leach_flipped"])]
        ax_b.scatter(
            sub["sbwa_efr60_km3_yr"], sub["tot_demand"],
            c=PARADOX_COLORS[st],
            s=np.clip(sub["salt_irrig_ha"] / 140000.0, 6.0, 68.0),
            alpha=0.84, edgecolors="#555555", linewidths=0.25, zorder=3,
        )

    sb_low = sb[(sb["sbwa_efr60_km3_yr"] < lims_b[0]) & (sb["tot_demand"] >= lims_b[0])].copy()
    if not sb_low.empty:
        ax_b.scatter(
            np.full(len(sb_low), lims_b[0] * 1.25), sb_low["tot_demand"],
            marker="<", c=[PARADOX_COLORS[s] for s in sb_low["paradox_status_amb"]],
            s=20, alpha=0.88, edgecolors="#444444", linewidths=0.25, zorder=5,
        )
        ax_b.annotate(
            f"{len(sb_low)} basins (<{lims_b[0]})",
            xy=(lims_b[0] * 1.35, float(sb_low["tot_demand"].max())),
            xytext=(0.011, 1.35),
            arrowprops={"arrowstyle": "->", "color": "#7A2424", "lw": 0.60, "shrinkA": 1, "shrinkB": 2},
            fontsize=5.7, color="#7A2424", ha="left", va="center", zorder=6,
        )

    fl_plot = sb_plot[sb_plot["leach_flipped"]].copy()
    for _, r_fl in fl_plot.iterrows():
        xv = float(r_fl["sbwa_efr60_km3_yr"])
        yv = float(r_fl["tot_demand"])
        y0_pre = float(r_fl["atotuse_pos"])
        if y0_pre >= lims_b[0]:
            ax_b.plot([xv, xv], [y0_pre, yv], "-", color="#111111", lw=1.0, zorder=5.5)
            ax_b.scatter([xv], [y0_pre], facecolor="white", edgecolor="#111111", linewidths=0.80, s=15, zorder=6)
    ax_b.scatter(
        fl_plot["sbwa_efr60_km3_yr"], fl_plot["tot_demand"],
        c=[PARADOX_COLORS[s] for s in fl_plot["paradox_status_amb"]],
        s=np.clip(fl_plot["salt_irrig_ha"] / 140000.0, 6.0, 68.0),
        edgecolors="#111111", linewidths=1.20, zorder=7,
    )

    ax_b.set_xscale("log")
    ax_b.set_yscale("log")
    ax_b.set_xlim(*lims_b)
    ax_b.set_ylim(*lims_b)
    ax_b.set_xticks([1e-2, 1e-1, 1e0, 1e1, 1e2])
    ax_b.set_yticks([1e-2, 1e-1, 1e0, 1e1, 1e2])
    ax_b.tick_params(labelsize=6.4)
    ax_b.set_xlabel(r"Sustainable blue-water availability (km$^3$ yr$^{-1}$)", fontsize=7.4, labelpad=2.2)
    ax_b.set_ylabel(r"Consumptive water use + leaching (km$^3$ yr$^{-1}$)", fontsize=7.4, labelpad=2.2)

    tag_map = {
        6376: ("Lower Nile", (3.6, 105.0), "right", "center"),
        6562: ("Upper Indus", (38.0, 225.0), "center", "bottom"),
        5705: ("Haihe–NCP", (2.6, 36.0), "right", "center"),
        6261: ("Imperial–Salton", (0.28, 4.6), "right", "center"),
        5560: ("Tarim", (1.6, 6.2), "right", "center"),
        6891: ("Ganges", (255.0, 72.0), "center", "top"),
    }
    for bid, (nm, (tx, ty), ha_al, va_al) in tag_map.items():
        row = sb_plot[sb_plot["aware_basin_idx"] == bid]
        if not row.empty:
            xv, yv = float(row["sbwa_efr60_km3_yr"].iloc[0]), float(row["tot_demand"].iloc[0])
            ax_b.plot([xv, tx], [yv, ty], "-", color="#555555", lw=0.45, zorder=4)
            ax_b.text(tx, ty, nm, ha=ha_al, va=va_al, fontsize=6.0, color="#111111", zorder=8)

    ax_b.text(
        0.035, 0.96,
        f"Exceeded: {int(sb['blue_water_exceeded_amb'].sum())}/{len(sb)} basins\n({float(sb.loc[sb['blue_water_exceeded_amb'], 'salt_irrig_ha'].sum()/1e6):.2f} Mha)",
        transform=ax_b.transAxes, va="top", fontsize=6.2, color=PARADOX_COLORS["PARADOX_INSOLVENT_BLUE_AND_GW"], fontweight="bold",
    )

    ms_05 = math.sqrt(float(np.clip(0.5e6 / 140000.0, 6.0, 68.0)))
    ms_40 = math.sqrt(float(np.clip(4.0e6 / 140000.0, 6.0, 68.0)))
    b_handles = [
        mpl.lines.Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor="#111111", markeredgewidth=0.80, markersize=4.0, label="Pre-leaching"),
        mpl.lines.Line2D([0], [0], marker="o", color="none", markerfacecolor=PARADOX_COLORS["PARADOX_BLUE_WATER_EXCEEDED"], markeredgecolor="#111111", markeredgewidth=1.05, markersize=5.0, label=rf"{int(sb.blue_water_exceeded_amb.sum())} exceeded; {len(fl_all)} newly"),
        mpl.lines.Line2D([0], [0], marker="o", color="none", markerfacecolor="#BDBDBD", markeredgecolor="#555555", markeredgewidth=0.3, markersize=ms_05, label="0.5 Mha"),
        mpl.lines.Line2D([0], [0], marker="o", color="none", markerfacecolor="#BDBDBD", markeredgecolor="#555555", markeredgewidth=0.3, markersize=ms_40, label="4 Mha"),
    ]
    ax_b.legend(handles=b_handles, loc="lower right", frameon=True, facecolor="white", edgecolor="none", framealpha=0.88, fontsize=5.6, handletextpad=0.25, labelspacing=0.28, borderpad=0.18)

    # -------------------------------------------------------------------------
    # Panel c: Top-12 Salt-Debt Basins (Linear Scale) + Embedded Global Budget Bar
    # -------------------------------------------------------------------------
    add_fig_panel_label(fig, X_RIGHT_LABEL, 0.536, "c")
    ax_c = fig.add_axes([X_COL2, 0.284, W_COL2, 0.244])
    top12 = sb_debt.sort_values("salt_debt_mt_yr", ascending=False).head(12).copy()
    top12["label"] = top12.apply(lambda r: BASIN_NAME_MAP.get(int(r["aware_basin_idx"]), f"{r['NAME']} #{int(r['aware_basin_idx'])}"), axis=1)
    top12_rev = top12.iloc[::-1].reset_index(drop=True)
    yc = np.arange(len(top12_rev))

    for yi in yc:
        if yi % 2 == 1:
            ax_c.axhspan(yi - 0.44, yi + 0.44, facecolor="#F7F8F9", zorder=0.5)

    ax_c.barh(yc, top12_rev["s_in_salt_cells_mt_yr"], height=0.60, facecolor="#EDE5E2", edgecolor="#C5B5AE", linewidth=0.42, zorder=2)
    bar_cols = [PARADOX_COLORS[s] for s in top12_rev["paradox_status_amb"]]
    ax_c.barh(yc, top12_rev["salt_debt_mt_yr"], height=0.60, color=bar_cols, alpha=0.92, zorder=3)

    for yi, debt, sin in zip(yc, top12_rev["salt_debt_mt_yr"], top12_rev["s_in_salt_cells_mt_yr"]):
        if sin > debt * 1.05:
            ax_c.text(sin + 0.65, yi, f"{debt:.1f} / {sin:.1f}", va="center", ha="left", fontsize=5.75, color="#222222")
        else:
            ax_c.text(debt + 0.65, yi, f"{debt:.1f}", va="center", ha="left", fontsize=5.75, color="#222222")

    ax_c.set_xlim(0.0, 45.0)
    ax_c.set_xticks([0, 10, 20, 30, 40])
    ax_c.set_ylim(-0.65, len(top12_rev) - 0.35)
    ax_c.set_yticks(yc)
    ax_c.set_yticklabels(top12_rev["label"], fontsize=5.6)
    ax_c.tick_params(axis="x", labelsize=6.4)
    ax_c.grid(axis="x", which="major", color="#ECEEEF", lw=0.5, zorder=1)
    ax_c.set_xlabel(r"Modeled salt debt (Mt yr$^{-1}$)", fontsize=7.4, labelpad=2.0)

    sd_acc = p1b["salt_debt_accounting"]
    tot_sin = float(sd_acc["total_irrigation_salt_input_salt_cells_mt_yr"])
    debt_mt = float(sd_acc["total_salt_debt_mt_yr"])
    by_st = sd_acc["salt_debt_by_status_mt_yr"]
    dual_mt = float(by_st["dual_insolvent"])
    bw_mt = float(by_st["blue_water_exceeded_only"])
    gw_sub_mt = float(by_st["gw_subsidized_salt_input"])
    sbwa_mt = float(by_st["within_sbwa_salt_input"])
    top12_share = float(top12["salt_debt_mt_yr"].sum() / debt_mt * 100.0)

    ax_c.text(
        0.975, 0.305,
        rf"Major-basin salt input ($\mathit{{n}}$ = {len(sb)}): {tot_sin:.1f} Mt yr$^{{-1}}$" + "\n" + rf"Unfunded debt: {debt_mt:.1f} Mt yr$^{{-1}}$ ({debt_mt / tot_sin * 100.0:.1f}%; top 12 = {top12_share:.1f}%)",
        transform=ax_c.transAxes, ha="right", va="bottom", fontsize=5.5, color="#222222",
    )

    ax_c_sub = ax_c.inset_axes([0.42, 0.185, 0.555, 0.092])
    ax_c_sub.set_xlim(0.0, 100.0)
    ax_c_sub.set_ylim(0.0, 1.0)
    ax_c_sub.set_xticks([])
    ax_c_sub.set_yticks([])
    ax_c_sub.axis("off")
    segs_c = [
        (dual_mt / tot_sin * 100.0, PARADOX_COLORS["PARADOX_INSOLVENT_BLUE_AND_GW"]),
        (bw_mt / tot_sin * 100.0, PARADOX_COLORS["PARADOX_BLUE_WATER_EXCEEDED"]),
        (gw_sub_mt / tot_sin * 100.0, PARADOX_COLORS["GW_DEPLETION_ONLY"]),
        (sbwa_mt / tot_sin * 100.0, PARADOX_COLORS["HYDROLOGICALLY_FEASIBLE"]),
    ]
    xc_acc = 0.0
    for pct_c, col_c in segs_c:
        ax_c_sub.add_patch(mpl.patches.Rectangle((xc_acc, 0.0), pct_c, 1.0, facecolor=col_c, edgecolor="white", linewidth=0.5))
        if pct_c >= 8.0:
            ax_c_sub.text(xc_acc + pct_c / 2.0, 0.48, f"{pct_c:.0f}%", ha="center", va="center", fontsize=5.4, fontweight="bold", color="white")
        xc_acc += pct_c

    c_handles = [
        mpl.patches.Patch(facecolor="#EDE5E2", edgecolor="#C5B5AE", linewidth=0.42, label="Total salt input (where > debt)"),
    ]
    ax_c.legend(handles=c_handles, loc="lower right", bbox_to_anchor=(0.99, 0.02), frameon=False, fontsize=5.45, ncol=1, handlelength=0.90, handleheight=0.75, handletextpad=0.35, borderpad=0.05)

    # -------------------------------------------------------------------------
    # Panel d: Sleek Horizontal Symlog Raincloud of DynQual 1991-2019 TDS Trends
    # -------------------------------------------------------------------------
    add_fig_panel_label(fig, X_RIGHT_LABEL, 0.236, "d")
    ax_d = fig.add_axes([X_COL2, 0.068, W_COL2, 0.160])
    dq_valid = sb_debt.dropna(subset=["dynqual_tds_slope_median_mgl_yr"]).copy()
    g_zero = dq_valid.loc[dq_valid["salt_debt_mt_yr"] <= 0, "dynqual_tds_slope_median_mgl_yr"].values
    g_pos = dq_valid.loc[dq_valid["salt_debt_mt_yr"] > 0, "dynqual_tds_slope_median_mgl_yr"].values

    ax_d.set_xscale("symlog", linthresh=2.0, linscale=1.2)
    ax_d.set_xlim(-38.0, 140.0)
    ax_d.set_ylim(-0.46, 1.84)
    ax_d.spines["left"].set_bounds(-0.34, 1.40)
    ax_d.grid(axis="x", which="major", color="#ECEEEF", lw=0.45, zorder=0.5)
    ax_d.axvline(0.0, ls="--", color="#555555", lw=0.75, zorder=1.0)

    tr_sy = ax_d.xaxis.get_transform()
    for y_idx, vals, col in [
        (0, g_zero, SLATE_ZERO),
        (1, g_pos, PARADOX_COLORS["PARADOX_INSOLVENT_BLUE_AND_GW"]),
    ]:
        vt = tr_sy.transform(vals)
        vt_grid = np.linspace(float(vt.min()), float(vt.max()), 280)
        dens = gaussian_kde(vt)(vt_grid)
        dens_h = dens / float(dens.max()) * 0.25
        x_grid = tr_sy.inverted().transform(vt_grid)
        ax_d.fill_between(x_grid, y_idx + 0.065, y_idx + 0.065 + dens_h, facecolor=col, edgecolor="none", alpha=0.28, zorder=2.0)
        ax_d.plot(x_grid, y_idx + 0.065 + dens_h, color=col, lw=0.90, zorder=3.0)

        rng = np.random.default_rng(20261124 + y_idx)
        yj = y_idx - 0.20 + rng.uniform(-0.075, 0.075, size=len(vals))
        ax_d.scatter(vals, yj, color=col, s=5.2, alpha=0.45, linewidths=0, zorder=3.0)

        q05, q25, q50, q75, q95 = np.percentile(vals, [5, 25, 50, 75, 95])
        ax_d.plot([q05, q95], [y_idx, y_idx], "-", color="#333333", lw=0.85, zorder=4.0)
        rect = mpl.patches.Rectangle((q25, y_idx - 0.048), q75 - q25, 0.096, facecolor=col, edgecolor="#222222", linewidth=0.70, alpha=0.90, zorder=5.0)
        ax_d.add_patch(rect)
        ax_d.plot([q50, q50], [y_idx - 0.048, y_idx + 0.048], "-", color="#FFFFFF", lw=1.45, zorder=6.0)
        ax_d.plot([q50, q50], [y_idx + 0.055, y_idx + 0.32], ":", color=col, lw=0.75, zorder=4.0)
        ax_d.text(q50, y_idx + 0.34, f"{q50:+.2f}", ha="center", va="bottom", fontsize=5.8, fontweight="bold", color=col, zorder=7.0)

    dq_st = p1b["dynqual_1991_2019_trend_consistency"]
    med_ratio = float(np.median(g_pos) / np.median(g_zero))
    ax_d.text(
        0.50, 0.99,
        rf"{med_ratio:.1f}× median · Mann–Whitney {fmt_p(dq_st['mannwhitney_twosided_p'])}",
        transform=ax_d.transAxes, ha="center", va="top", fontsize=5.65, color="#111111", zorder=7.0,
    )

    ax_d.set_xticks([-25, -5, 0, 5, 25, 100])
    ax_d.set_xticklabels(["−25", "−5", "0", "+5", "+25", "+100"], fontsize=6.4)
    ax_d.set_yticks([0, 1])
    ax_d.set_yticklabels([rf"Salt debt = 0" + "\n" + rf"($\mathit{{n}}$ = {len(g_zero)})", rf"Salt debt > 0" + "\n" + rf"($\mathit{{n}}$ = {len(g_pos)})"], fontsize=6.1)
    ax_d.set_xlabel(r"Surface-water TDS trend (mg L$^{-1}$ yr$^{-1}$)", fontsize=7.4, labelpad=2.2)

    SRC_DIR.mkdir(parents=True, exist_ok=True)
    salt_cells[["r", "c", "lat", "lon", "salt_irrig_ha", "b_stat"]].to_csv(SRC_DIR / "Fig1A_basin_closure_map.csv", index=False)
    sb[["aware_basin_idx", "NAME", "salt_irrig_ha", "sbwa_efr60_km3_yr", "tot_demand", "paradox_status_amb", "leach_flipped"]].to_csv(SRC_DIR / "Fig1B_basin_closure_scatter.csv", index=False)
    top12[["aware_basin_idx", "label", "salt_debt_mt_yr", "s_in_salt_cells_mt_yr", "salt_irrig_ha", "paradox_status_amb"]].to_csv(SRC_DIR / "Fig1C_top12_salt_debt.csv", index=False)
    dq_valid[["aware_basin_idx", "NAME", "salt_debt_mt_yr", "dynqual_tds_slope_median_mgl_yr"]].to_csv(SRC_DIR / "Fig1D_dynqual_trend_by_debt.csv", index=False)

    save_pub_figure(fig, FIG_DIR / "Fig1_Leaching_Paradox_Salt_Debt.pdf", FIG_DIR / "Fig1_Leaching_Paradox_Salt_Debt.png", "Fig1")


def build_fig2() -> None:
    p2b = json.loads((table_input("pillar2_armA_ecw_results_current.json")).read_text(encoding="utf-8"))
    arma = pd.read_csv(HARMONIZED / "armA_extracted_current.csv", low_memory=False)
    u = arma[arma["qc_gate_status"].isin(["PASS_READY", "PASS_NO_VARIANCE"])].copy()
    for col in ["ecw_treatment_dsm", "ecw_control_dsm", "mean_treatment", "mean_control", "n_replicates"]:
        u[col] = pd.to_numeric(u[col], errors="coerce")
    u = u.dropna(subset=["ecw_treatment_dsm", "ecw_control_dsm", "mean_treatment", "mean_control"]).copy()
    u = u[(u["mean_treatment"] > 0) & (u["mean_control"] > 0) & (u["ecw_treatment_dsm"] > u["ecw_control_dsm"])].copy()
    u["lnRR"] = np.log(u["mean_treatment"] / u["mean_control"])
    metric_aliases = {"seed_cotton_yield": "grain_yield", "fruit_yield": "grain_yield", "tuber_yield": "grain_yield", "marketable_yield": "grain_yield", "crop_yield": "grain_yield", "lint_yield": "grain_yield", "fresh_yield": "grain_yield", "cane_yield": "grain_yield", "WUE": "wue_or_iwue", "IWUE": "wue_or_iwue", "soil_ECe": "soil_EC", "soil_EC1_5": "soil_EC"}
    u["metric_std"] = u["metric_name"].replace(metric_aliases)
    fam_counts = u.groupby(["family_id", "metric_std"])["lnRR"].transform("count").clip(lower=1)
    u["w_obs"] = u["n_replicates"].fillna(3.0).clip(lower=1.0) / fam_counts

    COL_RISK = "#A61C28"
    COL_REF = "#2B6E8C"
    COL_FAO = "#C96A1B"
    COL_DEPTH = ["#D98A2B", "#2B7A8B", "#2C3E6B"]

    fig = plt.figure(figsize=(180 / 25.4, 144 / 25.4))

    # Top row: [y0=0.562, y1=0.958] with flush 0.002 top KDE strips; Bottom row: [y0=0.080, y1=0.448]
    ax_a_top = fig.add_axes([0.072, 0.922, 0.420, 0.036])
    ax_a = fig.add_axes([0.072, 0.562, 0.420, 0.358])
    add_fig_panel_label(fig, 0.008, 0.984, "a")

    ax_b_top = fig.add_axes([0.576, 0.922, 0.404, 0.036])
    ax_b = fig.add_axes([0.576, 0.562, 0.404, 0.358])
    add_fig_panel_label(fig, 0.506, 0.984, "b")

    ax_c = fig.add_axes([0.102, 0.080, 0.236, 0.368])
    add_fig_panel_label(fig, 0.008, 0.484, "c")
    ax_d = fig.add_axes([0.458, 0.080, 0.244, 0.368])
    add_fig_panel_label(fig, 0.354, 0.484, "d")
    ax_e = fig.add_axes([0.788, 0.080, 0.194, 0.368])
    add_fig_panel_label(fig, 0.716, 0.484, "e")

    # -------------------------------------------------------------------------
    # Panel a: Relative Crop Yield vs Irrigation ECw + Top Sampling Density KDE
    # -------------------------------------------------------------------------
    da = p2b["dose_response_absolute"]
    lg_c = p2b["logistic_sensitivity"]["continuous_sigmoidal_p_ge_1"]
    model = da["field_continuous_harvest"]
    ecw90_cont = model["ecw_star_90pct_dsm"]
    ec_grid = np.asarray(model["curve_ecw_dsm"])
    bins = [0, 2, 3, 4, 6, 8, 12, 20]
    ya = u[u.metric_std.isin(["grain_yield", "shoot_biomass"]) & u.irrigation_scheduling.eq("continuous_brackish")].copy()
    field = ya[ya.metric_std.eq("grain_yield") & ya.experiment_type.str.lower().eq("field")].copy()
    field["w_obs"] = field.n_replicates.fillna(3).clip(lower=1) / field.groupby("family_id").family_id.transform("size")
    points, curves, bin_rows = [], [], []
    for label, data, fit, color, line in [
        ("Field harvest", field, model, COL_RISK, "-"),
        ("Harvest + biomass", ya, da["continuous"], COL_REF, "--"),
    ]:
        t, b = fit["t_dsm"], fit["b_per_dsm"]
        def predict(x):
            return np.maximum(1 - b * np.maximum(0, x - t), 0.05)
        draws = data.copy()
        draws["plotted_response"] = np.exp(draws.lnRR) * predict(draws.ecw_control_dsm)
        draws["series"] = label
        draws["shown"] = (label == "Field harvest") & draws.ecw_treatment_dsm.le(16) & draws.plotted_response.between(.35, 1.35)
        points.append(draws)
        kde = gaussian_kde(data.ecw_treatment_dsm.clip(0,16))(ec_grid)
        pred = predict(ec_grid)
        curve = pd.DataFrame({"series":label, "ecw_dsm":ec_grid, "relative_output":pred, "dose_density":kde,
                              "ecw90":fit["ecw_star_90pct_dsm"], "threshold_lo":fit["ecw_star_90pct_ci95"][0],
                              "threshold_hi":fit["ecw_star_90pct_ci95"][1]})
        ax_a_top.plot(ec_grid, kde, color=color, lw=.8, ls=line)
        if label == "Field harvest":
            curve["lo"], curve["hi"] = fit["curve_lo"], fit["curve_hi"]
            ax_a.fill_between(ec_grid, curve.lo, curve.hi, color=color, alpha=.14)
            shown = draws[draws.shown]
            ax_a.scatter(shown.ecw_treatment_dsm, shown.plotted_response, c=color, alpha=.11, s=8, lw=0)

            # Scheme B: crop-group and control-salinity adjusted marginal binned geometric means
            draws["log_ry"] = np.log(draws["plotted_response"])
            draws["band"] = pd.cut(draws["ecw_treatment_dsm"], bins).astype(str)
            m_adj = smf.wls(
                "log_ry ~ C(band) + C(crop_group) + ecw_control_dsm",
                data=draws,
                weights=draws["w_obs"],
            ).fit(cov_type="cluster", cov_kwds={"groups": draws["family_id"]})
            ctrl_wmean = float(np.average(draws["ecw_control_dsm"], weights=draws["w_obs"]))
            m_design = m_adj.model.data.design_info

            for band, g in draws.groupby(pd.cut(draws.ecw_treatment_dsm, bins), observed=True):
                w = g.w_obs
                value = np.average(np.log(g.plotted_response), weights=w)
                scores = (w * (np.log(g.plotted_response) - value)).groupby(g.family_id).sum()
                n = len(scores)
                se = np.sqrt(n / (n - 1) * sum(scores ** 2)) / w.sum() if n > 1 else np.nan
                xx = np.average(g.ecw_treatment_dsm, weights=w)
                raw_lo, raw_mid, raw_hi = np.exp([value - 1.96 * se, value, value + 1.96 * se])

                df_b = draws[["crop_group"]].copy()
                df_b["band"] = str(band)
                df_b["ecw_control_dsm"] = ctrl_wmean
                X_b = dmatrix(m_design, df_b, return_type="dataframe")
                x_bar = (X_b.mul(draws["w_obs"].values, axis=0).sum(axis=0) / draws["w_obs"].sum()).values
                mu_adj = float(x_bar @ m_adj.params.values)
                se_adj = float(np.sqrt(x_bar @ m_adj.cov_params().values @ x_bar))
                adj_lo, adj_mid, adj_hi = np.exp([mu_adj - 1.96 * se_adj, mu_adj, mu_adj + 1.96 * se_adj])

                bin_rows.append(dict(
                    bin=str(band), mean_ecw=xx,
                    mean_ry=adj_mid, ci_lo=adj_lo, ci_hi=adj_hi,
                    raw_mean_ry=raw_mid, raw_ci_lo=raw_lo, raw_ci_hi=raw_hi,
                    k=len(g), n_clusters=n, series=label,
                    adjustment="crop_group_and_control_ecw_wls",
                ))
                ax_a.errorbar(xx, adj_mid, yerr=[[adj_mid - adj_lo], [adj_hi - adj_mid]], fmt="o", color=color, ms=3.5, capsize=2, zorder=4)
        curves.append(curve)
        ax_a.plot(ec_grid,pred,color=color,ls=line,lw=1.6,label=f"{label}  {fit['ecw_star_90pct_dsm']:.2f}")

    # Continuous sigmoidal comparator (p = 1.0 lower bound)
    pred_sig = 1.0 / (1.0 + (np.maximum(ec_grid, 1e-4) / float(lg_c["ec50_dsm"])) ** float(lg_c["p_shape"]))
    sig_ci = lg_c.get("ecw_star_90pct_ci95", [np.nan, np.nan])
    curves.append(pd.DataFrame({
        "series": "Continuous sigmoidal",
        "ecw_dsm": ec_grid,
        "relative_output": pred_sig,
        "dose_density": np.nan,
        "ecw90": lg_c["ecw_star_90pct_dsm"],
        "threshold_lo": sig_ci[0],
        "threshold_hi": sig_ci[1],
    }))
    ax_a.plot(ec_grid, pred_sig, color="#4A5568", ls=":", lw=1.25, label=f"Cont. sigmoidal  {float(lg_c['ecw_star_90pct_dsm']) + 1e-6:.2f}", zorder=3)
    ax_a.text(14.37, 0.795, "9 tolerant-crop\nfamilies", ha="center", va="bottom", fontsize=5.1, color="#555555")

    pd.concat(points)[["series","contrast_id","family_id","control_arm_id","metric_name","experiment_type","ecw_control_dsm",
                       "ecw_treatment_dsm","lnRR","w_obs","plotted_response","shown"]].to_csv(SRC_DIR/"Fig2A_observations.csv",index=False)
    pd.concat(curves).to_csv(SRC_DIR/"Fig2A_fitted_curves.csv",index=False)
    ax_a_top.set(xlim=(0,16),xticks=[],yticks=[])
    ax_a_top.spines[["top","right","left"]].set_visible(False)
    ax_a.axhline(.90,color="#555555",ls=":",lw=.7)
    ax_a.axvline(ecw90_cont,color=COL_RISK,ls=":",lw=.7)
    ax_a.text(15.7,.92,"90%",ha="right",fontsize=6)
    ax_a.set(xlim=(0,16),ylim=(.35,1.35),ylabel="Relative production",xlabel=r"Irrigation water salinity, $\mathrm{EC_w}$ (dS m$^{-1}$)")
    ax_a.legend(loc="upper right",frameon=False,fontsize=5.5,title="90% production threshold (dS m$^{-1}$)",title_fontsize=5.5)
    soil_points, soil_curves = [], []
    # -------------------------------------------------------------------------
    # Panel b: Soil EC Change (Log-Ratio) + Sampling KDE + 326-Pair Inset
    # -------------------------------------------------------------------------
    sec_ece = p2b["soil_EC_absolute"]["saturated_paste_ECe_only"]
    sec_15 = p2b["soil_EC_absolute"]["extract_1_5_only"]
    se_all_df = u[u["metric_std"] == "soil_EC"].copy()

    max_kde_b = 0.0
    for meth_k, col_k in [("saturated_paste_ECe", COL_RISK), ("extract_1_5", COL_REF)]:
        vals_ec = se_all_df.loc[se_all_df["soil_ec_method_std"] == meth_k, "ecw_treatment_dsm"].clip(0.0, 16.0).values
        dens_b = gaussian_kde(vals_ec)(ec_grid)
        ax_b_top.fill_between(ec_grid, 0.0, dens_b, facecolor=col_k, alpha=0.22, edgecolor="none")
        ax_b_top.plot(ec_grid, dens_b, color=col_k, lw=0.85)
        max_kde_b = max(max_kde_b, float(dens_b.max()))
    ax_b_top.set_xlim(0.0, 16.0)
    ax_b_top.set_ylim(0.0, max_kde_b * 1.15)
    ax_b_top.set_xticks([])
    ax_b_top.set_yticks([])
    for sp in ("top", "left", "right"):
        ax_b_top.spines[sp].set_visible(False)
    ax_b_top.spines["bottom"].set_linewidth(0.45)
    ax_b_top.spines["bottom"].set_color("#CCCCCC")

    sec_rows = []
    for meth_key, sec_d, col_m, mkr_m, ls_m, lbl_m in [
        ("saturated_paste_ECe", sec_ece, COL_RISK, "^", "-", rf"$\mathrm{{EC_e}}$ paste ($\mathit{{k}}$={sec_ece['k']}, ref {sec_ece['median_ctrl_ecw_dsm']:.2f}; $\mathit{{\beta}}$={sec_ece['power_exponent_b']:.2f})"),
        ("extract_1_5", sec_15, COL_REF, "s", "--", rf"$\mathrm{{EC}}_{{1:5}}$ extract ($\mathit{{k}}$={sec_15['k']}, ref {sec_15['median_ctrl_ecw_dsm']:.2f}; $\mathit{{\beta}}$={sec_15['power_exponent_b']:.2f})"),
    ]:
        b_pow = float(sec_d["power_exponent_b"])
        b_lo, b_hi = [float(v) for v in sec_d["power_exponent_b_ci95"]]
        ec_ref = float(sec_d["median_ctrl_ecw_dsm"])
        ec_grid_b = np.linspace(ec_ref, 16.0, 300)
        ratio_curve = (ec_grid_b / ec_ref) ** b_pow
        ratio_lo = (ec_grid_b / ec_ref) ** b_lo
        ratio_hi = (ec_grid_b / ec_ref) ** b_hi

        se_df = se_all_df[se_all_df["soil_ec_method_std"] == meth_key].copy()
        se_df["lnRR_std"] = se_df["lnRR"].values + b_pow * np.log(np.maximum(se_df["ecw_control_dsm"].values, 0.05) / ec_ref)
        y_obs_b = np.exp(se_df["lnRR_std"].values)
        m_obs_b = (y_obs_b >= 0.73) & (y_obs_b <= 24.0) & (se_df["ecw_treatment_dsm"].values <= 16.0)
        soil_points.append(se_df.assign(plotted_ratio=y_obs_b, shown=m_obs_b, series=meth_key))
        soil_curves.append(pd.DataFrame({"method":meth_key,"ecw_dsm":ec_grid_b,"predicted_ratio":ratio_curve,
                                         "lo":ratio_lo,"hi":ratio_hi,"reference_ecw":ec_ref}))
        ax_b.scatter(
            se_df.loc[m_obs_b, "ecw_treatment_dsm"].values,
            y_obs_b[m_obs_b],
            color=col_m,
            s=np.clip(se_df.loc[m_obs_b, "w_obs"].values * 7.0, 3.0, 12.0),
            alpha=0.10,
            linewidths=0,
            zorder=1.8,
        )

        se_df["ec_bin"] = pd.cut(se_df["ecw_treatment_dsm"], bins=bins)
        m_rows = []
        for bn, grp in se_df.groupby("ec_bin", observed=True):
            if len(grp) < 3:
                continue
            w = grp["w_obs"].values
            vals = grp["lnRR_std"].values
            xm = float(np.sum(w * grp["ecw_treatment_dsm"].values) / np.sum(w))
            ym = float(np.sum(w * vals) / np.sum(w))
            cl = grp["family_id"].values
            uniq = np.unique(cl)
            se_c = math.sqrt((len(uniq) / max(len(uniq) - 1, 1)) * sum(float(np.sum(w[cl == c] * (vals[cl == c] - ym))) ** 2 for c in uniq) / (np.sum(w) ** 2)) if len(uniq) >= 2 else 0.1
            pct_m = (math.exp(ym) - 1.0) * 100.0
            ci_lo_b = (math.exp(ym - 1.96 * se_c) - 1.0) * 100.0
            ci_hi_b = (math.exp(ym + 1.96 * se_c) - 1.0) * 100.0
            m_rows.append({"method_scale": meth_key, "bin": str(bn), "mean_ecw": xm, "pct": pct_m, "ci_lo": ci_lo_b, "ci_hi": ci_hi_b, "k": int(len(grp)), "n_clusters": int(len(uniq))})
        mdf = pd.DataFrame(m_rows)
        sec_rows.extend(m_rows)
        ax_b.fill_between(ec_grid_b, ratio_lo, ratio_hi, color=col_m, alpha=0.14, zorder=2.2)
        ax_b.plot(ec_grid_b, ratio_curve, ls=ls_m, color=col_m, lw=1.65, label=lbl_m, zorder=3.2)
        if not mdf.empty:
            y_rat = 1.0 + mdf["pct"].values / 100.0
            r_lo = 1.0 + mdf["ci_lo"].values / 100.0
            r_hi = 1.0 + mdf["ci_hi"].values / 100.0
            yerr_b = [y_rat - r_lo, r_hi - y_rat]
            ax_b.errorbar(
                mdf["mean_ecw"], y_rat, yerr=yerr_b,
                fmt=mkr_m, color=col_m, markeredgecolor="white", markeredgewidth=0.50,
                markersize=4.1, elinewidth=0.85, capsize=1.8, alpha=0.95, zorder=4.5,
            )

    pd.concat(soil_points)[["series","contrast_id","family_id","control_arm_id","unit","ecw_control_dsm","ecw_treatment_dsm",
                                "lnRR","w_obs","plotted_ratio","shown"]].to_csv(SRC_DIR/"Fig2B_observations.csv",index=False)
    pd.concat(soil_curves).to_csv(SRC_DIR/"Fig2B_fitted_curves.csv",index=False)
    sec_bdf = pd.DataFrame(sec_rows)
    ax_b.set_yscale("log")
    ax_b.grid(axis="y", which="major", color="#EEF1F4", lw=0.45, zorder=0.5)
    ax_b.axhline(1.0, ls="-", color="#BBBBBB", lw=0.5, zorder=1)
    ax_b.axhline(1.5, ls=":", color=COL_RISK, lw=0.65, alpha=0.60, zorder=1.5)
    ax_b.axhline(2.0, ls=":", color=COL_RISK, lw=0.65, alpha=0.60, zorder=1.5)

    ec50_ece = float(sec_ece["ecw_plus50pct_dsm"])
    ec100_ece = float(sec_ece["ecw_plus100pct_dsm"])
    ec50_15 = float(sec_15["ecw_plus50pct_dsm"])
    ax_b.scatter([ec50_ece, ec100_ece], [1.5, 2.0], color=COL_RISK, s=20, zorder=5)
    ax_b.scatter([ec50_15], [1.5], color=COL_REF, s=20, zorder=5)
    ax_b.annotate(rf"$\mathrm{{EC_e}}$ +50%: {ec50_ece:.2f}", xy=(ec50_ece, 1.5), xytext=(0.75, 0.80), fontsize=5.3, color=COL_RISK, fontweight="bold", arrowprops={"arrowstyle": "->", "color": COL_RISK, "lw": 0.55})
    ax_b.annotate(rf"$\mathrm{{EC_e}}$ +100%: {ec100_ece:.2f}", xy=(ec100_ece, 2.0), xytext=(0.65, 3.35), fontsize=5.3, color=COL_RISK, fontweight="bold", arrowprops={"arrowstyle": "->", "color": COL_RISK, "lw": 0.55})
    ax_b.annotate(rf"$\mathrm{{EC_{{1:5}}}}$ +50%: {ec50_15:.2f}", xy=(ec50_15, 1.5), xytext=(5.6, 0.82), fontsize=5.3, color=COL_REF, fontweight="bold", arrowprops={"arrowstyle": "->", "color": COL_REF, "lw": 0.55})
    ax_b.text(15.6, 0.80, "Common-ref +50% ratio: 1.25×", ha="right", va="bottom", fontsize=5.2, color="#555555")

    ax_b.legend(loc="upper left", frameon=False, fontsize=5.3, handlelength=1.5, handletextpad=0.35)
    ax_b.set_xlim(0, 16.0)
    ax_b.set_ylim(0.72, 25.0)
    ax_b.set_yticks([0.8, 1.0, 1.5, 2.0, 3.0, 6.0, 16.0])
    ax_b.set_yticklabels(["−20", "0", "+50", "+100", "+200", "+500", "+1,500"])
    ax_b.yaxis.set_minor_locator(mpl.ticker.NullLocator())
    ax_b.tick_params(labelsize=6.3)
    ax_b.set_xlabel(r"Irrigation water salinity, $\mathrm{EC_w}$ (dS m$^{-1}$)", fontsize=7.4, labelpad=2.2)
    ax_b.set_ylabel("Soil EC change (%, log scale)", fontsize=7.4, labelpad=2.2)

    # Embedded micro-inset: 326 multi-year pre/post soil EC pairs (18 families)
    pp_raw = pd.read_csv(HARMONIZED / "armA_multiyear_soil_ec_pre_post_pairs_current.csv", low_memory=False)
    for c_pp in ["ecw_dsm", "soil_ec_pre_val", "soil_ec_post_val"]:
        pp_raw[c_pp] = pd.to_numeric(pp_raw[c_pp], errors="coerce")
    pp_v = pp_raw.dropna(subset=["ecw_dsm", "soil_ec_pre_val", "soil_ec_post_val"]).copy()
    pp_v = pp_v[(pp_v["soil_ec_pre_val"] > 0) & (pp_v["soil_ec_post_val"] > 0)].copy()
    pp_v["ratio"] = pp_v["soil_ec_post_val"] / pp_v["soil_ec_pre_val"]
    r_lo_pp = pp_v.loc[pp_v["ecw_dsm"] < ecw90_cont, "ratio"].values
    r_hi_pp = pp_v.loc[pp_v["ecw_dsm"] >= ecw90_cont, "ratio"].values

    ax_b_ins = ax_b.inset_axes([0.13, 0.54, 0.54, 0.25])
    matched = next(r for r in p2b["monitoring_checks"] if r["threshold"] == "field_harvest")
    ax_b.text(.78,.83, f"Greater buildup\nin {matched['families_higher']}/{matched['paired_families']} matched\nfamilies",
              ha="center", va="center", transform=ax_b.transAxes, fontsize=5.2, color="#454d52")
    ax_b_ins.set_facecolor((1.0, 1.0, 1.0, 0.88))
    ax_b_ins.set_xscale("log")
    ax_b_ins.set_xlim(0.42, 28.0)
    ax_b_ins.set_ylim(-0.52, 1.55)
    ax_b_ins.axvline(1.0, ls="--", color="#666666", lw=0.65, zorder=1)
    for y_i, arr_r, col_i in [(0, r_lo_pp, COL_REF), (1, r_hi_pp, COL_RISK)]:
        rng_i = np.random.default_rng(20261125 + y_i)
        y_j = y_i - 0.19 + rng_i.uniform(-0.08, 0.08, size=len(arr_r))
        ax_b_ins.scatter(np.clip(arr_r, 0.44, 24.0), y_j, color=col_i, s=3.2, alpha=0.30, linewidths=0, zorder=2)
        q10, q25, q50, q75, q90 = np.percentile(arr_r, [10, 25, 50, 75, 90])
        ax_b_ins.plot([q10, q90], [y_i, y_i], "-", color="#333333", lw=0.75, zorder=3)
        ax_b_ins.add_patch(mpl.patches.Rectangle((q25, y_i - 0.09), q75 - q25, 0.18, facecolor=col_i, edgecolor="#222222", linewidth=0.55, alpha=0.90, zorder=4))
        ax_b_ins.plot([q50, q50], [y_i - 0.09, y_i + 0.09], "-", color="white", lw=1.35, zorder=5)
        pct_inc = float((arr_r > 1.0).mean() * 100.0)
        ax_b_ins.text(q90 * 1.12, y_i + 0.02, f"{q50:.2f}× ({pct_inc:.0f}%↑)", va="center", ha="left", fontsize=5.2, fontweight="bold", color=col_i, zorder=6)
    ax_b_ins.text(0.0, 1.04, rf"Post/pre soil salinity ({pp_v['family_id'].nunique()} fam., $\mathit{{n}}$={len(pp_v)})", transform=ax_b_ins.transAxes, ha="left", va="bottom", fontsize=5.2, color="#222222")
    ax_b_ins.set_xticks([0.5, 1.0, 2.0, 5.0, 10.0])
    ax_b_ins.set_xticklabels(["0.5×", "1×", "2×", "5×", "10×"], fontsize=5.2)
    ax_b_ins.xaxis.set_minor_locator(mpl.ticker.NullLocator())
    ax_b_ins.set_yticks([0, 1])
    ax_b_ins.set_yticklabels([f"<{ecw90_cont:.2f}", f"≥{ecw90_cont:.2f}"], fontsize=5.2)
    ax_b_ins.tick_params(length=2.0, pad=1.5)
    for sp in ("top", "right"):
        ax_b_ins.spines[sp].set_visible(False)
    for sp in ("bottom", "left"):
        ax_b_ins.spines[sp].set_linewidth(0.45)
        ax_b_ins.spines[sp].set_color("#888888")

    # -------------------------------------------------------------------------
    # Panel c: 6 Crop Groups vs FAO-29 ECe90/1.5 (Top-Strip Legend)
    # -------------------------------------------------------------------------
    cg_d = p2b["crop_group_vs_fao29"]["by_crop_group7"]
    cg_list = [("wheat", "Wheat"), ("cotton", "Cotton"), ("vegetables", "Vegetables"), ("fruit", "Fruit"), ("maize", "Maize"), ("oilseed_legume", "Oilseed/legume")]
    yc = np.arange(len(cg_list))[::-1]
    ax_c.grid(axis="x", color="#EEF1F4", lw=0.45, zorder=0.3)
    for yi in yc:
        if yi % 2 == 1:
            ax_c.axhspan(yi - 0.42, yi + 0.42, facecolor="#F6F8FA", zorder=0.4)
    ax_c.axvline(ecw90_cont, ls=":", color=COL_RISK, lw=0.75, alpha=0.72, zorder=1.5)

    crop_rows = []
    for yi, (ck, lbl) in zip(yc, cg_list):
        r = cg_d[ck]
        e90 = r["ecw_star_90pct_dsm"]
        lo, hi = r["ecw_star_90pct_ci95"]
        f_ref = r["fao29_ecw_ref_dsm"]
        k_emp = r["k_emp_ecw90_over_ece90"]
        ax_c.plot([e90, f_ref], [yi, yi], "-", color="#88929E", lw=1.05, zorder=2)
        ax_c.errorbar(
            e90, yi, xerr=[[e90 - lo], [hi - e90]],
            fmt="o", color=COL_REF, markeredgecolor="white", markeredgewidth=0.55,
            markersize=4.7, elinewidth=1.05, capsize=2.0, zorder=4,
        )
        ax_c.scatter([f_ref], [yi], facecolor="white", edgecolor=COL_FAO, s=26, lw=1.15, zorder=5)
        ax_c.text(10.55, yi, f"{k_emp:.2f}", va="center", ha="left", fontsize=5.6, color="#333333")
        crop_rows.append({"crop_group": ck, "ecw90": e90, "ci_lo": lo, "ci_hi": hi, "fao29_ecw_ref": f_ref, "k_emp": k_emp, "k": r["k"], "n_clusters": r["n_clusters"]})
    ax_c.text(10.55, 5.52, r"$\mathit{k}_{\mathrm{emp}}$", va="center", ha="left", fontsize=5.7, color="#333333", fontweight="bold")
    ax_c.set_yticks(yc)
    ax_c.set_yticklabels([x[1] for x in cg_list], fontsize=6.0)
    ax_c.set_xlim(0.0, 12.2)
    ax_c.set_xticks([0, 2, 4, 6, 8, 10])
    ax_c.spines["bottom"].set_bounds(0.0, 10.0)
    ax_c.set_ylim(-0.52, 5.65)
    ax_c.tick_params(axis="x", labelsize=6.3)
    ax_c.set_xlabel(r"Salinity limit, $\mathrm{EC}_{\mathrm{w},90}^*$ (dS m$^{-1}$)", fontsize=7.4, labelpad=2.2)
    c_handles = [
        mpl.lines.Line2D([0], [0], marker="o", color="none", markerfacecolor=COL_REF, markeredgecolor="white", markeredgewidth=0.55, markersize=4.9, label="Empirical"),
        mpl.lines.Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor=COL_FAO, markeredgewidth=1.15, markersize=4.6, label="FAO-29"),
    ]
    ax_c.legend(handles=c_handles, loc="lower left", bbox_to_anchor=(-0.02, 1.01), ncol=2, frameon=False, fontsize=5.3, columnspacing=0.80, handletextpad=0.30, borderaxespad=0.0)

    # -------------------------------------------------------------------------
    # Panel d: FAO-48 Orthogonal Moderator Forest (Untruncated t-CIs + Tier Tags)
    # -------------------------------------------------------------------------
    adj_all = p2b["adjusted_scheduling_and_tech"]
    adj_s_y = adj_all["scheduling_adjusted_cr1"]
    adj_s_e = adj_all["soil_ec_scheduling_adjusted_cr1"]
    adj_f_y = adj_all["technology_fao48_adjusted_cr1"]
    adj_f_e = adj_all["soil_ec_technology_fao48_adjusted_cr1"]
    bry = p2b["bridge_moderator_tests"]["yield_combined"]
    bre = p2b["bridge_moderator_tests"]["soil_EC"]
    mod_items = [
        ("Cyclic", adj_s_y["terms"].get("excess_x_cyclic"), adj_s_e["terms"].get("dECw_x_cyclic")),
        ("Blended", adj_s_y["terms"].get("excess_x_blended"), adj_s_e["terms"].get("dECw_x_blended")),
        ("Surface drip", adj_f_y["terms"].get("excess_x_surface_drip"), adj_f_e["terms"].get("dECw_x_surface_drip")),
        ("Subsurface drip", adj_f_y["terms"].get("excess_x_subsurface_drip"), adj_f_e["terms"].get("dECw_x_subsurface_drip")),
        ("Ca / chemical", bry["amendment_co_applied"]["terms"].get("delta_slope_ca_chemical"), None),
        ("Organic", bry["amendment_co_applied"]["terms"].get("delta_slope_organic"), bre["amendment_co_applied"]["terms"].get("delta_slope_organic")),
        ("Surface mulching", bry["mulching"]["terms"].get("delta_slope_mulched"), bre["mulching"]["terms"].get("delta_slope_mulched")),
        ("Subsurface drainage", bry["subsurface_drainage"]["terms"].get("delta_slope_subdrain"), bre["subsurface_drainage"]["terms"].get("delta_slope_subdrain")),
    ]
    t_crit_y = float(stats.t.ppf(0.975, df=max(adj_s_y["n_clusters"] - 1, 1)))
    t_crit_e = float(stats.t.ppf(0.975, df=max(adj_s_e["n_clusters"] - 1, 1)))
    ym = np.arange(len(mod_items))[::-1]
    ax_d.grid(axis="x", color="#EEF1F4", lw=0.45, zorder=0.3)
    ax_d.axvline(0.0, ls="--", color="#444444", lw=0.75, zorder=2)
    ax_d.axhline(5.5, ls="--", color="#DCDFE4", lw=0.55, zorder=1)
    ax_d.axhline(3.5, ls="--", color="#DCDFE4", lw=0.55, zorder=1)
    mod_rows = []
    for yi, (lbl, ry_t, re_t) in zip(ym, mod_items):
        for tdict, off, col, mk, ep_nm, t_cr in [
            (ry_t, 0.14, COL_REF, "o", "Yield", t_crit_y),
            (re_t, -0.14, COL_RISK, "s", "Soil EC", t_crit_e),
        ]:
            if not tdict:
                if ep_nm == "Soil EC" and lbl == "Ca / chemical":
                    ax_d.text(0.008, yi - 0.14, "No Soil EC", va="center", ha="left", fontsize=5.2, style="italic", color="#7F8C8D")
                continue
            c_v, se_v, p_v = tdict["coef"], tdict["se_cr"], tdict["p_val"]
            sig = p_v < 0.05
            ax_d.errorbar(c_v, yi + off, xerr=t_cr * se_v, fmt=mk, color=col, mfc=col if sig else "white", mec=col, markersize=3.9, elinewidth=0.9, capsize=1.6, zorder=4)
            mod_rows.append({"moderator": lbl, "endpoint": ep_nm, "delta_slope": c_v, "se_cr": se_v, "t_crit": t_cr, "p_val": p_v, "sig_p05": sig})
    ax_d.text(0.146, 7.22, "Scheduling", ha="right", va="center", fontsize=5.2, style="italic", color="#6C7A89")
    ax_d.text(0.146, 5.22, "Delivery", ha="right", va="center", fontsize=5.2, style="italic", color="#6C7A89")
    ax_d.text(0.146, 3.18, "Soil & drainage", ha="right", va="center", fontsize=5.2, style="italic", color="#6C7A89")
    ax_d.text(0.0, 1.015, r"$\leftarrow$ Soil EC (−: less salt)", transform=ax_d.transAxes, ha="left", va="bottom", fontsize=5.2, color=COL_RISK)
    ax_d.text(1.0, 1.015, r"Yield (+: less loss) $\rightarrow$", transform=ax_d.transAxes, ha="right", va="bottom", fontsize=5.2, color=COL_REF)
    ax_d.scatter([], [], color=COL_REF, marker="o", s=16, label="Yield")
    ax_d.scatter([], [], color=COL_RISK, marker="s", s=16, label="Soil EC")
    ax_d.scatter([], [], facecolor="white", edgecolor="#333333", marker="o", s=15, label=r"Open: $\mathit{P}$ $\geq$ 0.05")
    ax_d.set_yticks(ym)
    ax_d.set_yticklabels([x[0] for x in mod_items], fontsize=5.8)
    ax_d.set_xlim(-0.188, 0.152)
    ax_d.set_xticks([-0.15, -0.10, -0.05, 0.0, 0.05, 0.10])
    ax_d.set_xticklabels(["−0.15", "−0.10", "−0.05", "0", "+0.05", "+0.10"], fontsize=6.0)
    ax_d.set_ylim(-0.60, len(mod_items) - 0.35)
    ax_d.set_xlabel(r"Salinity-sensitivity contrast (lnRR dS$^{-1}$ m)", fontsize=7.4, labelpad=2.2)
    ax_d.legend(loc="upper left", frameon=False, fontsize=5.2, handletextpad=0.30, labelspacing=0.22)

    # -------------------------------------------------------------------------
    # Panel e: Predicted Within-Study Soil EC Slope by Depth (Observed Years 1-3)
    # -------------------------------------------------------------------------
    yr_d = p2b["within_study_year_x_depth"]
    pslp = yr_d["predicted_slopes"]
    yrs_x = [1, 2, 3]
    dep_cfg = [
        ("topsoil_0_20", "0–20 cm", COL_DEPTH[0], "o-", -0.11),
        ("rootzone_0_40_60", "0–40/60 cm", COL_DEPTH[1], "s--", 0.0),
        ("subsoil_40_100", "40–100 cm", COL_DEPTH[2], "D:", 0.11),
    ]
    yd_rows = []
    ax_e.grid(axis="y", color="#EEF1F4", lw=0.45, zorder=0.5)
    ax_e.axhspan(-0.044, 0.0, facecolor="#F5F6F8", zorder=0.3)
    for dk, lbl, col, fmt, off in dep_cfg:
        ys_v = [pslp[f"year_{yr}"][dk]["slope"] for yr in yrs_x]
        lo_v = [pslp[f"year_{yr}"][dk]["slope"] - pslp[f"year_{yr}"][dk]["ci95"][0] for yr in yrs_x]
        hi_v = [pslp[f"year_{yr}"][dk]["ci95"][1] - pslp[f"year_{yr}"][dk]["slope"] for yr in yrs_x]
        ax_e.errorbar(
            np.array(yrs_x) + off, ys_v, yerr=[lo_v, hi_v],
            fmt=fmt, color=col, markeredgecolor="white", markeredgewidth=0.45,
            markersize=4.3, lw=1.25, capsize=2.0, label=lbl, zorder=4,
        )
        for yr, sv, lv, hv in zip(yrs_x, ys_v, lo_v, hi_v):
            yd_rows.append({"year": yr, "depth": dk, "slope": sv, "ci_lo": sv - lv, "ci_hi": sv + hv})
    ax_e.axhline(0.0, ls="--", color="#666666", lw=0.65, zorder=1)
    ax_e.set_xticks(yrs_x)
    ax_e.set_xticklabels(["1", "2", "3"], fontsize=6.3)
    ax_e.set_xlim(0.55, 3.45)
    ax_e.set_ylim(-0.044, 0.145)
    ax_e.tick_params(axis="y", labelsize=6.3)
    ax_e.set_ylabel(r"Soil EC sensitivity (lnRR dS$^{-1}$ m)", fontsize=7.4, labelpad=2.2)
    ax_e.set_xlabel("Observation year", fontsize=7.4, labelpad=2.2)
    p_int = yr_d["terms"]["dECw_x_subsoil_x_logYr"]["p_val"]
    ax_e.text(0.04, 0.97, rf"Subsoil $\times$ year {fmt_p(p_int)}", transform=ax_e.transAxes, va="top", ha="left", fontsize=5.45, color="#222222")
    ax_e.legend(loc="lower right", frameon=False, fontsize=5.2, handletextpad=0.35, labelspacing=0.22)

    pd.DataFrame(bin_rows).to_csv(SRC_DIR / "Fig2A_yield_absolute_ecw.csv", index=False)
    sec_bdf.to_csv(SRC_DIR / "Fig2B_soil_ec_absolute_ecw.csv", index=False)
    pd.DataFrame(crop_rows).to_csv(SRC_DIR / "Fig2C_crop_groups_vs_fao29.csv", index=False)
    pd.DataFrame(mod_rows).to_csv(SRC_DIR / "Fig2D_moderator_forest.csv", index=False)
    pd.DataFrame(yd_rows).to_csv(SRC_DIR / "Fig2E_within_study_year_depth.csv", index=False)

    relevant = u[u.metric_std.isin(["grain_yield","shoot_biomass","soil_EC"])]
    relevant[["contrast_id","family_id","control_arm_id","metric_name","crop_group","experiment_type","irrigation_scheduling",
              "irrigation_mode","drainage_regime","mulching_flag","amendment_co_applied","study_year_index","soil_depth_stratum",
              "ecw_control_dsm","ecw_treatment_dsm","mean_control","mean_treatment","unit","lnRR","w_obs"]].to_csv(
              SRC_DIR/"Fig2CDE_eligible_contrasts.csv",index=False)
    save_pub_figure(fig, FIG_DIR / "Fig2_ECw_Thresholds_ArmA.pdf", FIG_DIR / "Fig2_ECw_Thresholds_ArmA.png", "Fig2")


def plot_amendment_strata(sub_df, axes):
    ax_b1, ax_b2, ax_b3, ax_b4 = axes
    COL_EX, COL_PRI = "#C96A1B", "#2B6E8C"
    two_w = sub_df[sub_df["stratum"].str.startswith("sodicity_x_amendment:")].copy()
    two_w["sod"] = two_w["stratum"].str.split("::").str[0].str.replace("sodicity_x_amendment:", "", regex=False)
    two_w["am"] = two_w["stratum"].str.split("::").str[1]
    am_short = {
        "Inorganic_Ca_Acid": "Ca / gypsum",
        "Combined_Inorg_Org": "Combined",
        "Organic_Carbon": "Organic",
        "Engineering_Drain_Drip": "Drain / drip",
        "Biotic_PGPR_Crop": "Biotic",
        "Other_or_Combined": "Multi-practice",
    }
    facet_specs = [
        (ax_b1, "ks_saturated", r"$\mathit{K_s}$", ["Multi-practice", "Ca / gypsum", "Organic"], "log", (-0.55, 2.02), [0, 100, 400], ["0%", "+100%", "+400%"], "bottom_right", "Change (%, lnRR scale)"),
        (ax_b2, "infiltration_rate_all", "Infiltration", ["Ca / gypsum", "Drain / drip", "Organic", "Combined", "Multi-practice"], "log", (-1.40, 3.22), [-50, 0, 200, 1500], ["−50%", "0%", "+200%", "+1,500%"], "top_right", "Change (%, lnRR scale)"),
        (ax_b3, "wue_or_iwue", "WUE / IWUE", ["Combined", "Organic", "Ca / gypsum", "Drain / drip", "Multi-practice", "Biotic"], "pct", (-28.0, 122.0), [0, 40, 80, 120], ["0%", "+40%", "+80%", "+120%"], "mid_right", "Change (%, linear)"),
        (ax_b4, "bulk_density", "Bulk density", ["Combined", "Organic", "Ca / gypsum", "Biotic", "Drain / drip"], "pct", (-20.5, 20.5), [-20, -10, 0, 10, 20], ["−20%", "−10%", "0%", "+10%", "+20%"], "top_right", "Change (%, linear)"),
    ]
    inv_short = {v: k for k, v in am_short.items()}
    mat_rows = []
    for ax_f, ep, f_title, row_lbls, sc_mode, xlim_f, xticks_raw, xtlbls_f, tag_side, xlab_f in facet_specs:
        ax_f.grid(axis="x", color="#EEF1F4", lw=0.45, zorder=0.4)
        ax_f.axvline(0.0, ls="--", color="#555555", lw=0.70, zorder=1.5)
        n_rf = len(row_lbls)
        yf_pos = np.arange(n_rf)[::-1]
        for r_idx, (yi_f, r_short) in enumerate(zip(yf_pos, row_lbls)):
            if r_idx % 2 == 1:
                ax_f.axhspan(yi_f - 0.44, yi_f + 0.44, facecolor="#F7F9FB", zorder=0.2)
            am_key = inv_short[r_short]
            sub_pair = two_w[(two_w["endpoint"] == ep) & (two_w["am"] == am_key)]
            has_both = len(sub_pair) == 2
            for sod_key, col_s, mkr_s, off_val in [
                ("sodic_or_saline_sodic_high_esp", COL_EX, "o", 0.16),
                ("saline_or_unspecified", COL_PRI, "s", -0.16),
            ]:
                m_r = sub_pair[sub_pair["sod"] == sod_key]
                if m_r.empty:
                    continue
                y_cur = yi_f + (off_val if has_both else 0.0)
                pval = float(m_r["p_val"].iloc[0]) if pd.notna(m_r["p_val"].iloc[0]) else 1.0
                sig = pval < 0.05
                kv = int(m_r["k"].iloc[0])
                ncl = int(m_r["n_clusters"].iloc[0])
                if sc_mode == "log":
                    val_x = float(m_r["mu_lnrr"].iloc[0])
                    ci_arr = json.loads(str(m_r["ci95"].iloc[0]))
                else:
                    val_x = float(m_r["pct_change"].iloc[0])
                    ci_arr = json.loads(str(m_r["pct_ci95"].iloc[0]))
                ax_f.plot([ci_arr[0], ci_arr[1]], [y_cur, y_cur], "-", color=col_s, lw=1.05, zorder=3.0)
                ax_f.scatter([val_x], [y_cur], marker=mkr_s, s=16.5, facecolor=col_s if sig else "white", edgecolor=col_s, linewidths=0.85, zorder=4.0)
                mat_rows.append({"endpoint": ep, "amendment": am_key, "sodicity": sod_key, "mu_lnrr": float(m_r["mu_lnrr"].iloc[0]), "pct_change": float(m_r["pct_change"].iloc[0]), "ci_low": float(ci_arr[0]), "ci_high": float(ci_arr[1]), "p_val": pval, "k": kv, "n_clusters": ncl})
        if sc_mode == "log":
            ax_f.set_xticks([math.log(1.0 + p / 100.0) for p in xticks_raw])
        else:
            ax_f.set_xticks(xticks_raw)
        ax_f.set_xticklabels(xtlbls_f, fontsize=5.25)
        ax_f.set_xlim(*xlim_f)
        ax_f.set_ylim(-0.55, n_rf - 0.35)
        ax_f.set_yticks(yf_pos)
        ax_f.set_yticklabels(row_lbls, fontsize=5.25)
        ax_f.tick_params(axis="y", length=2.0, pad=2.0)
        ax_f.tick_params(axis="x", length=2.2, pad=1.5)
        if tag_side == "top_right":
            ax_f.text(0.97, 0.94, f_title, transform=ax_f.transAxes, ha="right", va="top", fontsize=5.4, fontweight="bold", color="#222222")
        elif tag_side == "mid_right":
            ax_f.text(0.97, 0.25, f_title, transform=ax_f.transAxes, ha="right", va="center", fontsize=5.4, fontweight="bold", color="#222222")
        elif tag_side == "bottom_right":
            ax_f.text(0.97, 0.06, f_title, transform=ax_f.transAxes, ha="right", va="bottom", fontsize=5.4, fontweight="bold", color="#222222")
        if xlab_f:
            ax_f.set_xlabel(xlab_f, fontsize=5.8, labelpad=1.6)

    b_handles = [
        mpl.lines.Line2D([0], [0], marker="o", color="none", markerfacecolor=COL_EX, markeredgecolor=COL_EX, markersize=4.3, label="Sodic (ESP ≥ 15%)"),
        mpl.lines.Line2D([0], [0], marker="s", color="none", markerfacecolor=COL_PRI, markeredgecolor=COL_PRI, markersize=4.1, label="Saline / unspecified"),
        mpl.lines.Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor="#333333", markeredgewidth=0.85, markersize=4.1, label=r"Open: $\mathit{P}$ $\geq$ 0.05"),
    ]
    ax_b1.legend(handles=b_handles, loc="lower left", bbox_to_anchor=(-0.06, 1.04), ncol=3, frameon=False, fontsize=5.2, columnspacing=0.95, handletextpad=0.25, borderaxespad=0.0)


    return pd.DataFrame(mat_rows)


def build_fig3() -> None:
    p2a = json.loads((table_input("pillar2_armB_hydraulic_results_current.json")).read_text(encoding="utf-8"))
    sub_df = pd.read_csv(table_input("pillar2_armB_subgroup_summary_current.csv"))
    salt_audit = pd.read_csv(HARMONIZED / "armB_leaching_7clusters_audit_current.csv")

    COL_PRI = "#2B6E8C"
    COL_PI = "#9AC1D4"
    COL_EX = "#C96A1B"
    COL_RISK = "#A61C28"
    COL_TEAL = "#1D7F70"

    fig = plt.figure(figsize=(179 / 25.4, 134 / 25.4))

    # Top row: Unified twin-column Panel a (Overall REML | Soil Clay Moderation)
    ax_a1 = fig.add_axes([0.140, 0.558, 0.448, 0.372])
    ax_a2 = fig.add_axes([0.614, 0.558, 0.372, 0.372])
    add_fig_panel_label(fig, 0.008, 0.980, "a")

    # Bottom row: Three decoupling mechanisms (b: salt transport, c: hyd vs EC, d: ET audit)
    ax_b = fig.add_axes([0.124, 0.088, 0.248, 0.348])
    add_fig_panel_label(fig, 0.008, 0.474, "b")

    ax_c = fig.add_axes([0.454, 0.088, 0.196, 0.310])
    ax_c_top = fig.add_axes([0.454, 0.400, 0.196, 0.036])
    ax_c_rgt = fig.add_axes([0.652, 0.088, 0.032, 0.310])
    add_fig_panel_label(fig, 0.388, 0.474, "c")

    ax_d = fig.add_axes([0.764, 0.088, 0.222, 0.348])
    add_fig_panel_label(fig, 0.696, 0.474, "d")

    # -------------------------------------------------------------------------
    # Panel a (Left): Overall Three-Level REML Forest on lnRR Scale
    # -------------------------------------------------------------------------
    eps_order = [
        ("ks_saturated", "Saturated hydraulic\n" + r"conductivity ($\mathit{K_s}$)"),
        ("infiltration_rate_all", "Infiltration rate"),
        ("wue_or_iwue", "Water-use efficiency\n(WUE / IWUE)"),
        ("bulk_density", "Soil bulk density (BD)"),
        ("direct_salt_export_or_storage_only", "Salt export / storage\n(excl. LF)"),
    ]
    h_map = {r["endpoint"]: r for r in p2a["headline_all_endpoints"]}
    leach_strat = p2a.get("leaching_efficiency_stratified", {})
    if "direct_salt_export_or_storage_only" in leach_strat:
        h_map["direct_salt_export_or_storage_only"] = {**leach_strat["direct_salt_export_or_storage_only"], "endpoint": "direct_salt_export_or_storage_only"}
    e_map = {r["endpoint"]: r for r in p2a["headline_exact_variance_sensitivity"]}
    y_pos = np.arange(len(eps_order))[::-1]  # 4, 3, 2, 1, 0

    for ax_top in (ax_a1, ax_a2):
        ax_top.grid(axis="x", color="#EEF1F4", lw=0.45, zorder=0.4)
        for yi in y_pos:
            if yi % 2 == 1:
                ax_top.axhspan(yi - 0.42, yi + 0.42, facecolor="#F6F8FA", zorder=0.3)
        ax_top.axhline(0.50, ls="--", color="#D0D7DE", lw=0.60, zorder=1.2)
        ax_top.axvline(0.0, ls="--", color="#555555", lw=0.75, zorder=1.5)

    f_rows = []
    for yi, (ep, _) in zip(y_pos, eps_order):
        rh = h_map.get(ep, {})
        re = e_map.get(ep, {})
        mu = rh.get("mu_lnrr")
        ci = rh.get("ci95")
        pi = rh.get("pi95")
        if mu is not None and pi is not None:
            ax_a1.plot([pi[0], pi[1]], [yi, yi], "-", color=COL_PI, lw=1.25, alpha=0.90, zorder=2)
        if mu is not None and ci is not None:
            ax_a1.plot([ci[0], ci[1]], [yi, yi], "-", color=COL_PRI, lw=2.8, solid_capstyle="round", zorder=3)
            ax_a1.scatter([mu], [yi], color=COL_PRI, edgecolor="white", linewidths=0.50, s=30, zorder=4)
        if re and re.get("mu_lnrr") is not None and ep != "direct_salt_export_or_storage_only":
            ax_a1.scatter([re["mu_lnrr"]], [yi - 0.19], color=COL_EX, edgecolor="white", linewidths=0.45, marker="D", s=18, zorder=5)

        pct_val = rh.get("pct_change")
        pct_str = f"{pct_val:+.1f}%".replace("-", "−") if pct_val is not None else "—"
        ax_a1.text(3.66, yi + 0.04, pct_str, va="bottom", ha="left", fontsize=5.3, fontweight="bold", color="#111111")
        if ep == "direct_salt_export_or_storage_only":
            sub_lbl = rf"($\mathit{{k}}$={rh.get('k')}, {rh.get('n_clusters')} cl; $\mathit{{P}}$={rh['p_val']:.3f})"
        else:
            sub_lbl = rf"($\mathit{{k}}$={rh.get('k')}, {rh.get('n_clusters')} cl)"
        ax_a1.text(3.66, yi - 0.04, sub_lbl, va="top", ha="left", fontsize=5.1, color="#555555")
        f_rows.append({"endpoint": ep, "k": rh.get("k"), "n_clusters": rh.get("n_clusters"), "mu_lnrr": mu, "ci95": ci, "pi95": pi, "I2_pct": rh.get("I2_pct"), "p_val": rh.get("p_val"), "exact_mu_lnrr": re.get("mu_lnrr") if ep != "direct_salt_export_or_storage_only" else None})

    lf_only = leach_strat.get("by_physical_category", {}).get("leaching_fraction_water_ratio", {})
    if lf_only.get("mu_lnrr") is not None:
        ax_a1.scatter([lf_only["mu_lnrr"]], [0.0 - 0.22], facecolor="white", edgecolor="#444444", marker="^", s=21, linewidths=0.95, zorder=5)

    pct_ticks = [-75, 0, 200, 2000]
    ax_a1.set_xticks([math.log(1.0 + p / 100.0) for p in pct_ticks])
    ax_a1.set_xticklabels(["−75%", "0%", "+200%", "+2,000%"], fontsize=5.6)
    ax_a1.set_xlim(-1.82, 6.75)
    ax_a1.spines["bottom"].set_bounds(-1.75, 3.55)
    ax_a1.set_ylim(-0.58, 4.55)
    ax_a1.set_yticks(y_pos)
    ax_a1.set_yticklabels([x[1] for x in eps_order], fontsize=5.75)
    ax_a1.set_xlabel("Overall amendment effect (% change, lnRR scale)", fontsize=6.6, labelpad=2.2)
    ax_a1.xaxis.set_label_coords(((-1.75 + 3.55) / 2.0 - (-1.82)) / (6.75 - (-1.82)), -0.115)
    a1_handles = [
        mpl.lines.Line2D([0], [0], color=COL_PRI, lw=2.5, marker="o", markersize=4.3, markerfacecolor=COL_PRI, markeredgecolor="white", markeredgewidth=0.45, label="95% CI (REML)"),
        mpl.lines.Line2D([0], [0], color=COL_PI, lw=1.25, label="95% PI"),
        mpl.lines.Line2D([0], [0], color="none", marker="D", markersize=4.0, markerfacecolor=COL_EX, markeredgecolor="white", markeredgewidth=0.4, label="Exact var."),
        mpl.lines.Line2D([0], [0], color="none", marker="^", markersize=4.3, markerfacecolor="white", markeredgecolor="#444444", markeredgewidth=0.9, label="LF ratio (1 cl)"),
    ]
    ax_a1.legend(handles=a1_handles, loc="lower left", bbox_to_anchor=(-0.02, 1.015), ncol=4, frameon=False, fontsize=5.05, columnspacing=0.55, handletextpad=0.25, borderaxespad=0.0)

    # -------------------------------------------------------------------------
    # Panel a (Right): Soil Clay-Texture Moderation Aligned Row-for-Row
    # -------------------------------------------------------------------------
    clay_df = pd.DataFrame(p2a["h2_clay_moderation"])
    c_eps = [
        ("ks_saturated", 4),
        ("infiltration_rate_all", 3),
        ("wue_or_iwue", 2),
        ("bulk_density", 1),
    ]
    clay_cfg = [
        ("clay:coarse_lt20", "Coarse (<20%)", "#D98A2B", "o", 0.21),
        ("clay:medium_20_35", "Medium (20–35%)", COL_PRI, "s", 0.0),
        ("clay:heavy_clay_ge35", "Heavy (≥35%)", COL_RISK, "D", -0.21),
    ]
    clay_export = []
    for ep, yi in c_eps:
        sub_c = clay_df[clay_df["endpoint"] == ep]
        has_med = not sub_c[sub_c["stratum"] == "clay:medium_20_35"].empty
        for ck, clbl, col_c, mk_c, off_c in clay_cfg:
            row_c = sub_c[sub_c["stratum"] == ck]
            if row_c.empty:
                continue
            y_cur = yi + ((0.14 if off_c > 0 else -0.14) if not has_med else off_c)
            mu_v = float(row_c["mu_lnrr"].iloc[0])
            ci_c = row_c["ci95"].iloc[0]
            if isinstance(ci_c, str):
                ci_c = json.loads(ci_c)
            pv_c = float(row_c["p_val"].iloc[0])
            ax_a2.plot([ci_c[0], ci_c[1]], [y_cur, y_cur], "-", color=col_c, lw=1.25, solid_capstyle="round", zorder=3.0)
            sig_c = pv_c < 0.05
            ax_a2.scatter(
                [mu_v], [y_cur], marker=mk_c, s=23,
                facecolor=col_c if sig_c else "white",
                edgecolor="white" if sig_c else col_c,
                linewidths=0.50 if sig_c else 0.85, zorder=4.0,
            )
            clay_export.append({"endpoint": ep, "stratum": ck, "mu_lnrr": mu_v, "ci95_low": float(ci_c[0]), "ci95_high": float(ci_c[1]), "pct_change": float(row_c["pct_change"].iloc[0]), "p_val": pv_c, "k": int(row_c["k"].iloc[0]), "n_clusters": int(row_c["n_clusters"].iloc[0])})

    ax_a2.text(1.44, 3.71, r"+323% ($\mathit{P}$ = 0.020)", ha="center", va="top", fontsize=5.15, color=COL_RISK, fontweight="bold")
    ax_a2.text(0.44, 4.24, r"+55% ($\mathit{P}$ = 0.007)", ha="left", va="bottom", fontsize=5.15, color="#B26B12")
    ax_a2.text(0.06, 4.00, "Med. (<3 cl)", ha="left", va="center", fontsize=5.05, style="italic", color="#7F8C8D")
    ax_a2.text(2.74, 0.16, r"$\leftarrow$ Salt transport stratified by metric in b", ha="right", va="center", fontsize=5.05, style="italic", color="#4A5A68")
    ax_a2.text(2.74, -0.20, r"$\mathit{K_s}$ vs ESP $\leq$ 100%: $\mathit{P}$ = 0.126 (ED Fig. 6b)", ha="right", va="center", fontsize=5.05, color="#555555")

    c_ticks_pct = [0, 50, 200, 1000]
    ax_a2.set_xticks([math.log(1.0 + p / 100.0) for p in c_ticks_pct])
    ax_a2.set_xticklabels(["0%", "+50%", "+200%", "+1,000%"], fontsize=5.6)
    ax_a2.set_xlim(-0.26, 2.82)
    ax_a2.set_ylim(-0.58, 4.55)
    ax_a2.set_yticks([4, 3, 2, 1])
    ax_a2.set_yticklabels([])
    ax_a2.spines["left"].set_color("#CBD4DC")
    ax_a2.spines["left"].set_linewidth(0.60)
    ax_a2.set_xlabel("Effect by soil clay content (% change, lnRR scale)", fontsize=6.6, labelpad=2.2)
    a2_handles = [
        mpl.lines.Line2D([0], [0], marker="o", color="none", markerfacecolor="#D98A2B", markeredgecolor="#D98A2B", markersize=4.1, label="Coarse (<20%)"),
        mpl.lines.Line2D([0], [0], marker="s", color="none", markerfacecolor=COL_PRI, markeredgecolor=COL_PRI, markersize=3.9, label="Medium (20–35%)"),
        mpl.lines.Line2D([0], [0], marker="D", color="none", markerfacecolor=COL_RISK, markeredgecolor=COL_RISK, markersize=3.8, label="Heavy (≥35% clay)"),
    ]
    ax_a2.legend(handles=a2_handles, loc="lower left", bbox_to_anchor=(-0.02, 1.015), ncol=3, frameon=False, fontsize=5.05, columnspacing=0.60, handletextpad=0.22, borderaxespad=0.0)

    # -------------------------------------------------------------------------
    # Panel b: Stratified Salt Transport Forest + Individual Contrast Strip
    # -------------------------------------------------------------------------
    cats = leach_strat["by_physical_category"]
    salt_obs = salt_audit.copy()
    salt_obs["pct_obs"] = (salt_obs["mean_treatment"].astype(float) / salt_obs["mean_control"].astype(float) - 1.0) * 100.0

    ax_b.grid(axis="x", color="#EEF1F4", lw=0.45, zorder=0.3)
    ax_b.axhspan(1.58, 2.42, facecolor="#F6F8FA", zorder=0.2)
    ax_b.axhspan(-0.52, 0.48, facecolor="#FDF7F0", zorder=0.2)
    ax_b.axvline(0.0, ls="--", color="#555555", lw=0.75, zorder=1.5)
    ax_b.axhline(0.50, ls="--", color="#B8C0C8", lw=0.65, zorder=1.4)

    rng_b = np.random.default_rng(20261126)
    salt_rows = []
    b_ytick_labels = []
    b_specs = [
        ("leaching_fraction_water_ratio", "Drainage / infiltration", "Drainage / infil.\n(1 family, LF)"),
        ("soil_salt_storage_change", "Change in soil salt storage", "Salt storage\n(3 families)"),
        ("salt_removal_per_unit_water", "Salt removal per unit water", "Salt removal\n(3 families)"),
        ("direct", "Direct salt outcomes combined", "Direct combined\n(6 families)"),
    ]
    for y, (key, name, y_disp) in enumerate(b_specs):
        r = leach_strat["direct_salt_export_or_storage_only"] if key == "direct" else cats[key]
        lo, hi = 100.0 * np.expm1(r["ci95"])
        val = float(r["pct_change"])
        color = COL_PRI if key == "direct" else (COL_EX if y == 0 else COL_TEAL)

        if key == "direct":
            obs_vals = salt_obs.loc[salt_obs["physical_category"] != "leaching_fraction_water_ratio", "pct_obs"].values
        else:
            obs_vals = salt_obs.loc[salt_obs["physical_category"] == key, "pct_obs"].values
        y_jit = y - 0.20 + rng_b.uniform(-0.055, 0.055, size=len(obs_vals))
        ax_b.scatter(obs_vals, y_jit, color=color, s=10.0, alpha=0.42, edgecolors="white", linewidths=0.25, zorder=2.5)

        ax_b.plot([lo, hi], [y, y], color=color, lw=2.3, solid_capstyle="round", zorder=3.5)
        mkr_b = "D" if key == "direct" else ("^" if y == 0 else "o")
        sz_b = 30 if key in ("direct", "leaching_fraction_water_ratio") else 27
        ax_b.scatter([val], [y], s=sz_b, marker=mkr_b, color=color, edgecolor="white", linewidths=0.55, zorder=4.5)

        val_str = f"{val:+.1f}%".replace("-", "−")
        if key == "leaching_fraction_water_ratio":
            sub_b = rf"($\mathit{{k}}$={r['k']}, desc.)"
        else:
            sub_b = rf"($\mathit{{k}}$={r['k']}, {fmt_p(r.get('p_val'))})"
        ax_b.text(183.0, y + 0.04, val_str, va="bottom", ha="left", fontsize=5.1, fontweight="bold", color="#111111")
        ax_b.text(183.0, y - 0.04, sub_b, va="top", ha="left", fontsize=4.85, color="#555555")

        b_ytick_labels.append(y_disp)
        salt_rows.append(dict(outcome=name, families=r["n_clusters"], contrasts=r["k"], estimate_pct=val, lower_pct=lo, upper_pct=hi))

    ax_b.set_yticks(range(4))
    ax_b.set_yticklabels(b_ytick_labels, fontsize=5.35)
    ax_b.set_xlim(-52.0, 290.0)
    ax_b.spines["bottom"].set_bounds(-50.0, 178.0)
    ax_b.set_ylim(-0.55, 3.55)
    ax_b.set_xticks([-50, 0, 50, 100, 150])
    ax_b.set_xticklabels(["−50%", "0%", "+50%", "+100%", "+150%"], fontsize=5.45)
    ax_b.set_xlabel("Change from control (%)", fontsize=6.6, labelpad=2.2)
    ax_b.xaxis.set_label_coords(((-50.0 + 178.0) / 2.0 - (-52.0)) / (290.0 - (-52.0)), -0.118)

    b_handles = [
        mpl.lines.Line2D([0], [0], color=COL_PRI, lw=2.2, marker="D", markersize=4.0, markerfacecolor=COL_PRI, markeredgecolor="white", markeredgewidth=0.45, label="REML 95% CI"),
        mpl.lines.Line2D([0], [0], color="none", marker="o", markersize=3.5, markerfacecolor=COL_TEAL, alpha=0.55, markeredgecolor="white", markeredgewidth=0.25, label="Contrasts"),
        mpl.lines.Line2D([0], [0], color=COL_EX, lw=2.2, marker="^", markersize=4.2, markerfacecolor=COL_EX, markeredgecolor="white", markeredgewidth=0.45, label="LF (1 cl)"),
    ]
    ax_b.legend(handles=b_handles, loc="lower left", bbox_to_anchor=(-0.03, 1.015), ncol=3, frameon=False, fontsize=4.85, columnspacing=0.42, handletextpad=0.20, borderaxespad=0.0)

    # -------------------------------------------------------------------------
    # Panel c: Marginal-Density Dual-Pathway Quadrant Scatter (Hydraulic vs EC)
    # -------------------------------------------------------------------------
    dp = p2a["dual_pathway_hydraulic_vs_ec"]
    fp_d = pd.DataFrame(dp["family_points"])
    ov = dp["overall"]
    pct_ret = round(((fp_d["hyd_lnrr"] > 0) & (fp_d["soil_EC_lnrr"] >= 0)).mean() * 100.0)
    pct_des = round(((fp_d["hyd_lnrr"] > 0) & (fp_d["soil_EC_lnrr"] < 0)).mean() * 100.0)

    x_d_grid = np.linspace(-1.42, 2.35, 200)
    dens_hyd = gaussian_kde(fp_d["hyd_lnrr"].values)(x_d_grid)
    ax_c_top.fill_between(x_d_grid, 0.0, dens_hyd, facecolor=COL_PRI, alpha=0.22, edgecolor="none")
    ax_c_top.plot(x_d_grid, dens_hyd, color=COL_PRI, lw=0.85)
    ax_c_top.axvline(0.0, ls=":", color="#777777", lw=0.65)
    ax_c_top.set_xlim(-1.42, 2.35)
    ax_c_top.set_ylim(0.0, float(dens_hyd.max()) * 1.15)
    ax_c_top.set_xticks([])
    ax_c_top.set_yticks([])
    for sp in ("top", "left", "right"):
        ax_c_top.spines[sp].set_visible(False)
    ax_c_top.spines["bottom"].set_linewidth(0.45)
    ax_c_top.spines["bottom"].set_color("#CCCCCC")

    y_d_grid = np.linspace(-2.25, 1.38, 200)
    dens_ec = gaussian_kde(fp_d["soil_EC_lnrr"].values)(y_d_grid)
    ax_c_rgt.fill_betweenx(y_d_grid, 0.0, dens_ec, facecolor=COL_RISK, alpha=0.20, edgecolor="none")
    ax_c_rgt.plot(dens_ec, y_d_grid, color=COL_RISK, lw=0.85)
    ax_c_rgt.axhline(0.0, ls=":", color="#777777", lw=0.65)
    ax_c_rgt.set_ylim(-2.25, 1.38)
    ax_c_rgt.set_xlim(0.0, float(dens_ec.max()) * 1.15)
    ax_c_rgt.set_xticks([])
    ax_c_rgt.set_yticks([])
    for sp in ("top", "right", "bottom"):
        ax_c_rgt.spines[sp].set_visible(False)
    ax_c_rgt.spines["left"].set_linewidth(0.45)
    ax_c_rgt.spines["left"].set_color("#CCCCCC")

    ax_c.grid(True, color="#EEF1F4", lw=0.45, zorder=0.5)
    ax_c.fill_between([0.0, 2.35], -2.25, 0.0, facecolor="#EBF6F4", alpha=0.52, zorder=0.2)
    ax_c.fill_between([0.0, 2.35], 0.0, 1.38, facecolor="#FCECE8", alpha=0.52, zorder=0.2)
    x_ols = np.array([-1.25, 2.15])
    ax_c.plot(x_ols, ov["ols_intercept"] + ov["ols_slope"] * x_ols, "--", color="#666666", lw=0.85, alpha=0.72, zorder=2.5)
    for reg, nm, col, mk in [
        ("sodic_or_saline_sodic_high_esp", "Sodic", COL_EX, "o"),
        ("saline_or_unspecified", "Saline / unspecified", COL_PRI, "s"),
    ]:
        sub_p = fp_d[fp_d["sodicity_regime"] == reg]
        ax_c.scatter(sub_p["hyd_lnrr"], sub_p["soil_EC_lnrr"], color=col, marker=mk, s=24, alpha=0.88, edgecolors="white", lw=0.40, label=nm, zorder=4)
    ax_c.axhline(0.0, ls=":", color="#777777", lw=0.65, zorder=1.5)
    ax_c.axvline(0.0, ls=":", color="#777777", lw=0.65, zorder=1.5)
    ov_rho = f"{ov['spearman_r']:+.3f}".replace("-", "−")
    ax_c.text(
        0.04, 0.96,
        rf"All $\mathit{{n}}$ = {ov['n_families']}: $\mathit{{\rho}}$ = {ov_rho}" + "\n" + rf"{fmt_p(ov['spearman_p'])}",
        transform=ax_c.transAxes, ha="left", va="top", fontsize=5.1, color="#222222",
    )
    ax_c.text(0.96, 0.95, f"EC rose ({pct_ret}%)", transform=ax_c.transAxes, ha="right", va="top", fontsize=5.05, style="italic", color=COL_RISK, alpha=0.85)
    ax_c.text(0.96, 0.22, f"EC fell ({pct_des}%)", transform=ax_c.transAxes, ha="right", va="bottom", fontsize=5.05, style="italic", color=COL_TEAL, alpha=0.90)
    ax_c.set_xlim(-1.42, 2.35)
    ax_c.set_ylim(-2.25, 1.38)
    ax_c.tick_params(labelsize=5.6)
    ax_c.set_xlabel("Infiltration change (lnRR)", fontsize=6.6, labelpad=2.2)
    ax_c.set_ylabel("Soil EC change (lnRR)", fontsize=6.6, labelpad=2.0)
    ax_c.legend(loc="lower left", frameon=False, fontsize=5.05, handletextpad=0.22, labelspacing=0.18)

    # -------------------------------------------------------------------------
    # Panel d: Clean Water-Use Ratio Rainclouds (Verified ET vs Supply IWUE)
    # -------------------------------------------------------------------------
    aud_sum = json.loads((table_input("armB_wue_et_64families_audit_summary.json")).read_text(encoding="utf-8"))
    audit_df = pd.read_csv(HARMONIZED / "armB_wue_et_64families_audit_current.csv")

    r_ab = np.exp(audit_df[audit_df["category"].isin(["A_direct_reported_ET", "B_inferred_Y_over_WUE_ET"])].groupby("family_id")["ln_et_ratio"].mean().dropna())
    r_a = np.exp(audit_df[audit_df["category"] == "A_direct_reported_ET"].groupby("family_id")["ln_et_ratio"].mean().dropna())
    r_c = np.exp(audit_df[audit_df["category"] == "C_IWUE_irrigation_supply"].groupby("family_id")["ln_irr_ratio"].mean().dropna())

    et_specs = [
        (2, r"Verified ET" + "\n" + rf"($\mathit{{n}}$ = {len(r_ab)})", f"Verified ET (n={len(r_ab)})", r_ab, aud_sum["verified_consumptive_ET_combined_AB"]["ratio_median"], aud_sum["verified_consumptive_ET_combined_AB"]["ratio_median_ci95"], COL_TEAL, True),
        (1, "Direct ET\n" + rf"($\mathit{{n}}$ = {len(r_a)})", f"Direct ET (n={len(r_a)})", r_a, aud_sum["A_direct_reported_ET"]["ratio_median"], aud_sum["A_direct_reported_ET"]["ratio_median_ci95"], "#2A8C7D", False),
        (0, r"Supply IWUE" + "\n" + rf"($\mathit{{n}}$ = {len(r_c)})", f"Supply IWUE (n={len(r_c)})", r_c, aud_sum["C_IWUE_irrigation_supply_ratio"]["ratio_median"], aud_sum["C_IWUE_irrigation_supply_ratio"]["ratio_median_ci95"], "#7D8C9B", False),
    ]

    ax_d.set_xscale("log")
    ax_d.set_xlim(0.38, 2.38)
    ax_d.set_ylim(-0.55, 2.74)
    ax_d.fill_between([0.38, 1.00], 0.50, 2.74, facecolor="#EBF6F4", alpha=0.52, zorder=0.2)
    ax_d.fill_between([1.00, 2.38], 0.50, 2.74, facecolor="#FCECE8", alpha=0.52, zorder=0.2)
    ax_d.fill_between([0.38, 2.38], -0.55, 0.50, facecolor="#F4F5F7", alpha=0.75, zorder=0.2)
    ax_d.axvline(1.0, ls="--", color="#222222", lw=0.95, zorder=1.5)
    ax_d.axhline(0.5, ls="--", color="#AAAAAA", lw=0.65, zorder=1.0)

    x_log_grid = np.linspace(math.log(0.40), math.log(1.80), 180)
    x_e_grid = np.exp(x_log_grid)
    et_export_rows = []
    for yi, ylbl, clean_lbl, arr_r, med_v, ci_v, col_r, is_main in et_specs:
        d_r = gaussian_kde(np.log(arr_r))(x_log_grid)
        d_scaled = d_r / max(float(d_r.max()), 1e-8) * 0.34
        ax_d.fill_between(x_e_grid, yi + 0.04, yi + 0.04 + d_scaled, facecolor=col_r, alpha=0.24, edgecolor=col_r, lw=0.80, zorder=2.2)
        rng_e = np.random.default_rng(20261125 + yi)
        y_jit = yi - 0.16 + rng_e.uniform(-0.07, 0.07, size=len(arr_r))
        ax_d.scatter(np.clip(arr_r, 0.40, 1.85), y_jit, color=col_r, s=13.0, alpha=0.70, edgecolors="white", linewidths=0.30, zorder=3.0)
        ax_d.plot([ci_v[0], ci_v[1]], [yi, yi], "-", color=col_r, lw=2.3, solid_capstyle="round", zorder=4.0)
        ax_d.scatter([med_v], [yi], marker="D", s=24, facecolor=col_r, edgecolor="white", linewidths=0.55, zorder=5.0)
        txt_r = f"{med_v:.3f} ({ci_v[0]:.2f}–{ci_v[1]:.2f})"
        ax_d.text(1.13, yi + 0.14, txt_r, ha="left", va="center", fontsize=5.1, color=col_r, fontweight="bold" if is_main else "normal", zorder=5.5)
        for family, rv in arr_r.items():
            et_export_rows.append({"family_id": family, "stratum": clean_lbl, "ratio": float(rv), "median": float(med_v), "ci95_low": float(ci_v[0]), "ci95_high": float(ci_v[1])})

    ax_d.text(0.03, 0.96, r"$\leftarrow$ ET$\downarrow$ (saving)", transform=ax_d.transAxes, ha="left", va="top", fontsize=5.05, color=COL_TEAL, fontweight="bold")
    ax_d.text(0.97, 0.96, r"ET$\uparrow$ (depletion) $\rightarrow$", transform=ax_d.transAxes, ha="right", va="top", fontsize=5.05, color=COL_RISK, fontweight="bold")
    ax_d.text(0.40, 0.43, r"Supply ratio ($\mathit{I}_{\mathrm{T}}/\mathit{I}_{\mathrm{C}}$)", ha="left", va="top", fontsize=5.05, style="italic", color="#55606E")
    ax_d.set_yticks([x[0] for x in et_specs])
    ax_d.set_yticklabels([x[1] for x in et_specs], fontsize=5.35)
    ax_d.set_xticks([0.5, 0.75, 1.0, 1.25, 1.5])
    ax_d.set_xticklabels(["0.5", "0.75", "1.0", "1.25", "1.5"], fontsize=5.6)
    ax_d.xaxis.set_minor_locator(mpl.ticker.NullLocator())
    ax_d.set_xlabel("Water-use ratio (T / C)", fontsize=6.6, labelpad=2.2)
    et_pts = pd.DataFrame(et_export_rows)

    forest = pd.DataFrame(f_rows)
    for interval in ("ci95","pi95"):
        forest[[interval+"_low",interval+"_high"]] = pd.DataFrame(forest[interval].tolist(),index=forest.index)
    forest.drop(columns=["ci95","pi95"]).to_csv(SRC_DIR / "Fig3A_reml_forest.csv", index=False)
    effects = pd.read_csv(grid_input("pillar2_armB_merged_current.csv"),low_memory=False)
    physical = effects.metric_bucket.str.startswith("infiltration_rate_",na=False) | effects.metric_bucket.isin(
        ["ks_saturated","bulk_density","wue_or_iwue","leaching_efficiency_or_fraction"])
    effects.loc[physical].to_csv(SRC_DIR/"Fig3A_amendment_contrasts.csv",index=False)
    salt_audit[["doi","contrast_index","treatment_label","control_label","mean_treatment","mean_control","unit",
                "physical_category","soil_depth_cm","sampling_period"]].to_csv(SRC_DIR/"Fig3B_salt_observations.csv",index=False)
    audit_df.drop(columns=["evidence_quote","local_pdf_path"],errors="ignore").to_csv(SRC_DIR/"Fig3D_denominator_records.csv",index=False)
    pd.DataFrame(salt_rows).to_csv(SRC_DIR / "Fig3B_salt_transport.csv", index=False)
    pd.DataFrame(clay_export).to_csv(SRC_DIR / "Fig3A_clay_moderation.csv", index=False)
    fp_d.to_csv(SRC_DIR / "Fig3C_dual_pathway_hyd_ec.csv", index=False)
    et_pts.to_csv(SRC_DIR / "Fig3D_comeasured_yield_wue_et.csv", index=False)

    save_pub_figure(fig, FIG_DIR / "Fig3_Hydraulic_Recovery_ArmB.pdf", FIG_DIR / "Fig3_Hydraulic_Recovery_ArmB.png", "Fig3")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Nature Water Main Figures 1-3 and Supplementary Fig. 1.")
    parser.add_argument("--only", choices=["fig1", "figs1", "fig2", "fig3", "all"], default="all")
    args = parser.parse_args()

    setup_mpl_style()
    print(f"Generating revised Figures (target={args.only})...")
    if args.only in ("all", "fig1", "figs1"):
        build_figS1_schematic()
    if args.only in ("all", "fig1"):
        build_fig1()
    if args.only in ("all", "fig2"):
        build_fig2()
    if args.only in ("all", "fig3"):
        build_fig3()
    print("Done.")


if __name__ == "__main__":
    prepare_outputs()
    main()
