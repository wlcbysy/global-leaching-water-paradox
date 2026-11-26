#!/usr/bin/env python3
"""
Stage 2: Observed District- & Basin-Scale Paired Water-Salt Budget Validation
and Two-Boundary (Field vs Basin) Salt Ledger.

Integrates three external empirical benchmarks strictly traceable to verified primary sources:
  1. La Violada Irrigation District (Ebro Basin, Spain):
     Barros, Isidoro & Aragues (2012) Agric. Ecosyst. Environ. 151:44-52
     (1982-1984, 1995-1998, and 2006-2008 / 2005-2008 three-decade paired balances;
     82% soil gypsum dissolution contribution & EC-to-TDS bias test on gypsiferous water).
  2. Hetao Irrigation District (Inner Mongolia, China):
     Ao et al. (2024) Agronomy 14(5):953 (doi:10.3390/agronomy14050953)
     (seasonal growing-period 35.8% vs autumn-leaching 55.4% salt discharge,
     Yellow River diversion ~2.80 Mt/yr vs outlet export 0.945-1.00 Mt/yr,
     and internal transfer to non-farmland saline sinks & shallow groundwater).
  3. USGS Upper Colorado River Basin (DOI 10.5066/P1K8G7JS; 205 WRTDS stations, 1986-2020):
     1991-2019 station-level annual TDS fluxes, baseflow salt-load shares,
     Lees Ferry (09380000) basin outlet benchmark vs Pillar 1/3 modeled UCRB,
     and explicit proximity overlap audit against Thorslund & van Vliet (2020) input stations.

Outputs:
  - tables/observed_district_basin_water_salt_ledger_current.csv
  - tables/observed_district_basin_water_salt_ledger_current.json
"""

from __future__ import annotations

from config import EXTERNAL_RAW_DATA_DIR, OUT_TABLES, table_input, grid_input, prepare_outputs

import json
import zipfile
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy import stats

USGS_DIR = EXTERNAL_RAW_DATA_DIR / "USGS_UCRB_SaltLoad"
THOR_RIVERS_CSV = EXTERNAL_RAW_DATA_DIR / "Thorslund_vanVliet_2020/Rivers_summary.csv"
P3_GRID = grid_input("pillar3_grid05_zoning_current.parquet")
P3_BASIN = table_input("pillar3_basin_zoning_current.csv")
P3_SUMMARY = table_input("pillar3_global_summary_current.json")

OUT_CSV = OUT_TABLES / "observed_district_basin_water_salt_ledger_current.csv"
OUT_JSON = OUT_TABLES / "observed_district_basin_water_salt_ledger_current.json"


def compute_usgs_ucrb_benchmark() -> Dict[str, Any]:
    atts = pd.read_csv(USGS_DIR / "ucol_sparrow_cal_station_atts.csv").dropna(subset=["station_id", "Latitude", "Longitude"]).copy()
    thor = pd.read_csv(THOR_RIVERS_CSV, low_memory=False)
    t_lat = thor["Lat"].values
    t_lon = thor["Lon"].values

    min_deg = []
    for _, r in atts.iterrows():
        d = np.sqrt((t_lat - r["Latitude"]) ** 2 + (t_lon - r["Longitude"]) ** 2)
        min_deg.append(float(np.nanmin(d)))
    atts["min_dist_thorslund_deg"] = min_deg
    atts["overlap_thorslund_005deg"] = atts["min_dist_thorslund_deg"] < 0.05
    atts["overlap_thorslund_001deg"] = atts["min_dist_thorslund_deg"] < 0.01

    zpath = USGS_DIR / "ucol_sparrow_cal_data.zip"
    st_rows: List[Dict[str, Any]] = []
    outlet_series: List[Dict[str, Any]] = []
    with zipfile.ZipFile(zpath) as zf:
        for nm in zf.namelist():
            with zf.open(nm) as f:
                df = pd.read_csv(f, usecols=["Date", "GenConc", "GenFlux", "bf_load", "bf_corrected_load", "bf_volume", "station_id"])
            df["Year"] = pd.to_datetime(df["Date"]).dt.year
            sub = df[(df["Year"] >= 1991) & (df["Year"] <= 2019)].copy()
            if sub.empty:
                continue
            sid = str(sub["station_id"].iloc[0])
            # GenFlux (kg/day) and GenConc (mg/L = 1e-3 kg/m3) give daily discharge km3/day = GenFlux / (GenConc * 1e6)
            sub["q_km3_day"] = np.where(
                (sub["GenConc"] > 0) & np.isfinite(sub["GenFlux"]) & np.isfinite(sub["GenConc"]),
                sub["GenFlux"] / (sub["GenConc"] * 1e-3) / 1e9,
                np.nan,
            )
            ann = sub.groupby("Year", as_index=False).agg(
                flux_mt=("GenFlux", lambda s: float(s.sum() / 1e9)),
                bf_load_mt=("bf_load", lambda s: float(s.sum() / 1e9)),
                bf_corr_mt=("bf_corrected_load", lambda s: float(s.sum() / 1e9)),
                tds_mean_mgl=("GenConc", "mean"),
                q_obs_km3=("q_km3_day", "sum"),
                q_bf_km3=("bf_volume", lambda s: float(s.sum() * 86400.0 / 1e9)),
            )
            if sid == "WQ_Combined_20733845":  # Lees Ferry outlet (Q_gage USGS-09380000)
                outlet_series = ann.to_dict(orient="records")
            slp, _, _, p_tr, _ = stats.linregress(ann["Year"].values, ann["flux_mt"].values) if len(ann) >= 10 else (np.nan, np.nan, np.nan, np.nan, np.nan)
            q_sum = float(ann["q_obs_km3"].sum())
            fw_tds = float(ann["flux_mt"].sum() / q_sum * 1000.0) if q_sum > 0 else np.nan
            st_rows.append({
                "station_id": sid,
                "n_years_1991_2019": int(len(ann)),
                "mean_q_km3_yr": round(float(ann["q_obs_km3"].mean()), 3),
                "sd_q_km3_yr": round(float(ann["q_obs_km3"].std()), 3),
                "mean_flux_mt_yr": round(float(ann["flux_mt"].mean()), 4),
                "sd_flux_mt_yr": round(float(ann["flux_mt"].std()), 4),
                "cv_flux_pct": round(float(ann["flux_mt"].std() / max(ann["flux_mt"].mean(), 1e-9) * 100.0), 1),
                "mean_bf_load_mt_yr": round(float(ann["bf_load_mt"].mean()), 4),
                "bf_share_pct": round(float(ann["bf_load_mt"].mean() / max(ann["flux_mt"].mean(), 1e-9) * 100.0), 1),
                "mean_tds_mgl": round(float(ann["tds_mean_mgl"].mean()), 1),
                "fw_tds_mgl": round(fw_tds, 1) if np.isfinite(fw_tds) else None,
                "trend_mt_yr2": round(float(slp), 5) if np.isfinite(slp) else None,
                "trend_p": round(float(p_tr), 4) if np.isfinite(p_tr) else None,
            })

    st_df = pd.DataFrame(st_rows).merge(
        atts[["station_id", "Latitude", "Longitude", "Q_gage", "overlap_thorslund_005deg", "overlap_thorslund_001deg"]],
        on="station_id", how="left",
    ).sort_values("mean_flux_mt_yr", ascending=False)

    # Key basin outlet and major tributaries (including non-overlapping vs overlapping counts)
    key_q_gages = ["USGS-09380000", "USGS-09180500", "USGS-09163500", "USGS-09315000", "USGS-09095500", "USGS-09261000"]
    key_stations = []
    for qg in key_q_gages:
        sub_k = st_df[st_df["Q_gage"] == qg]
        if not sub_k.empty:
            key_stations.append(sub_k.iloc[0].to_dict())

    non_overlap = st_df[~st_df["overlap_thorslund_005deg"].fillna(True)]
    return {
        "doi": "10.5066/P1K8G7JS",
        "period_matched": "1991-2019",
        "total_stations": int(len(st_df)),
        "thorslund_overlap_audit": {
            "within_005deg_count": int(atts["overlap_thorslund_005deg"].sum()),
            "within_001deg_count": int(atts["overlap_thorslund_001deg"].sum()),
            "disjoint_gt_005deg_count": int((~atts["overlap_thorslund_005deg"]).sum()),
            "disjoint_stations_top5": non_overlap.head(5)[["station_id", "Q_gage", "mean_flux_mt_yr", "mean_tds_mgl"]].to_dict(orient="records"),
            "circularity_note": "197 of 205 USGS stations lie within 0.05 deg (107 within 0.01 deg) of Thorslund & van Vliet (2020) river stations used for ambient ECw gap-filling; therefore USGS WRTDS river salt loads serve as an order-of-magnitude and interannual-variability benchmark of total catchment salt export (geogenic + irrigation), not an independent station-disjoint ambient-EC test or sole validator of cropland salt debt.",
        },
        "lees_ferry_outlet_09380000": st_df[st_df["Q_gage"] == "USGS-09380000"].iloc[0].to_dict(),
        "key_tributary_stations": key_stations,
        "lees_ferry_annual_series": outlet_series,
    }


def build_la_violada_and_hetao_records(grid: pd.DataFrame, basins: pd.DataFrame, usgs: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Build structured Two-Boundary (Field vs Basin) water-salt ledger entries from
    verified primary literature alongside matched Pillar 1/3 model cells/basins.
    """
    # Matched Ebro basin (5470) & La Violada 0.5-deg cell (42.25, -0.75 / 41.75, -0.75)
    ebro_b = basins[basins["aware_basin_idx"] == 5470].iloc[0]
    ebro_cell = grid[(grid["lat"] == 41.75) & (grid["lon"] == -0.75)].iloc[0]

    # Matched Hetao cells (40.25-41.25 N, 106.75-109.25 E in basin 5820) & Yellow River basin (5820)
    ht_cells = grid[(grid["lat"].between(40.25, 41.5)) & (grid["lon"].between(106.5, 109.5)) & (grid["aware_basin_idx"] == 5820)]
    yr_b = basins[basins["aware_basin_idx"] == 5820].iloc[0]

    # Matched Upper Colorado basin (6360)
    ucrb_b = basins[basins["aware_basin_idx"] == 6360].iloc[0]
    lf = usgs["lees_ferry_outlet_09380000"]

    # Helper for La Violada hm3 conversion from mm * ha * 1e-5
    # 1980s: area=3913 ha, I=1045 mm -> 40.89 hm3; Q=942 mm -> 36.86 hm3; Q*=819 mm -> 32.05 hm3; SLQ=75.35 kt, SLI=7.771 kt, SLCS=2.153 kt, SLOI=1.180 kt, SLP=0.955 kt
    # 1990s: area=3731 ha, I=1002 mm -> 37.38 hm3; Q=1160 mm -> 43.28 hm3; Q*=1071 mm -> 39.96 hm3; SLQ=74.03 kt, SLI=8.746 kt
    # 2000s: area=3213 ha, I=550 mm -> 17.67 hm3; Q=520 mm -> 16.71 hm3; Q*=453 mm -> 14.55 hm3; SLQ=32.667 kt, SLI=4.463 kt
    records = [
        {
            "case_id": "LaViolada_1982_1984_HY",
            "site_or_basin": "La Violada Irrigation District, Ebro Basin (Spain)",
            "scale_type": "District_Paired_Observed",
            "period_and_season": "1982-1984 Hydrological Year (flood irrigation)",
            "source_citation": "Barros, Isidoro & Aragues (2012) Agric. Ecosyst. Environ. 151:44-52, Tables 1-3",
            "irrigated_area_ha": 3913.0,
            "irrigation_diversion_km3_yr": round(3913.0 * 1045.0 * 1e-8, 5),
            "irrigation_depth_mm_yr": 1045.0,
            "drainage_outflow_km3_yr": round(3913.0 * 942.0 * 1e-8, 5),
            "drainage_depth_mm_yr": 942.0,
            "drainage_fraction_pct": 58.0,  # IS DRF reported in Sec 3.1
            "irrigation_tds_mgl": 190.0,
            "irrigation_ec_dsm": 0.32,
            "drainage_flow_weighted_tds_mgl": 2055.0,
            "drainage_sample_ec_dsm": 2.32,
            "ec_to_tds_fao29_underestimate_pct": round((640.0 * 2.32 / 2055.0 - 1.0) * 100.0, 1),
            "field_salt_input_kt_yr": 7.771,
            "basin_external_salt_input_kt_yr": round(7.771 + 2.153 + 1.180 + 0.955, 3),  # SLI + SLCS + SLOI + SLP
            "basin_outlet_salt_export_kt_yr": 75.350,
            "geogenic_gypsum_salt_share_pct": 82.0,
            "geogenic_salt_export_est_kt_yr": round(75.350 - (7.771 + 2.153 + 1.180 + 0.955), 3),
            "internal_transfer_or_storage_kt_yr": 0.0,
            "drainage_destination": "La Violada Gully D-14 -> Gallego/Ebro River",
            "model_matched_area_ha": round(float(ebro_cell["salt_irrig_ha"]), 1),
            "model_matched_sin_pre_kt_yr": round(float(ebro_cell["s_in_pre_mt_yr"] * 1e3), 1),
            "note": "Closed impervious Tertiary clay base; 82% (79-84%) of exported salt is geogenic gypsum dissolution; 640*EC underestimates flow-weighted TDS by 27.7%.",
        },
        {
            "case_id": "LaViolada_1995_1998_HY",
            "site_or_basin": "La Violada Irrigation District, Ebro Basin (Spain)",
            "scale_type": "District_Paired_Observed",
            "period_and_season": "1995-1998 Hydrological Year (pre-modernization flood)",
            "source_citation": "Barros, Isidoro & Aragues (2012) Agric. Ecosyst. Environ. 151:44-52, Tables 1-3",
            "irrigated_area_ha": 3731.0,
            "irrigation_diversion_km3_yr": round(3731.0 * 1002.0 * 1e-8, 5),
            "irrigation_depth_mm_yr": 1002.0,
            "drainage_outflow_km3_yr": round(3731.0 * 1160.0 * 1e-8, 5),
            "drainage_depth_mm_yr": 1160.0,
            "drainage_fraction_pct": 72.0,
            "irrigation_tds_mgl": 234.0,
            "irrigation_ec_dsm": 0.37,
            "drainage_flow_weighted_tds_mgl": 1708.0,
            "drainage_sample_ec_dsm": 1.86,
            "ec_to_tds_fao29_underestimate_pct": round((640.0 * 1.86 / 1708.0 - 1.0) * 100.0, 1),
            "field_salt_input_kt_yr": 8.746,
            "basin_external_salt_input_kt_yr": round(8.746 + 2.402 + 1.197 + 0.953, 3),
            "basin_outlet_salt_export_kt_yr": 74.030,
            "geogenic_gypsum_salt_share_pct": 82.0,
            "geogenic_salt_export_est_kt_yr": round(74.030 - (8.746 + 2.402 + 1.197 + 0.953), 3),
            "internal_transfer_or_storage_kt_yr": 0.0,
            "drainage_destination": "La Violada Gully D-14 -> Gallego/Ebro River",
            "model_matched_area_ha": round(float(ebro_cell["salt_irrig_ha"]), 1),
            "model_matched_sin_pre_kt_yr": round(float(ebro_cell["s_in_pre_mt_yr"] * 1e3), 1),
            "note": "Canal seepage adds 2.40 kt/yr (18% of external salt inputs); 640*EC underestimates gypsiferous drainage TDS by 30.3%.",
        },
        {
            "case_id": "LaViolada_2006_2008_HY",
            "site_or_basin": "La Violada Irrigation District, Ebro Basin (Spain)",
            "scale_type": "District_Paired_Observed",
            "period_and_season": "2006-2008 Hydrological Year (improved flood/modernization)",
            "source_citation": "Barros, Isidoro & Aragues (2012) Agric. Ecosyst. Environ. 151:44-52, Tables 1-3",
            "irrigated_area_ha": 3213.0,
            "irrigation_diversion_km3_yr": round(3213.0 * 550.0 * 1e-8, 5),
            "irrigation_depth_mm_yr": 550.0,
            "drainage_outflow_km3_yr": round(3213.0 * 520.0 * 1e-8, 5),
            "drainage_depth_mm_yr": 520.0,
            "drainage_fraction_pct": 50.0,
            "irrigation_tds_mgl": 245.0,
            "irrigation_ec_dsm": 0.42,
            "drainage_flow_weighted_tds_mgl": 1953.0,
            "drainage_sample_ec_dsm": 2.25,
            "ec_to_tds_fao29_underestimate_pct": round((640.0 * 2.25 / 1953.0 - 1.0) * 100.0, 1),
            "field_salt_input_kt_yr": 4.463,
            "basin_external_salt_input_kt_yr": round(4.463 + 0.942 + 0.851 + 0.598, 3),
            "basin_outlet_salt_export_kt_yr": 32.667,
            "geogenic_gypsum_salt_share_pct": 82.0,
            "geogenic_salt_export_est_kt_yr": round(32.667 - (4.463 + 0.942 + 0.851 + 0.598), 3),
            "internal_transfer_or_storage_kt_yr": 0.0,
            "drainage_destination": "La Violada Gully D-14 -> Gallego/Ebro River (Ebro basin model S_in=1798.8 kt/yr)",
            "model_matched_area_ha": round(float(ebro_b["salt_irrig_ha"]), 1),
            "model_matched_sin_pre_kt_yr": round(float(ebro_b["s_in_pre_mt_yr"] * 1e3), 1),
            "note": "Halving irrigation/drainage depth halved gypsum-driven salt export (19.5 -> 9.9 Mg/ha/yr IRF), proving return-flow volume reduction cuts geogenic salt export.",
        },
        {
            "case_id": "Hetao_HID_1990_2013_Annual_and_Seasonal",
            "site_or_basin": "Hetao Irrigation District, Yellow River (China)",
            "scale_type": "District_Seasonal_Observed_and_SWAT_Salt",
            "period_and_season": "1990-2013 avg (35.8% growing-season vs 55.4% autumn-leaching discharge)",
            "source_citation": "Ao et al. (2024) Agronomy 14(5):953 (doi:10.3390/agronomy14050953)",
            "irrigated_area_ha": 508000.0,  # 7.62 million mu / 15 mu/ha
            "irrigation_diversion_km3_yr": 5.0,  # ~5.0 billion m3/yr (~35-40% in autumn watering)
            "irrigation_depth_mm_yr": round(5.0 * 1e9 / (508000.0 * 1e4) * 1e3, 1),
            "drainage_outflow_km3_yr": round(14.87 * 86400 * 365 / 1e9, 3),  # avg 14.14-15.59 m3/s at Red Ohb outlet
            "drainage_depth_mm_yr": round((14.87 * 86400 * 365 / 1e9) * 1e9 / (508000.0 * 1e4) * 1e3, 1),
            "drainage_fraction_pct": round((14.87 * 86400 * 365 / 1e9) / 5.0 * 100.0, 1),
            "irrigation_tds_mgl": 597.0,
            "irrigation_ec_dsm": None,
            "drainage_flow_weighted_tds_mgl": 1873.0,
            "drainage_sample_ec_dsm": None,
            "ec_to_tds_fao29_underestimate_pct": None,
            "field_salt_input_kt_yr": 2800.0,  # 280 x 10^4 t/yr Yellow River diversion (Sec 2.1); 1280 kt/yr SWAT-Salt field irrigation
            "basin_external_salt_input_kt_yr": 2800.0,
            "basin_outlet_salt_export_kt_yr": 945.0,  # 94.5 x 10^4 t/yr (1990-2013 Sec 2.1); 1000 kt/yr (1996-2013 Sec 3.1.3)
            "geogenic_gypsum_salt_share_pct": 0.0,
            "geogenic_salt_export_est_kt_yr": 0.0,
            "internal_transfer_or_storage_kt_yr": round(2800.0 - 945.0, 1),  # 1,855 kt/yr (1.86 Mt/yr) retained in shallow GW & non-farmland saline sinks
            "drainage_destination": "Red Ohb Drain -> Wuliangsuhai Lake (35.8% growing, 55.4% autumn); ~66% retained internally in shallow GW & bare saline land",
            "model_matched_area_ha": round(float(ht_cells["salt_irrig_ha"].sum()), 1),
            "model_matched_sin_pre_kt_yr": round(float(ht_cells["s_in_pre_mt_yr"].sum() * 1e3), 1),
            "note": "Seasonal audit shows 55.4% of salt export occurs during autumn leaching (35-40% of Yellow River diversion). Cropland groundwater TDS=2946 mg/L vs bare non-farmland EC=18.98 dS/m & Mollic Solonchaks TDS=5015 mg/L. Raw continuous monthly series require institutional data request (published seasonal/annual totals used).",
        },
        {
            "case_id": "USGS_UpperColorado_LeesFerry_1991_2019",
            "site_or_basin": "Upper Colorado River Basin at Lees Ferry (09380000, USA)",
            "scale_type": "Basin_Outlet_WRTDS_Observed",
            "period_and_season": "1991-2019 annual (29 yr; 205 UCRB stations)",
            "source_citation": "McDonnell, Longley & Wise (2025) USGS Data Release (doi:10.5066/P1K8G7JS)",
            "irrigated_area_ha": round(float(ucrb_b["salt_irrig_ha"]), 1),
            "irrigation_diversion_km3_yr": round(float(ucrb_b["atotuse_km3_yr"]), 3),
            "irrigation_depth_mm_yr": None,
            "drainage_outflow_km3_yr": round(float(lf["mean_q_km3_yr"]), 3),
            "drainage_outflow_sd_km3_yr": round(float(lf["sd_q_km3_yr"]), 3),
            "drainage_depth_mm_yr": None,
            "drainage_fraction_pct": None,
            "irrigation_tds_mgl": None,
            "irrigation_ec_dsm": None,
            "drainage_flow_weighted_tds_mgl": lf["fw_tds_mgl"],
            "drainage_arithmetic_tds_mgl": lf["mean_tds_mgl"],
            "drainage_sample_ec_dsm": None,
            "ec_to_tds_fao29_underestimate_pct": None,
            "field_salt_input_kt_yr": round(float(ucrb_b["s_in_pre_mt_yr"] * 1e3), 1),
            "basin_external_salt_input_kt_yr": None,
            "basin_outlet_salt_export_kt_yr": round(float(lf["mean_flux_mt_yr"] * 1e3), 1),
            "geogenic_gypsum_salt_share_pct": None,
            "geogenic_salt_export_est_kt_yr": None,
            "internal_transfer_or_storage_kt_yr": None,
            "drainage_destination": "Colorado River at Lees Ferry (total catchment export = geogenic weathering + irrigation return flows)",
            "model_matched_area_ha": round(float(ucrb_b["salt_irrig_ha"]), 1),
            "model_matched_sin_pre_kt_yr": round(float(ucrb_b["s_in_pre_mt_yr"] * 1e3), 1),
            "model_matched_basin6360_qtot_km3_yr": round(float(ucrb_b["qtot_km3_yr"]), 3),
            "note": f"Lees Ferry (09380000) observed 1991-2019 Q = {lf['mean_q_km3_yr']:.2f} +/- {lf['sd_q_km3_yr']:.2f} km3/yr, FW-TDS = {lf['fw_tds_mgl']:.1f} mg/L (arith. {lf['mean_tds_mgl']:.1f} mg/L), flux = {lf['mean_flux_mt_yr']:.2f} +/- {lf['sd_flux_mt_yr']:.2f} Mt/yr; AWARE basin #6360 spans full Upper+Lower Colorado (qtot = {ucrb_b['qtot_km3_yr']:.2f} km3/yr, cropland S_in_pre = {ucrb_b['s_in_pre_mt_yr']:.2f} Mt/yr). Tributaries show 97-98% baseflow load; 197/205 USGS sites within 0.05 deg of Thorslund.",
        },
    ]
    return records


def main() -> None:
    usgs = compute_usgs_ucrb_benchmark()
    grid = pd.read_parquet(P3_GRID)
    basins = pd.read_csv(P3_BASIN, low_memory=False)
    p3 = json.loads(P3_SUMMARY.read_text(encoding="utf-8"))

    records = build_la_violada_and_hetao_records(grid, basins, usgs)
    df_out = pd.DataFrame(records)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUT_CSV, index=False)

    # Global Two-Boundary Summary from Pillar 3
    gs = p3["global_salt_affected_croplands_savings"]
    sb_maj = basins[basins["salt_irrig_ha"] >= 10000.0]
    sb_net_in = np.maximum(sb_maj["s_in_pre_mt_yr"] + sb_maj["delta_s_basin_new_mt_yr"], 0.0)
    basin_debt_cap = round(float((sb_maj["u_post"] * sb_net_in).sum()), 2)
    basin_debt_act = round(float((sb_maj["u_post_actual"] * sb_net_in).sum()), 2)
    global_ledger = {
        "field_boundary_baseline_salt_input_all_salt_cells_mt_yr": round(float(basins["s_in_pre_mt_yr"].sum()), 2),
        "field_boundary_baseline_salt_input_376_major_basins_mt_yr": round(float(sb_maj["s_in_pre_mt_yr"].sum()), 2),
        "field_boundary_added_salt_load_mt_yr": gs["delta_s_salt_import_mt_yr"],
        "delta_s_field_recycled_gross_mt_yr": gs.get("delta_s_field_recycled_gross_mt_yr"),
        "delta_s_basin_gw_gross_mt_yr": gs.get("delta_s_basin_gw_gross_mt_yr"),
        "delta_s_fresh_displaced_mt_yr": gs.get("delta_s_fresh_displaced_mt_yr"),
        "field_boundary_recycled_return_flow_salt_mt_yr": gs.get("delta_s_field_recycled_mt_yr"),
        "basin_boundary_new_shallow_gw_salt_mt_yr": gs.get("delta_s_basin_new_mt_yr"),
        "field_salt_debt_pre_mt_yr": gs["salt_debt_pre_mt_yr"],
        "field_salt_debt_post_equal_prod_mt_yr": gs["salt_debt_post_mt_yr"],
        "field_salt_debt_post_actual_et_mt_yr": gs.get("salt_debt_post_actual_mt_yr"),
        "basin_boundary_salt_debt_post_equal_prod_mt_yr": basin_debt_cap,
        "basin_boundary_salt_debt_post_actual_et_mt_yr": basin_debt_act,
        "interpretation": (
            f"At the field boundary, reusing internal return flows reapplies dissolved salts onto crops "
            f"(+{gs.get('delta_s_field_recycled_gross_mt_yr'):.2f} Mt/yr recycled gross) and lifts field salt-export obligation from "
            f"{gs['salt_debt_pre_mt_yr']:.2f} to {gs['salt_debt_post_mt_yr']:.2f} Mt/yr (equal-production capped) or "
            f"{gs.get('salt_debt_post_actual_mt_yr'):.2f} Mt/yr (actual-management ET); at the basin boundary, internal return-flow "
            f"recycling is an internal transfer rather than net new basin salt creation, whereas shallow brackish groundwater "
            f"mobilization and freshwater displacement sum to net basin-boundary shift ({gs.get('delta_s_basin_new_mt_yr'):.2f} Mt/yr)."
        ),
    }

    payload = {
        "script": "validate_observed_water_salt_budgets.py",
        "global_two_boundary_salt_ledger": global_ledger,
        "usgs_ucrb_benchmark": usgs,
        "district_and_basin_records": records,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved Two-Boundary Observed Ledger -> {OUT_CSV} & {OUT_JSON}")


if __name__ == "__main__":
    prepare_outputs()
    main()
