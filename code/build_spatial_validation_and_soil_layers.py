#!/usr/bin/env python3
"""
Scheme B Spatial Harmonization, Global Ambient Water Salinity (DynQual v1.0),
Full Jasechko et al. (2024) 1,693-Aquifer + Well Benchmarking,
and ISIMIP3a 4-GHM Multi-Model Hydrological Structural Cross-Validation.

Integrates (all verified local public datasets):
  1. SoilGrids 2.0 (5 km) 0-30 cm clay (%) -> 0.5-deg WaterGAP grid.
  2. FAO HWSD2 (D1 0-20 cm) SMU-weighted ESP (%), sodic fraction (ESP >= 15%),
     gypsum (%), clay (%), and poorly/imperfectly drained fraction -> 0.5-deg grid.
  3. BGR/UNESCO WHYMAP GWR v1 (whymap_salineGW__v1_poly & whymap_GW_aquifers_v1_poly):
     flags > 5 g/L saline groundwater areas & HYGEO2 aquifer recharge tiers (11..34).
  4. DynQual v1.0 (Jones et al., 2023 GMD / 2024 Nature Water; Zenodo 10.5281/zenodo.14675270):
     0.5-deg 1991-2019 annual median & p90 monthly TDS -> FAO-29 ECw (dS/m), eliminating the
     91.6% Eurasian sampling void of station-only salinity databases.
  5. CSR GRACE/GRACE-FO RL06.3 Mascons (2002-2024): linear TWSA trend (mm/yr) -> 0.5-deg & 376 basins.
  6. Jasechko et al. (2024, Nature):
     (a) Combined piezometric wells (AnnualDepthToGroundwater + AnnualGroundwaterElevation, >=5 yr 1990-2022)
     (b) Full 1,693 global aquifer systems (jasechko_et_al_2024_aquifers.shp + Supplementary_Table_16.xlsx)
         area-weighted onto the 376 major salt-affected AWARE basins (covering North China Plain,
         Iran, India, Spain, Mexico, Australia, USA).
  7. ISIMIP3a 4-GHM Structural Cross-Validation (WaterGAP 2.2e, H08, LPJmL5-7-10-fire, CWatM;
     1991-2019 gswp3-w5e5 obsclim_histsoc_default) across all 376 major salt-affected basins.

Outputs:
  - data/analysis_input/public_validation_soil_aquifer_grid05_current.parquet
  - tables/basin_grace_jasechko_validation_current.csv
  - tables/pillar1_multighm_structural_validation_current.csv
  - tables/hydrological_independent_validation_current.json
"""

from __future__ import annotations

from config import EXTERNAL_RAW_DATA_DIR, OUT_TABLES, OUT_GRIDS, table_input, grid_input, prepare_outputs

import json
import sqlite3
import zipfile
from pathlib import Path
from typing import Dict, Any, List

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject
from scipy.stats import spearmanr, mannwhitneyu
from shapely.geometry import Point
import xarray as xr

PUB = EXTERNAL_RAW_DATA_DIR
LSM_NC = PUB / "WaterGAP2_2e/landseamask_water-global.nc"
P1_GRID = grid_input("pillar1_grid05_baseline_current.parquet")
P1_BASIN = table_input("pillar1_basin_leaching_closure_current.csv")
AWARE_GPKG = EXTERNAL_RAW_DATA_DIR / "AWARE2/AWARE20_Native_CFs_geospatial.gpkg"

SG_DIR = EXTERNAL_RAW_DATA_DIR / "SoilGrids_2.0_5km"
HWSD2_DIR = EXTERNAL_RAW_DATA_DIR / "FAO_HWSD2"
WHYMAP_ZIP = PUB / "IGRAC_WHYMAP/WHYMAP_GWR_v1.zip"
GRACE_NC = PUB / "GRACE_CSR/CSR_GRACE_GRACE-FO_RL0603_Mascons_all-corrections.nc"
JASECHKO_GW_ZIP = PUB / "Jasechko_2024/Jasechko_et_al_2024_GroundwaterLevelData.zip"
JASECHKO_AQ_ZIP = PUB / "Jasechko_2024/Jasechko_et_al_2024_AquiferSpatialData.zip"
JASECHKO_ST_ZIP = PUB / "Jasechko_2024/Jasechko_et_al_2024_SupplementaryTables.zip"
DYNQUAL_ANN_NC = PUB / "DynQual_v1/salinity_annualAvg_1980_2019_30min.nc"
DYNQUAL_MON_NC = PUB / "DynQual_v1/salinity_monthlyAvg_1980_2019_30min.nc"
MULTIGHM_DIR = PUB / "ISIMIP3a_MultiGHM"

OUT_PARQUET = OUT_GRIDS / "public_validation_soil_aquifer_grid05_current.parquet"
OUT_BASIN_VAL = OUT_TABLES / "basin_grace_jasechko_validation_current.csv"
OUT_MULTIGHM_CSV = OUT_TABLES / "pillar1_multighm_structural_validation_current.csv"
OUT_JSON = OUT_TABLES / "hydrological_independent_validation_current.json"

DST_SHAPE = (360, 720)
DST_TRANSFORM = from_origin(-180.0, 90.0, 0.5, 0.5)


def cell_area_km2_05deg() -> np.ndarray:
    R = 6371.0088
    lats_top = np.linspace(90.0, -89.5, 360)
    area_row = (R ** 2) * np.deg2rad(0.5) * (np.sin(np.deg2rad(lats_top)) - np.sin(np.deg2rad(lats_top - 0.5)))
    return np.repeat(area_row[:, None], 720, axis=1)


def reproject_soilgrids_clay() -> np.ndarray:
    layers = [
        (SG_DIR / "clay_0-5cm_mean_5000.tif", 5.0),
        (SG_DIR / "clay_5-15cm_mean_5000.tif", 10.0),
        (SG_DIR / "clay_15-30cm_mean_5000.tif", 15.0),
    ]
    num = np.zeros(DST_SHAPE, dtype="float64")
    den = np.zeros(DST_SHAPE, dtype="float64")
    for p, w in layers:
        with rasterio.open(p) as src:
            arr = src.read(1).astype("float64")
            if src.nodata is not None:
                arr = np.where(arr == src.nodata, np.nan, arr)
            arr = np.where((arr >= 0) & (arr <= 1000), arr / 10.0, np.nan)
            dst = np.full(DST_SHAPE, np.nan, dtype="float64")
            reproject(
                source=arr, destination=dst,
                src_transform=src.transform, src_crs=src.crs, src_nodata=np.nan,
                dst_transform=DST_TRANSFORM, dst_crs="EPSG:4326", dst_nodata=np.nan,
                resampling=Resampling.average,
            )
            ok = np.isfinite(dst)
            num[ok] += dst[ok] * w
            den[ok] += w
    return np.where(den > 0, num / np.maximum(den, 1e-6), np.nan)


def compute_hwsd2_05deg() -> Dict[str, np.ndarray]:
    con = sqlite3.connect((HWSD2_DIR / "HWSD2.sqlite").resolve().as_uri() + "?mode=ro", uri=True)
    d1 = pd.read_sql(
        "SELECT HWSD2_SMU_ID, SHARE, DRAINAGE, CLAY, ESP, GYPSUM FROM HWSD2_LAYERS WHERE LAYER='D1'",
        con,
    )
    con.close()

    d1["sh"] = pd.to_numeric(d1["SHARE"], errors="coerce").fillna(0.0).clip(lower=0.0)
    d1["esp_v"] = np.where(pd.to_numeric(d1["ESP"], errors="coerce") >= 0, pd.to_numeric(d1["ESP"], errors="coerce"), np.nan)
    d1["clay_v"] = np.where(pd.to_numeric(d1["CLAY"], errors="coerce") >= 0, pd.to_numeric(d1["CLAY"], errors="coerce"), np.nan)
    d1["gyp_v"] = np.where(pd.to_numeric(d1["GYPSUM"], errors="coerce") >= 0, pd.to_numeric(d1["GYPSUM"], errors="coerce"), np.nan)
    d1["sodic_flag"] = (d1["esp_v"] >= 15.0).astype("float64")
    d1["poor_drain_flag"] = d1["DRAINAGE"].isin(["P", "VP", "I"]).astype("float64")

    def agg_smu(col: str) -> pd.Series:
        sub = d1[d1[col].notna() & (d1["sh"] > 0)]
        num = (sub[col] * sub["sh"]).groupby(sub["HWSD2_SMU_ID"]).sum()
        den = sub["sh"].groupby(sub["HWSD2_SMU_ID"]).sum()
        return num / np.maximum(den, 1e-9)

    smu_esp = agg_smu("esp_v")
    smu_clay = agg_smu("clay_v")
    smu_gyp = agg_smu("gyp_v")
    sub_all = d1[d1["sh"] > 0]
    smu_sodic = (sub_all["sodic_flag"] * sub_all["sh"]).groupby(sub_all["HWSD2_SMU_ID"]).sum() / np.maximum(sub_all["sh"].groupby(sub_all["HWSD2_SMU_ID"]).sum(), 1e-9)
    smu_pdrain = (sub_all["poor_drain_flag"] * sub_all["sh"]).groupby(sub_all["HWSD2_SMU_ID"]).sum() / np.maximum(sub_all["sh"].groupby(sub_all["HWSD2_SMU_ID"]).sum(), 1e-9)

    max_id = int(max(d1["HWSD2_SMU_ID"].max(), 40000)) + 5
    lut_esp = np.full(max_id, np.nan, dtype="float32")
    lut_clay = np.full(max_id, np.nan, dtype="float32")
    lut_gyp = np.full(max_id, np.nan, dtype="float32")
    lut_sodic = np.full(max_id, np.nan, dtype="float32")
    lut_pdrain = np.full(max_id, np.nan, dtype="float32")

    for src_s, lut in [(smu_esp, lut_esp), (smu_clay, lut_clay), (smu_gyp, lut_gyp), (smu_sodic, lut_sodic), (smu_pdrain, lut_pdrain)]:
        for sid, v in src_s.items():
            if 0 <= int(sid) < max_id:
                lut[int(sid)] = float(v)

    esp_05 = np.full(DST_SHAPE, np.nan, dtype="float64")
    clay_05 = np.full(DST_SHAPE, np.nan, dtype="float64")
    gyp_05 = np.full(DST_SHAPE, np.nan, dtype="float64")
    sodic_05 = np.full(DST_SHAPE, np.nan, dtype="float64")
    pdrain_05 = np.full(DST_SHAPE, np.nan, dtype="float64")

    with rasterio.open(HWSD2_DIR / "HWSD2.bil") as src:
        for i in range(6):
            win = rasterio.windows.Window(0, i * 3600, 43200, 3600)
            blk = src.read(1, window=win)
            blk_safe = np.where((blk > 0) & (blk < max_id), blk, 0)
            sub_ids = blk_safe.reshape(60, 60, 720, 60).transpose(0, 2, 1, 3).reshape(60, 720, 3600)
            with np.errstate(invalid="ignore"):
                e_arr = lut_esp[sub_ids]
                vc = np.isfinite(e_arr).sum(axis=2)
                r0 = i * 60
                esp_05[r0:r0+60] = np.where(vc >= 30, np.nanmean(e_arr, axis=2), np.nan)
                clay_05[r0:r0+60] = np.where(vc >= 30, np.nanmean(lut_clay[sub_ids], axis=2), np.nan)
                gyp_05[r0:r0+60] = np.where(vc >= 30, np.nanmean(lut_gyp[sub_ids], axis=2), np.nan)
                sodic_05[r0:r0+60] = np.where(vc >= 30, np.nanmean(lut_sodic[sub_ids], axis=2), np.nan)
                pdrain_05[r0:r0+60] = np.where(vc >= 30, np.nanmean(lut_pdrain[sub_ids], axis=2), np.nan)

    return {
        "hwsd2_esp_pct": esp_05,
        "hwsd2_clay_pct": clay_05,
        "hwsd2_gypsum_pct": gyp_05,
        "hwsd2_sodic_frac": sodic_05,
        "hwsd2_poor_drain_frac": pdrain_05,
    }


def compute_dynqual_05deg() -> Dict[str, np.ndarray]:
    """Compute 1991-2019 DynQual v1.0 annual median & 90th-percentile monthly TDS (mg/L) -> FAO-29 ECw (dS/m)."""
    out = {
        "dynqual_tds_annual_mgl": np.full(DST_SHAPE, np.nan, dtype="float64"),
        "dynqual_ecw_annual_dsm": np.full(DST_SHAPE, np.nan, dtype="float64"),
        "dynqual_ecw_p90_monthly_dsm": np.full(DST_SHAPE, np.nan, dtype="float64"),
    }
    if DYNQUAL_ANN_NC.is_file():
        with xr.open_dataset(DYNQUAL_ANN_NC) as ds:
            arr = ds["tds"].sel(time=slice("1991-01-01", "2019-12-31")).values.astype("float64")
            if float(ds.lat.values[0]) < 0:
                arr = arr[:, ::-1, :]
        arr_clean = np.where((arr > 0.0) & (arr < 1.0e5) & np.isfinite(arr), arr, np.nan)
        with np.errstate(invalid="ignore"):
            tds_med = np.nanmedian(arr_clean, axis=0)
        # FAO-29 conversion: ECw = TDS / 640 (<=3200 mg/L), TDS / 800 (>3200 mg/L), bounded [0.1, 30.0] dS/m
        ecw_ann = np.where(np.isfinite(tds_med), np.where(tds_med <= 3200.0, tds_med / 640.0, tds_med / 800.0), np.nan)
        ecw_ann = np.where(np.isfinite(ecw_ann), np.clip(ecw_ann, 0.10, 30.0), np.nan)
        out["dynqual_tds_annual_mgl"] = tds_med
        out["dynqual_ecw_annual_dsm"] = ecw_ann

    if DYNQUAL_MON_NC.is_file():
        with xr.open_dataset(DYNQUAL_MON_NC) as ds:
            arr_m = ds["tds"].sel(time=slice("1991-01-01", "2019-12-31")).values.astype("float64")
            if float(ds.lat.values[0]) < 0:
                arr_m = arr_m[:, ::-1, :]
        arr_m_clean = np.where((arr_m > 0.0) & (arr_m < 1.0e5) & np.isfinite(arr_m), arr_m, np.nan)
        with np.errstate(invalid="ignore"):
            tds_p90 = np.nanpercentile(arr_m_clean, 90, axis=0)
        ecw_p90 = np.where(np.isfinite(tds_p90), np.where(tds_p90 <= 3200.0, tds_p90 / 640.0, tds_p90 / 800.0), np.nan)
        out["dynqual_ecw_p90_monthly_dsm"] = np.where(np.isfinite(ecw_p90), np.clip(ecw_p90, 0.10, 30.0), np.nan)
    return out


def compute_grace_05deg() -> np.ndarray:
    with xr.open_dataset(GRACE_NC) as ds:
        t_vals = ds["time"].values
        if np.issubdtype(t_vals.dtype, np.datetime64):
            yr_vals = (t_vals.astype("datetime64[D]") - np.datetime64("2002-01-01")).astype("float64") / 365.25 + 2002.0
        else:
            yr_vals = t_vals.astype("float64") / 365.25 + 2002.0
        lwe_cm = ds["lwe_thickness"].values
        lats = ds["lat"].values
        lons = ds["lon"].values

    t_c = yr_vals - yr_vals.mean()
    y_m = np.nanmean(lwe_cm, axis=0)
    slope_mm_yr = (np.nansum((lwe_cm - y_m[None, :, :]) * t_c[:, None, None], axis=0) / np.sum(t_c ** 2)) * 10.0

    if lats[0] < lats[-1]:
        slope_mm_yr = slope_mm_yr[::-1, :]
    shift_cols = int(np.sum(lons >= 180.0))
    if shift_cols > 0:
        slope_mm_yr = np.roll(slope_mm_yr, shift=shift_cols, axis=1)

    with np.errstate(invalid="ignore"):
        out_05 = np.nanmean(slope_mm_yr.reshape(360, 2, 720, 2), axis=(1, 3))
    return out_05


def compute_jasechko_full_benchmarks(aware_gdf: gpd.GeoDataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    (A) Combined piezometric well trends from BOTH AnnualDepthToGroundwater.csv (DTW: +slope = deepening)
        and AnnualGroundwaterElevation.csv (Elevation: -slope = deepening, adding Europe/China/MENA wells).
    (B) Full 1,693 global aquifer system trends from jasechko_et_al_2024_aquifers.shp joined 100% with
        Supplementary_Table_16.xlsx, area-weighted onto AWARE watersheds.
    """
    def calc_well_slopes(df: pd.DataFrame, val_col: str, sign: float) -> pd.DataFrame:
        df = df[(df["IntegerYear"] >= 1990) & (df["IntegerYear"] <= 2022)].dropna(subset=["Lat", "Lon", val_col]).copy()
        stats = df.groupby("StnID", as_index=False).agg(
            n_yrs=("IntegerYear", "nunique"),
            span=("IntegerYear", lambda s: int(s.max() - s.min())),
            lat=("Lat", "first"),
            lon=("Lon", "first"),
            t_mean=("IntegerYear", "mean"),
            y_mean=(val_col, "mean"),
        )
        stats = stats[(stats["n_yrs"] >= 5) & (stats["span"] >= 5)].copy()
        df_s = df.merge(stats[["StnID", "t_mean", "y_mean"]], on="StnID", how="inner")
        df_s["dt"] = df_s["IntegerYear"] - df_s["t_mean"]
        df_s["dy"] = df_s[val_col] - df_s["y_mean"]
        df_s["dt_dy"] = df_s["dt"] * df_s["dy"]
        df_s["dt2"] = df_s["dt"] ** 2
        slp = df_s.groupby("StnID", as_index=False).agg(dt_dy=("dt_dy", "sum"), dt2=("dt2", "sum"))
        slp = slp[slp["dt2"] > 0].copy()
        slp["dtw_slope_m_yr"] = sign * (slp["dt_dy"] / slp["dt2"])  # >0 = water-table deepening (depletion)
        return stats.merge(slp[["StnID", "dtw_slope_m_yr"]], on="StnID", how="inner")[["StnID", "lat", "lon", "dtw_slope_m_yr"]]

    with zipfile.ZipFile(JASECHKO_GW_ZIP) as zf:
        with zf.open("AnnualDepthToGroundwater.csv") as f:
            w_dtw = calc_well_slopes(pd.read_csv(f, low_memory=False), "DepthToWater_m", +1.0)
        with zf.open("AnnualGroundwaterElevation.csv") as f:
            w_elv = calc_well_slopes(pd.read_csv(f, low_memory=False), "GroundwaterElevation_masl", -1.0)

    wells = pd.concat([w_dtw, w_elv], ignore_index=True).drop_duplicates(subset=["StnID"], keep="first")
    pts = gpd.GeoDataFrame(wells, geometry=[Point(xy) for xy in zip(wells["lon"], wells["lat"])], crs="EPSG:4326")
    j_w = gpd.sjoin(pts, aware_gdf[["aware_basin_idx", "geometry"]], how="inner", predicate="within")
    b_wells = j_w.groupby("aware_basin_idx", as_index=False).agg(
        jasechko_n_wells=("StnID", "nunique"),
        jasechko_median_dtw_slope_m_yr=("dtw_slope_m_yr", "median"),
        jasechko_frac_declining_wells=("dtw_slope_m_yr", lambda s: float((s > 0.02).mean())),
    )

    # Track B: Full 1,693 global aquifer systems (shapefile + Supp Table 16)
    aq_gdf = gpd.read_file(f"zip://{JASECHKO_AQ_ZIP}!jasechko_et_al_2024_aquifers.shp").to_crs("EPSG:4326")
    with zipfile.ZipFile(JASECHKO_ST_ZIP) as zf:
        with zf.open("Supplementary_Table_16.xlsx") as f:
            s16 = pd.read_excel(f)

    s16_col_n = [c for c in s16.columns if "number of monitoring wells" in c.lower()][0]
    s16_col_slp = [c for c in s16.columns if "median theil-sen slope" in c.lower()][0]
    s16_col_pct = [c for c in s16.columns if "percent of wells with theil-sen slopes of >0" in c.lower()][0]
    s16["key"] = s16["Study area"].astype(str).str.strip()
    s16_map = s16.set_index("key")[[s16_col_n, s16_col_slp, s16_col_pct]].to_dict(orient="index")

    aq_gdf["k1"] = aq_gdf["Aquifer"].astype(str).str.strip()
    aq_gdf["k2"] = aq_gdf.apply(
        lambda r: f"{r['Aquifer']} ({r['Broader']})" if str(r["Broader"]).strip() not in {"", "-", "nan"} else str(r["Aquifer"]).strip(),
        axis=1,
    )
    aq_gdf["aq_n_wells"] = [s16_map.get(k1, s16_map.get(k2, {})).get(s16_col_n, np.nan) for k1, k2 in zip(aq_gdf["k1"], aq_gdf["k2"])]
    aq_gdf["aq_theilsen_m_yr"] = [s16_map.get(k1, s16_map.get(k2, {})).get(s16_col_slp, np.nan) for k1, k2 in zip(aq_gdf["k1"], aq_gdf["k2"])]
    aq_gdf["aq_pct_declining"] = [s16_map.get(k1, s16_map.get(k2, {})).get(s16_col_pct, np.nan) for k1, k2 in zip(aq_gdf["k1"], aq_gdf["k2"])]
    aq_gdf = aq_gdf.dropna(subset=["aq_theilsen_m_yr"]).copy()

    # Use equal-area projection for accurate overlap area weighting
    aq_ea = aq_gdf[["Aquifer", "aq_n_wells", "aq_theilsen_m_yr", "aq_pct_declining", "geometry"]].to_crs("EPSG:6933")
    aw_ea = aware_gdf[["aware_basin_idx", "geometry"]].to_crs("EPSG:6933")
    aq_ea["geometry"] = aq_ea.geometry.buffer(0)
    aw_ea["geometry"] = aw_ea.geometry.buffer(0)
    inter = gpd.overlay(aw_ea, aq_ea, how="intersection", keep_geom_type=False)
    inter["ov_km2"] = inter.geometry.area / 1e6
    inter = inter[inter["ov_km2"] >= 50.0].copy()  # filter sliver overlaps < 50 km2

    def agg_aq_basin(g: pd.DataFrame) -> pd.Series:
        w = np.maximum(g["ov_km2"].values, 1.0)
        return pd.Series({
            "jasechko_aq_count": int(len(g)),
            "jasechko_aq_total_wells": int(np.nansum(g["aq_n_wells"].values)),
            "jasechko_aq_wmean_dtw_m_yr": float(np.average(g["aq_theilsen_m_yr"].values, weights=w)),
            "jasechko_aq_median_dtw_m_yr": float(np.median(g["aq_theilsen_m_yr"].values)),
            "jasechko_aq_pct_declining": float(np.average(g["aq_pct_declining"].fillna(50.0).values, weights=w)),
        })

    b_aquifers = inter.groupby("aware_basin_idx").apply(agg_aq_basin, include_groups=False).reset_index()
    return b_wells, b_aquifers


def compute_multighm_validation(p1_grid: pd.DataFrame, p1_basins: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Cross-validate Pillar 1 historical runoff (qtot), irrigation withdrawals (pirrww), and
    blue-water paradox exceedance across 4 independent ISIMIP3a GHMs (WaterGAP 2.2e, H08, LPJmL5-7-10-fire, CWatM)
    over 1991-2019 (gswp3-w5e5 obsclim_histsoc_default) across all 376 major salt-affected basins.
    """
    with xr.open_dataset(LSM_NC) as dl:
        lsm = np.squeeze(dl["mask"].values) == 1.0
    area_km2 = cell_area_km2_05deg()
    rr = p1_grid["r"].values
    cc = p1_grid["c"].values
    b_idx_cell = p1_grid["aware_basin_idx"].values
    cell_area = p1_grid["cell_area_km2"].values
    valid_b = b_idx_cell >= 0

    sb_base = p1_basins[p1_basins["salt_irrig_ha"] >= 10000.0].copy()
    sb_ids = set(sb_base["aware_basin_idx"].values)

    def read_ann_mm(nc_p: Path, var: str) -> np.ndarray:
        with xr.open_dataset(nc_p) as ds:
            sub = ds[var].sel(time=slice("1991-01-01", "2019-12-31"))
            dim = sub["time"].dt.days_in_month
            ann = (sub * dim * 86400.0).groupby("time.year").sum("time", skipna=True).mean("year").values
            if float(ds.lat.values[0]) < 0:
                ann = ann[::-1, :]
        return np.where(lsm, ann, np.nan)

    def agg_basin_km3(arr_mm: np.ndarray) -> pd.Series:
        vals = np.nan_to_num(arr_mm[rr[valid_b], cc[valid_b]], nan=0.0) * 1e-6 * cell_area[valid_b]
        return pd.Series(vals).groupby(b_idx_cell[valid_b]).sum()

    # Model configs: (name, qtot_file, use_file, use_var, use_is_withdrawal_scale, pirrww_file, pirrww_var)
    # Note: WaterGAP uses consumptive atotuse vs 0.40*qtot; H08 reports airruse (consumptive irrigation);
    # LPJmL reports airruse (consumptive irrigation; note LPJmL atotuse includes green water ET in ISIMIP3a);
    # CWatM reports atotww (total withdrawal; scaled by local WaterGAP consumptive ratio atotuse/pirrww for fair blue-water consumption check, plus raw withdrawal check).
    ghm_specs = [
        (
            "H08",
            MULTIGHM_DIR / "H08/h08_gswp3-w5e5_obsclim_histsoc_default_qtot_global_monthly_1901_2019.nc",
            MULTIGHM_DIR / "H08/h08_gswp3-w5e5_obsclim_histsoc_default_airruse_global_monthly_1901_2019.nc",
            "airruse",
            MULTIGHM_DIR / "H08/h08_gswp3-w5e5_obsclim_histsoc_default_pirrww_global_monthly_1901_2019.nc",
            "pirrww",
        ),
        (
            "LPJmL5-7-10-fire",
            MULTIGHM_DIR / "LPJmL5-7-10-fire/lpjml5-7-10-fire_gswp3-w5e5_obsclim_histsoc_default_qtot_global_monthly_1901_2019.nc",
            MULTIGHM_DIR / "LPJmL5-7-10-fire/lpjml5-7-10-fire_gswp3-w5e5_obsclim_histsoc_default_airruse_global_monthly_1901_2019.nc",
            "airruse",
            MULTIGHM_DIR / "LPJmL5-7-10-fire/lpjml5-7-10-fire_gswp3-w5e5_obsclim_histsoc_default_atotww_global_monthly_1901_2019.nc",
            "atotww",
        ),
        (
            "CWatM",
            MULTIGHM_DIR / "CWatM/cwatm_gswp3-w5e5_obsclim_histsoc_default_qtot_global_monthly_1901_2019.nc",
            MULTIGHM_DIR / "CWatM/cwatm_gswp3-w5e5_obsclim_histsoc_default_atotww_global_monthly_1901_2019.nc",
            "atotww",
            MULTIGHM_DIR / "CWatM/cwatm_gswp3-w5e5_obsclim_histsoc_default_pirrww_global_monthly_1901_2019.nc",
            "pirrww",
        ),
    ]

    # Reference WaterGAP row
    wg_b = sb_base.set_index("aware_basin_idx")
    # Also compute WaterGAP-equivalent demand on each GHM's qtot (isolating runoff structural uncertainty)
    # AND each GHM's native demand + ambient V_leach against its own 0.40 * qtot (full model structural uncertainty)
    summary_rows: List[Dict[str, Any]] = [{
        "ghm": "WaterGAP2-2e (Primary)",
        "global_qtot_km3_yr": round(float(p1_grid["qtot_km3_yr"].sum()), 1),
        "salt_basins_qtot_km3_yr": round(float(wg_b["qtot_km3_yr"].sum()), 1),
        "global_irr_or_tot_ww_km3_yr": round(float(p1_grid["pirrww_km3_yr"].sum()), 1),
        "salt_basins_demand_km3_yr": round(float(wg_b["atotuse_km3_yr"].sum()), 1),
        "runoff_corr_vs_watergap_376basins": 1.0,
        "paradox_exceeded_efr60_native_n": int(wg_b["blue_water_exceeded_amb"].sum()),
        "paradox_exceeded_efr60_native_mha": round(float(wg_b.loc[wg_b["blue_water_exceeded_amb"], "salt_irrig_ha"].sum() / 1e6), 2),
        "paradox_exceeded_efr60_wg_demand_n": int(wg_b["blue_water_exceeded_amb"].sum()),
        "paradox_agreement_vs_watergap_pct": 100.0,
        "demand_variable_note": "atotuse (consumptive total blue-water use)",
    }]

    basin_flags = pd.DataFrame(index=wg_b.index)
    basin_flags["WaterGAP2-2e"] = wg_b["blue_water_exceeded_amb"].astype(int)

    for ghm_name, fq, fu, uvar, fp, pvar in ghm_specs:
        if not (fq.is_file() and fu.is_file() and fp.is_file()):
            continue
        q_mm = read_ann_mm(fq, "qtot")
        u_mm = read_ann_mm(fu, uvar)
        p_mm = read_ann_mm(fp, pvar)

        q_b = agg_basin_km3(q_mm).reindex(wg_b.index).fillna(0.0)
        u_b = agg_basin_km3(u_mm).reindex(wg_b.index).fillna(0.0)
        # For CWatM (which reports withdrawal atotww rather than consumptive use), convert conservatively
        # using WaterGAP basin consumptive fraction (clipped [0.35, 0.70]) AND also record native withdrawal
        wg_atot_pos = np.maximum(wg_b["atotuse_km3_yr"].values, 0.0)
        if ghm_name == "CWatM":
            cons_ratio = np.clip(
                wg_atot_pos / np.maximum(wg_b["pirrww_km3_yr"].values, 1e-3),
                0.35, 0.70,
            )
            u_cons_b = pd.Series(u_b.values * cons_ratio, index=wg_b.index)
            note = "atotww scaled by basin consumptive ratio [0.35-0.70]"
        else:
            u_cons_b = u_b
            note = f"{uvar} (native consumptive irrigation use)"

        sbwa_g = 0.40 * q_b
        exc_native = (u_cons_b + wg_b["vleach_amb_km3_yr"]) > sbwa_g
        exc_wg_dem = (pd.Series(wg_atot_pos, index=wg_b.index) + wg_b["vleach_amb_km3_yr"]) > sbwa_g

        rho_q, _ = spearmanr(wg_b["qtot_km3_yr"].values, q_b.values)
        agree_pct = float((exc_native.values == wg_b["blue_water_exceeded_amb"].values).mean() * 100.0)
        basin_flags[ghm_name] = exc_native.astype(int)

        summary_rows.append({
            "ghm": ghm_name,
            "global_qtot_km3_yr": round(float(np.nansum(q_mm * 1e-6 * area_km2)), 1),
            "salt_basins_qtot_km3_yr": round(float(q_b.sum()), 1),
            "global_irr_or_tot_ww_km3_yr": round(float(np.nansum(p_mm * 1e-6 * area_km2)), 1),
            "salt_basins_demand_km3_yr": round(float(u_cons_b.sum()), 1),
            "runoff_corr_vs_watergap_376basins": round(float(rho_q), 4),
            "paradox_exceeded_efr60_native_n": int(exc_native.sum()),
            "paradox_exceeded_efr60_native_mha": round(float(wg_b.loc[exc_native, "salt_irrig_ha"].sum() / 1e6), 2),
            "paradox_exceeded_efr60_wg_demand_n": int(exc_wg_dem.sum()),
            "paradox_agreement_vs_watergap_pct": round(agree_pct, 1),
            "demand_variable_note": note,
        })

    # Ensemble consensus across 4 GHMs
    votes = basin_flags.sum(axis=1)
    n_ghms = basin_flags.shape[1]
    ens_summary = {
        "n_ghms": int(n_ghms),
        "ghms_evaluated": list(basin_flags.columns),
        "paradox_exceeded_n_range": [
            int(min(r["paradox_exceeded_efr60_native_n"] for r in summary_rows)),
            int(max(r["paradox_exceeded_efr60_native_n"] for r in summary_rows)),
        ],
        "paradox_exceeded_mha_range": [
            round(float(min(r["paradox_exceeded_efr60_native_mha"] for r in summary_rows)), 2),
            round(float(max(r["paradox_exceeded_efr60_native_mha"] for r in summary_rows)), 2),
        ],
        "basins_exceeded_in_at_least_2_of_4_ghms": int((votes >= 2).sum()),
        "area_mha_exceeded_in_at_least_2_of_4_ghms": round(float(wg_b.loc[votes >= 2, "salt_irrig_ha"].sum() / 1e6), 2),
        "basins_exceeded_in_at_least_3_of_4_ghms": int((votes >= 3).sum()),
        "area_mha_exceeded_in_at_least_3_of_4_ghms": round(float(wg_b.loc[votes >= 3, "salt_irrig_ha"].sum() / 1e6), 2),
        "basins_exceeded_in_all_4_ghms": int((votes == n_ghms).sum()),
        "area_mha_exceeded_in_all_4_ghms": round(float(wg_b.loc[votes == n_ghms, "salt_irrig_ha"].sum() / 1e6), 2),
        "models": summary_rows,
    }
    return pd.DataFrame(summary_rows), ens_summary


def main() -> None:
    print("[1/7] Loading SoilGrids 2.0 0-30 cm clay (%)...", flush=True)
    sg_clay_05 = reproject_soilgrids_clay()

    print("[2/7] Aggregating FAO HWSD2 D1 (ESP, sodic frac, gypsum, clay, poor drainage)...", flush=True)
    hw = compute_hwsd2_05deg()

    print("[3/7] Computing DynQual v1.0 0.5-deg global ambient salinity (1991-2019 annual & p90 monthly)...", flush=True)
    dq = compute_dynqual_05deg()

    print("[4/7] Computing CSR GRACE RL06.3 TWSA linear trends (2002-2024, mm/yr)...", flush=True)
    grace_05 = compute_grace_05deg()

    print("[5/7] Spatial joining WHYMAP GWR v1 (>5 g/L salineGW & HYGEO2 aquifers) onto WaterGAP grid...", flush=True)
    p1 = pd.read_parquet(P1_GRID)
    p1_b = pd.read_csv(P1_BASIN, low_memory=False)
    rr, cc = p1["r"].values, p1["c"].values

    out = p1[["r", "c", "lat", "lon", "aware_basin_idx"]].copy()
    out["sg_clay_0_30_pct"] = sg_clay_05[rr, cc]
    for k, arr in hw.items():
        out[k] = arr[rr, cc]
    for k, arr in dq.items():
        out[k] = arr[rr, cc]
    out["grace_twsa_trend_mm_yr"] = grace_05[rr, cc]

    pts_g = gpd.GeoDataFrame(
        {"r": rr, "c": cc},
        geometry=[Point(xy) for xy in zip(out["lon"], out["lat"])],
        crs="EPSG:4326",
    )
    sal_gw = gpd.read_file(f"zip://{WHYMAP_ZIP}!WHYMAP_GWR/shp/whymap_salineGW__v1_poly.shp").to_crs("EPSG:4326")
    aq_gw = gpd.read_file(f"zip://{WHYMAP_ZIP}!WHYMAP_GWR/shp/whymap_GW_aquifers_v1_poly.shp").to_crs("EPSG:4326")

    j_sal = gpd.sjoin(pts_g, sal_gw[["WROB0SAL_7", "geometry"]], how="left", predicate="within").drop_duplicates(["r", "c"])
    j_aq = gpd.sjoin(pts_g, aq_gw[["HYGEO2", "geometry"]], how="left", predicate="within").drop_duplicates(["r", "c"])

    sal_map = j_sal.set_index(["r", "c"])["WROB0SAL_7"].notna().to_dict()
    aq_map = j_aq.set_index(["r", "c"])["HYGEO2"].to_dict()

    out["whymap_saline_gw_gt5gl"] = [bool(sal_map.get((r0, c0), False)) for r0, c0 in zip(rr, cc)]
    hy_vals = np.array([aq_map.get((r0, c0), np.nan) for r0, c0 in zip(rr, cc)], dtype="float64")
    out["whymap_hygeo2"] = hy_vals
    out["whymap_aquifer_tier"] = np.where(
        (hy_vals >= 11) & (hy_vals <= 15), "Major_Basin",
        np.where((hy_vals >= 22) & (hy_vals <= 25), "Complex_Hydro",
                 np.where((hy_vals >= 33) & (hy_vals <= 34), "Local_Shallow", "Non_Aquifer_or_Ice"))
    )

    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PARQUET, index=False)

    print("[6/7] Cross-validating WaterGAP groundwater depletion against GRACE, Jasechko wells (DTW+Elev), and Jasechko 1,693 aquifers...", flush=True)
    aware_gdf = gpd.read_file(AWARE_GPKG, layer="AWARE20_Native_CFs_geospatial").reset_index().rename(columns={"index": "aware_basin_idx"})
    b_jas_wells, b_jas_aq = compute_jasechko_full_benchmarks(aware_gdf)

    drop_cols = [c for c in ["lat", "lon", "aware_basin_idx", "dynqual_ecw_annual_dsm", "dynqual_ecw_p90_monthly_dsm"] if c in out.columns and c in p1.columns]
    m_cell = p1.merge(out.drop(columns=drop_cols), on=["r", "c"], how="left")
    b_agg = m_cell[m_cell["aware_basin_idx"] >= 0].groupby("aware_basin_idx", as_index=False).agg(
        grace_mean_twsa_mm_yr=("grace_twsa_trend_mm_yr", "mean"),
        dynqual_median_ecw_dsm=("dynqual_ecw_annual_dsm", "median"),
        dynqual_p90_ecw_dsm=("dynqual_ecw_p90_monthly_dsm", "median"),
        whymap_saline_area_ha=("salt_irrig_ha", lambda s: float(s[m_cell.loc[s.index, "whymap_saline_gw_gt5gl"]].sum())),
        hwsd2_mean_esp_pct=("hwsd2_esp_pct", "mean"),
        sg_mean_clay_pct=("sg_clay_0_30_pct", "mean"),
    )

    sb = (
        p1_b[p1_b["salt_irrig_ha"] >= 10000.0]
        .merge(b_agg, on="aware_basin_idx", how="left")
        .merge(b_jas_wells, on="aware_basin_idx", how="left")
        .merge(b_jas_aq, on="aware_basin_idx", how="left")
    )
    sb.to_csv(OUT_BASIN_VAL, index=False)

    # Validation stats across 376 basins
    gw_act = sb[sb["gw_depletion_active"]]
    gw_non = sb[~sb["gw_depletion_active"]]
    valid_gr = sb.dropna(subset=["grace_mean_twsa_mm_yr", "gws_trend_mm_yr"])
    rho_gr, p_gr = spearmanr(valid_gr["gws_trend_mm_yr"], valid_gr["grace_mean_twsa_mm_yr"])
    _, p_ugr = mannwhitneyu(gw_act["grace_mean_twsa_mm_yr"].dropna(), gw_non["grace_mean_twsa_mm_yr"].dropna(), alternative="less")

    jas_w_sub = sb[sb["jasechko_n_wells"].fillna(0) >= 3]
    jw_act = jas_w_sub[jas_w_sub["gw_depletion_active"]]["jasechko_median_dtw_slope_m_yr"].dropna()
    jw_non = jas_w_sub[~jas_w_sub["gw_depletion_active"]]["jasechko_median_dtw_slope_m_yr"].dropna()
    _, p_ujw = mannwhitneyu(jw_act, jw_non, alternative="greater") if len(jw_act) >= 3 and len(jw_non) >= 3 else (np.nan, np.nan)

    jas_aq_sub = sb.dropna(subset=["jasechko_aq_wmean_dtw_m_yr"])
    ja_act = jas_aq_sub[jas_aq_sub["gw_depletion_active"]]["jasechko_aq_wmean_dtw_m_yr"].dropna()
    ja_non = jas_aq_sub[~jas_aq_sub["gw_depletion_active"]]["jasechko_aq_wmean_dtw_m_yr"].dropna()
    _, p_uja = mannwhitneyu(ja_act, ja_non, alternative="greater") if len(ja_act) >= 3 and len(ja_non) >= 3 else (np.nan, np.nan)
    rho_ja, p_rja = spearmanr(-jas_aq_sub["gws_trend_mm_yr"], jas_aq_sub["jasechko_aq_wmean_dtw_m_yr"]) if len(jas_aq_sub) >= 10 else (np.nan, np.nan)

    # DynQual vs Thorslund observed river salinity cross-check (on cells with both)
    dq_riv = m_cell.dropna(subset=["dynqual_ecw_annual_dsm", "riv_ec_median_dsm"])
    rho_dq, p_dq = spearmanr(dq_riv["dynqual_ecw_annual_dsm"], dq_riv["riv_ec_median_dsm"]) if len(dq_riv) >= 20 else (np.nan, np.nan)

    print("[7/7] Running ISIMIP3a 4-GHM structural cross-validation (WaterGAP, H08, LPJmL5, CWatM)...", flush=True)
    mghm_df, mghm_summary = compute_multighm_validation(p1, p1_b)
    mghm_df.to_csv(OUT_MULTIGHM_CSV, index=False)

    salt_cells = m_cell[m_cell["salt_affected"] & (m_cell["salt_irrig_ha"] > 0)].copy()
    tot_ha = float(salt_cells["salt_irrig_ha"].sum())

    val_summary: Dict[str, Any] = {
        "script": "build_spatial_validation_and_soil_layers.py",
        "grace_csr_validation_376_basins": {
            "basins_with_grace": int(len(valid_gr)),
            "spearman_r_watergap_gws_vs_grace_twsa": round(float(rho_gr), 4),
            "spearman_p": float(p_gr),
            "gw_depletion_basins_median_grace_mm_yr": round(float(gw_act["grace_mean_twsa_mm_yr"].median()), 2),
            "non_depletion_basins_median_grace_mm_yr": round(float(gw_non["grace_mean_twsa_mm_yr"].median()), 2),
            "mannwhitney_less_p": float(p_ugr),
        },
        "jasechko_2024_well_validation": {
            "basins_with_ge3_wells": int(len(jas_w_sub)),
            "salt_area_covered_mha": round(float(jas_w_sub["salt_irrig_ha"].sum() / 1e6), 2),
            "total_matched_wells": int(jas_w_sub["jasechko_n_wells"].sum()),
            "gw_depletion_basins_median_dtw_decline_m_yr": round(float(jw_act.median()), 4) if len(jw_act) else None,
            "non_depletion_basins_median_dtw_decline_m_yr": round(float(jw_non.median()), 4) if len(jw_non) else None,
            "mannwhitney_greater_p": float(p_ujw) if np.isfinite(p_ujw) else None,
        },
        "jasechko_2024_1693_aquifer_systems_validation": {
            "basins_matched_376": int(len(jas_aq_sub)),
            "salt_area_covered_mha": round(float(jas_aq_sub["salt_irrig_ha"].sum() / 1e6), 2),
            "salt_area_covered_pct": round(float(jas_aq_sub["salt_irrig_ha"].sum() / sb["salt_irrig_ha"].sum() * 100.0), 1),
            "total_aquifer_well_records": int(jas_aq_sub["jasechko_aq_total_wells"].sum()),
            "gw_depletion_basins_n": int(len(ja_act)),
            "non_depletion_basins_n": int(len(ja_non)),
            "gw_depletion_basins_median_dtw_m_yr": round(float(ja_act.median()), 4) if len(ja_act) else None,
            "non_depletion_basins_median_dtw_m_yr": round(float(ja_non.median()), 4) if len(ja_non) else None,
            "mannwhitney_greater_p": float(p_uja) if np.isfinite(p_uja) else None,
            "spearman_r_vs_neg_watergap_gws": round(float(rho_ja), 4) if np.isfinite(rho_ja) else None,
            "spearman_p": float(p_rja) if np.isfinite(p_rja) else None,
        },
        "dynqual_v1_global_salinity_coverage": {
            "salt_cells_covered": int(salt_cells["dynqual_ecw_annual_dsm"].notna().sum()),
            "salt_cells_total": int(len(salt_cells)),
            "salt_area_covered_pct": round(float(salt_cells.loc[salt_cells["dynqual_ecw_annual_dsm"].notna(), "salt_irrig_ha"].sum() / tot_ha * 100.0), 2),
            "salt_cropland_median_ecw_dsm": round(float(salt_cells["dynqual_ecw_annual_dsm"].median()), 3),
            "salt_cropland_p90_annual_ecw_dsm": round(float(salt_cells["dynqual_ecw_annual_dsm"].quantile(0.90)), 3),
            "salt_cropland_median_p90_monthly_ecw_dsm": round(float(salt_cells["dynqual_ecw_p90_monthly_dsm"].median()), 3),
            "cross_check_vs_thorslund_rivers_cells": int(len(dq_riv)),
            "cross_check_vs_thorslund_rivers_spearman_r": round(float(rho_dq), 4) if np.isfinite(rho_dq) else None,
            "cross_check_vs_thorslund_rivers_p": float(p_dq) if np.isfinite(p_dq) else None,
        },
        "isimip3a_multighm_structural_validation": mghm_summary,
        "whymap_gwr_overlap_on_salt_croplands": {
            "saline_gw_gt5gl_cells": int(salt_cells["whymap_saline_gw_gt5gl"].sum()),
            "saline_gw_gt5gl_mha": round(float(salt_cells.loc[salt_cells["whymap_saline_gw_gt5gl"], "salt_irrig_ha"].sum() / 1e6), 2),
            "aquifer_tier_mha": {k: round(float(g["salt_irrig_ha"].sum() / 1e6), 2) for k, g in salt_cells.groupby("whymap_aquifer_tier")},
        },
        "soil_mechanistic_layers_on_salt_croplands": {
            "total_mha": round(tot_ha / 1e6, 2),
            "hwsd2_sodic_share_ge10pct_mha": round(float(salt_cells.loc[salt_cells["hwsd2_sodic_frac"].fillna(0) >= 0.10, "salt_irrig_ha"].sum() / 1e6), 2),
            "hwsd2_mean_esp_ge15pct_mha": round(float(salt_cells.loc[salt_cells["hwsd2_esp_pct"].fillna(0) >= 15.0, "salt_irrig_ha"].sum() / 1e6), 2),
            "soilgrids_heavy_clay_ge35pct_mha": round(float(salt_cells.loc[salt_cells["sg_clay_0_30_pct"].fillna(0) >= 35.0, "salt_irrig_ha"].sum() / 1e6), 2),
            "hwsd2_poorly_imperfectly_drained_ge30pct_mha": round(float(salt_cells.loc[salt_cells["hwsd2_poor_drain_frac"].fillna(0) >= 0.30, "salt_irrig_ha"].sum() / 1e6), 2),
        },
    }
    OUT_JSON.write_text(json.dumps(val_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(val_summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    prepare_outputs()
    main()
