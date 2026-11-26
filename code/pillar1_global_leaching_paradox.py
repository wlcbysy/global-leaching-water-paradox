#!/usr/bin/env python3
"""
Pillar 1: Global Physical Leaching-Water Paradox, Groundwater Depletion Redlines,
and Aligned CMIP6/ISIMIP3b Future Basin Hydrological Closure.

Strictly read-only on upstream files and newly acquired Public_Spatial datasets.
Uses ISIMIP land-sea mask (67,420 land cells), aggregates local runoff (qtot) within
AWARE 2.0 native watersheds (never summing routed dis), applies exact atotusegw sign
convention (net groundwater mining = atotusegw > 0), and evaluates 5-GCM ISIMIP3b
+ Hassani (2021) CMIP6 salinity projections + HSWUD China cross-validation.
"""

from __future__ import annotations

from config import climate_models, EXTERNAL_RAW_DATA_DIR, BASEMAP, OUT_TABLES, OUT_GRIDS, prepare_outputs

import json
import zipfile
from pathlib import Path
from typing import Any, Dict, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject
from shapely.geometry import Point
import xarray as xr

PROJECT_PUB_DIR = EXTERNAL_RAW_DATA_DIR

# Input paths (all strictly read-only)
WG_3A_DIR = PUB_DIR / "WaterGAP2_2e/ISIMIP3a_historical"
WG_3B_DIR = PUB_DIR / "WaterGAP2_2e/ISIMIP3b_future"
LSM_NC = PUB_DIR / "WaterGAP2_2e/landseamask_water-global.nc"
THORSLUND_DIR = PUB_DIR / "Thorslund_vanVliet_2020"
DYNQUAL_ANN_NC = PUB_DIR / "DynQual_v1/salinity_annualAvg_1980_2019_30min.nc"
DYNQUAL_MON_NC = PUB_DIR / "DynQual_v1/salinity_monthlyAvg_1980_2019_30min.nc"
HASSANI_PRED_ZIP = PUB_DIR / "Hassani_2021/Predictions.zip"

SPAM_HARV_ZIP = EXTERNAL_RAW_DATA_DIR / "SPAM2020/spam2020V2r2_global_harvested_area.geotiff.zip"
AWARE_GPKG = EXTERNAL_RAW_DATA_DIR / "AWARE2/AWARE20_Native_CFs_geospatial.gpkg"
GMIA_TIF = EXTERNAL_RAW_DATA_DIR / "FAO/AQUAMAPS_GMIA.tif"
GLOSIS_EXS_TIF = EXTERNAL_RAW_DATA_DIR / "FAO/GLOSIS_EXS.tif"
WANG_ECE_TIF = EXTERNAL_RAW_DATA_DIR / "Wang/ECe_2017_Wang.tif"
GAEZ_ET0_TIF = EXTERNAL_RAW_DATA_DIR / "GAEZv4/GAEZ-V4.RES01-ET0.TP-2010.tif"
GAEZ_PRC_TIF = EXTERNAL_RAW_DATA_DIR / "GAEZv4/GAEZ-V4.RES01-PRC.TP-2010.tif"
HSWUD_NC = EXTERNAL_RAW_DATA_DIR / "HSWUD/HSWUD_irr_v20260711.nc"
NE_COUNTRIES_SHP = BASEMAP / "ne_110m_admin_0_countries.zip"

# Outputs
OUT_GRID_PARQUET = OUT_GRIDS / "pillar1_grid05_baseline_current.parquet"
OUT_BASIN_CSV = OUT_TABLES / "pillar1_basin_leaching_closure_current.csv"
OUT_SUMMARY_JSON = OUT_TABLES / "pillar1_global_summary_current.json"

DST_SHAPE = (360, 720)
DST_TRANSFORM = from_origin(-180.0, 90.0, 0.5, 0.5)
DST_CRS = "EPSG:4326"

# FAO-29 / Maas-Hoffman parameters: (ECe_threshold_100pct dS/m, slope b % per dS/m)
CROP_SALT_PARAMS: Dict[str, Tuple[float, float]] = {
    "BARL": (8.0, 5.0),
    "COTT": (7.7, 5.2),
    "SUGB": (7.0, 5.9),
    "SORG": (6.8, 16.0),
    "RAPE": (6.5, 13.0),
    "WHEA": (6.0, 7.1),
    "OCER": (5.5, 7.0),
    "SOYB": (5.0, 20.0),
    "OFIB": (5.0, 10.0),
    "COWP": (4.9, 12.0),
    "SUNF": (4.8, 7.9),
    "PMIL": (4.0, 16.0),
    "MILL": (4.0, 16.0),
    "CNUT": (4.0, 10.0),
    "OOIL": (3.5, 12.0),
    "GROU": (3.2, 29.0),
    "RICE": (3.0, 12.0),
    "OILP": (3.0, 10.0),
    "SESA": (3.0, 12.0),
    "TOMA": (2.5, 9.9),
    "REST": (2.5, 12.0),
    "TOBA": (2.0, 12.0),
    "RUBB": (2.0, 12.0),
    "VEGE": (1.8, 12.0),
    "TROF": (1.8, 14.0),
    "MAIZ": (1.7, 12.0),
    "SUGC": (1.7, 5.9),
    "POTA": (1.7, 12.0),
    "CITR": (1.7, 16.0),
    "TEMF": (1.6, 15.0),
    "SWPO": (1.5, 11.0),
    "CASS": (1.5, 14.0),
    "YAMS": (1.5, 14.0),
    "ORTS": (1.5, 14.0),
    "CHIC": (1.5, 16.0),
    "BANA": (1.5, 15.0),
    "PLNT": (1.5, 15.0),
    "COFF": (1.5, 15.0),
    "RCOF": (1.5, 15.0),
    "COCO": (1.5, 15.0),
    "TEAS": (1.5, 15.0),
    "PIGE": (1.3, 15.0),
    "LENT": (1.3, 15.0),
    "OPUL": (1.3, 16.0),
    "ONIO": (1.2, 16.0),
    "BEAN": (1.0, 19.0),
}


def cell_area_km2_05deg() -> np.ndarray:
    R = 6371.0088
    lats_top = np.linspace(90.0, -89.5, 360)
    lats_bot = lats_top - 0.5
    area_row = (R ** 2) * np.deg2rad(0.5) * (np.sin(np.deg2rad(lats_top)) - np.sin(np.deg2rad(lats_bot)))
    return np.repeat(area_row[:, None], 720, axis=1)


def reproject_to_05deg(tif_path: Path, resampling: Resampling = Resampling.average) -> np.ndarray:
    with rasterio.open(tif_path) as src:
        src_arr = src.read(1).astype("float64")
        if src.nodata is not None:
            src_arr = np.where(src_arr == src.nodata, np.nan, src_arr)
        dst_arr = np.full(DST_SHAPE, np.nan, dtype="float64")
        reproject(
            source=src_arr,
            destination=dst_arr,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=np.nan,
            dst_transform=DST_TRANSFORM,
            dst_crs=DST_CRS,
            dst_nodata=np.nan,
            resampling=resampling,
        )
        return dst_arr


def compute_annual_flux_mm_yr(nc_path: Path, var: str, y0: int, y1: int, lsm: np.ndarray) -> np.ndarray:
    """Read monthly flux (kg m-2 s-1 == mm s-1) and return period-mean annual total (mm/yr)."""
    with xr.open_dataset(nc_path) as ds:
        sub = ds[var].sel(time=slice(f"{y0}-01-01", f"{y1}-12-31"))
        dim = sub["time"].dt.days_in_month
        # skipna=False preserves ocean NaNs; apply land mask explicitly
        ann = (sub * dim * 86400.0).groupby("time.year").sum("time", skipna=True).mean("year").values
        return np.where(lsm, ann, np.nan)


def compute_gws_trend_mm_yr(nc_path: Path, y0: int, y1: int, lsm: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute period-mean groundwstor (mm) and OLS linear trend (mm/yr) over y0..y1."""
    with xr.open_dataset(nc_path) as ds:
        sub = ds["groundwstor"].sel(time=slice(f"{y0}-01-01", f"{y1}-12-31"))
        ann_yr = sub.groupby("time.year").mean("time")
        yrs = ann_yr["year"].values.astype("float64")
        vals = ann_yr.values  # shape (n_yr, 360, 720)
        mean_mm = np.nanmean(vals, axis=0)
        # Vectorized OLS slope across years
        t_c = yrs - yrs.mean()
        denom = float(np.sum(t_c ** 2))
        y_c = vals - mean_mm[None, :, :]
        slope = np.nansum(y_c * t_c[:, None, None], axis=0) / denom
        return np.where(lsm, mean_mm, np.nan), np.where(lsm, slope, np.nan)


def build_spam_crop_thresholds(lsm: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate SPAM2020 irrigated harvested area and crop-weighted ECe_t & ECe_90 to 0.5-deg."""
    total_ha = np.zeros(DST_SHAPE, dtype="float64")
    w_ecet = np.zeros(DST_SHAPE, dtype="float64")
    w_ece90 = np.zeros(DST_SHAPE, dtype="float64")

    with zipfile.ZipFile(SPAM_HARV_ZIP) as zf:
        members = [m for m in zf.namelist() if m.endswith("_I.tif")]

    for m in members:
        code = m.split("_")[-2]
        ecet, slope_b = CROP_SALT_PARAMS.get(code, (2.5, 12.0))
        ece90 = ecet + 10.0 / slope_b
        with rasterio.open(f"/vsizip/{SPAM_HARV_ZIP}/{m}") as src:
            a = src.read(1).astype("float64")
            a = np.where((a > 0) & np.isfinite(a), a, 0.0)
            # Sum exact 6x6 blocks (2160x4320 -> 360x720)
            a_05 = a.reshape(360, 6, 720, 6).sum(axis=(1, 3))
        total_ha += a_05
        w_ecet += a_05 * ecet
        w_ece90 += a_05 * ece90

    has_crop = lsm & (total_ha > 0)
    ecet_05 = np.where(has_crop, w_ecet / np.maximum(total_ha, 1e-9), np.nan)
    ece90_05 = np.where(has_crop, w_ece90 / np.maximum(total_ha, 1e-9), np.nan)
    return np.where(lsm, total_ha, np.nan), ecet_05, ece90_05


def bin_thorslund_to_05deg() -> Dict[str, np.ndarray]:
    """Bin Thorslund & van Vliet (2020) groundwater & river EC stations (uS/cm -> dS/m) to 0.5-deg."""
    gw = pd.read_csv(
        THORSLUND_DIR / "Groundwaters_summary.csv",
        usecols=["Lat", "Lon", "median", "median_EC_conv"],
        low_memory=False,
    )
    gw["ec_dsm"] = gw["median"].fillna(gw["median_EC_conv"]) / 1000.0
    gw = gw.dropna(subset=["Lat", "Lon", "ec_dsm"])
    gw = gw[(gw["Lat"] >= -90) & (gw["Lat"] <= 90) & (gw["Lon"] >= -180) & (gw["Lon"] <= 180) & (gw["ec_dsm"] > 0.01) & (gw["ec_dsm"] <= 100.0)].copy()

    gw["r"] = np.clip(np.floor((90.0 - gw["Lat"].values) / 0.5).astype(int), 0, 359)
    gw["c"] = np.clip(np.floor((gw["Lon"].values + 180.0) / 0.5).astype(int), 0, 719)

    gw_n = np.zeros(DST_SHAPE, dtype="float64")
    gw_med = np.full(DST_SHAPE, np.nan, dtype="float64")
    gw_med_scr625 = np.full(DST_SHAPE, np.nan, dtype="float64")
    gw_p75 = np.full(DST_SHAPE, np.nan, dtype="float64")
    gw_frac_1_3 = np.full(DST_SHAPE, np.nan, dtype="float64")
    gw_frac_1_5 = np.full(DST_SHAPE, np.nan, dtype="float64")
    gw_frac_gt8 = np.full(DST_SHAPE, np.nan, dtype="float64")

    for (r, c), grp in gw.groupby(["r", "c"]):
        vals = grp["ec_dsm"].values
        gw_n[r, c] = len(vals)
        gw_med[r, c] = np.median(vals)
        vals_scr = vals[vals <= 6.25]
        if len(vals_scr) > 0:
            gw_med_scr625[r, c] = np.median(vals_scr)
        gw_p75[r, c] = np.percentile(vals, 75)
        gw_frac_1_3[r, c] = np.mean((vals >= 1.0) & (vals < 3.0))
        gw_frac_1_5[r, c] = np.mean((vals >= 1.0) & (vals < 5.0))
        gw_frac_gt8[r, c] = np.mean(vals >= 8.0)

    riv = pd.read_csv(
        THORSLUND_DIR / "Rivers_summary.csv",
        usecols=["Lat", "Lon", "median"],
        low_memory=False,
    )
    riv["ec_dsm"] = riv["median"] / 1000.0
    riv = riv.dropna(subset=["Lat", "Lon", "ec_dsm"])
    riv = riv[(riv["Lat"] >= -90) & (riv["Lat"] <= 90) & (riv["Lon"] >= -180) & (riv["Lon"] <= 180) & (riv["ec_dsm"] > 0.01) & (riv["ec_dsm"] <= 100.0)].copy()
    riv["r"] = np.clip(np.floor((90.0 - riv["Lat"].values) / 0.5).astype(int), 0, 359)
    riv["c"] = np.clip(np.floor((riv["Lon"].values + 180.0) / 0.5).astype(int), 0, 719)

    riv_n = np.zeros(DST_SHAPE, dtype="float64")
    riv_med = np.full(DST_SHAPE, np.nan, dtype="float64")
    riv_med_scr625 = np.full(DST_SHAPE, np.nan, dtype="float64")
    for (r, c), grp in riv.groupby(["r", "c"]):
        vals = grp["ec_dsm"].values
        riv_n[r, c] = len(vals)
        riv_med[r, c] = np.median(vals)
        vals_scr = vals[vals <= 6.25]
        if len(vals_scr) > 0:
            riv_med_scr625[r, c] = np.median(vals_scr)

    return {
        "gw_n_stations": gw_n,
        "gw_ec_median_dsm": gw_med,
        "gw_ec_median_scr625_dsm": gw_med_scr625,
        "gw_ec_p75_dsm": gw_p75,
        "gw_frac_brackish_1_3": gw_frac_1_3,
        "gw_frac_brackish_1_5": gw_frac_1_5,
        "gw_frac_hypersaline_gt8": gw_frac_gt8,
        "riv_n_stations": riv_n,
        "riv_ec_median_dsm": riv_med,
        "riv_ec_median_scr625_dsm": riv_med_scr625,
        "gw_total_valid_stations": int(len(gw)),
        "riv_total_valid_stations": int(len(riv)),
    }


def load_hassani_cmip6_ensembles() -> Dict[str, np.ndarray]:
    """Compute CMIP6 multi-GCM median ECe (dS/m) for 2020, 2050 (2040-2069), 2080 (2070-2099)."""
    out: Dict[str, np.ndarray] = {}
    with zipfile.ZipFile(HASSANI_PRED_ZIP) as zf:
        for ssp in ["ssp245", "ssp585"]:
            members = sorted([m for m in zf.namelist() if f"CMIP6_{ssp}/" in m and m.endswith(".txt")])
            if not members:
                continue
            y2020_cols = [str(y) for y in range(2005, 2025)]
            y2050_cols = [str(y) for y in range(2040, 2070)]
            y2080_cols = [str(y) for y in range(2070, 2100)]
            use_cols = ["X", "Y"] + y2020_cols + y2050_cols + y2080_cols
            g_2020 = []
            g_2050 = []
            g_2080 = []
            for m in members:
                with zf.open(m) as f:
                    df = pd.read_csv(f, usecols=use_cols)
                r_idx = np.clip(np.floor((90.0 - df["Y"].values) / 0.5).astype(int), 0, 359)
                c_idx = np.clip(np.floor((df["X"].values + 180.0) / 0.5).astype(int), 0, 719)
                a20 = np.full(DST_SHAPE, np.nan, dtype="float64")
                a50 = np.full(DST_SHAPE, np.nan, dtype="float64")
                a80 = np.full(DST_SHAPE, np.nan, dtype="float64")
                a20[r_idx, c_idx] = df[y2020_cols].mean(axis=1).values
                a50[r_idx, c_idx] = df[y2050_cols].mean(axis=1).values
                a80[r_idx, c_idx] = df[y2080_cols].mean(axis=1).values
                g_2020.append(a20)
                g_2050.append(a50)
                g_2080.append(a80)
            with np.errstate(invalid="ignore"):
                out[f"hassani_{ssp}_2020"] = np.nanmedian(np.stack(g_2020, axis=0), axis=0)
                out[f"hassani_{ssp}_2050"] = np.nanmedian(np.stack(g_2050, axis=0), axis=0)
                out[f"hassani_{ssp}_2080"] = np.nanmedian(np.stack(g_2080, axis=0), axis=0)
    return out


def calc_leaching(
    ecw_arr: np.ndarray,
    irr_net_mm: np.ndarray,
    peff_mm: np.ndarray,
    ece_max: np.ndarray,
    irrig_active: np.ndarray,
    area_km2: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Rhoades / FAO-29 dilution-adjusted leaching requirement & volumetric leaching demand (km3/yr).
    Returns: (LR_eff, V_leach_km3_yr, unleachable_mask)
    """
    ec_aw = np.where(
        irrig_active & np.isfinite(ecw_arr) & ((irr_net_mm + peff_mm) > 0),
        (irr_net_mm * ecw_arr) / np.maximum(irr_net_mm + peff_mm, 1e-6),
        np.nan,
    )
    denom = 5.0 * ece_max - ec_aw
    unleachable = irrig_active & np.isfinite(ec_aw) & np.isfinite(ece_max) & (denom <= 0.1 * ec_aw)
    lr = np.where(
        irrig_active & np.isfinite(ec_aw) & np.isfinite(ece_max),
        np.where(unleachable, 0.80, np.clip(ec_aw / np.maximum(denom, 1e-6), 0.0, 0.80)),
        np.nan,
    )
    # V_leach = V_irr_net * LR / (1 - LR)
    v_irr_km3 = irr_net_mm * 1e-6 * area_km2
    v_leach_km3 = np.where(irrig_active & np.isfinite(lr), v_irr_km3 * (lr / (1.0 - lr)), np.nan)
    return lr, v_leach_km3, unleachable


def run_future_isimip3b(
    lsm: np.ndarray,
    area_km2: np.ndarray,
    basin_grid: np.ndarray,
    vleach_amb_grid: np.ndarray,
    vleach_ec3_grid: np.ndarray,
    salt_basins_idx: set,
) -> Dict[str, Any]:
    """Discover all complete ISIMIP3b GCMs and compute 2050 (2040-2069) & 2080 (2070-2099) ensembles + basin shifts."""
    if not WG_3B_DIR.is_dir():
        raise FileNotFoundError(f"Missing WaterGAP future directory {WG_3B_DIR.name}")

    out_summary: Dict[str, Any] = {"available": True, "scenarios": {}}
    gcms = climate_models(WG_3B_DIR)

    # Pre-index valid basin masks for salt-affected basins
    valid_b_mask = basin_grid >= 0
    b_ids_all = basin_grid[valid_b_mask]
    v_amb_all = np.nan_to_num(vleach_amb_grid[valid_b_mask], nan=0.0)
    v_ec3_all = np.nan_to_num(vleach_ec3_grid[valid_b_mask], nan=0.0)
    b_vleach_amb = pd.Series(v_amb_all).groupby(b_ids_all).sum()
    b_vleach_ec3 = pd.Series(v_ec3_all).groupby(b_ids_all).sum()

    for ssp in ["ssp126", "ssp370", "ssp585"]:
        gcm_records = []
        for gcm in gcms:
            gdir = WG_3B_DIR / gcm
            f_q = gdir / f"watergap2-2e_{gcm}_w5e5_{ssp}_2015soc-from-histsoc_default_qtot_global_monthly_2015_2100.nc"
            f_p = gdir / f"watergap2-2e_{gcm}_w5e5_{ssp}_2015soc-from-histsoc_default_pirrww_global_monthly_2015_2100.nc"
            f_a = gdir / f"watergap2-2e_{gcm}_w5e5_{ssp}_2015soc-from-histsoc_default_atotuse_global_monthly_2015_2100.nc"
            f_gw = gdir / f"watergap2-2e_{gcm}_w5e5_{ssp}_2015soc-from-histsoc_default_atotusegw_global_monthly_2015_2100.nc"
            f_qr = gdir / f"watergap2-2e_{gcm}_w5e5_{ssp}_2015soc-from-histsoc_default_qr_global_monthly_2015_2100.nc"
            f_gws = gdir / f"watergap2-2e_{gcm}_w5e5_{ssp}_2015soc-from-histsoc_default_groundwstor_global_monthly_2015_2100.nc"
            if not (f_q.is_file() and f_p.is_file() and f_a.is_file() and f_gw.is_file() and f_qr.is_file() and f_gws.is_file()):
                continue
            try:
                rec = {"gcm": gcm}
                for h_name, (y0, y1) in [("2050", (2040, 2069)), ("2080", (2070, 2099))]:
                    q_f = compute_annual_flux_mm_yr(f_q, "qtot", y0, y1, lsm)
                    p_f = compute_annual_flux_mm_yr(f_p, "pirrww", y0, y1, lsm)
                    a_f = compute_annual_flux_mm_yr(f_a, "atotuse", y0, y1, lsm)
                    gw_f = compute_annual_flux_mm_yr(f_gw, "atotusegw", y0, y1, lsm)
                    qr_f = compute_annual_flux_mm_yr(f_qr, "qr", y0, y1, lsm)
                    _, gws_slp = compute_gws_trend_mm_yr(f_gws, y0, y1, lsm)
                    rec[h_name] = {
                        "qtot_km3": float(np.nansum(q_f * 1e-6 * area_km2)),
                        "pirrww_km3": float(np.nansum(p_f * 1e-6 * area_km2)),
                        "atotuse_km3": float(np.nansum(a_f * 1e-6 * area_km2)),
                        "atotusegw_mining_km3": float(np.nansum(np.maximum(gw_f, 0.0) * 1e-6 * area_km2)),
                        "qr_km3": float(np.nansum(qr_f * 1e-6 * area_km2)),
                        "qtot_grid": q_f,
                        "pirrww_grid": p_f,
                        "atotuse_grid": a_f,
                        "gw_grid": gw_f,
                        "qr_grid": qr_f,
                        "gws_slp_grid": gws_slp,
                    }
                gcm_records.append(rec)
            except Exception as e:
                raise RuntimeError(f"Failed climate calculation for {gcm} / {ssp}") from e

        if not gcm_records:
            continue

        ssp_res = {"n_gcms": len(gcm_records), "gcms_used": [r["gcm"] for r in gcm_records]}
        for h_name in ["2050", "2080"]:
            q_med = np.nanmedian(np.stack([r[h_name]["qtot_grid"] for r in gcm_records], axis=0), axis=0)
            a_med = np.nanmedian(np.stack([r[h_name]["atotuse_grid"] for r in gcm_records], axis=0), axis=0)
            gw_med = np.nanmedian(np.stack([r[h_name]["gw_grid"] for r in gcm_records], axis=0), axis=0)
            qr_med = np.nanmedian(np.stack([r[h_name]["qr_grid"] for r in gcm_records], axis=0), axis=0)
            slp_med = np.nanmedian(np.stack([r[h_name]["gws_slp_grid"] for r in gcm_records], axis=0), axis=0)

            # Basin-level future closure under ensemble median
            df_b = pd.DataFrame({
                "b": b_ids_all,
                "area": area_km2[valid_b_mask],
                "qtot": np.nan_to_num(q_med[valid_b_mask], nan=0.0) * 1e-6 * area_km2[valid_b_mask],
                "atot": np.nan_to_num(a_med[valid_b_mask], nan=0.0) * 1e-6 * area_km2[valid_b_mask],
                "gw_pos": np.maximum(np.nan_to_num(gw_med[valid_b_mask], nan=0.0), 0.0) * 1e-6 * area_km2[valid_b_mask],
                "gw_net": np.nan_to_num(gw_med[valid_b_mask], nan=0.0) * 1e-6 * area_km2[valid_b_mask],
                "qr": np.nan_to_num(qr_med[valid_b_mask], nan=0.0) * 1e-6 * area_km2[valid_b_mask],
                "gws_vol": np.nan_to_num(slp_med[valid_b_mask], nan=0.0) * 1e-6 * area_km2[valid_b_mask],
            }).groupby("b").sum()
            df_b = df_b.loc[df_b.index.isin(salt_basins_idx)].copy()
            df_b["sbwa"] = 0.40 * df_b["qtot"]
            df_b["v_amb"] = b_vleach_amb.reindex(df_b.index).fillna(0.0)
            df_b["v_ec3"] = b_vleach_ec3.reindex(df_b.index).fillna(0.0)
            df_b["gws_mm_yr"] = df_b["gws_vol"] / (df_b["area"] * 1e-6)
            df_b["gw_dep"] = (df_b["gw_pos"] > 0.05 * df_b["qr"]) & ((df_b["gw_net"] > df_b["qr"]) | (df_b["gws_mm_yr"] < -1.0))
            df_b_atot_pos = np.maximum(df_b["atot"], 0.0)
            df_b["bw_exc_amb"] = (df_b_atot_pos + df_b["v_amb"]) > df_b["sbwa"]
            df_b["bw_exc_ec3"] = (df_b_atot_pos + df_b["v_ec3"]) > df_b["sbwa"]

            ssp_res[h_name] = {
                "global_qtot_km3_median": round(float(np.median([r[h_name]["qtot_km3"] for r in gcm_records])), 1),
                "global_qtot_km3_iqr": [
                    round(float(np.percentile([r[h_name]["qtot_km3"] for r in gcm_records], 25)), 1),
                    round(float(np.percentile([r[h_name]["qtot_km3"] for r in gcm_records], 75)), 1),
                ],
                "global_pirrww_km3_median": round(float(np.median([r[h_name]["pirrww_km3"] for r in gcm_records])), 1),
                "global_pirrww_km3_iqr": [
                    round(float(np.percentile([r[h_name]["pirrww_km3"] for r in gcm_records], 25)), 1),
                    round(float(np.percentile([r[h_name]["pirrww_km3"] for r in gcm_records], 75)), 1),
                ],
                "global_atotuse_km3_median": round(float(np.median([r[h_name]["atotuse_km3"] for r in gcm_records])), 1),
                "global_gw_mining_pos_km3_median": round(float(np.median([r[h_name]["atotusegw_mining_km3"] for r in gcm_records])), 1),
                "salt_basins_blue_exceeded_amb": int(df_b["bw_exc_amb"].sum()),
                "salt_basins_blue_exceeded_ec3": int(df_b["bw_exc_ec3"].sum()),
                "salt_basins_insolvent_blue_and_gw_amb": int((df_b["bw_exc_amb"] & df_b["gw_dep"]).sum()),
                "salt_basins_insolvent_blue_and_gw_ec3": int((df_b["bw_exc_ec3"] & df_b["gw_dep"]).sum()),
            }
        out_summary["scenarios"][ssp] = ssp_res

    return out_summary


def main() -> None:
    print("=== Pillar 1: Global Physical Leaching-Water Paradox & Basin Closure ===")
    with xr.open_dataset(LSM_NC) as dl:
        lsm = np.squeeze(dl["mask"].values) == 1.0
    assert lsm.shape == DST_SHAPE, f"Unexpected lsm shape: {lsm.shape}"
    area_km2 = cell_area_km2_05deg()

    # 1. Historical WaterGAP 2.2e ISIMIP3a (1991-2019)
    print("[1/6] Loading WaterGAP 2.2e ISIMIP3a historical hydrology (1991-2019)...")
    qtot_mm = compute_annual_flux_mm_yr(WG_3A_DIR / "watergap2-2e_gswp3-w5e5_obsclim_histsoc_default_qtot_global_monthly_1901_2019.nc", "qtot", 1991, 2019, lsm)
    atotuse_mm = compute_annual_flux_mm_yr(WG_3A_DIR / "watergap2-2e_gswp3-w5e5_obsclim_histsoc_default_atotuse_global_monthly_1901_2019.nc", "atotuse", 1991, 2019, lsm)
    atotusegw_mm = compute_annual_flux_mm_yr(WG_3A_DIR / "watergap2-2e_gswp3-w5e5_obsclim_histsoc_default_atotusegw_global_monthly_1901_2019.nc", "atotusegw", 1991, 2019, lsm)
    pirrww_mm = compute_annual_flux_mm_yr(WG_3A_DIR / "watergap2-2e_gswp3-w5e5_obsclim_histsoc_default_pirrww_global_monthly_1901_2019.nc", "pirrww", 1991, 2019, lsm)
    pirruse_mm = compute_annual_flux_mm_yr(WG_3A_DIR / "watergap2-2e_gswp3-w5e5_obsclim_histsoc_default_pirruse_global_monthly_1901_2019.nc", "pirruse", 1991, 2019, lsm)
    qr_mm = compute_annual_flux_mm_yr(WG_3A_DIR / "watergap2-2e_gswp3-w5e5_obsclim_histsoc_default_qr_global_monthly_1901_2019.nc", "qr", 1991, 2019, lsm)
    gws_mean_mm, gws_trend_mm_yr = compute_gws_trend_mm_yr(WG_3A_DIR / "watergap2-2e_gswp3-w5e5_obsclim_histsoc_default_groundwstor_global_monthly_1901_2019.nc", 1991, 2019, lsm)
    with xr.open_dataset(WG_3A_DIR / "watergap2-2e_gswp3-w5e5_obsclim_histsoc_default_dis_global_monthly_1901_2019.nc") as ds_dis:
        sub_d = ds_dis["dis"].sel(time=slice("1991-01-01", "2019-12-31"))
        dim_d = sub_d["time"].dt.days_in_month
        dis_km3_yr = np.where(lsm, (sub_d * dim_d * 86400.0 * 1e-9).groupby("time.year").sum("time", skipna=True).mean("year").values, np.nan)

    # 2. Soil Salinity, GMIA, Climate & Crop Salt Tolerance Composite
    print("[2/6] Aggregating SPAM2020 (46 irrigated crops), GLOSIS, Wang2017, GMIA, and GAEZ...")
    gmia_05 = np.where(lsm, reproject_to_05deg(GMIA_TIF, Resampling.average), np.nan)
    et0_05 = np.where(lsm, reproject_to_05deg(GAEZ_ET0_TIF, Resampling.average), np.nan)
    prc_05 = np.where(lsm, reproject_to_05deg(GAEZ_PRC_TIF, Resampling.average), np.nan)
    peff_mm = np.minimum(0.70 * np.maximum(prc_05, 0.0), 0.85 * np.maximum(et0_05, 0.0))

    with rasterio.open(GLOSIS_EXS_TIF) as src:
        exs_raw = src.read(1)
    exs_blk = exs_raw.reshape(360, 6, 720, 6).transpose(0, 2, 1, 3).reshape(360, 720, 36)
    soil_valid = np.isin(exs_blk, [1, 2, 3, 4, 5])
    saline_soil = np.isin(exs_blk, [2, 3, 4, 5])
    glosis_sal_frac = np.where(lsm & (soil_valid.sum(axis=2) >= 6), saline_soil.sum(axis=2) / np.maximum(soil_valid.sum(axis=2), 1), np.nan)

    wang_mean_05 = np.full(DST_SHAPE, np.nan, dtype="float64")
    with rasterio.open(WANG_ECE_TIF) as src:
        for i in range(6):
            win = rasterio.windows.Window(0, i * 3600, 43200, 3600)
            blk = src.read(1, window=win).astype("float32") / 1000.0
            blk_v = np.where((blk > 0.0) & (blk <= 60.0) & np.isfinite(blk), blk, np.nan)
            sub = blk_v.reshape(60, 60, 720, 60).transpose(0, 2, 1, 3).reshape(60, 720, 3600)
            vcnt = np.isfinite(sub).sum(axis=2)
            with np.errstate(invalid="ignore"):
                wang_mean_05[i * 60 : (i + 1) * 60, :] = np.where(vcnt >= 30, np.nanmean(sub, axis=2), np.nan)
    wang_mean_05 = np.where(lsm, wang_mean_05, np.nan)

    spam_irrig_ha, crop_ecet_05, crop_ece90_05 = build_spam_crop_thresholds(lsm)

    # 3. Thorslund & van Vliet (2020) + DynQual v1.0 (Jones et al. 2023/2024) Ambient Water EC
    # Pre-map land cells to AWARE 2.0 native basins so low-discharge DynQual numerical spikes (>5000 mg/L TDS / >6.25 dS/m)
    # are replaced by the non-spike basin median before computing physical leaching requirements.
    print("[3/6] Binning Thorslund & van Vliet (2020) stations & loading DynQual v1.0 0.5-deg global salinity...")
    th = bin_thorslund_to_05deg()

    aware_gdf = gpd.read_file(AWARE_GPKG, layer="AWARE20_Native_CFs_geospatial").reset_index().rename(columns={"index": "aware_basin_idx"})
    month_cols = [f"CF_{m}" for m in ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]]
    aware_gdf["CF_max_monthly"] = aware_gdf[month_cols].max(axis=1)

    rr, cc = np.where(lsm)
    lons_c = -180.0 + (cc + 0.5) * 0.5
    lats_c = 90.0 - (rr + 0.5) * 0.5
    pts = gpd.GeoDataFrame(
        {"r": rr, "c": cc},
        geometry=[Point(xy) for xy in zip(lons_c, lats_c)],
        crs="EPSG:4326",
    )
    joined = gpd.sjoin(pts, aware_gdf[["aware_basin_idx", "CF_annual_agri", "CF_annual_unspecified", "CF_max_monthly", "geometry"]], how="left", predicate="within")
    joined = joined.drop_duplicates(subset=["r", "c"], keep="first")

    basin_grid = np.full(DST_SHAPE, -1, dtype="int32")
    aware_agri_grid = np.full(DST_SHAPE, np.nan, dtype="float64")
    basin_grid[joined["r"].values, joined["c"].values] = joined["aware_basin_idx"].fillna(-1).astype("int32").values
    aware_agri_grid[joined["r"].values, joined["c"].values] = joined["CF_annual_agri"].values

    def screen_dynqual_grid(raw_ecw_2d: np.ndarray, cap_dsm: float = 6.25) -> np.ndarray:
        """Screen PCR-GLOBWB2 low-discharge numerical TDS spikes (>5,000 mg/L = 6.25 dS/m) with basin-median valid DynQual."""
        vals = raw_ecw_2d[rr, cc].copy()
        valid = np.where(np.isfinite(vals) & (vals <= cap_dsm), vals, np.nan)
        tmp_s = pd.Series(valid)
        b_med = tmp_s.groupby(basin_grid[rr, cc]).transform("median")
        filled = tmp_s.fillna(b_med).values
        out = np.full(DST_SHAPE, np.nan, dtype="float64")
        out[rr, cc] = np.where(np.isfinite(filled), np.clip(filled, 0.20, cap_dsm), np.nan)
        return out

    dq_ecw_ann_raw = np.full(DST_SHAPE, np.nan, dtype="float64")
    dq_ecw_ann = np.full(DST_SHAPE, np.nan, dtype="float64")
    dq_ecw_p90 = np.full(DST_SHAPE, np.nan, dtype="float64")
    if DYNQUAL_ANN_NC.is_file():
        with xr.open_dataset(DYNQUAL_ANN_NC) as ds_dq:
            tds_a = ds_dq["tds"].sel(time=slice("1991-01-01", "2019-12-31")).values.astype("float64")
            if float(ds_dq.lat.values[0]) < 0:
                tds_a = tds_a[:, ::-1, :]
        tds_a = np.where((tds_a > 0.0) & (tds_a < 1.0e5) & np.isfinite(tds_a), tds_a, np.nan)
        with np.errstate(invalid="ignore"):
            tds_med = np.nanmedian(tds_a, axis=0)
        dq_ecw_ann_raw = np.where(np.isfinite(tds_med), np.where(tds_med <= 3200.0, tds_med / 640.0, tds_med / 800.0), np.nan)
        dq_ecw_ann_raw = np.where(np.isfinite(dq_ecw_ann_raw), np.clip(dq_ecw_ann_raw, 0.20, 30.0), np.nan)
        dq_ecw_ann = screen_dynqual_grid(dq_ecw_ann_raw, cap_dsm=6.25)
    if DYNQUAL_MON_NC.is_file():
        with xr.open_dataset(DYNQUAL_MON_NC) as ds_dqm:
            tds_m = ds_dqm["tds"].sel(time=slice("1991-01-01", "2019-12-31")).values.astype("float64")
            if float(ds_dqm.lat.values[0]) < 0:
                tds_m = tds_m[:, ::-1, :]
        tds_m = np.where((tds_m > 0.0) & (tds_m < 1.0e5) & np.isfinite(tds_m), tds_m, np.nan)
        with np.errstate(invalid="ignore"):
            tds_p90 = np.nanpercentile(tds_m, 90, axis=0)
        dq_p90_raw = np.where(np.isfinite(tds_p90), np.where(tds_p90 <= 3200.0, tds_p90 / 640.0, tds_p90 / 800.0), np.nan)
        dq_p90_raw = np.where(np.isfinite(dq_p90_raw), np.clip(dq_p90_raw, 0.20, 30.0), np.nan)
        dq_ecw_p90 = screen_dynqual_grid(dq_p90_raw, cap_dsm=6.25)

    # 4. Physical Leaching Requirement (LR) & V_leach across salinity regimes
    print("[4/6] Computing physical leaching requirements V_leach across ECw scenarios...")
    irrig_active = lsm & ((np.nan_to_num(gmia_05, nan=0.0) >= 1.0) | (np.nan_to_num(spam_irrig_ha, nan=0.0) >= 500.0)) & (np.nan_to_num(pirruse_mm, nan=0.0) > 1.0)
    salt_affected_grid = irrig_active & (
        (np.nan_to_num(glosis_sal_frac, nan=0.0) >= 0.10) | (np.nan_to_num(wang_mean_05, nan=0.0) >= 2.0)
    )

    # Primary ambient water salinity: source-weighted blend of surface water (screened Thorslund rivers <=6.25 dS/m
    # gap-filled by spike-screened DynQual v1.0 <=6.25 dS/m, 0.70 dS/m residual fallback) and screened agricultural
    # groundwater (Thorslund GW <=6.25 dS/m, fallback to surface water), weighted by WaterGAP net GW share f_gw_irrig.
    ecw_ambient_station_fb07 = np.where(
        np.isfinite(th["gw_ec_median_dsm"]),
        th["gw_ec_median_dsm"],
        np.where(np.isfinite(th["riv_ec_median_dsm"]), th["riv_ec_median_dsm"], 0.70),
    )
    ecw_ambient_station_fb07 = np.clip(ecw_ambient_station_fb07, 0.30, 30.0)

    ecw_sw = np.where(
        np.isfinite(th["riv_ec_median_scr625_dsm"]),
        th["riv_ec_median_scr625_dsm"],
        np.where(np.isfinite(dq_ecw_ann), dq_ecw_ann, 0.70),
    )
    ecw_sw = np.clip(ecw_sw, 0.20, 6.25)

    gw_pos_mm = np.maximum(np.nan_to_num(atotusegw_mm, nan=0.0), 0.0)
    denom_use_mm = np.maximum(
        np.maximum(np.nan_to_num(pirruse_mm, nan=0.0), np.nan_to_num(atotuse_mm, nan=0.0)),
        1e-6,
    )
    f_gw_irrig = np.clip(gw_pos_mm / denom_use_mm, 0.0, 1.0)
    ecw_gw_irr = np.where(
        np.isfinite(th["gw_ec_median_scr625_dsm"]),
        np.clip(th["gw_ec_median_scr625_dsm"], 0.20, 6.25),
        ecw_sw,
    )
    ecw_ambient = f_gw_irrig * ecw_gw_irr + (1.0 - f_gw_irrig) * ecw_sw
    ecw_ambient = np.clip(ecw_ambient, 0.20, 6.25)

    ecw_station_first = np.where(
        np.isfinite(th["gw_ec_median_dsm"]),
        th["gw_ec_median_dsm"],
        np.where(
            np.isfinite(th["riv_ec_median_dsm"]),
            th["riv_ec_median_dsm"],
            np.where(np.isfinite(dq_ecw_ann), dq_ecw_ann, 0.70),
        ),
    )
    ecw_ambient_both625 = np.clip(ecw_station_first, 0.20, 6.25)
    ecw_ambient_raw_dq = np.clip(
        np.where(
            np.isfinite(th["gw_ec_median_dsm"]),
            th["gw_ec_median_dsm"],
            np.where(np.isfinite(th["riv_ec_median_dsm"]), th["riv_ec_median_dsm"], np.where(np.isfinite(dq_ecw_ann_raw), dq_ecw_ann_raw, 0.70)),
        ),
        0.20,
        30.0,
    )
    ecw_dynqual_only = np.clip(np.where(np.isfinite(dq_ecw_ann), dq_ecw_ann, 0.70), 0.20, 6.25)
    ecw_dynqual_p90 = np.clip(np.where(np.isfinite(dq_ecw_p90), dq_ecw_p90, ecw_ambient_both625), 0.20, 6.25)

    ece_max_use = np.where(np.isfinite(crop_ece90_05), crop_ece90_05, 3.5)
    lr_amb, vleach_amb_km3, unl_amb = calc_leaching(ecw_ambient, pirruse_mm, peff_mm, ece_max_use, irrig_active, area_km2)
    _, vleach_fb07_km3, _ = calc_leaching(ecw_ambient_station_fb07, pirruse_mm, peff_mm, ece_max_use, irrig_active, area_km2)
    _, vleach_both625_km3, _ = calc_leaching(ecw_ambient_both625, pirruse_mm, peff_mm, ece_max_use, irrig_active, area_km2)
    _, vleach_raw_dq_km3, _ = calc_leaching(ecw_ambient_raw_dq, pirruse_mm, peff_mm, ece_max_use, irrig_active, area_km2)
    _, vleach_dq_only_km3, _ = calc_leaching(ecw_dynqual_only, pirruse_mm, peff_mm, ece_max_use, irrig_active, area_km2)
    _, vleach_dq_p90_km3, _ = calc_leaching(ecw_dynqual_p90, pirruse_mm, peff_mm, ece_max_use, irrig_active, area_km2)
    lr_ec1, vleach_ec1_km3, unl_ec1 = calc_leaching(np.full(DST_SHAPE, 1.0), pirruse_mm, peff_mm, ece_max_use, irrig_active, area_km2)
    lr_ec2, vleach_ec2_km3, unl_ec2 = calc_leaching(np.full(DST_SHAPE, 2.0), pirruse_mm, peff_mm, ece_max_use, irrig_active, area_km2)
    lr_ec3, vleach_ec3_km3, unl_ec3 = calc_leaching(np.full(DST_SHAPE, 3.0), pirruse_mm, peff_mm, ece_max_use, irrig_active, area_km2)
    lr_ec5, vleach_ec5_km3, unl_ec5 = calc_leaching(np.full(DST_SHAPE, 5.0), pirruse_mm, peff_mm, ece_max_use, irrig_active, area_km2)

    # Strict zero-yield-loss sensitivity (ECe_t)
    _, vleach_amb_ecet_km3, _ = calc_leaching(ecw_ambient, pirruse_mm, peff_mm, np.where(np.isfinite(crop_ecet_05), crop_ecet_05, 2.5), irrig_active, area_km2)
    # Zero rainfall dilution sensitivity (Peff=0)
    _, vleach_ec3_nopeff_km3, _ = calc_leaching(np.full(DST_SHAPE, 3.0), pirruse_mm, np.zeros(DST_SHAPE), ece_max_use, irrig_active, area_km2)

    # 5. Aggregating grid volumes onto AWARE 2.0 native watersheds & Hassani 2021...
    print("[5/6] Aggregating grid volumes onto AWARE 2.0 native watersheds & Hassani 2021...")

    # Hassani 2021 CMIP6 ECe
    hassani = load_hassani_cmip6_ensembles()

    # Build grid-level dataframe of all 67,420 land cells
    grid_df = pd.DataFrame({
        "r": rr,
        "c": cc,
        "lat": lats_c,
        "lon": lons_c,
        "aware_basin_idx": basin_grid[rr, cc],
        "aware_cf_agri": aware_agri_grid[rr, cc],
        "cell_area_km2": area_km2[rr, cc],
        "gmia_pct": gmia_05[rr, cc],
        "spam_irrig_ha": spam_irrig_ha[rr, cc],
        "irrig_active": irrig_active[rr, cc],
        "salt_affected": salt_affected_grid[rr, cc],
        "glosis_sal_frac": glosis_sal_frac[rr, cc],
        "wang_ece_2017_dsm": wang_mean_05[rr, cc],
        "crop_ecet_dsm": crop_ecet_05[rr, cc],
        "crop_ece90_dsm": crop_ece90_05[rr, cc],
        "et0_mm_yr": et0_05[rr, cc],
        "prc_mm_yr": prc_05[rr, cc],
        "peff_mm_yr": peff_mm[rr, cc],
        "qtot_mm_yr": qtot_mm[rr, cc],
        "atotuse_mm_yr": atotuse_mm[rr, cc],
        "atotusegw_mm_yr": atotusegw_mm[rr, cc],
        "pirrww_mm_yr": pirrww_mm[rr, cc],
        "pirruse_mm_yr": pirruse_mm[rr, cc],
        "qr_mm_yr": qr_mm[rr, cc],
        "gws_mean_mm": gws_mean_mm[rr, cc],
        "gws_trend_mm_yr": gws_trend_mm_yr[rr, cc],
        "gw_n_stations": th["gw_n_stations"][rr, cc],
        "gw_ec_median_dsm": th["gw_ec_median_dsm"][rr, cc],
        "gw_frac_brackish_1_3": th["gw_frac_brackish_1_3"][rr, cc],
        "gw_frac_brackish_1_5": th["gw_frac_brackish_1_5"][rr, cc],
        "riv_ec_median_dsm": th["riv_ec_median_dsm"][rr, cc],
        "dynqual_ecw_annual_dsm": dq_ecw_ann[rr, cc],
        "dynqual_ecw_p90_monthly_dsm": dq_ecw_p90[rr, cc],
        "f_gw_irrig": f_gw_irrig[rr, cc],
        "ecw_sw_dsm": ecw_sw[rr, cc],
        "ecw_ambient_dsm": ecw_ambient[rr, cc],
        "lr_ambient": lr_amb[rr, cc],
        "vleach_amb_km3_yr": vleach_amb_km3[rr, cc],
        "vleach_fb07_km3_yr": vleach_fb07_km3[rr, cc],
        "vleach_both625_km3_yr": vleach_both625_km3[rr, cc],
        "vleach_raw_dq_km3_yr": vleach_raw_dq_km3[rr, cc],
        "vleach_dq_only_km3_yr": vleach_dq_only_km3[rr, cc],
        "vleach_dq_p90_km3_yr": vleach_dq_p90_km3[rr, cc],
        "vleach_ec1_km3_yr": vleach_ec1_km3[rr, cc],
        "vleach_ec2_km3_yr": vleach_ec2_km3[rr, cc],
        "vleach_ec3_km3_yr": vleach_ec3_km3[rr, cc],
        "vleach_ec5_km3_yr": vleach_ec5_km3[rr, cc],
        "dis_km3_yr": dis_km3_yr[rr, cc],
        "unleachable_amb": unl_amb[rr, cc],
        "unleachable_ec3": unl_ec3[rr, cc],
        "unleachable_ec5": unl_ec5[rr, cc],
    })
    for k_h, arr_h in hassani.items():
        grid_df[k_h] = arr_h[rr, cc]

    # Volumetric columns per cell
    grid_df["qtot_km3_yr"] = grid_df["qtot_mm_yr"].fillna(0.0) * 1e-6 * grid_df["cell_area_km2"]
    grid_df["atotuse_km3_yr"] = grid_df["atotuse_mm_yr"].fillna(0.0) * 1e-6 * grid_df["cell_area_km2"]
    grid_df["atotusegw_mining_km3_yr"] = np.maximum(grid_df["atotusegw_mm_yr"].fillna(0.0), 0.0) * 1e-6 * grid_df["cell_area_km2"]
    grid_df["atotusegw_net_km3_yr"] = grid_df["atotusegw_mm_yr"].fillna(0.0) * 1e-6 * grid_df["cell_area_km2"]
    grid_df["qr_km3_yr"] = grid_df["qr_mm_yr"].fillna(0.0) * 1e-6 * grid_df["cell_area_km2"]
    grid_df["pirrww_km3_yr"] = grid_df["pirrww_mm_yr"].fillna(0.0) * 1e-6 * grid_df["cell_area_km2"]
    grid_df["pirruse_km3_yr"] = grid_df["pirruse_mm_yr"].fillna(0.0) * 1e-6 * grid_df["cell_area_km2"]
    grid_df["gws_trend_km3_yr"] = grid_df["gws_trend_mm_yr"].fillna(0.0) * 1e-6 * grid_df["cell_area_km2"]
    grid_df["salt_irrig_ha"] = np.where(grid_df["salt_affected"], grid_df["spam_irrig_ha"].fillna(0.0), 0.0)
    grid_df["irrig_ha_active"] = np.where(grid_df["irrig_active"], grid_df["spam_irrig_ha"].fillna(0.0), 0.0)
    for col_v in ["vleach_amb_km3_yr", "vleach_ec1_km3_yr", "vleach_ec2_km3_yr", "vleach_ec3_km3_yr", "vleach_ec5_km3_yr"]:
        scol = col_v.replace("_km3_yr", "_salt_only_km3_yr")
        grid_df[scol] = np.where(grid_df["salt_affected"], grid_df[col_v].fillna(0.0), 0.0)

    OUT_GRID_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    grid_df.to_parquet(OUT_GRID_PARQUET, index=False)

    # Basin aggregation (only valid AWARE basins)
    b_sub = grid_df[grid_df["aware_basin_idx"] >= 0].copy()
    basin_df = b_sub.groupby("aware_basin_idx", as_index=False).agg(
        n_cells=("r", "size"),
        lat_centroid=("lat", "mean"),
        lon_centroid=("lon", "mean"),
        basin_area_km2=("cell_area_km2", "sum"),
        irrig_ha_active=("irrig_ha_active", "sum"),
        salt_irrig_ha=("salt_irrig_ha", "sum"),
        aware_cf_agri=("aware_cf_agri", "first"),
        qtot_km3_yr=("qtot_km3_yr", "sum"),
        atotuse_km3_yr=("atotuse_km3_yr", "sum"),
        atotusegw_mining_km3_yr=("atotusegw_mining_km3_yr", "sum"),
        atotusegw_net_km3_yr=("atotusegw_net_km3_yr", "sum"),
        qr_km3_yr=("qr_km3_yr", "sum"),
        pirrww_km3_yr=("pirrww_km3_yr", "sum"),
        pirruse_km3_yr=("pirruse_km3_yr", "sum"),
        gws_trend_km3_yr=("gws_trend_km3_yr", "sum"),
        vleach_amb_km3_yr=("vleach_amb_km3_yr", "sum"),
        vleach_ec1_km3_yr=("vleach_ec1_km3_yr", "sum"),
        vleach_ec2_km3_yr=("vleach_ec2_km3_yr", "sum"),
        vleach_ec3_km3_yr=("vleach_ec3_km3_yr", "sum"),
        vleach_ec5_km3_yr=("vleach_ec5_km3_yr", "sum"),
        vleach_amb_salt_only_km3_yr=("vleach_amb_salt_only_km3_yr", "sum"),
        vleach_ec1_salt_only_km3_yr=("vleach_ec1_salt_only_km3_yr", "sum"),
        vleach_ec2_salt_only_km3_yr=("vleach_ec2_salt_only_km3_yr", "sum"),
        vleach_ec3_salt_only_km3_yr=("vleach_ec3_salt_only_km3_yr", "sum"),
        vleach_ec5_salt_only_km3_yr=("vleach_ec5_salt_only_km3_yr", "sum"),
        basin_max_dis_km3_yr=("dis_km3_yr", "max"),
        gw_n_stations=("gw_n_stations", "sum"),
        gw_ec_median_dsm=("gw_ec_median_dsm", "median"),
        gw_frac_brackish_1_5=("gw_frac_brackish_1_5", "mean"),
        hassani_2020_ece=("hassani_ssp585_2020", "mean"),
        hassani_2050_ssp585_ece=("hassani_ssp585_2050", "mean"),
        hassani_2080_ssp585_ece=("hassani_ssp585_2080", "mean"),
    )
    basin_df["gws_trend_mm_yr"] = basin_df["gws_trend_km3_yr"] / (basin_df["basin_area_km2"] * 1e-6)
    basin_df["sbwa_efr60_km3_yr"] = 0.40 * basin_df["qtot_km3_yr"]
    basin_df["sbwa_efr50_km3_yr"] = 0.50 * basin_df["qtot_km3_yr"]
    basin_df["sbwa_efr80_km3_yr"] = 0.20 * basin_df["qtot_km3_yr"]

    basin_df["atotuse_pos_km3_yr"] = np.maximum(basin_df["atotuse_km3_yr"], 0.0)
    basin_df["gw_depletion_active"] = (basin_df["atotusegw_mining_km3_yr"] > 0.05 * basin_df["qr_km3_yr"]) & (
        (basin_df["atotusegw_net_km3_yr"] > basin_df["qr_km3_yr"]) | (basin_df["gws_trend_mm_yr"] < -1.0)
    )
    basin_df["blue_water_exceeded_amb"] = (basin_df["atotuse_pos_km3_yr"] + basin_df["vleach_amb_km3_yr"]) > basin_df["sbwa_efr60_km3_yr"]
    basin_df["blue_water_exceeded_ec3"] = (basin_df["atotuse_pos_km3_yr"] + basin_df["vleach_ec3_km3_yr"]) > basin_df["sbwa_efr60_km3_yr"]

    def classify_paradox(r: pd.Series, bw_col: str) -> str:
        bw = bool(r[bw_col])
        gw = bool(r["gw_depletion_active"])
        if bw and gw:
            return "PARADOX_INSOLVENT_BLUE_AND_GW"
        if bw:
            return "PARADOX_BLUE_WATER_EXCEEDED"
        if gw:
            return "GW_DEPLETION_ONLY"
        return "HYDROLOGICALLY_FEASIBLE"

    basin_df["paradox_status_amb"] = basin_df.apply(lambda r: classify_paradox(r, "blue_water_exceeded_amb"), axis=1)
    basin_df["paradox_status_ec3"] = basin_df.apply(lambda r: classify_paradox(r, "blue_water_exceeded_ec3"), axis=1)
    basin_df["scarcity_weighted_vleach_amb_km3_weq"] = basin_df["vleach_amb_km3_yr"] * basin_df["aware_cf_agri"].fillna(1.0)
    basin_df["scarcity_weighted_vleach_ec3_km3_weq"] = basin_df["vleach_ec3_km3_yr"] * basin_df["aware_cf_agri"].fillna(1.0)

    # Attach country name via Natural Earth countries centroid join
    if NE_COUNTRIES_SHP.is_file():
        ne = gpd.read_file(NE_COUNTRIES_SHP)[["NAME", "CONTINENT", "geometry"]]
        b_pts = gpd.GeoDataFrame(
            {"aware_basin_idx": basin_df["aware_basin_idx"]},
            geometry=[Point(xy) for xy in zip(basin_df["lon_centroid"], basin_df["lat_centroid"])],
            crs="EPSG:4326",
        )
        b_cntry = gpd.sjoin(b_pts, ne, how="left", predicate="within").drop_duplicates("aware_basin_idx")
        basin_df = basin_df.merge(b_cntry[["aware_basin_idx", "NAME", "CONTINENT"]], on="aware_basin_idx", how="left")

    OUT_BASIN_CSV.parent.mkdir(parents=True, exist_ok=True)
    basin_df.to_csv(OUT_BASIN_CSV, index=False)

    # 6. ISIMIP3b Future Ensembles & HSWUD China Cross-Validation
    print("[6/6] Computing ISIMIP3b multi-GCM ensembles & HSWUD China cross-validation...")
    salt_basins = basin_df[basin_df["salt_irrig_ha"] >= 10000.0]  # Basins with >= 10,000 ha salt-affected irrigated land
    fut = run_future_isimip3b(lsm, area_km2, basin_grid, vleach_amb_km3, vleach_ec3_km3, set(salt_basins["aware_basin_idx"]))

    # HSWUD China cross-validation (1991-2019 mean irrigation water use)
    hswud_val = {}
    if HSWUD_NC.is_file():
        with xr.open_dataset(HSWUD_NC) as ds_h:
            h_sub = ds_h["irr"].sel(time=slice("1991-01-01", "2019-12-31"))
            h_sub = h_sub.where((h_sub >= 0.0) & (h_sub < 1e10))
            # units: 10^8 m3 == 0.1 km3 per cell-month
            h_ann_km3 = h_sub.groupby("time.year").sum("time", skipna=True).mean("year").values * 0.1
            h_lats = ds_h["lat"].values
            h_lons = ds_h["lon"].values
        h_total_km3 = float(np.nansum(h_ann_km3))

        # Compare over China bounding box (lat 18..53.5, lon 73.5..135) against WaterGAP pirrww & atotuse
        cn_mask = (grid_df["lat"] >= 18.0) & (grid_df["lat"] <= 53.5) & (grid_df["lon"] >= 73.5) & (grid_df["lon"] <= 135.0)
        hswud_val = {
            "hswud_china_1991_2019_mean_km3_yr": round(h_total_km3, 2),
            "watergap_china_box_pirrww_km3_yr": round(float(grid_df.loc[cn_mask, "pirrww_km3_yr"].sum()), 2),
            "watergap_china_box_atotuse_km3_yr": round(float(grid_df.loc[cn_mask, "atotuse_km3_yr"].sum()), 2),
        }
        # Sub-regional checks (Tarim / Xinjiang, North China Plain, Hetao/Yellow)
        regions = {
            "Tarim_Xinjiang": ((36, 44), (75, 90)),
            "Hetao_UpperYellow": ((39.5, 41.5), (105.5, 110.5)),
            "North_China_Plain": ((34, 40), (113, 120)),
        }
        reg_cmp = {}
        lon_g, lat_g = np.meshgrid(h_lons, h_lats)
        for rname, ((la0, la1), (lo0, lo1)) in regions.items():
            hm = (lat_g >= la0) & (lat_g <= la1) & (lon_g >= lo0) & (lon_g <= lo1)
            wm = (grid_df["lat"] >= la0) & (grid_df["lat"] <= la1) & (grid_df["lon"] >= lo0) & (grid_df["lon"] <= lo1)
            reg_cmp[rname] = {
                "hswud_km3_yr": round(float(np.nansum(h_ann_km3[hm])), 2),
                "watergap_pirrww_km3_yr": round(float(grid_df.loc[wm, "pirrww_km3_yr"].sum()), 2),
                "watergap_atotuse_km3_yr": round(float(grid_df.loc[wm, "atotuse_km3_yr"].sum()), 2),
            }
        hswud_val["regional_benchmarks"] = reg_cmp

    fut_clean = fut
    global_summary = {
        "script": "pillar1_global_leaching_paradox.py",
        "baseline_period": "1991-2019 (WaterGAP 2.2e ISIMIP3a gswp3-w5e5)",
        "global_hydrology_1991_2019": {
            "land_cells": int(lsm.sum()),
            "qtot_km3_yr": round(float(grid_df["qtot_km3_yr"].sum()), 1),
            "atotuse_km3_yr": round(float(grid_df["atotuse_km3_yr"].sum()), 1),
            "atotusegw_mining_pos_km3_yr": round(float(grid_df["atotusegw_mining_km3_yr"].sum()), 1),
            "atotusegw_net_km3_yr": round(float(grid_df["atotusegw_net_km3_yr"].sum()), 1),
            "qr_km3_yr": round(float(grid_df["qr_km3_yr"].sum()), 1),
            "pirrww_km3_yr": round(float(grid_df["pirrww_km3_yr"].sum()), 1),
            "pirruse_km3_yr": round(float(grid_df["pirruse_km3_yr"].sum()), 1),
        },
        "global_irrigated_and_salinity_scope": {
            "irrig_active_cells": int(grid_df["irrig_active"].sum()),
            "salt_affected_irrig_cells": int(grid_df["salt_affected"].sum()),
            "spam2020_total_irrig_mha": round(float(grid_df["irrig_ha_active"].sum() / 1e6), 2),
            "spam2020_salt_affected_irrig_mha": round(float(grid_df["salt_irrig_ha"].sum() / 1e6), 2),
            "thorslund_gw_valid_stations": th["gw_total_valid_stations"],
            "thorslund_riv_valid_stations": th["riv_total_valid_stations"],
        },
        "physical_leaching_demand_V_leach_km3_yr": {
            "ambient_observed_water_ec": round(float(np.nansum(vleach_amb_km3)), 2),
            "ambient_station_fallback07_sensitivity": round(float(np.nansum(vleach_fb07_km3)), 2),
            "ambient_both_capped_625_sensitivity": round(float(np.nansum(vleach_both625_km3)), 2),
            "ambient_dynqual_only_annual_sensitivity": round(float(np.nansum(vleach_dq_only_km3)), 2),
            "ambient_dynqual_p90_dryseason_sensitivity": round(float(np.nansum(vleach_dq_p90_km3)), 2),
            "ambient_dynqual_unscreened_raw_upper_bound": round(float(np.nansum(vleach_raw_dq_km3)), 2),
            "ambient_strict_ecet_100pct_yield": round(float(np.nansum(vleach_amb_ecet_km3)), 2),
            "counterfactual_ecw_1_dsm": round(float(np.nansum(vleach_ec1_km3)), 2),
            "counterfactual_ecw_2_dsm": round(float(np.nansum(vleach_ec2_km3)), 2),
            "counterfactual_ecw_3_dsm": round(float(np.nansum(vleach_ec3_km3)), 2),
            "counterfactual_ecw_3_no_peff_sensitivity": round(float(np.nansum(vleach_ec3_nopeff_km3)), 2),
            "counterfactual_ecw_5_dsm": round(float(np.nansum(vleach_ec5_km3)), 2),
            "unleachable_cells_amb": int(unl_amb.sum()),
            "unleachable_cells_ec3": int(unl_ec3.sum()),
            "unleachable_cells_ec5": int(unl_ec5.sum()),
            "salt_affected_only_vleach_km3_yr": {
                "ambient_observed_water_ec": round(float(grid_df["vleach_amb_salt_only_km3_yr"].sum()), 2),
                "ambient_station_fallback07_sensitivity": round(float(np.where(grid_df["salt_affected"], grid_df["vleach_fb07_km3_yr"].fillna(0.0), 0.0).sum()), 2),
                "ambient_both_capped_625_sensitivity": round(float(np.where(grid_df["salt_affected"], grid_df["vleach_both625_km3_yr"].fillna(0.0), 0.0).sum()), 2),
                "ambient_dynqual_only_annual_sensitivity": round(float(np.where(grid_df["salt_affected"], grid_df["vleach_dq_only_km3_yr"].fillna(0.0), 0.0).sum()), 2),
                "ambient_dynqual_p90_dryseason_sensitivity": round(float(np.where(grid_df["salt_affected"], grid_df["vleach_dq_p90_km3_yr"].fillna(0.0), 0.0).sum()), 2),
                "ambient_dynqual_unscreened_raw_upper_bound": round(float(np.where(grid_df["salt_affected"], grid_df["vleach_raw_dq_km3_yr"].fillna(0.0), 0.0).sum()), 2),
                "counterfactual_ecw_1_dsm": round(float(grid_df["vleach_ec1_salt_only_km3_yr"].sum()), 2),
                "counterfactual_ecw_2_dsm": round(float(grid_df["vleach_ec2_salt_only_km3_yr"].sum()), 2),
                "counterfactual_ecw_3_dsm": round(float(grid_df["vleach_ec3_salt_only_km3_yr"].sum()), 2),
                "counterfactual_ecw_5_dsm": round(float(grid_df["vleach_ec5_salt_only_km3_yr"].sum()), 2),
            },
        },
        "whole_basin_mass_conservation_check_top20_basins": {
            "top20_corr_qtot_vs_max_dis": round(float(basin_df.nlargest(20, "basin_area_km2")[["qtot_km3_yr", "basin_max_dis_km3_yr"]].corr().iloc[0, 1]), 4),
            "top20_sum_qtot_km3_yr": round(float(basin_df.nlargest(20, "basin_area_km2")["qtot_km3_yr"].sum()), 1),
            "top20_sum_max_dis_km3_yr": round(float(basin_df.nlargest(20, "basin_area_km2")["basin_max_dis_km3_yr"].sum()), 1),
        },
        "salt_affected_basins_ge_10kha": {
            "n_basins": int(len(salt_basins)),
            "salt_irrig_mha": round(float(salt_basins["salt_irrig_ha"].sum() / 1e6), 2),
            "paradox_status_ambient_counts": {k: int(v) for k, v in salt_basins["paradox_status_amb"].value_counts().to_dict().items()},
            "paradox_status_ec3_counts": {k: int(v) for k, v in salt_basins["paradox_status_ec3"].value_counts().to_dict().items()},
            "salt_irrig_mha_by_status_ambient": {
                k: round(float(grp["salt_irrig_ha"].sum() / 1e6), 2)
                for k, grp in salt_basins.groupby("paradox_status_amb")
            },
            "salt_irrig_mha_by_status_ec3": {
                k: round(float(grp["salt_irrig_ha"].sum() / 1e6), 2)
                for k, grp in salt_basins.groupby("paradox_status_ec3")
            },
            "efr_sensitivity_exceeded_basins_amb": {
                "efr_50pct": int(((salt_basins["atotuse_pos_km3_yr"] + salt_basins["vleach_amb_km3_yr"]) > salt_basins["sbwa_efr50_km3_yr"]).sum()),
                "efr_60pct": int(salt_basins["blue_water_exceeded_amb"].sum()),
                "efr_80pct": int(((salt_basins["atotuse_pos_km3_yr"] + salt_basins["vleach_amb_km3_yr"]) > salt_basins["sbwa_efr80_km3_yr"]).sum()),
            },
        },
        "hassani_2021_cmip6_salt_irrig_mean_ece_dsm": {
            k: round(float(grid_df.loc[grid_df["salt_affected"], k].mean()), 3)
            for k in sorted(hassani.keys())
        },
        "isimip3b_future_hydrology": fut_clean,
        "hswud_china_cross_validation": hswud_val,
    }

    with OUT_SUMMARY_JSON.open("w", encoding="utf-8") as f:
        json.dump(global_summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(global_summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    prepare_outputs()
    main()
