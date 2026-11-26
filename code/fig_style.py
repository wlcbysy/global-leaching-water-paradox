#!/usr/bin/env python3
"""
Shared publication-grade plotting utilities for the Nature Water manuscript.
Enforces global Arial typography, pdf.fonttype=42, Okabe-Ito/NPG colorblind-safe palettes,
SI negative-exponent labels, zero subplot titles, and automatic text-overlap auditing.
"""

from __future__ import annotations

from config import BASEMAP

from pathlib import Path
from typing import List, Tuple

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from pyproj import Transformer
from shapely.geometry import Polygon, box

NE_COUNTRIES_SHP = BASEMAP / "ne_110m_admin_0_countries.zip"
NE_LAND_50M = BASEMAP / "ne_50m_land.zip"
NE_LAKES_50M = BASEMAP / "ne_50m_lakes.zip"
NE_BORDERS_50M = BASEMAP / "ne_50m_admin_0_boundary_lines_land.zip"
ROBIN_CRS = "+proj=robin +lon_0=0 +datum=WGS84 +units=m +no_defs"

# Colorblind-safe unified journal palette
PALETTE = {
    "zone1": "#0072B2",      # Safe brackish window (Deep Blue)
    "zone2": "#E69F00",      # Hydraulic recovery priority (Amber)
    "zone3": "#D55E00",      # Water-salt dual redline (Vermilion/Crimson)
    "blue": "#0072B2",
    "teal": "#009E73",
    "amber": "#E69F00",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "navy": "#2B4162",
    "gray": "#7F7F7F",
    "lightgray": "#D9D9D9",
    "land": "#F2F2F0",
    "border": "#B8B8B2",
}

PARADOX_COLORS = {
    "HYDROLOGICALLY_FEASIBLE": "#3B8EA5",
    "GW_DEPLETION_ONLY": "#E9A825",
    "PARADOX_BLUE_WATER_EXCEEDED": "#E86F51",
    "PARADOX_INSOLVENT_BLUE_AND_GW": "#A61C28",
}

NW_FIG1_PALETTE = {
    "all_irrig": "#2166AC",
    "salt_cells": "#B2182B",
    "lever_blend": "#2166AC",
    "lever_hyd": "#2A8C6E",
    "lever_cap": "#4B3F72",
    "land": "#EFEFEB",
    "coast": "#A6AAA4",
    "border": "#D4D7D2",
    "grid": "#E3E6EA",
}


def setup_mpl_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "mathtext.fontset": "stixsans",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "font.size": 8.0,
        "axes.labelsize": 8.5,
        "axes.titlesize": 9.0,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.65,
        "ytick.major.width": 0.65,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "text.color": "#111111",
        "axes.labelcolor": "#111111",
        "xtick.color": "#111111",
        "ytick.color": "#111111",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.06,
    })


def add_panel_label(ax: mpl.axes.Axes, letter: str, x: float = -0.10, y: float = 1.05) -> None:
    ax.text(
        x, y, letter,
        transform=ax.transAxes,
        fontsize=11.0,
        fontweight="bold",
        fontfamily="sans-serif",
        va="bottom",
        ha="left",
        color="#000000",
    )


def add_fig_panel_label(fig: mpl.figure.Figure, x: float, y: float, letter: str) -> None:
    fig.text(
        x, y, letter,
        fontsize=10.5,
        fontweight="bold",
        fontfamily="sans-serif",
        va="top",
        ha="left",
        color="#000000",
    )


def get_clipped_land_50m(clip_lon=(-138.0, 168.0), clip_lat=(-48.0, 64.0)) -> gpd.GeoDataFrame:
    clip_poly = box(clip_lon[0], clip_lat[0], clip_lon[1], clip_lat[1])
    land = gpd.read_file(NE_LAND_50M)
    return gpd.clip(land, clip_poly)


def plot_robinson_basemap(
    ax: mpl.axes.Axes,
    xlim_lon: Tuple[float, float] = (-138.0, 168.0),
    ylim_lat: Tuple[float, float] = (-48.0, 64.0),
    graticule_lats: Tuple[int, ...] = (-30, 0, 30, 60),
    curved_frame: bool = True,
) -> gpd.GeoDataFrame:
    clip_poly = box(xlim_lon[0], ylim_lat[0], xlim_lon[1], ylim_lat[1])
    land_wgs = get_clipped_land_50m(xlim_lon, ylim_lat)
    land_rob = land_wgs.to_crs(ROBIN_CRS)
    tr = Transformer.from_crs("EPSG:4326", ROBIN_CRS, always_xy=True)

    if curved_frame:
        lats_edge = np.linspace(ylim_lat[0], ylim_lat[1], 140)
        lons_edge = np.linspace(xlim_lon[0], xlim_lon[1], 140)
        ring_lon = np.concatenate([
            np.full_like(lats_edge, xlim_lon[0]),
            lons_edge,
            np.full_like(lats_edge, xlim_lon[1]),
            lons_edge[::-1],
        ])
        ring_lat = np.concatenate([
            lats_edge,
            np.full_like(lons_edge, ylim_lat[1]),
            lats_edge[::-1],
            np.full_like(lons_edge, ylim_lat[0]),
        ])
        rx, ry = tr.transform(ring_lon, ring_lat)
        globe_poly = Polygon(zip(rx, ry))
        globe_gs = gpd.GeoSeries([globe_poly], crs=ROBIN_CRS)
        globe_gs.plot(ax=ax, facecolor="#F4F8FB", edgecolor="none", zorder=0.2)

        for mlon in (-120, -60, 0, 60, 120):
            if xlim_lon[0] < mlon < xlim_lon[1]:
                mx, my = tr.transform(np.full_like(lats_edge, float(mlon)), lats_edge)
                ax.plot(mx, my, ls=":", color="#E1EAF0", lw=0.35, zorder=0.4)

        for glat in graticule_lats:
            if ylim_lat[0] < glat < ylim_lat[1]:
                xl, yg = tr.transform(xlim_lon[0], float(glat))
                xr, _ = tr.transform(xlim_lon[1], float(glat))
                ax.plot([xl, xr], [yg, yg], ls="--", color="#DCE5EC", lw=0.42, zorder=0.4)

        land_rob.plot(ax=ax, facecolor="#ECEEE9", edgecolor="#9AA199", linewidth=0.32, zorder=1.0)
        if NE_BORDERS_50M.is_file():
            borders = gpd.clip(gpd.read_file(NE_BORDERS_50M), clip_poly).to_crs(ROBIN_CRS)
            borders.plot(ax=ax, color="#D6D9D3", linewidth=0.20, zorder=1.4)
        if NE_LAKES_50M.is_file():
            lakes = gpd.clip(gpd.read_file(NE_LAKES_50M), clip_poly).to_crs(ROBIN_CRS)
            lakes.plot(ax=ax, facecolor="#F4F8FB", edgecolor="#B8C5CE", linewidth=0.18, zorder=1.6)

        globe_gs.plot(ax=ax, facecolor="none", edgecolor="#BCC7D1", linewidth=0.65, zorder=6.0)
        bx0, by0, bx1, by1 = globe_poly.bounds
        pad_x = (bx1 - bx0) * 0.006
        pad_y = (by1 - by0) * 0.010
        ax.set_xlim(bx0 - pad_x, bx1 + pad_x)
        ax.set_ylim(by0 - pad_y, by1 + pad_y)
        for sp in ("top", "right", "bottom", "left"):
            ax.spines[sp].set_visible(False)
    else:
        land_rob.plot(
            ax=ax,
            facecolor=NW_FIG1_PALETTE["land"],
            edgecolor=NW_FIG1_PALETTE["coast"],
            linewidth=0.30,
            zorder=1.0,
        )
        if NE_BORDERS_50M.is_file():
            borders = gpd.clip(gpd.read_file(NE_BORDERS_50M), clip_poly).to_crs(ROBIN_CRS)
            borders.plot(ax=ax, color=NW_FIG1_PALETTE["border"], linewidth=0.20, zorder=1.4)
        if NE_LAKES_50M.is_file():
            lakes = gpd.clip(gpd.read_file(NE_LAKES_50M), clip_poly).to_crs(ROBIN_CRS)
            lakes.plot(ax=ax, facecolor="#FFFFFF", edgecolor="#BFC6CE", linewidth=0.18, zorder=1.6)
        x_min, _ = tr.transform(xlim_lon[0], 0.0)
        x_max, _ = tr.transform(xlim_lon[1], 0.0)
        _, y_min = tr.transform(0.0, ylim_lat[0])
        _, y_max = tr.transform(0.0, ylim_lat[1])
        for lat in graticule_lats:
            _, yg = tr.transform(0.0, float(lat))
            ax.plot([x_min, x_max], [yg, yg], ls="--", color=NW_FIG1_PALETTE["grid"], lw=0.40, zorder=0.5)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        for sp in ("top", "right", "bottom", "left"):
            ax.spines[sp].set_visible(True)
            ax.spines[sp].set_linewidth(0.45)
            ax.spines[sp].set_color("#CFD4D9")

    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    return land_wgs


def plot_world_basemap(ax: mpl.axes.Axes, xlim=(-130, 155), ylim=(-45, 62)) -> None:
    if NE_COUNTRIES_SHP.is_file():
        world = gpd.read_file(NE_COUNTRIES_SHP)
        world = world[world["NAME"] != "Antarctica"]
        world.plot(ax=ax, facecolor=PALETTE["land"], edgecolor=PALETTE["border"], linewidth=0.35, zorder=1)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ("top", "right", "bottom", "left"):
        ax.spines[sp].set_visible(True)
        ax.spines[sp].set_linewidth(0.55)
        ax.spines[sp].set_color("#888888")


def fmt_p(p: float | None) -> str:
    if p is None or not np.isfinite(p):
        return "NA"
    if p < 0.001:
        return r"$\mathit{P}$ < 0.001"
    return rf"$\mathit{{P}}$ = {p:.3f}"


def audit_text_overlaps(fig: mpl.figure.Figure, name: str) -> int:
    """Detect bounding-box overlaps among all rendered visible text artists within a figure."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes: List[Tuple[str, object]] = []

    def _add_artist(txt_obj, tag: str) -> None:
        if not txt_obj.get_visible():
            return
        s = txt_obj.get_text().strip()
        if not s:
            return
        bb = txt_obj.get_window_extent(renderer=renderer)
        if bb.width > 1 and bb.height > 1:
            boxes.append((f"{tag}:{s.replace(chr(10), ' ')[:30]}", bb))

    for t in fig.texts:
        _add_artist(t, "fig")

    for leg in fig.legends:
        if leg.get_visible():
            for t in leg.get_texts():
                _add_artist(t, "fleg")

    for ax_idx, ax in enumerate(fig.get_axes()):
        if not ax.get_visible():
            continue
        for t in ax.texts:
            _add_artist(t, f"ax{ax_idx}")
        _add_artist(ax.xaxis.label, f"ax{ax_idx}_xl")
        _add_artist(ax.yaxis.label, f"ax{ax_idx}_yl")
        xmin, xmax = sorted(ax.get_xlim())
        ymin, ymax = sorted(ax.get_ylim())
        for t in ax.get_xticklabels():
            pos = t.get_position()
            if xmin - 1e-6 <= pos[0] <= xmax + 1e-6:
                _add_artist(t, f"ax{ax_idx}_xt")
        for t in ax.get_yticklabels():
            pos = t.get_position()
            if ymin - 1e-6 <= pos[1] <= ymax + 1e-6:
                _add_artist(t, f"ax{ax_idx}_yt")
        leg = ax.get_legend()
        if leg is not None and leg.get_visible():
            for t in leg.get_texts():
                _add_artist(t, f"ax{ax_idx}_leg")

    overlaps = 0
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            b1 = boxes[i][1].expanded(0.95, 0.90)
            b2 = boxes[j][1].expanded(0.95, 0.90)
            if b1.overlaps(b2):
                overlaps += 1
                print(f"  [OVERLAP WARN {name}] '{boxes[i][0]}' <-> '{boxes[j][0]}'")
    print(f"  [AUDIT {name}] checked={len(boxes)} text elements, overlaps={overlaps}")
    return overlaps


def save_pub_figure(fig: mpl.figure.Figure, out_pdf: Path, out_png: Path, audit_name: str) -> None:
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    audit_text_overlaps(fig, audit_name)
    fig.savefig(out_pdf, format="pdf")
    fig.savefig(out_png, format="png", dpi=600)
    plt.close(fig)
    # Preserve vector text and the full tight drawing when the crop slightly exceeds two columns.
    import fitz
    from PIL import Image
    import io
    limit_pt = 179.8 * 72 / 25.4
    with fitz.open(out_pdf) as original:
        bounds = original[0].rect
        if bounds.width <= limit_pt:
            return
        scale = limit_pt / bounds.width
        with fitz.open() as scaled:
            page = scaled.new_page(width=limit_pt, height=bounds.height*scale)
            page.show_pdf_page(page.rect, original, 0)
            pdf_bytes = scaled.tobytes(garbage=4, deflate=True)
    out_pdf.write_bytes(pdf_bytes)
    with Image.open(out_png) as source:
        pixels = (round(source.width*scale),round(source.height*scale))
        raster = source.resize(pixels, Image.Resampling.LANCZOS)
    raster.save(out_png, dpi=(600,600))
