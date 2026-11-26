"""Paths shared by the analysis, figures and quality checks."""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_RAW_DATA_DIR = Path(os.environ.get("EXTERNAL_RAW_DATA_DIR", REPO_ROOT / "external_raw_data"))
HARMONIZED = REPO_ROOT / "data/harmonized"
ANALYSIS_INPUT = REPO_ROOT / "data/analysis_input"
TABLES = REPO_ROOT / "tables"
SOURCE_DATA = REPO_ROOT / "source_data"
QA = REPO_ROOT / "qa"
BASEMAP = REPO_ROOT / "data/basemap"
OUTPUT = REPO_ROOT / "outputs"
OUT_TABLES = OUTPUT / "tables"
OUT_GRIDS = OUTPUT / "analysis_input"
OUT_PANELS = OUTPUT / "source_data"
OUT_FIGURES = OUTPUT / "figures"


def table_input(name):
    result = OUT_TABLES / name
    return result if result.is_file() else TABLES / name


def grid_input(name):
    result = OUT_GRIDS / name
    return result if result.is_file() else ANALYSIS_INPUT / name


def panel_input(name):
    result = OUT_PANELS / name
    return result if result.is_file() else SOURCE_DATA / name


def prepare_outputs():
    for directory in (OUT_TABLES, OUT_GRIDS, OUT_PANELS, OUT_FIGURES, OUTPUT / "qa"):
        directory.mkdir(parents=True, exist_ok=True)

def climate_models(directory):
    """Verify all monthly future drivers before calculating the six ensemble summaries."""
    models = ("gfdl-esm4", "ipsl-cm6a-lr", "mpi-esm1-2-hr", "mri-esm2-0", "ukesm1-0-ll")
    variables = ("qtot", "pirrww", "atotuse", "atotusegw", "qr", "groundwstor")
    for model in models:
        for scenario in ("ssp126", "ssp370", "ssp585"):
            for variable in variables:
                name = f"watergap2-2e_{model}_w5e5_{scenario}_2015soc-from-histsoc_default_{variable}_global_monthly_2015_2100.nc"
                if not (directory / model / name).is_file():
                    raise FileNotFoundError(f"Missing WaterGAP driver {model}/{name}")
    return models
