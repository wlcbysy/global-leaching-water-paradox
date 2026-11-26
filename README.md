# Supply, soil hydraulic and consumptive limits govern the global leaching-water paradox

**Shu-Yang Yu** (College of Agronomy, Hebei Agricultural University), **Jingsong Du** and **Shaobo Li** (School of Computer Science, Jining Normal University).

Code and harmonized data accompanying an Analysis prepared for *Nature Water*. The study asks when irrigation with brackish water can replace freshwater withdrawals, and how much of a withdrawal saving survives as reduced basin depletion after leaching. Field comparisons relate irrigation-water electrical conductivity (ECw) to crop production and separately measured soil salinity. A second synthesis estimates hydraulic responses and distinguishes consumptive evapotranspiration (ET) from irrigation delivered. The spatial analysis links those estimates to a 0.5° grid and basin water and salt budgets.

The field-harvest model estimates 90% of the fitted low-salinity plateau at 3.519 dS/m (95% family-bootstrap interval 2.899–4.282, 730 contrasts in 96 families). The broader continuous-irrigation production model supplies the 3.547 dS/m accounting reference. At that reference, modeled annual basin depletion relief is +72.72 km³ under an equal-production water cap and −6.36 km³ using the observed ET ratio. The negative value denotes increased depletion.

## Repository contents

| Path | Contents |
| --- | --- |
| `code/` | Empirical models, spatial processing, validation, figures, formatted table extracts and Excel builder |
| `data/harmonized/` | Six evidence and audit tables. Arm A includes 4,216 extracted contrasts, of which 3,775 pass the model filters. The Arm B synthesis starts from 6,879 harmonized rows. |
| `data/analysis_input/` | Four Parquet files and a second, identical copy of the Arm B table. These contain the baseline grid, soil and aquifer covariates, baseline zoning and the SSP5-8.5 mid-century zoning grid. |
| `data/basemap/` | Four Natural Earth public-domain vector layers in zipped shapefile format |
| `tables/` | 22 machine-readable result and validation tables (CSV or JSON) |
| `source_data/` | 69 panel CSVs and 13 Excel workbooks for Figures 1–5 and Extended Data Figures 1–8. Each workbook's Contents sheet maps worksheets to their source files. |
| `qa/` | The completed extraction checklist and summary, and the original manuscript-package quality report. Its package hashes describe that package at the time of inspection. |
| `outputs/` | Created by the commands below. Contains regenerated tables, figures, panels, workbooks and optional blank recoding forms. |

One family without a DOI is represented by a stable identifier in every affected family and shared-control field. `lat_dd` and `lon_dd` preserve the extracted coordinates. `analysis_lat_dd` and `analysis_lon_dd` give the coordinates used after study-level geographic reconciliation. Effect sizes, units, family membership, common controls and repeated observations are retained. The 202-row author audit covers 28 sampled study arms and 27 distinct DOIs. The recorded findings are 191 exact matches and 11 figure-digitization rounding differences.

## Environment

The reference run used CPython 3.14.8 and Node.js 26.0.0. The Excel entry point uses Node's standard library and the same Python environment as the analysis. Python package versions are pinned in `requirements.txt`. The examples use a POSIX shell from the repository root. A virtual environment with compatible geospatial wheels or the corresponding GDAL, GEOS and PROJ libraries is required.

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
export PYTHON="$(command -v python)"
export MPLBACKEND=Agg
```

## Level 1 — Models, summaries, figures and source workbooks

The released evidence, gridded inputs and basin covariates support an offline run. The code fits the Arm A thresholds and family-level tests, the Arm B clustered REML models and ET audit bootstrap, and the baseline closure and sensitivity calculations. Future figures and uncertainty summaries use the released per-model and per-threshold outputs. Level 2 recalculates those outputs from monthly climate forcings.

```sh
python code/pillar2_armA_ecw_threshold_model.py
python code/pillar2_armB_hydraulic_model.py
python code/pillar1b_salt_debt_and_closure_sensitivity.py
python code/pillar3_safe_windows_and_zoning.py
python code/water_threshold_sensitivity.py
python code/refresh_analysis_tables.py
python code/generate_main_figures_1_3.py
python code/generate_main_figures_4_5_ed.py
node code/build_figure_source_workbooks.mjs
python code/verify_all_pillars_and_gates.py
```

The Arm A run includes the harvest-only model, matched monitoring and family-level SAR checks. `python code/field_threshold_checks.py` runs that subset independently. Arm B recomputes the effects from the harmonized rows and the ET ratios from the 64-family audit. The archived source-selection flow, mapped trial coordinates and district observation summaries enter the displays as evidence. The two generators together write 13 PDFs and 13 PNGs to `outputs/figures/`. Source workbooks use regenerated panel CSVs when available. The verifier checks every indexed worksheet against its CSV, including identifier types and numeric precision.

For an optional extraction check, run the following from the same environment.

```sh
python code/validate_12_gates.py --arm A --input data/harmonized/armA_extracted_current.csv --output-csv outputs/qa/armA_validated.csv --output-report outputs/qa/armA_gates.json
python code/generate_10pct_human_audit_checklist.py
```

The second command generates a blank recoding form for the registered sample. The completed observations are preserved in `qa/`.

### Geographic coverage reported in two project outputs

Extended Data Figure 4 and its released site and PCA tables contain 486 Arm B site coordinates and 296 matched training cells. Recalculation from these coordinates places 2,614 of 2,783 evaluated cells inside the area of applicability (96.92% of evaluated irrigated area). The original Pillar 3 summary and manuscript caption refer to 295 training cells, 2,606 cells inside and 96.84% coverage. The repository preserves both original records. The regenerated full-evidence diagnostic and figure follow the released site coordinates. One ancillary geographic comparison, restricted to the original brackish-water screening subset, is carried forward from the archived summary because membership of that subset is unavailable in the public site table.

## Level 2 — Rebuild the spatial and climate analysis

Place the provider files in the layout and versions specified in `DATA_SOURCES.md`. Set `EXTERNAL_RAW_DATA_DIR` to their common directory, or use `external_raw_data/` at the repository root. Use the original 0.5° global-water land mask. The loaders use the original units, projections, variable names and basin ordering.

```sh
export EXTERNAL_RAW_DATA_DIR="/path/to/public-spatial-inputs"
python code/pillar1_global_leaching_paradox.py
python code/build_spatial_validation_and_soil_layers.py
python code/pillar2_armA_ecw_threshold_model.py
python code/pillar2_armB_hydraulic_model.py
python code/pillar1b_salt_debt_and_closure_sensitivity.py
python code/pillar3_safe_windows_and_zoning.py
python code/extract_gcm_uncertainty_table.py
python code/validate_observed_water_salt_budgets.py
python code/refresh_analysis_tables.py
python code/generate_main_figures_1_3.py
python code/generate_main_figures_4_5_ed.py
node code/build_figure_source_workbooks.mjs
python code/verify_all_pillars_and_gates.py
```

The 1991–2019 baseline uses 67,420 land cells. Five climate models, three pathways and two time horizons give 30 paired climate runs. Seven ECw thresholds are evaluated in each run (210 cases). The 2050 and 2080 windows cover 2040–2069 and 2070–2099. The salinity projections paired with the water-model pathways are specified in the source catalogue. Raw monthly model outputs, original study articles and working extraction materials are obtained from their respective providers.

## Citation and reuse

Cite Yu, S.-Y., Du, J. & Li, S., *Supply, soil hydraulic and consumptive limits govern the global leaching-water paradox*, code and harmonized data, v1.0.0. The version release links to the citable archive. [GitHub release](https://github.com/wlcbysy/global-leaching-water-paradox/releases/tag/v1.0.0)

Code is licensed under MIT. Author-created harmonized data and figure source data are licensed under CC BY 4.0. Embedded third-party components retain their original terms. [License](LICENSE) · [Dataset references and file specifications](DATA_SOURCES.md)

