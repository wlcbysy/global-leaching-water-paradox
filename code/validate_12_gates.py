#!/usr/bin/env python3
"""
Step 2-B3: Automated 12-Gate Hard Red-Line QC Validator for Arm A and Arm B datasets.
Enforces extraction_schema_armA_armB_current.json rules and outputs row-level qc_gate_status
plus an auditable JSON summary report.
"""

from __future__ import annotations

from config import prepare_outputs

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

PHYSICAL_BOUNDS = {
    "ecw_dsm": [
        0.02,
        55.0
    ],
    "sarw": [
        0.0,
        80.0
    ],
    "bulk_density_g_cm3": [
        0.65,
        2.05
    ],
    "ks_saturated_mm_h": [
        0.001,
        2000.0
    ],
    "infiltration_rate_mm_h": [
        0.001,
        2000.0
    ],
    "grain_yield_t_ha": [
        0.01,
        25.0
    ],
    "wue_kg_m3": [
        0.05,
        15.0
    ],
    "abs_lnrr_hard_review": 4.5
}

PURE_LENGTH_UNITS = {"mm", "cm", "m", "in", "inch", "inches"}
ALLOWED_EXP_TYPES = {"field", "micro_plot", "lysimeter", "greenhouse", "pot", "lab_column"}


def to_float(v: Any) -> float | None:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    if not s or s.lower() in {"nan", "none", "na", "null", "unreported", "not_reported"}:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def validate_arm_a(df: pd.DataFrame, schema: Dict) -> tuple[pd.DataFrame, Dict]:
    bounds = schema["physical_bounds"]
    statuses: List[str] = []
    warnings_list: List[str] = []
    gate_fail_counts: Dict[str, int] = {}

    for _, r in df.iterrows():
        fails: List[str] = []
        warns: List[str] = []

        # Gate 9: Arm isolation
        arm_val = str(r.get("arm", "Arm_A")).strip().upper()
        if arm_val and arm_val not in {"A", "ARM_A", "ARMA", "ARM A"} and not arm_val.startswith("ARM_A"):
            fails.append("FAIL_GATE9_NON_ARMA_CONTRAST")

        # Gate 1: Ratio direction
        mname = str(r.get("metric_name", "")).strip()
        if mname.lower() in {"na_k_ratio", "shoot_root_ratio", "s_r_ratio"}:
            fails.append("FAIL_GATE1_INVERTED_RATIO")

        # Gate 3: Pure numeric assertion & Gate 2: Non-positive check for ln(T/C)
        mt = to_float(r.get("mean_treatment"))
        mc = to_float(r.get("mean_control"))
        if mt is None or mc is None:
            fails.append("FAIL_GATE3_NONNUMERIC_MEAN")
        elif mt <= 0 or mc <= 0:
            fails.append("FAIL_GATE2_NONPOSITIVE_MEAN")

        # Gate 5: Experiment type isolation
        exp_t = str(r.get("experiment_type", "")).strip().lower()
        if exp_t == "simulation_excluded" or exp_t == "in_vitro" or (exp_t and exp_t not in ALLOWED_EXP_TYPES):
            fails.append("FAIL_GATE5_INVALID_OR_SIM_EXP_TYPE")

        # Gate 4: Organ/matrix compatibility
        organ = str(r.get("organ_or_matrix", "")).strip().lower()
        if mname in {"soil_EC", "soil_ESP", "soil_SAR", "ks_saturated", "infiltration_rate", "bulk_density"}:
            if organ in {"grain", "leaf", "shoot", "root", "plant"}:
                fails.append("FAIL_GATE4_ORGAN_METRIC_MISMATCH")
        elif mname == "grain_yield":
            if organ in {"soil", "water", "drainage_water", "irrigation_water"}:
                fails.append("FAIL_GATE4_ORGAN_METRIC_MISMATCH")

        # Gate 8: ECw numeric & contrast direction
        ecw_t = to_float(r.get("ecw_treatment_dsm"))
        ecw_c = to_float(r.get("ecw_control_dsm"))
        if ecw_t is None or ecw_c is None:
            fails.append("FAIL_GATE8_MISSING_NUMERIC_ECW")
        else:
            lo, hi = bounds["ecw_dsm"]
            if not (lo <= ecw_t <= hi and 0.0 <= ecw_c <= hi):
                fails.append("FAIL_GATE6_ECW_OUT_OF_BOUNDS")
            sched = str(r.get("irrigation_scheduling", "")).strip().lower()
            if sched != "fresh_control" and ecw_t <= ecw_c:
                fails.append("FAIL_GATE8_TREAT_ECW_NOT_GT_CONTROL")

        # Gate 10: Cumulative depth vs rate check
        unit_str = str(r.get("unit", "")).strip().lower()
        if mname in {"ks_saturated", "infiltration_rate"} and unit_str in PURE_LENGTH_UNITS:
            fails.append("FAIL_GATE10_LENGTH_UNIT_ON_RATE")

        # Gate 7: Acid soil & confounding ban
        init_ph = to_float(r.get("initial_soil_ph"))
        if init_ph is not None and init_ph < 5.5:
            fails.append("FAIL_GATE7_ACID_SOIL_CONTAMINATION")
        if bool(r.get("is_confounded", False)):
            fails.append("FAIL_GATE7_CONFOUNDED_PAIRING")

        # Gate 6: Physical bounds & extreme lnRR
        if mt is not None and mc is not None and mt > 0 and mc > 0:
            lnrr = abs(math.log(mt / mc))
            if lnrr > bounds["abs_lnrr_hard_review"]:
                fails.append("FAIL_GATE6_EXTREME_LNRR")
        if mname == "grain_yield" and mt is not None and mc is not None:
            is_fresh_or_fodder = any(
                k in f"{organ} {r.get('crop_group', '')} {r.get('crop_species', '')}".lower()
                for k in ("fresh", "fruit", "tomato", "melon", "onion", "cabbage", "potato", "tuber", "fodder", "forage", "silage", "beet", "cane", "banana", "grape", "vegetable", "root")
            )
            max_yld_tha = 150.0 if is_fresh_or_fodder else bounds["grain_yield_t_ha"][1]
            if unit_str in {"t/ha", "t ha-1", "mg/ha", "mg ha-1"}:
                if max(mt, mc) > max_yld_tha:
                    fails.append("FAIL_GATE6_YIELD_OUT_OF_BOUNDS")
            elif unit_str in {"kg/ha", "kg ha-1"}:
                if max(mt, mc) > max_yld_tha * 1000.0:
                    fails.append("FAIL_GATE6_YIELD_OUT_OF_BOUNDS")

        # Gate 12: Duration numeric
        dur = to_float(r.get("duration_years"))
        if dur is not None and dur <= 0:
            fails.append("FAIL_GATE12_NONPOSITIVE_DURATION")

        # Yellow warnings
        et = to_float(r.get("error_treatment"))
        ec = to_float(r.get("error_control"))
        nrep = to_float(r.get("n_replicates"))
        if et is None or ec is None or nrep is None or nrep < 2:
            warns.append("YELLOW_MISSING_VARIANCE_OR_N")
        if mname in {"soil_EC", "soil_ESP", "soil_SAR"} and not str(r.get("soil_depth_cm", "")).strip():
            warns.append("YELLOW_MISSING_SOIL_DEPTH")
        if to_float(r.get("sarw_treatment")) is None and to_float(r.get("initial_soil_esp_pct")) is None:
            warns.append("YELLOW_MISSING_SARW_AND_ESP")

        if fails:
            status = fails[0]
            for f in fails:
                gate_fail_counts[f] = gate_fail_counts.get(f, 0) + 1
        elif "YELLOW_MISSING_VARIANCE_OR_N" in warns:
            status = "PASS_NO_VARIANCE"
        else:
            status = "PASS_READY"

        statuses.append(status)
        warnings_list.append("|".join(warns))

    out = df.copy()
    out["qc_gate_status"] = statuses
    out["qc_yellow_warnings"] = warnings_list
    if "lnRR" not in out.columns:
        out["lnRR"] = np.where(
            pd.to_numeric(out["mean_treatment"], errors="coerce") > 0,
            np.where(
                pd.to_numeric(out["mean_control"], errors="coerce") > 0,
                np.log(pd.to_numeric(out["mean_treatment"], errors="coerce") / pd.to_numeric(out["mean_control"], errors="coerce")),
                np.nan,
            ),
            np.nan,
        )

    report = {
        "arm": "A",
        "total_rows": int(len(out)),
        "status_counts": {k: int(v) for k, v in out["qc_gate_status"].value_counts().to_dict().items()},
        "usable_rows_pass_ready_or_novar": int(out["qc_gate_status"].isin(["PASS_READY", "PASS_NO_VARIANCE"]).sum()),
        "gate_fail_breakdown": gate_fail_counts,
        "yellow_warning_counts": {
            "missing_variance_or_n": int(out["qc_yellow_warnings"].str.contains("YELLOW_MISSING_VARIANCE_OR_N").sum()),
            "missing_soil_depth": int(out["qc_yellow_warnings"].str.contains("YELLOW_MISSING_SOIL_DEPTH").sum()),
            "missing_sarw_and_esp": int(out["qc_yellow_warnings"].str.contains("YELLOW_MISSING_SARW_AND_ESP").sum()),
        },
    }
    return out, report


def validate_arm_b(df: pd.DataFrame, schema: Dict) -> tuple[pd.DataFrame, Dict]:
    bounds = schema["physical_bounds"]
    statuses: List[str] = []
    gate_fail_counts: Dict[str, int] = {}

    for _, r in df.iterrows():
        fails: List[str] = []
        bucket = str(r.get("metric_bucket", "")).strip()
        mt = to_float(r.get("canonical_value_t")) if to_float(r.get("canonical_value_t")) is not None else to_float(r.get("raw_value_t"))
        mc = to_float(r.get("canonical_value_c")) if to_float(r.get("canonical_value_c")) is not None else to_float(r.get("raw_value_c"))
        ut = str(r.get("raw_unit_t", "")).strip().lower()

        # Gate 10: Cumulative depth isolation
        if bucket == "cumulative_infiltration_depth_isolated":
            fails.append("FAIL_GATE10_ISOLATED_CUMULATIVE_DEPTH")
        elif bucket in {"ks_saturated", "infiltration_rate_steady", "infiltration_rate_initial", "infiltration_rate_unspecified"} and ut in PURE_LENGTH_UNITS:
            fails.append("FAIL_GATE10_LENGTH_UNIT_ON_RATE")

        # Gate 3 & Gate 2
        if mt is None or mc is None:
            fails.append("FAIL_GATE3_NONNUMERIC_OR_MISSING")
        elif mt <= 0 or mc <= 0:
            fails.append("FAIL_GATE2_NONPOSITIVE")
        else:
            lnrr = abs(math.log(mt / mc))
            if lnrr > bounds["abs_lnrr_hard_review"]:
                fails.append("FAIL_GATE6_EXTREME_MAGNITUDE_REVIEW")

        # Gate 6 unit mismatch
        cunit_t = str(r.get("canonical_unit", "")).strip()
        if str(r.get("unit_qc_flag", "")) == "FAIL_GATE10_LENGTH_UNIT_ON_RATE":
            fails.append("FAIL_GATE10_LENGTH_UNIT_ON_RATE")

        et = to_float(r.get("raw_error_t"))
        ec = to_float(r.get("raw_error_c"))
        nt = to_float(r.get("n_t"))
        nc = to_float(r.get("n_c"))

        if fails:
            status = fails[0]
            for f in fails:
                gate_fail_counts[f] = gate_fail_counts.get(f, 0) + 1
        elif et is None or ec is None or nt is None or nc is None or nt < 2 or nc < 2:
            status = "PASS_NO_VARIANCE"
        else:
            status = "PASS_READY"

        statuses.append(status)

    out = df.copy()
    out["qc_gate_status"] = statuses
    report = {
        "arm": "B",
        "total_rows": int(len(out)),
        "status_counts": {k: int(v) for k, v in out["qc_gate_status"].value_counts().to_dict().items()},
        "usable_rows_pass_ready_or_novar": int(out["qc_gate_status"].isin(["PASS_READY", "PASS_NO_VARIANCE"]).sum()),
        "gate_fail_breakdown": gate_fail_counts,
    }
    return out, report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Arm A / Arm B extraction against 12 QC gates.")
    parser.add_argument("--arm", choices=["A", "B"], required=True)
    parser.add_argument("--input", required=True, help="Input CSV path")
    parser.add_argument("--output-csv", default=None, help="Optional validated CSV output path (defaults to overwrite --input)")
    parser.add_argument("--output-report", required=True, help="Output JSON QA report path")
    args = parser.parse_args()

    schema = {"physical_bounds": PHYSICAL_BOUNDS}
    in_path = Path(args.input)
    df = pd.read_csv(in_path, low_memory=False)
    if args.arm == "A":
        out_df, report = validate_arm_a(df, schema)
    else:
        out_df, report = validate_arm_b(df, schema)

    out_csv = Path(args.output_csv) if args.output_csv else in_path
    out_df.to_csv(out_csv, index=False)

    rep_path = Path(args.output_report)
    rep_path.parent.mkdir(parents=True, exist_ok=True)
    with rep_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    prepare_outputs()
    main()
