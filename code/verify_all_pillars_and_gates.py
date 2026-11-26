#!/usr/bin/env python3
"""
End-to-End Verification & Sanity Audit Suite.
Validates harmonized evidence, statistical and spatial outputs,
three-tier water conservation constraints, Arm A/B consistency, Salt Debt metadata, and independent hydrological comparisons.
"""

from __future__ import annotations

from config import HARMONIZED, table_input, grid_input, prepare_outputs

import json
import math
import re
import csv
from itertools import zip_longest
from openpyxl import load_workbook
from config import SOURCE_DATA, panel_input, OUT_PANELS
from typing import List

import pandas as pd



def same_value(raw, value, header):
    if raw == '':
        return value is None
    if isinstance(value,bool):
        return raw==str(value)
    if isinstance(value,(int,float)):
        if re.search(r'(?:^|_)(?:id|doi|code|identifier)(?:$|_)',header,re.I):
            return False
        try:
            return math.isclose(float(raw),value,rel_tol=1e-13,abs_tol=1e-15)
        except ValueError:
            return False
    return raw==value


def verify_workbook(path):
    book=load_workbook(path,read_only=True,data_only=False)
    index=book['Contents']
    assert 'Source Data | ' in index['A1'].value
    index_rows=list(index.values)[4:]
    expected={p.name for p in SOURCE_DATA.glob(path.stem.removeprefix('Source_Data_')+'*.csv')}
    assert {r[2] for r in index_rows}==expected
    cells=records=0
    for panel,sheet,src,count in index_rows:
        data=book[sheet].values
        source_path = SOURCE_DATA / src if path.parent == SOURCE_DATA else panel_input(src)
        with source_path.open(encoding="utf-8-sig", newline='') as f:
            lines=csv.reader(f)
            header=next(lines)
            assert list(next(data))==header,(src,'header')
            n=0
            for n,(raw,stored) in enumerate(zip_longest(lines,data),1):
                assert raw is not None and stored is not None and len(raw)==len(stored),(src,n)
                for col,(val,actual) in enumerate(zip(raw,stored)):
                    assert same_value(val,actual,header[col]),(src,n,header[col],val,actual)
                cells+=len(raw)
            assert n==count,(src,n,count)
            records+=n
    book.close()
    return dict(worksheets=1+len(index_rows),panel_tables=len(index_rows),data_rows=records,verified_data_cells=cells)


def main() -> None:
    failures: List[str] = []

    arma = pd.read_csv(HARMONIZED / "armA_extracted_current.csv", low_memory=False)
    if not arma.qc_gate_status.isin(["PASS_READY", "PASS_NO_VARIANCE"]).any():
        failures.append("Arm A has no eligible contrasts")

    p2a_merged = grid_input("pillar2_armB_merged_current.csv")
    if p2a_merged.is_file():
        m_df = pd.read_csv(p2a_merged, low_memory=False)
        if (m_df["metric_bucket"] == "cumulative_infiltration_depth_isolated").any():
            failures.append("Gate 10 violation: cumulative_infiltration_depth_isolated leaked into pillar2_armB_merged_current.csv")

    # 4. Pillar 1B & Pillar 2 checks
    p1b_path = table_input("pillar1b_salt_debt_closure_sensitivity_current.json")
    if not p1b_path.is_file():
        failures.append("Missing pillar1b_salt_debt_closure_sensitivity_current.json")
        p1b_json = {}
    else:
        p1b_json = json.loads(p1b_path.read_text(encoding="utf-8"))
        if "salt_debt_definition" not in p1b_json.get("salt_debt_accounting", {}):
            failures.append("pillar1b JSON missing salt_debt_definition under salt_debt_accounting")

    p2a_json = json.loads((table_input("pillar2_armB_hydraulic_results_current.json")).read_text(encoding="utf-8"))
    for r in p2a_json.get("headline_all_endpoints", []):
        if r.get("status") == "OK" and not math.isfinite(r.get("mu_lnrr", float("nan"))):
            failures.append(f"Non-finite REML estimate in Pillar 2A for {r.get('endpoint')}")

    p2b_json = json.loads((table_input("pillar2_armA_ecw_results_current.json")).read_text(encoding="utf-8"))
    if p2b_json.get("usable_rows_total", 0) <= 0:
        failures.append("Pillar 2B usable_rows_total <= 0")
    dose_abs = p2b_json.get("dose_response_absolute", {})
    for sub_k in ["pooled", "continuous", "cyclic", "field_only", "exact_variance", "exact_plus_lsd", "grain_only", "wue"]:
        ci = (dose_abs.get(sub_k) or {}).get("ecw_star_90pct_ci95")
        if not (isinstance(ci, list) and len(ci) == 2 and all(math.isfinite(float(x)) for x in ci)):
            failures.append(f"Arm A dose_response_absolute.{sub_k} missing finite ecw_star_90pct_ci95")
    for sub_k in ["pooled", "continuous"]:
        hb = (dose_abs.get(sub_k) or {}).get("ecw90_hit_bound_pct", 999.0)
        if float(hb) >= 20.0:
            failures.append(f"Arm A {sub_k} ecw90_hit_bound_pct={hb} >= 20%")

    # 5. Pillar 3 three-tier accounting, ECw* consistency & Three-Zone conservation
    p3_json = json.loads((table_input("pillar3_global_summary_current.json")).read_text(encoding="utf-8"))
    gs = p3_json["global_salt_affected_croplands_savings"]
    ep = p3_json["empirical_parameters_used"]
    wat = p3_json["water_accounting_three_tier"]

    B_val = float(gs["gross_brackish_substitution_km3_yr"])
    P_val = float(gs["leaching_penalty_P_km3_yr"])
    dF_val = float(gs["net_brackish_fresh_saving_deltaF_km3_yr"])
    if dF_val > B_val + 1e-6 or P_val < -1e-6:
        failures.append(f"Brackish mass-balance violation: deltaF={dF_val}, B={B_val}, P={P_val}")
    if float(wat["consumption_tier_km3_yr"]["return_flow_basin_credit"]) != 0.0 or float(gs["return_flow_basin_credit_km3_yr"]) != 0.0:
        failures.append("Return-flow basin credit must equal 0.0")
    if float(ep["ecw_star_baseline_used_dsm"]) != float(dose_abs["continuous"]["ecw_star_90pct_dsm"]):
        failures.append("Pillar 3 ecw_star_baseline_used_dsm does not match Arm A continuous ecw_star_90pct_dsm")
    if float(ep["ecw_star_managed_cyclic_used_dsm"]) != float(dose_abs["cyclic"]["ecw_star_90pct_dsm"]):
        failures.append("Pillar 3 ecw_star_managed_cyclic_used_dsm does not match Arm A cyclic ecw_star_90pct_dsm")

    tot_mha = gs["salt_irrig_mha"]
    z_sum = sum(v["salt_irrig_mha"] for v in p3_json["three_zone_partitioning"].values())
    if abs(z_sum - tot_mha) > 0.25 or abs(tot_mha - 97.12) > 0.25:
        failures.append(f"Baseline zone area mismatch: sum={z_sum} vs total={tot_mha} vs expected 97.12")

    for fs in p3_json.get("future_zoning_2050_2080", []):
        fz = sum(v["salt_irrig_mha"] for v in fs.get("zone_partitioning", {}).values())
        if abs(fz - fs["salt_irrig_mha"]) > 0.25:
            failures.append(f"Future {fs.get('scenario')} zone area mismatch: sum={fz} vs total={fs['salt_irrig_mha']}")

    # 5b. Stage 1-3 Risk Decomposition, Audited 64-Family ET, Dual Closure, Observed Ledger, 7-Cluster Leaching, ECe vs EC1:5
    rb = p3_json.get("risk_intersection_breakdown_mha", {})
    z3_mha = float(p3_json["three_zone_partitioning"]["Zone_III_Water_Salt_Dual_Redline"]["salt_irrig_mha"])
    z1_mha = float(p3_json["three_zone_partitioning"]["Zone_I_Safe_Brackish_Substitution_Window"]["salt_irrig_mha"])
    if abs(rb.get("zone3_bw_only_mha", 0) + rb.get("zone3_sal_arid_only_mha", 0) + rb.get("zone3_dual_bw_and_sal_arid_mha", 0) - z3_mha) > 0.05:
        failures.append("Risk intersection Zone III decomposition does not sum to Zone III total")
    if abs(rb.get("zone1_safe_window_gw_sustainable_mha", 0) + rb.get("zone1_local_potential_gw_constrained_mha", 0) - z1_mha) > 0.05:
        failures.append("Zone I GW-sustainable vs GW-constrained areas do not sum to Zone I total")
    rb_act = p3_json.get("risk_intersection_actual_et_mha", {})
    if abs(rb_act.get("zone3_bw_only_mha", 0) + rb_act.get("zone3_sal_arid_only_mha", 0) + rb_act.get("zone3_dual_bw_and_sal_arid_mha", 0) - rb_act.get("zone3_at_least_one_risk_total_mha", -1)) > 0.05:
        failures.append("Actual-ET risk intersection decomposition failed conservation")

    dual_c = p3_json.get("dual_scenario_basin_closure", {})
    act_c = dual_c.get("actual_management_observed_et", {})
    cap_c = dual_c.get("equal_production_water_capped", {})
    if act_c.get("relieved_mha") != 0.0 or not all(math.isfinite(float(x)) for x in act_c.get("net_basin_depletion_relief_ci95", [])):
        failures.append("Actual-management dual closure missing finite CI or nonzero relieved_mha when 0 rescued")
    if not all(math.isfinite(float(x)) for x in cap_c.get("net_basin_depletion_relief_ci95", [])):
        failures.append("Equal-production dual closure missing finite net_basin_depletion_relief_ci95")

    audit64_p = table_input("armB_wue_et_64families_audit_summary.json")
    if not audit64_p.is_file():
        failures.append("Missing armB_wue_et_64families_audit_summary.json")
    else:
        a64 = json.loads(audit64_p.read_text(encoding="utf-8"))
        if a64.get("total_families_audited") != 64 or a64.get("verified_consumptive_ET_combined_AB", {}).get("n_families") != 18:
            failures.append("64-family WUE/ET audit mismatch (expected 64 total, 18 verified A+B)")

    leach7_p = HARMONIZED / "armB_leaching_7clusters_audit_current.csv"
    if not leach7_p.is_file() or len(pd.read_csv(leach7_p)) != 33:
        failures.append("armB_leaching_7clusters_audit_current.csv missing or != 33 rows")
    if p2a_json.get("leaching_efficiency_stratified", {}).get("direct_salt_export_or_storage_only", {}).get("n_clusters") != 6:
        failures.append("Arm B direct_salt_export_or_storage_only expected 6 clusters")

    sec_abs = p2b_json.get("soil_EC_absolute", {})
    for sk in ["saturated_paste_ECe_only", "extract_1_5_only"]:
        if not math.isfinite(float((sec_abs.get(sk) or {}).get("power_exponent_b", float("nan")))):
            failures.append(f"Missing Arm A native soil_EC_absolute.{sk}")

    obs_p = table_input("observed_district_basin_water_salt_ledger_current.json")
    obs_csv_p = table_input("observed_district_basin_water_salt_ledger_current.csv")
    if not (obs_p.is_file() and obs_csv_p.is_file()):
        failures.append("Missing observed_district_basin_water_salt_ledger_current JSON/CSV")
    else:
        obs_d = json.loads(obs_p.read_text(encoding="utf-8"))
        u_b = obs_d.get("usgs_ucrb_benchmark", {})
        gl = obs_d.get("global_two_boundary_salt_ledger", {})
        if u_b.get("total_stations") != 205 or u_b.get("thorslund_overlap_audit", {}).get("within_005deg_count") != 197:
            failures.append("USGS UCRB station/overlap audit count mismatch")
        s_comp = (gl.get("delta_s_field_recycled_gross_mt_yr") or 0) + (gl.get("delta_s_basin_gw_gross_mt_yr") or 0) + (gl.get("delta_s_fresh_displaced_mt_yr") or 0)
        if abs(s_comp - (gl.get("field_boundary_added_salt_load_mt_yr") or 0)) > 0.1:
            failures.append(f"Two-boundary salt decomposition sum mismatch: {s_comp} vs {gl.get('field_boundary_added_salt_load_mt_yr')}")

    # 6. Plan A + B spatial validation & supply sensitivity audit
    val_p = table_input("hydrological_independent_validation_current.json")
    sens_p = table_input("pillar3_supply_sensitivity_current.csv")
    grid_val_p = grid_input("public_validation_soil_aquifer_grid05_current.parquet")
    if not val_p.is_file():
        failures.append("Missing hydrological_independent_validation_current.json")
        val_d = {}
    else:
        val_d = json.loads(val_p.read_text(encoding="utf-8"))
        gr = val_d.get("grace_csr_validation_376_basins", {})
        ja = val_d.get("jasechko_2024_1693_aquifer_systems_validation", {})
        dq = val_d.get("dynqual_v1_global_salinity_coverage", {})
        mg = val_d.get("isimip3a_multighm_structural_validation", {})
        if gr.get("basins_with_grace", 0) != 376 or gr.get("mannwhitney_less_p", 1.0) >= 0.001:
            failures.append("GRACE validation check failed")
        if ja.get("basins_matched_376", 0) < 150 or ja.get("mannwhitney_greater_p", 1.0) >= 0.001:
            failures.append("Jasechko 1,693-aquifer global validation check failed")
        if dq.get("salt_area_covered_pct", 0.0) < 98.0:
            failures.append("DynQual v1.0 global salinity coverage check failed")
        if mg.get("n_ghms", 0) < 4:
            failures.append("ISIMIP3a 4-GHM structural validation check failed")
    if not sens_p.is_file() or len(pd.read_csv(sens_p)) < 12:
        failures.append("Missing or incomplete pillar3_supply_sensitivity_current.csv")
    if not grid_val_p.is_file():
        failures.append("Missing public_validation_soil_aquifer_grid05_current.parquet")

    archived_books = sorted(SOURCE_DATA.glob("Source_Data_*.xlsx"))
    if len(archived_books) != 13:
        failures.append("Expected 13 figure workbooks")
    book_counts = []
    for archived in archived_books:
        regenerated = OUT_PANELS / archived.name
        # A regenerated workbook uses regenerated panel CSVs where available.
        # The archived workbook always belongs to the archived source CSVs.
        book_counts.append(verify_workbook(regenerated if regenerated.is_file() else archived))
    if sum(row["panel_tables"] for row in book_counts) != 69:
        failures.append("Expected 69 workbook-to-CSV comparisons")
    report = {
        "panel_tables_checked": sum(row["panel_tables"] for row in book_counts),
        "workbook_data_cells_checked": sum(row["verified_data_cells"] for row in book_counts),
        "pillar2A_total_rows": p2a_json.get("total_merged_rows"),
        "pillar2B_usable_rows": p2b_json.get("usable_rows_total"),
        "pillar2B_usable_studies": p2b_json.get("usable_studies_total"),
        "pillar2B_countries": len(p2b_json.get("countries_represented", {})),
        "pillar3_baseline_total_mha": tot_mha,
        "pillar3_baseline_zone_sum_mha": round(z_sum, 2),
        "pillar3_gross_brackish_B_km3_yr": B_val,
        "pillar3_leaching_penalty_P_km3_yr": P_val,
        "pillar3_net_brackish_deltaF_km3_yr": dF_val,
        "pillar3_hydraulic_cons_km3_yr": gs.get("hydraulic_recovery_cons_km3_yr"),
        "pillar3_relieved_basins": gs.get("paradox_basins_relieved_to_blue_feasible"),
        "pillar3_future_scenarios_count": len(p3_json.get("future_zoning_2050_2080", [])),
        "grace_basins_verified": val_d.get("grace_csr_validation_376_basins", {}).get("basins_with_grace"),
        "jasechko_matched_wells": val_d.get("jasechko_2024_well_validation", {}).get("total_matched_wells"),
        "failures": failures,
        "passed": len(failures) == 0,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    prepare_outputs()
    main()
