"""Reconstruct a blank recoding form from the published stratified audit sample.

The sampling frame and the authors' completed assessment are recorded in qa/.
This entry point verifies the registration and writes a blank form to outputs/qa/.
"""

from config import QA, OUTPUT, prepare_outputs
import json
import pandas as pd


def main():
    rows = pd.read_csv(QA / "extraction_10pct_human_audit_checklist_current.csv", dtype=str, keep_default_na=False)
    summary = json.loads((QA / "extraction_10pct_human_audit_summary_current.json").read_text())
    totals = summary["totals"]
    signs = rows.numeric_match_status.value_counts()
    if len(rows) != totals["total_contrasts_to_audit"] or rows.doi_norm.nunique() != totals["unique_dois_audited"]:
        raise ValueError("Published audit sample counts differ from its registration")
    if signs.get("EXACT", 0) != 191 or signs.get("MINOR_ROUNDING", 0) != 11:
        raise ValueError("Published audit outcomes differ from the recorded checks")
    for column in rows:
        if column.startswith("verified_") or column in {"audit_notes", "country_match_status", "numeric_match_status", "auditor_initials"}:
            rows[column] = ""
    dest = OUTPUT / "qa" / "recoding_form.csv"
    rows.to_csv(dest, index=False)
    print(f"Prepared {len(rows)} rows from {rows.doi_norm.nunique()} DOIs")


if __name__ == "__main__":
    prepare_outputs()
    main()
