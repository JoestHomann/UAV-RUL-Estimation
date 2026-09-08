"""Write ../evidence_ledger.csv from evidence_claims_source.csv.

One row per numerical claim or substantive empirical conclusion in
UAV_RUL_Project_20min.pptx. ``evidence_claims_source.csv`` is the editable
source of truth; this script validates it and copies it into place, so the
ledger and the deck cannot drift apart silently.

``status`` is one of:
  verified   - recomputed here, or read directly from the named artifact key
  documented - taken from project documentation; no separate artifact key was
               reachable for it in this session
  unverified - reported by a person, with no artifact establishing provenance
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "evidence_claims_source.csv"
OUTPUT = HERE.parent / "evidence_ledger.csv"

FIELDS = ["claim_id", "slide_number", "claim", "source_path",
          "table_row_or_json_key", "experiment_identity", "data_scope",
          "target_policy", "evaluation_support", "aggregation", "comparator",
          "status", "limitation"]

VALID_STATUS = {"verified", "documented", "unverified"}
SLIDE_ORDER = {str(number): number for number in range(1, 15)}
SLIDE_ORDER.update({f"B{number}": 100 + number for number in range(1, 7)})


def main():
    rows = list(csv.DictReader(SOURCE.open(encoding="utf-8")))
    if not rows:
        raise SystemExit("evidence_claims_source.csv is empty")

    seen = set()
    previous = 0
    for index, row in enumerate(rows, start=1):
        if list(row) != FIELDS:
            raise SystemExit(f"row {index}: unexpected fields {list(row)}")
        expected = f"C{index:02d}"
        if row["claim_id"] != expected:
            raise SystemExit(f"row {index}: claim_id {row['claim_id']!r} "
                             f"should be {expected!r}")
        if row["claim_id"] in seen:
            raise SystemExit(f"duplicate claim id {row['claim_id']}")
        seen.add(row["claim_id"])
        if row["slide_number"] not in SLIDE_ORDER:
            raise SystemExit(f"row {index}: unknown slide {row['slide_number']}")
        if SLIDE_ORDER[row["slide_number"]] < previous:
            raise SystemExit(f"row {index}: slides are out of order")
        previous = SLIDE_ORDER[row["slide_number"]]
        if row["status"] not in VALID_STATUS:
            raise SystemExit(f"row {index}: bad status {row['status']!r}")
        if not row["claim"].strip() or not row["source_path"].strip():
            raise SystemExit(f"row {index}: empty claim or source path")

    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    groups = defaultdict(list)
    for row in rows:
        groups[row["slide_number"]].append(row["claim_id"])
    print(f"wrote {OUTPUT} with {len(rows)} claims")
    for slide in sorted(groups, key=lambda value: SLIDE_ORDER[value]):
        ids = groups[slide]
        print(f"  slide {slide:>3}: {ids[0]}-{ids[-1]} ({len(ids)})")


if __name__ == "__main__":
    main()
