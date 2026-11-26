"""Apply a visible list of corrections to your E2 review marks, then show the new score.

Read every line below. If you disagree with one, delete that line (or change it)
BEFORE running. Nothing is hidden: this file is the whole set of changes.

Run:  python scripts/e2_corrections.py
"""
import csv
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "data" / "derived" / "e2_wc_ch6_review.csv"
FIELDS = ["chunk_id", "span", "label", "score", "correct"]

# Entities to change from 1 (right) to 0 (wrong): (chunk_id, span)
MARK_WRONG = [
    ("wc_ch6-0044", "major powers"),      # generic phrase, not one specific organization
    ("wc_ch6-0044", "nuclear powers"),    # generic phrase
    ("wc_ch6-0047", "Communists"),        # a general group of people, not a specific organization
    ("wc_ch6-0052", "visa"),              # a generic word, not a specific named document
    ("wc_ch6-0071", "Soviet official"),   # unnamed, generic person
    ("wc_ch6-0015", "Irving and Dallas"), # borderline: two places merged into one span
]

# Clear named things the model did NOT find, counted once per distinct entity.
MISSED = {
    "wc_ch6-0012": 1,  # Oswald (never tagged in this chunk)
    "wc_ch6-0013": 1,  # Depository Co.
    "wc_ch6-0016": 2,  # the President; Friday
    "wc_ch6-0047": 3,  # United States; Far East; 1959
}


def main():
    with open(CSV_PATH, newline="", encoding="utf-8-sig") as f:
        rows = [{k: r.get(k, "") for k in FIELDS} for r in csv.DictReader(f)]

    changed = 0
    for chunk_id, span in MARK_WRONG:
        for row in rows:
            if (row["chunk_id"] == chunk_id and row["span"].strip().lower() == span.lower()
                    and row["correct"].strip() != "m"):
                if row["correct"].strip() != "0":
                    row["correct"] = "0"
                    changed += 1
                break
        else:
            print(f"Could not find: {chunk_id} / {span}")

    # Replace any earlier 'missed' rows for these chunks, so re-running is safe.
    rows = [r for r in rows if not (r["correct"].strip().lower() == "m" and r["chunk_id"] in MISSED)]
    for chunk_id, count in MISSED.items():
        for _ in range(count):
            rows.append({"chunk_id": chunk_id, "span": "(missed)", "label": "",
                         "score": "", "correct": "m"})

    with open(CSV_PATH, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Changed {changed} marks to wrong; recorded {sum(MISSED.values())} missed entities.")
    print("Now run: python scripts\\e2_ner.py score")


if __name__ == "__main__":
    main()