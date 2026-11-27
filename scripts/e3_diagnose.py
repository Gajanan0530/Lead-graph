"""Show WHY claims failed the automatic checks, so we fix the right thing.

Run:  python scripts/e3_diagnose.py
"""
import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "derived"
CHECKS = ["evidence_in_text", "subject_in_text", "object_in_text", "linked_to_evidence"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="wc_ch6")
    parser.add_argument("--show", type=int, default=12, help="how many failed claims to print")
    args = parser.parse_args()

    claims = json.loads((OUT_DIR / f"e3_{args.source}_claims.json").read_text(encoding="utf-8"))
    failed = [c for c in claims if not c["checks"]["grounded"]]
    print(f"{len(claims)} claims, {len(failed)} failed at least one check\n")

    print("Which checks failed (a claim can fail several):")
    for name in CHECKS:
        print(f"  {name}: {sum(not c['checks'][name] for c in failed)}")

    patterns = Counter(tuple(n for n in CHECKS if not c["checks"][n]) for c in failed)
    print("\nMost common failure patterns:")
    for pattern, count in patterns.most_common(5):
        print(f"  {count:>3} x {', '.join(pattern)}")

    print(f"\nFirst {args.show} failed claims:")
    for c in failed[:args.show]:
        bad = [n for n in CHECKS if not c["checks"][n]]
        print(f"- [{c['chunk_id']}] {c['subject']} | {c['predicate']} | {c['object']}")
        print(f"    failed: {', '.join(bad)}")
        print(f"    quote: {c['evidence'][:140]}")


if __name__ == "__main__":
    main()