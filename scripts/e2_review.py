"""Fast hand-review of the E2 entity sheet, one chunk at a time, in the terminal.

Run:  python scripts/e2_review.py
For each chunk you see the text and a numbered list of the entities the model
found. You type the numbers of the WRONG ones (or just press Enter if all are
right), then how many clear entities the model MISSED. Your answers are saved
into the same CSV that `e2_ner.py score` reads. You can quit with q and resume
later; finished chunks are skipped.
"""
import argparse
import csv
import re
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "derived"
FIELDS = ["chunk_id", "span", "label", "score", "correct"]


def load_chunk_texts(path):
    """Read the review_chunks file into {chunk_id: text}."""
    texts, current = {}, None
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^=== (\S+) ===$", line)
        if match:
            current = match.group(1)
            texts[current] = []
        elif current is not None and line.strip():
            texts[current].append(line.strip())
    return {k: " ".join(v) for k, v in texts.items()}


def save(path, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def ask_wrong(count):
    """Ask which entity numbers are wrong. Returns a set, or None to quit."""
    while True:
        answer = input("Numbers of WRONG entities (space-separated), Enter = all right, q = save & quit: ").strip().lower()
        if answer == "q":
            return None
        try:
            numbers = {int(x) for x in answer.replace(",", " ").split()}
        except ValueError:
            print("  Please type numbers only, like: 3 7 12")
            continue
        if any(n < 1 or n > count for n in numbers):
            print(f"  Numbers must be between 1 and {count}.")
            continue
        return numbers


def ask_missed():
    while True:
        answer = input("How many clear entities did the model MISS here? (number, Enter = 0): ").strip()
        if answer == "":
            return 0
        if answer.isdigit():
            return int(answer)
        print("  Please type a whole number.")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--source", default="wc_ch6")
    args = parser.parse_args()

    csv_path = OUT_DIR / f"e2_{args.source}_review.csv"
    texts = load_chunk_texts(OUT_DIR / f"e2_{args.source}_review_chunks.txt")
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        rows = [{k: r.get(k, "") for k in FIELDS} for r in csv.DictReader(f)]

    chunk_ids = list(texts)  # order of the review file
    for number, chunk_id in enumerate(chunk_ids, 1):
        mine = [r for r in rows if r["chunk_id"] == chunk_id and r["correct"].strip().lower() != "m"]
        done_missed = any(r["chunk_id"] == chunk_id and r["correct"].strip().lower() == "m" for r in rows)
        if mine and all(r["correct"].strip() for r in mine):
            continue  # this chunk is finished
        print("\n" + "=" * 100)
        print(f"Chunk {number} of {len(chunk_ids)}   ({chunk_id})")
        print("=" * 100)
        print(textwrap.fill(texts[chunk_id], 100))
        print()
        for k, r in enumerate(mine, 1):
            print(f"{k:>3}. [{r['label']}]  {r['span']}")
        print()
        wrong = ask_wrong(len(mine)) if mine else set()
        if wrong is None:
            save(csv_path, rows)
            print("Saved. Run this script again to continue.")
            return
        missed = ask_missed()
        for k, r in enumerate(mine, 1):
            r["correct"] = "0" if k in wrong else "1"
        if not done_missed:
            for _ in range(missed):
                rows.append({"chunk_id": chunk_id, "span": "(missed)", "label": "", "score": "", "correct": "m"})
        save(csv_path, rows)
    print("\nAll chunks reviewed. Now run: python scripts\\e2_ner.py score")


if __name__ == "__main__":
    main()