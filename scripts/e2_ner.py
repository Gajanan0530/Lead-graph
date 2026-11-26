"""Experiment E2: how well does a small local model (GLiNER) find names in our text?

Two commands:
  python scripts/e2_ner.py run     -> runs the model on a sample, writes a review sheet
  python scripts/e2_ner.py score   -> reads your corrections and prints precision/recall

Why we measure: a model that sounds smart can still be wrong. We only trust
what we have measured on text we checked by hand.
"""
import argparse
import csv
import json
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "derived"

# The model. Only the v2.1 GLiNER checkpoints are Apache-2.0 (older ones are non-commercial).
MODEL_NAME = "urchade/gliner_medium-v2.1"
# The kinds of things we ask the model to find.
LABELS = ["person", "organization", "government agency", "location", "date", "event", "document"]

CHUNK_WORDS = 160  # target chunk size; the model reads about 384 tokens at a time

# Lines that are website boilerplate, not report text.
BOILERPLATE = [
    re.compile(r"^Have a question about JFK Assassination Records", re.I),
    re.compile(r"^Page \d+$", re.I),
]


def read_blocks(path):
    """Read the saved text file and return its useful lines."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines
            if ln.strip() and not any(p.match(ln.strip()) for p in BOILERPLATE)]


def split_long(block, limit=220):
    """Split a very long line into sentences so no piece is too long."""
    if len(block.split()) <= limit:
        return [block]
    return [s for s in re.split(r"(?<=[.!?])\s+", block) if s]


def make_chunks(blocks, source_id, target=CHUNK_WORDS):
    """Join lines into chunks of about `target` words, never cutting mid-sentence."""
    pieces = [p for b in blocks for p in split_long(b)]
    chunks, current, count = [], [], 0
    for piece in pieces:
        words = len(piece.split())
        if current and count + words > target:
            chunks.append(" ".join(current))
            current, count = [], 0
        current.append(piece)
        count += words
    if current:
        chunks.append(" ".join(current))
    return [{"chunk_id": f"{source_id}-{i:04d}", "text": t} for i, t in enumerate(chunks)]


def run(args):
    from gliner import GLiNER  # imported here so `score` works without the model installed

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    chunks = make_chunks(read_blocks(RAW_DIR / f"{args.source}.txt"), args.source)
    print(f"{args.source}: {len(chunks)} chunks of about {CHUNK_WORDS} words")

    print(f"Loading model {MODEL_NAME} (the first run downloads it; please wait)...")
    started = time.time()
    model = GLiNER.from_pretrained(MODEL_NAME)
    print(f"Model ready in {time.time() - started:.0f} s")

    # Pick chunks spread across the document, the same ones every time (seed 42).
    count = min(args.max_chunks, len(chunks))
    chosen = sorted(random.Random(42).sample(range(len(chunks)), count))

    results, seconds = [], []
    for n, index in enumerate(chosen, 1):
        chunk = chunks[index]
        t0 = time.time()
        found = model.predict_entities(chunk["text"], LABELS, threshold=args.threshold)
        seconds.append(time.time() - t0)
        for e in found:
            results.append({"chunk_id": chunk["chunk_id"], "text": e["text"],
                            "label": e["label"], "score": round(float(e["score"]), 3)})
        if n % 10 == 0 or n == count:
            print(f"  processed {n}/{count} chunks")

    print(f"Average {sum(seconds) / len(seconds):.2f} s per chunk; "
          f"{len(results)} entities found")

    by_label = defaultdict(Counter)
    for r in results:
        by_label[r["label"]][r["text"].lower()] += 1
    for label in LABELS:
        top = ", ".join(f"{t} ({c})" for t, c in by_label[label].most_common(8))
        print(f"\n{label}: {sum(by_label[label].values())} found\n  top: {top}")

    (OUT_DIR / f"e2_{args.source}_entities.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")

    # Review sheet: the first N sampled chunks, one row per found entity.
    review_ids = {chunks[i]["chunk_id"] for i in chosen[:args.review_chunks]}
    with open(OUT_DIR / f"e2_{args.source}_review.csv", "w", newline="",
              encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["chunk_id", "span", "label", "score", "correct"])
        for r in results:
            if r["chunk_id"] in review_ids:
                writer.writerow([r["chunk_id"], r["text"], r["label"], r["score"], ""])
    with open(OUT_DIR / f"e2_{args.source}_review_chunks.txt", "w", encoding="utf-8") as f:
        for i in chosen[:args.review_chunks]:
            f.write(f"=== {chunks[i]['chunk_id']} ===\n{chunks[i]['text']}\n\n")
    print("\nReview files written to", OUT_DIR)


def score(args):
    path = OUT_DIR / f"e2_{args.source}_review.csv"
    right = wrong = missed = blank = 0
    per_label = defaultdict(lambda: [0, 0])  # label -> [right, wrong]
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            mark = (row.get("correct") or "").strip().lower()
            if mark == "1":
                right += 1
                per_label[row["label"]][0] += 1
            elif mark == "0":
                wrong += 1
                per_label[row["label"]][1] += 1
            elif mark == "m":
                missed += 1
            else:
                blank += 1
    if blank:
        print(f"Warning: {blank} rows have no mark and were ignored.")
    if right + wrong == 0:
        print("No marked rows yet. Fill the 'correct' column with 1, 0 or m.")
        return
    precision = right / (right + wrong)
    recall = right / (right + missed) if right + missed else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    print(f"Marked: {right} right, {wrong} wrong, {missed} missed")
    print(f"Precision {precision:.2f}   Recall {recall:.2f}   F1 {f1:.2f}")
    for label, (r, w) in sorted(per_label.items()):
        print(f"  {label}: precision {r / (r + w):.2f} ({r + w} rows)")
    print("PASS (F1 >= 0.60)" if f1 >= 0.60 else "BELOW TARGET (F1 < 0.60)")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("command", choices=["run", "score"])
    parser.add_argument("--source", default="wc_ch6")
    parser.add_argument("--max-chunks", type=int, default=40)
    parser.add_argument("--review-chunks", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    run(args) if args.command == "run" else score(args)


if __name__ == "__main__":
    main()