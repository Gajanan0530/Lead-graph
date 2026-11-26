"""Experiment E3: can a small local model extract claims we can verify against the text?

Commands:
  python scripts/e3_claims.py run      -> model extracts claims, code checks each one
  python scripts/e3_claims.py review   -> you check the grounded claims by hand
  python scripts/e3_claims.py score    -> show the numbers again

Idea in one line: the model PROPOSES claims; plain code DISPOSES of any claim whose
quoted evidence is not really in the source text. Then you check what is left.
"""
import argparse
import json
import random
import re
import sys
import textwrap
import time
from pathlib import Path
from typing import List

from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e2_ner import OUT_DIR, RAW_DIR, make_chunks, read_blocks  # reuse our chunker

DEFAULT_MODEL = "qwen2.5:7b"


# ---- What we ask the model to return (the "schema") --------------------------------
class Claim(BaseModel):
    subject: str = Field(description="Who or what the claim is about: a name that appears in the passage.")
    predicate: str = Field(description="Short verb phrase for what the passage says about the subject.")
    object: str = Field(description="The person, thing, place, date or fact the subject is linked to, worded as in the passage.")
    evidence: str = Field(description="ONE exact quote copied word for word from the passage that supports the claim.")
    attributed_to: str = Field(description="Who says this, if the passage names a source (for example the Commission or a witness). Empty string if none.")


class ClaimList(BaseModel):
    claims: List[Claim]


SYSTEM = """You read one passage from an official government report and list the factual claims it makes.
Rules:
1. Use only what the passage says. Never add outside knowledge and never guess.
2. Each claim links a subject to an object. Use names exactly as written in the passage.
3. 'evidence' must be ONE quote copied exactly, word for word, from the passage (at most 40 words). Do not change it, shorten it with dots, or join sentences.
4. If the passage says someone concluded, testified or stated something, put that person or body in attributed_to.
5. Skip vague or generic statements. If nothing qualifies, return an empty list.
6. Return at most 6 claims, the most specific first."""


# ---- Text normalising and the automatic checks -------------------------------------
FOOTNOTE = re.compile(r'(?<=[.,;:!?"\u201d\u2019])\d{1,3}(?=\s|$)')  # footnote digits glued to punctuation


def norm(text):
    """Lower-case, drop footnote digits and punctuation, squash spaces, so quotes compare fairly."""
    text = FOOTNOTE.sub("", text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def check_claim(claim, chunk_norm):
    ev, sub, obj = norm(claim["evidence"]), norm(claim["subject"]), norm(claim["object"])
    checks = {
        "evidence_in_text": bool(ev) and ev in chunk_norm,
        "subject_in_text": bool(sub) and sub in chunk_norm,
        "object_in_text": bool(obj) and obj in chunk_norm,
        "linked_to_evidence": bool(ev) and ((bool(sub) and sub in ev) or (bool(obj) and obj in ev)),
    }
    checks["grounded"] = all(checks.values())
    return checks


def ask_model(model, passage, schema):
    from ollama import chat
    user = f"Passage:\n{passage}\n\nYour answer must match this JSON schema:\n{json.dumps(schema)}"
    kwargs = dict(model=model, format=schema,
                  messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
                  options={"temperature": 0, "num_ctx": 4096, "num_predict": 700})
    try:
        response = chat(think=False, **kwargs)   # ask the model to skip any "thinking" phase
    except TypeError:
        response = chat(**kwargs)                # older library without the think option
    except Exception as error:
        if "think" in str(error).lower():        # this model has no thinking mode: retry without it
            response = chat(**kwargs)
        else:
            raise
    return response.message.content


def run(args):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    chunks = make_chunks(read_blocks(RAW_DIR / f"{args.source}.txt"), args.source)
    chunks = [c for c in chunks if len(c["text"].split()) >= 80]
    chosen = sorted(random.Random(7).sample(range(len(chunks)), min(args.max_chunks, len(chunks))))
    schema = ClaimList.model_json_schema()
    print(f"{args.source}: using {len(chosen)} of {len(chunks)} chunks; model {args.model}")

    results, chunk_store, seconds, format_errors = [], {}, [], 0
    for n, index in enumerate(chosen, 1):
        chunk = chunks[index]
        chunk_store[chunk["chunk_id"]] = chunk["text"]
        started = time.time()
        try:
            raw = ask_model(args.model, chunk["text"], schema)
        except Exception as error:
            print(f"\nCould not talk to the model: {error}")
            print("Check that Ollama is running (look for its icon near the clock) and that you ran: "
                  f"ollama pull {args.model}")
            return
        seconds.append(time.time() - started)
        try:
            claims = [c.model_dump() for c in ClaimList.model_validate_json(raw).claims]
        except Exception:
            format_errors += 1
            claims = []
        chunk_norm = norm(chunk["text"])
        seen = set()
        for claim in claims:
            key = (norm(claim["subject"]), norm(claim["predicate"]), norm(claim["object"]))
            if key in seen:
                continue
            seen.add(key)
            results.append({"chunk_id": chunk["chunk_id"], **claim, "checks": check_claim(claim, chunk_norm)})
        print(f"  chunk {n}/{len(chosen)} done ({seconds[-1]:.1f} s, {len(claims)} claims)")

    (OUT_DIR / f"e3_{args.source}_claims.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (OUT_DIR / f"e3_{args.source}_chunks.json").write_text(json.dumps(chunk_store, indent=2), encoding="utf-8")

    total = len(results)
    print(f"\nAverage {sum(seconds) / len(seconds):.1f} s per chunk; {format_errors} chunks returned unreadable output")
    print(f"{total} claims extracted ({total / len(chosen):.1f} per chunk)")
    if total:
        for name in ["evidence_in_text", "subject_in_text", "object_in_text", "linked_to_evidence", "grounded"]:
            count = sum(r["checks"][name] for r in results)
            print(f"  {name}: {count}/{total} = {count / total:.0%}")
        failed = [r for r in results if not r["checks"]["grounded"]][:5]
        if failed:
            print("\nExamples that failed the automatic checks:")
            for r in failed:
                bad = [k for k, v in r["checks"].items() if not v and k != "grounded"]
                print(f"  - {r['subject']} | {r['predicate']} | {r['object']}   (failed: {', '.join(bad)})")
                print(f"    quote: {r['evidence'][:120]}")
    print("\nNext: python scripts\\e3_claims.py review")


def marks_path(source):
    return OUT_DIR / f"e3_{source}_marks.json"


def load_marks(source):
    p = marks_path(source)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def review(args):
    claims = json.loads((OUT_DIR / f"e3_{args.source}_claims.json").read_text(encoding="utf-8"))
    texts = json.loads((OUT_DIR / f"e3_{args.source}_chunks.json").read_text(encoding="utf-8"))
    marks = load_marks(args.source)
    by_chunk = {}
    for i, c in enumerate(claims):
        if c["checks"]["grounded"]:
            by_chunk.setdefault(c["chunk_id"], []).append(i)
    chunk_ids = list(by_chunk)
    for number, chunk_id in enumerate(chunk_ids, 1):
        indices = by_chunk[chunk_id]
        if all(str(i) in marks for i in indices):
            continue
        print("\n" + "=" * 100)
        print(f"Chunk {number} of {len(chunk_ids)}   ({chunk_id})")
        print("=" * 100)
        print(textwrap.fill(texts[chunk_id], 100))
        print()
        for k, i in enumerate(indices, 1):
            c = claims[i]
            who = f"   [attributed to: {c['attributed_to']}]" if c["attributed_to"].strip() else ""
            print(f"{k:>3}. {c['subject']}  |  {c['predicate']}  |  {c['object']}{who}")
            print(f"       quote: \"{c['evidence']}\"")
        print()
        while True:
            answer = input("Numbers of claims that are NOT faithful (space-separated), Enter = all faithful, q = save & quit: ").strip().lower()
            if answer == "q":
                marks_path(args.source).write_text(json.dumps(marks), encoding="utf-8")
                print("Saved. Run this command again to continue.")
                return
            try:
                bad = {int(x) for x in answer.replace(",", " ").split()}
            except ValueError:
                print("  Please type numbers only, like: 2 5")
                continue
            if any(b < 1 or b > len(indices) for b in bad):
                print(f"  Numbers must be between 1 and {len(indices)}.")
                continue
            break
        for k, i in enumerate(indices, 1):
            marks[str(i)] = 0 if k in bad else 1
        marks_path(args.source).write_text(json.dumps(marks), encoding="utf-8")
    print("\nAll grounded claims reviewed.")
    score(args)


def score(args):
    claims = json.loads((OUT_DIR / f"e3_{args.source}_claims.json").read_text(encoding="utf-8"))
    marks = load_marks(args.source)
    total = len(claims)
    grounded = [i for i, c in enumerate(claims) if c["checks"]["grounded"]]
    reviewed = [i for i in grounded if str(i) in marks]
    faithful = sum(marks[str(i)] for i in reviewed)
    print(f"Claims extracted: {total}; passed automatic checks: {len(grounded)} ({len(grounded) / total:.0%})" if total else "No claims.")
    if reviewed:
        rate = faithful / len(reviewed)
        print(f"Human review: {faithful}/{len(reviewed)} grounded claims faithful ({rate:.0%})")
        print(f"End to end (faithful out of ALL extracted claims): {faithful}/{total} = {faithful / total:.0%}")
        ok = len(grounded) / total >= 0.70 and rate >= 0.75
        print("PASS (grounded >= 70% and faithful >= 75%)" if ok else "BELOW TARGET (need grounded >= 70% and faithful >= 75%)")
    else:
        print("No human review marks yet. Run: python scripts\\e3_claims.py review")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("command", choices=["run", "review", "score"])
    parser.add_argument("--source", default="wc_ch6")
    parser.add_argument("--max-chunks", type=int, default=20)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()
    {"run": run, "review": review, "score": score}[args.command](args)


if __name__ == "__main__":
    main()