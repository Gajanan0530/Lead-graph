"""E3b: fairer, smarter checks on the claims we already extracted (no model run needed).

Commands:
  python scripts/e3_v2.py recheck   -> re-run the automatic checks with better rules
  python scripts/e3_v2.py review    -> review only grounded claims you have not marked yet
  python scripts/e3_v2.py score     -> show the numbers

The model PROPOSES claims; these plain-code checks DISPOSE of the ones whose quote
does not really support them. Your hand marks tell us how good the checks are.
"""
import argparse
import json
import re
import shutil
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "derived"

STOP = set("a an the of to in on at by for with and or but as is are was were be been being that this these "
           "those it its his her their he she they them him from into than then there which who whom whose "
           "has have had not no".split())
HEDGES = ["apparently", "seems", "seemed", "appears to", "may have", "might", "possibly", "probably", "alleg",
          "rumor", "speculat", "claim", "suggest", "no indication", "no evidence", "unlikely", "believed",
          "could have", "charge was made", "purported", "reportedly"]
SAY_VERBS = ["testified", "stated", "said", "told", "reported", "informed", "advised", "indicated",
             "related", "described", "recalled", "asserted", "declared", "concluded"]

# Footnote numbers glued to punctuation, e.g. "less.126" or '...less.126"'
FOOTNOTE = re.compile(r'(?<=[.,;:!?"\u201d\u2019)])\d{1,3}(?=[\s"\u201d\u2019)]|$)')
QUOTE_CHARS = "\"'\u201c\u201d\u2018\u2019 "


def norm_strict(text):
    text = FOOTNOTE.sub("", text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def norm_loose(text):
    """Like strict, but also drops stray 1-3 digit numbers (footnotes with spaces around them)."""
    return " ".join(w for w in norm_strict(text).split() if not re.fullmatch(r"\d{1,3}", w))


def stem(word):
    for suffix in ("ing", "ed", "es", "s"):
        if len(word) > len(suffix) + 2 and word.endswith(suffix):
            word = word[:-len(suffix)]
            break
    if len(word) > 4 and word.endswith("e"):
        word = word[:-1]
    return word


def tokens(text):
    return [stem(w) for w in norm_strict(text).split() if w not in STOP]


def coverage(part, reference_tokens):
    """Share of the important words in `part` that also appear in the reference."""
    wanted = tokens(part)
    if not wanted:
        return 0.0
    ref = set(reference_tokens)
    return sum(w in ref for w in wanted) / len(wanted)


def evidence_found(evidence, chunk_strict, chunk_loose):
    ev = evidence.strip(QUOTE_CHARS)
    parts = [p.strip(QUOTE_CHARS) for p in re.split(r"\[\s*\.\.\.\s*\]|\.\.\.|\u2026", ev)]
    parts = [p for p in parts if len(p.split()) >= 3]
    if not parts:
        return False
    return all(norm_strict(p) in chunk_strict or norm_loose(p) in chunk_loose for p in parts)


def numbers(text):
    return set(re.findall(r"\d+", text))


def stance_of(evidence, attribution_is_ok, attribution):
    low = evidence.lower()
    if any(h in low for h in HEDGES):
        return "hedged"
    if any(v in low for v in SAY_VERBS) or (attribution.strip() and attribution_is_ok):
        return "attributed"
    return "narrative"


def check_claim(claim, chunk_text):
    chunk_strict, chunk_loose = norm_strict(chunk_text), norm_loose(chunk_text)
    chunk_tokens = tokens(chunk_text)
    ev_tokens = set(tokens(claim["evidence"]))
    attribution = claim.get("attributed_to", "")
    a_ok = not attribution.strip() or (
        not re.fullmatch(r"[\d\s.,]+", attribution.strip()) and coverage(attribution, chunk_tokens) >= 0.5)
    checks = {
        "evidence_in_text": evidence_found(claim["evidence"], chunk_strict, chunk_loose),
        "numbers_ok": numbers(claim["evidence"]) <= numbers(chunk_text),
        "subject_ok": coverage(claim["subject"], chunk_tokens) >= 0.8,
        "object_ok": coverage(claim["object"], chunk_tokens) >= 0.8,
        "supported_by_quote": coverage(claim["subject"] + " " + claim["object"], list(ev_tokens)) >= 0.7,
    }
    checks["grounded"] = all(checks.values())
    return checks, a_ok, stance_of(claim["evidence"], a_ok, attribution)


def load(source):
    claims = json.loads((OUT_DIR / f"e3_{source}_claims.json").read_text(encoding="utf-8"))
    texts = json.loads((OUT_DIR / f"e3_{source}_chunks.json").read_text(encoding="utf-8"))
    return claims, texts


def marks_path(source):
    return OUT_DIR / f"e3_{source}_marks.json"


def load_marks(source):
    p = marks_path(source)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def recheck(args):
    claims, texts = load(args.source)
    path = OUT_DIR / f"e3_{args.source}_claims.json"
    backup = OUT_DIR / f"e3_{args.source}_claims_v1_backup.json"
    if not backup.exists():
        shutil.copy(path, backup)
    marks = load_marks(args.source)

    for c in claims:
        if "checks_v1" not in c:
            c["checks_v1"] = c["checks"]
        checks, a_ok, stance = check_claim(c, texts[c["chunk_id"]])
        c["checks"], c["attribution_ok"], c["stance"] = checks, a_ok, stance
    path.write_text(json.dumps(claims, indent=2), encoding="utf-8")

    total = len(claims)
    v1 = [i for i, c in enumerate(claims) if c["checks_v1"]["grounded"]]
    v2 = [i for i, c in enumerate(claims) if c["checks"]["grounded"]]
    print(f"Claims: {total}")
    print(f"Passed old checks: {len(v1)} ({len(v1) / total:.0%})   Pass new checks: {len(v2)} ({len(v2) / total:.0%})")
    print(f"Newly passing: {len(set(v2) - set(v1))}   Newly rejected: {len(set(v1) - set(v2))}")

    # How good are the new checks? Compare with YOUR marks on claims that were reviewed.
    kept_ok = kept_bad = rej_ok = rej_bad = 0
    for i in v1:
        if str(i) not in marks:
            continue
        kept = claims[i]["checks"]["grounded"]
        good = marks[str(i)] == 1
        kept_ok += kept and good
        kept_bad += kept and not good
        rej_ok += (not kept) and good
        rej_bad += (not kept) and not good
    print("\nHow the new checks treat the claims you already marked:")
    print(f"  kept and faithful: {kept_ok}   kept but unfaithful: {kept_bad}")
    print(f"  rejected and faithful (a pity): {rej_ok}   rejected and unfaithful (caught!): {rej_bad}")

    stances = {}
    for i in v2:
        stances[claims[i]["stance"]] = stances.get(claims[i]["stance"], 0) + 1
    print(f"\nStance of passing claims: {stances}")
    bad_attr = [c for c in claims if not c["attribution_ok"]]
    print(f"Claims whose 'attributed_to' looks wrong (e.g. a footnote number): {len(bad_attr)}")

    still = [c for c in claims if not c["checks"]["grounded"]][:8]
    if still:
        print("\nSome claims that still fail:")
        for c in still:
            why = [k for k, v in c["checks"].items() if not v and k != "grounded"]
            print(f"- [{c['chunk_id']}] {c['subject']} | {c['predicate']} | {c['object']}  (failed: {', '.join(why)})")
    print("\nNext: python scripts\\e3_v2.py review")


def review(args):
    claims, texts = load(args.source)
    marks = load_marks(args.source)
    todo = {}
    for i, c in enumerate(claims):
        if c["checks"]["grounded"] and str(i) not in marks:
            todo.setdefault(c["chunk_id"], []).append(i)
    if not todo:
        print("Nothing left to review.")
        score(args)
        return
    ids = list(todo)
    for number, chunk_id in enumerate(ids, 1):
        indices = todo[chunk_id]
        print("\n" + "=" * 100)
        print(f"Chunk {number} of {len(ids)}   ({chunk_id})   [{len(indices)} new claims to check]")
        print("=" * 100)
        print(textwrap.fill(texts[chunk_id], 100))
        print()
        for k, i in enumerate(indices, 1):
            c = claims[i]
            who = f"  [says: {c['attributed_to']}]" if c["attributed_to"].strip() and c["attribution_ok"] else ""
            print(f"{k:>3}. ({c['stance']})  {c['subject']}  |  {c['predicate']}  |  {c['object']}{who}")
            print(f"       quote: \"{c['evidence'].strip(chr(34) + ' ')}\"")
        print()
        while True:
            answer = input("Numbers of claims that are NOT faithful (space-separated), Enter = all faithful, q = save & quit: ").strip().lower()
            if answer == "q":
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
    print("\nAll new claims reviewed.")
    score(args)


def score(args):
    claims, _ = load(args.source)
    marks = load_marks(args.source)
    total = len(claims)
    grounded = [i for i, c in enumerate(claims) if c["checks"]["grounded"]]
    reviewed = [i for i in grounded if str(i) in marks]
    faithful = sum(marks[str(i)] for i in reviewed)
    print(f"Claims extracted: {total}; pass automatic checks: {len(grounded)} ({len(grounded) / total:.0%})")
    if reviewed:
        rate = faithful / len(reviewed)
        print(f"Human review: {faithful}/{len(reviewed)} kept claims faithful ({rate:.0%}); {len(grounded) - len(reviewed)} kept claims not yet reviewed")
        print(f"End to end (faithful out of ALL extracted claims): {faithful}/{total} = {faithful / total:.0%}")
        ok = len(grounded) / total >= 0.70 and rate >= 0.85
        print("PASS (kept >= 70% of claims and >= 85% of kept claims faithful)" if ok
              else "BELOW TARGET (need kept >= 70% and faithful >= 85%)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("command", choices=["recheck", "review", "score"])
    parser.add_argument("--source", default="wc_ch6")
    args = parser.parse_args()
    {"recheck": recheck, "review": review, "score": score}[args.command](args)


if __name__ == "__main__":
    main()