"""Validate a VLM evidence cache and report coverage against DGM4 metadata.

Checks the things that cause *silent* training-time failures:
* coverage: fraction of training samples that actually have an OK record
  (uncovered samples train with valid=0, i.e. no VLM signal);
* parse_status breakdown;
* prompt_version / model_id consistency;
* word indices in range (max flagged index < #caption words);
* hash alignment: recomputes the key from --ann with the given
  --max-words/--lowercase and confirms it matches the stored text_sha1. A
  mismatch here means the cache was built with different normalisation than
  training will use -> every lookup would miss.

Usage:
    python tools/validate_vlm_cache.py \
        --cache vlm_cache/qwen25_3b_train_div5.jsonl \
        --ann ../../datasets/DGM4/metadata/train.json \
        --dataset-division 5 --max-words 50 --lowercase
"""

import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset.utils import pre_caption  # noqa: E402
from dataset.vlm_cache import MANIP_CLASSES, hash_caption  # noqa: E402


def load_jsonl(path):
    recs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--ann", nargs="+", default=None,
                    help="Optional: metadata to measure coverage + hash alignment against.")
    ap.add_argument("--dataset-division", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-words", type=int, default=50)
    lc = ap.add_mutually_exclusive_group()
    lc.add_argument("--lowercase", dest="lowercase", action="store_true")
    lc.add_argument("--no-lowercase", dest="lowercase", action="store_false")
    ap.set_defaults(lowercase=True)
    args = ap.parse_args()

    recs = load_jsonl(args.cache)
    print(f"== cache: {args.cache}")
    print(f"   records: {len(recs)}")

    keys = [f"{r.get('image')}::{r.get('text_sha1')}" for r in recs]
    dup = [k for k, c in Counter(keys).items() if c > 1]
    print(f"   unique keys: {len(set(keys))}  duplicate keys: {len(dup)}")

    status = Counter(r.get("parse_status", "missing") for r in recs)
    print("   parse_status:", dict(status))
    n_ok = status.get("ok", 0)
    print(f"   ok fraction: {n_ok}/{len(recs)} = {100*n_ok/max(1,len(recs)):.1f}%")

    print("   prompt_version:", dict(Counter(r.get("prompt_version") for r in recs)))
    print("   model_id:", dict(Counter(r.get("model_id") for r in recs)))

    # teacher signal summary (OK records only)
    ok = [r for r in recs if r.get("parse_status") == "ok"]
    if ok:
        import statistics as st
        fp = [float(r.get("fake_probability", 0)) for r in ok]
        n_flagged = [len(r.get("unsupported_word_indices", []) or []) for r in ok]
        print(f"   fake_probability: mean={st.mean(fp):.3f} "
              f"min={min(fp):.3f} max={max(fp):.3f}")
        print(f"   flagged words/sample: mean={st.mean(n_flagged):.2f} "
              f"max={max(n_flagged)}  (samples with >=1 flag: "
              f"{sum(1 for x in n_flagged if x)}/{len(ok)})")
        for ci, c in enumerate(MANIP_CLASSES):
            vals = [float(r.get("manipulation_probs", {}).get(c, 0)) for r in ok]
            print(f"     P({c}): mean={st.mean(vals):.3f}")

    # coverage + hash alignment against metadata
    if args.ann:
        ann = []
        for f in args.ann:
            ann += json.load(open(f, "r", encoding="utf-8"))
        if args.dataset_division and args.dataset_division > 1:
            ann = ann[: int(len(ann) / args.dataset_division)]
        if args.limit:
            ann = ann[: args.limit]

        cache_keys = set(keys)
        covered = 0
        oob = 0  # records whose flagged indices exceed caption length
        rec_by_key = {f"{r.get('image')}::{r.get('text_sha1')}": r for r in recs}
        for a in ann:
            cap = pre_caption(a["text"], args.max_words, lowercase=args.lowercase)
            k = f"{a['image']}::{hash_caption(cap)}"
            if k in cache_keys:
                covered += 1
                r = rec_by_key[k]
                nw = len(cap.split(" "))
                for idx in r.get("unsupported_word_indices", []) or []:
                    if not (0 <= int(idx) < nw):
                        oob += 1
                        break
        print(f"== coverage vs {args.ann} (division={args.dataset_division})")
        print(f"   {covered}/{len(ann)} = {100*covered/max(1,len(ann)):.1f}% of samples covered")
        print(f"   records with out-of-range word indices: {oob}")
        if covered == 0:
            print("   !! ZERO coverage -> hash mismatch likely. Check --max-words/--lowercase "
                  "match the training config and the cache build settings.")

    print("== OK" if n_ok else "== WARNING: no OK records")


if __name__ == "__main__":
    main()
