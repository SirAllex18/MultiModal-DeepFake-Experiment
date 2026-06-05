"""Quantify how well the offline VLM cache aligns with DGM4 ground truth.

The smoke test proves the *plumbing* works; this proves whether the teacher
signal is worth distilling. It joins a VLM cache with the DGM4 annotations
(by the same image::sha1(pre_caption) key the trainer uses) and reports:

1. Sequence-level separation (ROC-AUC) of VLM scores between text-manipulated
   captions (GT fake_cls contains text_swap / text_attribute) and genuine ones.
   AUC ~0.50 = no signal; >0.60 = weak; >0.70 = usable for distillation.
2. Token-level alignment: do the VLM's flagged word indices land on the GT
   `fake_text_pos` tokens, vs a random-flagging baseline (precision/recall).

No numpy/sklearn needed — AUC is an exact all-pairs (Mann-Whitney) computation,
fine for the few-hundred-sample smoke sets this is meant for.

Usage:
  python tools/vlm_signal_alignment.py \
      --cache vlm_cache/qwen25_7b_sample.jsonl \
      --ann ../../datasets/DGM4/metadata/train.json \
      --dataset-division 5 --limit 200 --max-words 50 --lowercase
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dataset.utils import pre_caption  # noqa: E402
from dataset.vlm_cache import hash_caption  # noqa: E402


def parse_fake_cls(fake_cls):
    """DGM4 fake_cls -> set of manipulation tags. 'orig'/'' -> empty set."""
    if fake_cls is None:
        return set()
    if isinstance(fake_cls, (list, tuple)):
        parts = fake_cls
    else:
        parts = str(fake_cls).split("&")
    tags = {p.strip() for p in parts if p and p.strip() and p.strip() != "orig"}
    return tags


def roc_auc(scores, labels):
    """Exact all-pairs AUC. labels in {0,1}. Ties count 0.5. Returns (auc, n_pos, n_neg)."""
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg:
        return float("nan"), len(pos), len(neg)
    wins = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return wins / (len(pos) * len(neg)), len(pos), len(neg)


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def load_cache(path):
    by_key = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            by_key[f"{rec.get('image')}::{rec.get('text_sha1')}"] = rec
    return by_key


def main():
    ap = argparse.ArgumentParser(description="VLM cache vs DGM4 GT alignment report.")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--ann", nargs="+", required=True)
    ap.add_argument("--dataset-division", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-words", type=int, default=50)
    lc = ap.add_mutually_exclusive_group()
    lc.add_argument("--lowercase", dest="lowercase", action="store_true")
    lc.add_argument("--no-lowercase", dest="lowercase", action="store_false")
    ap.set_defaults(lowercase=True)
    args = ap.parse_args()

    ann = []
    for f in args.ann:
        ann += json.load(open(f, "r", encoding="utf-8"))
    if args.dataset_division and args.dataset_division > 1:
        ann = ann[: int(len(ann) / args.dataset_division)]
    if args.limit:
        ann = ann[: int(args.limit)]

    cache = load_cache(args.cache)

    rows = []          # one per matched, ok record
    n_missing = n_badparse = 0
    for a in ann:
        cap = pre_caption(a["text"], args.max_words, lowercase=args.lowercase)
        key = f"{a['image']}::{hash_caption(cap)}"
        rec = cache.get(key)
        if rec is None:
            n_missing += 1
            continue
        if rec.get("parse_status") != "ok":
            n_badparse += 1
            continue
        tags = parse_fake_cls(a.get("fake_cls"))
        mp = rec.get("manipulation_probs", {}) or {}
        ts = float(mp.get("text_swap", 0.0))
        ta = float(mp.get("text_attribute", 0.0))
        words = cap.split(" ")
        flagged = {int(i) for i in rec.get("unsupported_word_indices", []) if 0 <= int(i) < len(words)}
        gt_pos = {int(i) for i in (a.get("fake_text_pos") or []) if 0 <= int(i) < len(words)}
        rows.append({
            "tags": tags,
            "has_text": ("text_swap" in tags) or ("text_attribute" in tags),
            "has_ts": "text_swap" in tags,
            "has_ta": "text_attribute" in tags,
            "is_orig": len(tags) == 0,
            "has_face": ("face_swap" in tags) or ("face_attribute" in tags),
            "fake_p": float(rec.get("fake_probability", 0.0)),
            "ts": ts, "ta": ta, "max_tsa": max(ts, ta),
            "n_words": len(words),
            "flagged": flagged,
            "gt_pos": gt_pos,
        })

    print(f"== matched {len(rows)} ok records  (missing {n_missing}, bad-parse {n_badparse})")
    if not rows:
        print("no matched records — check --max-words/--lowercase/--dataset-division match the build")
        return

    n_text = sum(r["has_text"] for r in rows)
    n_orig = sum(r["is_orig"] for r in rows)
    n_faceonly = sum(r["has_face"] and not r["has_text"] for r in rows)
    print(f"   GT breakdown: text-manip={n_text}  orig={n_orig}  face-only={n_faceonly}  "
          f"(ts={sum(r['has_ts'] for r in rows)} ta={sum(r['has_ta'] for r in rows)})")

    print("\n== group means (higher should be the manipulated group)")
    print(f"   fake_probability : orig={mean([r['fake_p'] for r in rows if r['is_orig']]):.3f}  "
          f"text-manip={mean([r['fake_p'] for r in rows if r['has_text']]):.3f}  "
          f"face-only={mean([r['fake_p'] for r in rows if r['has_face'] and not r['has_text']]):.3f}")
    print(f"   max(P_ts,P_ta)   : orig={mean([r['max_tsa'] for r in rows if r['is_orig']]):.3f}  "
          f"text-manip={mean([r['max_tsa'] for r in rows if r['has_text']]):.3f}")

    print("\n== sequence-level separation (ROC-AUC; 0.50=none, >0.60 weak, >0.70 usable)")

    def report_auc(name, score_key, pos_pred, neg_pred):
        s = [r[score_key] for r in rows if pos_pred(r) or neg_pred(r)]
        y = [1 if pos_pred(r) else 0 for r in rows if pos_pred(r) or neg_pred(r)]
        auc, npos, nneg = roc_auc(s, y)
        print(f"   {name:<46} AUC={auc:.3f}  (pos={npos}, neg={nneg})")

    report_auc("fake_prob: text-manip vs orig", "fake_p",
               lambda r: r["has_text"], lambda r: r["is_orig"])
    report_auc("max(P_ts,P_ta): text-manip vs orig", "max_tsa",
               lambda r: r["has_text"], lambda r: r["is_orig"])
    report_auc("fake_prob: text-manip vs (orig+face-only)", "fake_p",
               lambda r: r["has_text"], lambda r: not r["has_text"])
    report_auc("P_text_swap: text_swap vs orig", "ts",
               lambda r: r["has_ts"], lambda r: r["is_orig"])
    report_auc("P_text_attribute: text_attribute vs orig", "ta",
               lambda r: r["has_ta"], lambda r: r["is_orig"])

    # ---- token-level alignment (for the E2 token-grounding distillation) ----
    tok_rows = [r for r in rows if r["gt_pos"] and r["has_text"]]
    inter = sum(len(r["flagged"] & r["gt_pos"]) for r in tok_rows)
    n_flag = sum(len(r["flagged"]) for r in tok_rows)
    n_gt = sum(len(r["gt_pos"]) for r in tok_rows)
    rand_prec = mean([len(r["gt_pos"]) / r["n_words"] for r in tok_rows])  # E[prec] of random flagging
    print("\n== token-level alignment on GT text-manipulated samples with positions")
    print(f"   samples={len(tok_rows)}  VLM-flagged tokens={n_flag}  GT tokens={n_gt}")
    if n_flag:
        print(f"   precision (flagged that are GT) = {inter}/{n_flag} = {inter / n_flag:.3f}  "
              f"(random baseline ~ {rand_prec:.3f})")
    if n_gt:
        print(f"   recall    (GT that were flagged) = {inter}/{n_gt} = {inter / n_gt:.3f}")
    print("\n(VLM scores beat their baselines => distillable signal; "
          "~baseline => the teacher adds noise, reconsider before the full build)")


if __name__ == "__main__":
    main()
