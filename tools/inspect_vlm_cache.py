"""Pretty-print VLM cache evidence for a few samples (qualitative spot checks).

Joins a DGM4 metadata file with the cache so you can eyeball, per sample:
the caption with numbered words, which words the VLM flagged (+ scores), the
manipulation probabilities, and the rationale. Optionally prints the resolved
image path so you can open it alongside.

Usage:
    python tools/inspect_vlm_cache.py \
        --cache vlm_cache/qwen25_3b_train_div5.jsonl \
        --ann ../../datasets/DGM4/metadata/train.json \
        --image-root ../../datasets --n 5 --max-words 50 --lowercase
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset.utils import pre_caption  # noqa: E402
from dataset.vlm_cache import MANIP_CLASSES, VLMCache  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--ann", nargs="+", required=True)
    ap.add_argument("--image-root", default=None)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--only-flagged", action="store_true",
                    help="Skip samples the VLM flagged no words for.")
    ap.add_argument("--max-words", type=int, default=50)
    lc = ap.add_mutually_exclusive_group()
    lc.add_argument("--lowercase", dest="lowercase", action="store_true")
    lc.add_argument("--no-lowercase", dest="lowercase", action="store_false")
    ap.set_defaults(lowercase=True)
    args = ap.parse_args()

    ann = []
    for f in args.ann:
        ann += json.load(open(f, "r", encoding="utf-8"))
    cache = VLMCache(args.cache, max_words=args.max_words)

    shown = 0
    i = args.start
    while shown < args.n and i < len(ann):
        a = ann[i]
        i += 1
        cap = pre_caption(a["text"], args.max_words, lowercase=args.lowercase)
        rec = cache.lookup(a["image"], cap)
        if rec is None:
            continue
        flagged = rec.get("unsupported_word_indices", []) or []
        if args.only_flagged and not flagged:
            continue

        words = cap.split(" ")
        scores = rec.get("word_scores", {}) or {}
        print("=" * 70)
        print(f"id={a.get('id')}  parse_status={rec.get('parse_status')}  "
              f"fake_cls(GT)={a.get('fake_cls')}  fake_text_pos(GT)={a.get('fake_text_pos')}")
        if args.image_root:
            print("image:", os.path.join(args.image_root, a["image"]))
        print("caption:")
        line = []
        for wi, w in enumerate(words):
            tag = f"*{w}*[{scores.get(str(wi), 1.0)}]" if wi in flagged else w
            line.append(tag)
        print("  " + " ".join(line))
        print("fake_probability:", rec.get("fake_probability"))
        print("manipulation_probs:",
              {c: rec.get("manipulation_probs", {}).get(c) for c in MANIP_CLASSES})
        print("rationale:", rec.get("rationale", ""))
        shown += 1

    print("=" * 70)
    print(f"shown {shown} records (* = VLM-flagged word; GT shown for your reference only)")


if __name__ == "__main__":
    main()
