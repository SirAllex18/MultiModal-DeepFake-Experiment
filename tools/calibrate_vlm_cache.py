"""Post-hoc calibration of a raw VLM evidence cache.

The Qwen teacher (prompt v2) separates GT well (ROC-AUC ~0.77 for text-manip vs
orig) but is badly *calibrated*: it over-flags, so even genuine captions get a
high text_swap/text_attribute probability. Distilling those raw values would
push HAMMER toward false positives on real samples.

This script applies a monotonic, rank-preserving remap (empirical CDF, i.e.
percentile) to the chosen probability fields, computed over all parsed ('ok')
records. Because the remap is strictly order-preserving, the ROC-AUC / ranking
the teacher actually got right is preserved EXACTLY, while the absolute scale is
spread to ~uniform[0,1] so genuine samples sit low and manipulated ones high.
The raw cache is left untouched (kept for reproducibility); training points at
this calibrated copy.

Calibrated fields: fake_probability, manipulation_probs.text_swap,
manipulation_probs.text_attribute. (word_scores / face_* are left as-is.)

Usage:
  python tools/calibrate_vlm_cache.py \
      --in  vlm_cache/qwen25_7b_train_div5.jsonl \
      --out vlm_cache/qwen25_7b_train_div5.calib.jsonl
"""
import argparse
import bisect
import json


def get_path(rec, path):
    cur = rec
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    try:
        return float(cur)
    except (TypeError, ValueError):
        return None


def set_path(rec, path, value):
    cur = rec
    for k in path[:-1]:
        cur = cur.setdefault(k, {})
    cur[path[-1]] = value


# (name, key-path) of the fields to calibrate.
TARGETS = [
    ("fake_probability", ["fake_probability"]),
    ("text_swap", ["manipulation_probs", "text_swap"]),
    ("text_attribute", ["manipulation_probs", "text_attribute"]),
]


def main():
    ap = argparse.ArgumentParser(description="Percentile-calibrate a VLM cache.")
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    records = []
    with open(args.inp, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    ok = [r for r in records if r.get("parse_status") == "ok"]
    print(f"[calib] {len(records)} records ({len(ok)} ok) from {args.inp}")

    # Build a sorted value list per target (over ok records only).
    sorted_vals = {}
    for name, path in TARGETS:
        vals = sorted(v for v in (get_path(r, path) for r in ok) if v is not None)
        sorted_vals[name] = vals
        if vals:
            print(f"[calib] {name:<16} n={len(vals)} raw mean="
                  f"{sum(vals)/len(vals):.3f} min={vals[0]:.3f} max={vals[-1]:.3f}")

    def cdf(name, v):
        vals = sorted_vals[name]
        if not vals:
            return v
        # fraction of values <= v  -> percentile in (0, 1]
        return bisect.bisect_right(vals, v) / len(vals)

    n_cal = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for r in records:
            if r.get("parse_status") == "ok":
                for name, path in TARGETS:
                    v = get_path(r, path)
                    if v is not None:
                        set_path(r, path, round(cdf(name, v), 4))
                r["calibrated"] = "percentile"
                n_cal += 1
            f.write(json.dumps(r) + "\n")

    print(f"[calib] wrote {n_cal} calibrated records -> {args.out}")
    for name, path in TARGETS:
        new = sorted(v for v in (get_path(r, path) for r in
                                 (x for x in records if x.get('parse_status') == 'ok'))
                     if v is not None)
        if new:
            print(f"[calib] {name:<16} calibrated mean={sum(new)/len(new):.3f}")
    print("[calib] AUC/ranking is unchanged by construction; verify with "
          "tools/vlm_signal_alignment.py on the calibrated file.")


if __name__ == "__main__":
    main()
