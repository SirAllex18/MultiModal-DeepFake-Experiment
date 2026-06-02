"""Offline smoke test for the VLM-distillation plumbing.

Exercises everything that does NOT need the vendored BERT encoder (which pins
old transformers) or a GPU:
  1. HAMMER.compute_vlm_losses: masking / class-weighting / NaN-safety.
  2. word -> subword token mapping (train.build_vlm_token_targets) alignment.
  3. DGM4_Dataset returning the 8-tuple + default_collate batching (if images
     are reachable; otherwise that part is skipped with a notice).

Usage:
  python tools/smoke_test_vlm_offline.py \
      --ann ../../Multimodal-fake-news-detection/DGM4/metadata/train.json \
      --image-root ../../Multimodal-fake-news-detection \
      --cache vlm_cache/test_dummy.jsonl
"""

import argparse
import json
import os
import sys

import torch
from torch.utils.data._utils.collate import default_collate

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset.utils import pre_caption  # noqa: E402
from dataset.vlm_cache import VLMCache, MANIP_CLASSES  # noqa: E402
from models.HAMMER import HAMMER  # noqa: E402  (class import is safe; xbert is lazy)
from train import text_input_adjust, build_vlm_token_targets, prepare_vlm_target  # noqa: E402


def ok(msg):
    print(f"  [ok] {msg}")


def test_compute_vlm_losses():
    print("== 1. compute_vlm_losses")

    class Stub:
        vlm_teacher_T = 1.0
        vlm_mlc_class_weight = torch.tensor([0.0, 0.0, 1.0, 1.0])

    B, L = 4, 10
    bic = torch.randn(B, 2)
    mlc = torch.randn(B, 4)
    tok = torch.randn(B, L, 2)
    vt = {
        "valid": torch.tensor([1.0, 1.0, 0.0, 1.0]),
        "fake_prob": torch.rand(B),
        "multicls_probs": torch.rand(B, 4),
        "token_scores": torch.rand(B, L),
        "token_mask": (torch.rand(B, L) > 0.3).float(),
    }
    lb, lm, lt = HAMMER.compute_vlm_losses(Stub(), bic, mlc, tok, vt)
    assert all(torch.isfinite(x) for x in (lb, lm, lt)), "non-finite loss"
    ok(f"finite losses: bic={lb.item():.4f} mlc={lm.item():.4f} tmg={lt.item():.4f}")

    # all-invalid batch must be finite zero, not NaN
    vt0 = {**vt, "valid": torch.zeros(B), "token_mask": torch.zeros(B, L)}
    lb0, lm0, lt0 = HAMMER.compute_vlm_losses(Stub(), bic, mlc, tok, vt0)
    assert torch.isfinite(lb0) and torch.isfinite(lm0) and torch.isfinite(lt0)
    assert lm0.item() == 0.0 and lt0.item() == 0.0, "invalid batch should give 0 mlc/tmg"
    ok(f"all-invalid batch -> bic={lb0.item():.4f} mlc={lm0.item():.4f} tmg={lt0.item():.4f} (no NaN)")

    # MLC must ignore FS/FA columns: changing only FS/FA targets must not move the loss
    vt_fsfa = {**vt}
    vt_fsfa["multicls_probs"] = vt["multicls_probs"].clone()
    vt_fsfa["multicls_probs"][:, 0:2] = 1.0 - vt_fsfa["multicls_probs"][:, 0:2]
    _, lm_b, _ = HAMMER.compute_vlm_losses(Stub(), bic, mlc, tok, vt_fsfa)
    assert abs(lm_b.item() - lm.item()) < 1e-6, "MLC leaked FS/FA columns"
    ok("MLC distillation restricted to TS/TA (FS/FA columns ignored)")

    # token loss with empty token fields -> 0
    lb2, lm2, lt2 = HAMMER.compute_vlm_losses(
        Stub(), bic, mlc, tok, {k: vt[k] for k in ("valid", "fake_prob", "multicls_probs")})
    assert lt2.item() == 0.0
    ok("missing token fields -> token loss 0 (guarded)")


def test_token_mapping(ann, cache, lowercase=True, max_words=50):
    print("== 2. word -> subword token mapping")
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)
    n = min(6, len(ann))
    caps = [pre_caption(a["text"], max_words, lowercase=lowercase) for a in ann[:n]]
    te = tok(caps, max_length=128, truncation=True, add_special_tokens=True,
             return_attention_mask=True, return_token_type_ids=False)
    fake_word_pos = [torch.zeros(max_words) for _ in range(n)]
    te, _ = text_input_adjust(te, fake_word_pos, torch.device("cpu"))

    ws = torch.stack([cache.build_target(a["image"], c)["word_scores"] for a, c in zip(ann[:n], caps)])
    wm = torch.stack([cache.build_target(a["image"], c)["word_mask"] for a, c in zip(ann[:n], caps)])

    token_dim = te.input_ids.shape[1] - 1
    ts, tm = build_vlm_token_targets(te, ws, wm, token_dim, torch.device("cpu"))
    assert ts.shape == (n, token_dim) and tm.shape == (n, token_dim)
    ok(f"token targets shape {tuple(ts.shape)} == [B, input_ids-1]")

    # token_mask must never fall on a padding position (attention_mask[:,1:]==0)
    attn_no_cls = te.attention_mask[:, 1:].float()
    leak = (tm * (1.0 - attn_no_cls)).sum().item()
    assert leak == 0.0, "token_mask leaked onto padding tokens"
    ok("token_mask never lands on padding (CLS/SEP convention aligned)")

    assert ((ts >= 0) & (ts <= 1)).all(), "token scores out of [0,1]"
    ok(f"token scores in [0,1]; flagged tokens/sample: "
       f"{[int(tm[i].sum().item()) for i in range(n)]}")

    # spot check: every masked token's score came from a real flagged word
    nonzero_consistent = ((ts > 0) <= (tm > 0)).all()
    assert nonzero_consistent, "nonzero score outside token_mask"
    ok("all nonzero token scores lie within token_mask")


def _find_one_image(image_root, max_dirs=4000):
    """Return the path to any image under image_root (bounded walk), or None."""
    if not image_root or not os.path.isdir(image_root):
        return None
    seen = 0
    for dirpath, _dirnames, filenames in os.walk(image_root):
        seen += 1
        for fn in filenames:
            if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                return os.path.join(dirpath, fn)
        if seen > max_dirs:
            break
    return None


def test_dataset(image_root, max_words=50):
    print("== 3. DGM4_Dataset 8-tuple + collate")
    import tempfile
    from dataset.dataset import DGM4_Dataset
    from dataset.vlm_cache import hash_caption
    from torchvision import transforms
    from PIL import Image

    img = _find_one_image(image_root)
    if img is None:
        print(f"  [skip] no image found under {image_root}; dataset I/O part skipped. "
              "(Runs fully on the VM where the DGM4 layout matches the metadata.)")
        return

    # Self-contained: build a 1-sample metadata + matching dummy cache around a
    # real on-disk image, so this exercises the actual DGM4_Dataset.__getitem__
    # (transforms, bbox, and the new vlm_target branch) regardless of layout.
    rel = os.path.relpath(img, image_root).replace(os.sep, "/")
    text = "a british court ruled today that the player can publish his book"
    cap = pre_caption(text, max_words, lowercase=True)
    rec = {
        "image": rel, "text_sha1": hash_caption(cap), "parse_status": "ok",
        "fake_probability": 0.7,
        "manipulation_probs": {"face_swap": 0.1, "face_attribute": 0.1,
                               "text_swap": 0.6, "text_attribute": 0.4},
        "unsupported_word_indices": [0, 3, 6],
        "word_scores": {"0": 0.8, "3": 0.6, "6": 0.9},
    }
    tmp = tempfile.mkdtemp(prefix="vlm_smoke_")
    ann_path = os.path.join(tmp, "meta.json")
    cache_path = os.path.join(tmp, "cache.jsonl")
    json.dump([{
        "id": 1, "image": rel, "text": text, "fake_cls": "text_swap",
        "fake_image_box": [10, 10, 80, 80], "fake_text_pos": [3],
    }], open(ann_path, "w"))
    open(cache_path, "w").write(json.dumps(rec) + "\n")

    config = {
        "image_root": image_root, "image_res": 256,
        "text_backbone": "bert", "vlm_distill": True, "vlm_cache_file": cache_path,
    }
    tf = transforms.Compose([
        transforms.Resize((256, 256), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
    ])
    ds = DGM4_Dataset(config=config, ann_file=[ann_path], transform=tf,
                      max_words=max_words, is_train=True)
    items = [ds[0]]
    assert len(items[0]) == 8, f"expected 8-tuple, got {len(items[0])}"
    ok(f"training dataset returns 8-tuple (resolved image: {rel})")

    batch = default_collate(items)
    image, label, caption, fbox, fwp, W, H, vlm_target = batch
    assert set(vlm_target.keys()) == {"valid", "fake_prob", "multicls_probs", "word_scores", "word_mask"}
    assert vlm_target["multicls_probs"].shape == (1, len(MANIP_CLASSES))
    assert vlm_target["word_scores"].shape == (1, max_words)
    assert vlm_target["valid"].item() == 1.0, "cache hit should yield valid=1"
    ok(f"default_collate batched vlm_target; cache hit valid={vlm_target['valid'].tolist()}, "
       f"TS/TA probs={vlm_target['multicls_probs'][0, 2:].tolist()}")

    # is_train=False must NOT attach vlm_target (eval path stays 7-tuple)
    ds_val = DGM4_Dataset(config=config, ann_file=[ann_path], transform=tf,
                          max_words=max_words, is_train=False)
    assert len(ds_val[0]) == 7, "val dataset must stay a 7-tuple"
    ok("val/eval dataset stays 7-tuple (VLM is train-only)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", default="../../Multimodal-fake-news-detection/DGM4/metadata/train.json")
    ap.add_argument("--image-root", default="../../Multimodal-fake-news-detection")
    ap.add_argument("--cache", default="vlm_cache/test_dummy.jsonl")
    ap.add_argument("--max-words", type=int, default=50)
    args = ap.parse_args()

    ann = json.load(open(args.ann, "r", encoding="utf-8"))

    # Bootstrap a small dummy cache if none exists, so the test is turnkey.
    if not os.path.isfile(args.cache):
        from tools.build_vlm_cache import DummyBackend, build_prompt, parse_vlm_json
        from dataset.vlm_cache import hash_caption, PROMPT_VERSION
        print(f"[bootstrap] {args.cache} missing -> generating dummy cache (25 samples)")
        os.makedirs(os.path.dirname(os.path.abspath(args.cache)) or ".", exist_ok=True)
        be = DummyBackend()
        with open(args.cache, "w", encoding="utf-8") as f:
            for a in ann[:25]:
                cap = pre_caption(a["text"], args.max_words, lowercase=True)
                _, words = build_prompt(cap)
                fields, status = parse_vlm_json(be.generate(None, cap, None, words), len(words))
                rec = {"id": a.get("id"), "image": a["image"], "text_sha1": hash_caption(cap),
                       "split": "train", "model_id": be.model_id,
                       "prompt_version": PROMPT_VERSION, "parse_status": status}
                rec.update(fields)
                f.write(json.dumps(rec) + "\n")

    cache = VLMCache(args.cache, max_words=args.max_words)

    test_compute_vlm_losses()
    test_token_mapping(ann, cache, lowercase=True, max_words=args.max_words)
    test_dataset(args.image_root, max_words=args.max_words)

    print("\nALL OFFLINE SMOKE TESTS PASSED")
    print("(The full BERT-HAMMER forward/backward must be verified on the VM with "
          "transformers==4.44.2.)")


if __name__ == "__main__":
    main()
