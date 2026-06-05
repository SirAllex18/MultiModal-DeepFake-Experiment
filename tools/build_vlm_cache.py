"""Offline VLM evidence cache builder (HAMMER+VLM, Stage 1).

Runs a vision-language model over DGM4 image/caption pairs and writes a JSONL
cache of *structured semantic evidence* that HAMMER later distills from during
training. This is the only component that needs the modern VLM stack
(``transformers`` with Qwen2.5-VL); it is meant to run in a separate conda
environment ("Environment B") and communicate with HAMMER purely through the
JSONL file.

Key contracts (see ``dataset/vlm_cache.py``):

* The cache key is ``image + "::" + sha1(prompt_caption)``.
* ``prompt_caption`` is ``pre_caption(ann["text"], max_words, lowercase)`` with
  the SAME ``max_words`` / ``lowercase`` the training run uses. For the BERT
  HAMMER baseline that means ``--max-words 50 --lowercase`` (bert-base-uncased
  is uncased). Get this wrong and every training-time lookup silently misses.

Prompt safety: the VLM sees ONLY the decoded image and the cleaned caption.
Never the image path (it encodes the manipulation method), ``fake_cls``,
``fake_image_box`` or ``fake_text_pos``.

Backends:
* ``qwen``  – Qwen2.5-VL-3B-Instruct in bf16 (default; fits a 16 GB 5080).
* ``dummy`` – deterministic heuristic, no model. Lets the whole downstream
              pipeline (loader, dataset, losses, smoke test) be exercised with
              no GPU/VLM installed.

Example (subset matching ``dataset_division: 5`` of train):
    python tools/build_vlm_cache.py \
        --ann ../../datasets/DGM4/metadata/train.json \
        --image-root ../../datasets \
        --out vlm_cache/qwen25_3b_train_div5.jsonl \
        --split train --dataset-division 5 \
        --backend qwen --model-id Qwen/Qwen2.5-VL-3B-Instruct \
        --max-words 50 --lowercase --resume
"""

import argparse
import json
import os
import re
import sys
import time

# Import the shared cache contract + the EXACT caption normaliser training uses.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset.utils import pre_caption  # noqa: E402
from dataset.vlm_cache import MANIP_CLASSES, PROMPT_VERSION, hash_caption  # noqa: E402


# --------------------------------------------------------------------------- #
# Prompt
# --------------------------------------------------------------------------- #
PROMPT_HEADER = """You are an expert news fact-checker. You are given a news photograph and its caption. Decide whether the CAPTION has been edited so that it no longer matches what the photograph actually shows.

Caption:
"{caption}"

Numbered caption words:
{numbered_words}

How to judge:
- Use ONLY what is visibly depicted in the photograph.
- Do NOT penalize names, places, or dates you cannot personally verify. Assume the named people / places / dates may be correct UNLESS the visible content clearly contradicts them. You are not verifying identities; you are checking whether the visible scene matches the words.
- You are looking for two kinds of TEXT edit:
  - text_swap: a concrete, depictable element (an action, object, setting/place type, event, count, or the people's visible activity) stated in the caption is clearly contradicted by the image.
  - text_attribute: a sentiment / emotion / attribute word (e.g. happy, angry, celebrating, mourning, injured, destroyed, crowded) that clearly conflicts with the mood or attributes visible in the image.
- Flag a word ONLY when the image gives POSITIVE visual evidence that it is wrong. If the image neither supports nor contradicts a word, do NOT flag it. Genuine captions are common; flagging every word is incorrect.

Scoring:
- text_swap / text_attribute: probability (0..1) that an edit of that type is present, based only on visible contradiction.
- fake_probability: overall probability (0..1) that the caption misrepresents the image.
- word_scores: for each flagged index, your confidence (0..1) that THAT specific word is contradicted by the image. Omit words you cannot judge.

Return STRICT JSON ONLY, no prose, with exactly this schema:
{{
  "fake_probability": <float 0..1>,
  "manipulation_probs": {{
    "text_swap": <float 0..1>,
    "text_attribute": <float 0..1>
  }},
  "unsupported_word_indices": [<int indices of specifically contradicted words>],
  "word_scores": {{"<word index>": <float 0..1>}},
  "rationale": "<one concise sentence citing the visual evidence>"
}}"""


def build_prompt(prompt_caption: str):
    """Return (prompt_text, words) for a pre-captioned string."""
    words = prompt_caption.split(" ")
    numbered = "\n".join(f"[{i}] {w}" for i, w in enumerate(words))
    return PROMPT_HEADER.format(caption=prompt_caption, numbered_words=numbered), words


# --------------------------------------------------------------------------- #
# JSON parsing / validation
# --------------------------------------------------------------------------- #
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_vlm_json(raw: str, num_words: int):
    """Parse + validate a VLM response. Returns (record_fields, status).

    status is "ok" or a short error tag. On "ok" the returned dict has the
    normalised, range-checked fields ready to merge into a cache record.
    """
    if not raw or not raw.strip():
        return {}, "empty"

    m = _JSON_RE.search(raw)
    if not m:
        return {}, "no_json"
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}, "parse_error"
    if not isinstance(obj, dict):
        return {}, "not_object"

    def f01(x, default=0.0):
        try:
            return max(0.0, min(1.0, float(x)))
        except (TypeError, ValueError):
            return default

    mp_in = obj.get("manipulation_probs", {}) or {}
    manip = {c: f01(mp_in.get(c, 0.0)) for c in MANIP_CLASSES}

    # Keep only in-range integer indices; clamp scores to [0,1].
    unsupported = []
    for idx in obj.get("unsupported_word_indices", []) or []:
        try:
            i = int(idx)
        except (TypeError, ValueError):
            continue
        if 0 <= i < num_words:
            unsupported.append(i)
    unsupported = sorted(set(unsupported))

    word_scores = {}
    for k, v in (obj.get("word_scores", {}) or {}).items():
        try:
            i = int(k)
        except (TypeError, ValueError):
            continue
        if 0 <= i < num_words:
            word_scores[str(i)] = f01(v, 0.0)

    fields = {
        "fake_probability": f01(obj.get("fake_probability", 0.0)),
        "manipulation_probs": manip,
        "unsupported_word_indices": unsupported,
        "word_scores": word_scores,
        "rationale": str(obj.get("rationale", ""))[:500],
    }
    return fields, "ok"


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #
class DummyBackend:
    """Deterministic, model-free evidence generator for offline testing.

    Produces schema-valid JSON whose content is a function of the caption hash,
    so runs are reproducible and the downstream pipeline can be validated end
    to end without a VLM. The numbers are NOT meaningful.
    """

    model_id = "dummy/heuristic-v1"

    def __init__(self, **_):
        pass

    def generate(self, image_path, prompt_caption, prompt_text, words, max_new_tokens=256):
        import hashlib

        h = int(hashlib.sha1(prompt_caption.encode("utf-8")).hexdigest(), 16)
        n = len(words)
        # Flag a deterministic subset of words (roughly every 7th, offset by hash).
        flagged = sorted({i for i in range(n) if n and (i + h) % 7 == 0})
        word_scores = {str(i): round(0.55 + 0.4 * (((h >> i) & 0xFF) / 255.0), 3) for i in flagged}
        fake_p = round(0.2 + 0.6 * ((h % 1000) / 1000.0), 3)
        obj = {
            "fake_probability": fake_p,
            "manipulation_probs": {
                "face_swap": round((h % 7) / 10.0, 3),
                "face_attribute": round((h % 5) / 10.0, 3),
                "text_swap": round(0.3 + 0.5 * ((h % 11) / 11.0), 3),
                "text_attribute": round(0.2 + 0.5 * ((h % 13) / 13.0), 3),
            },
            "unsupported_word_indices": flagged,
            "word_scores": word_scores,
            "rationale": "dummy backend: deterministic evidence for pipeline testing",
        }
        return json.dumps(obj)


class QwenBackend:
    """Qwen2.5-VL backend (default 3B bf16). VM/GPU only.

    Loads lazily so importing this module (e.g. for the dummy backend or for
    unit tests) never requires transformers/torch with VLM support.
    """

    def __init__(self, model_id="Qwen/Qwen2.5-VL-3B-Instruct", device="cuda",
                 dtype="bfloat16", max_image_side=896, load_in_4bit=False):
        import torch
        from transformers import AutoProcessor

        self.model_id = model_id
        self.device = device
        self.max_image_side = int(max_image_side)
        torch_dtype = {"bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
                       "float16": torch.float16, "fp16": torch.float16}.get(dtype, torch.bfloat16)

        # Qwen2.5-VL class name differs across transformers versions; try both.
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration as model_cls
        except ImportError:
            from transformers import AutoModelForVision2Seq as model_cls  # fallback

        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            quant = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            )
            self.model = model_cls.from_pretrained(
                model_id, quantization_config=quant, device_map="auto"
            ).eval()
        else:
            self.model = model_cls.from_pretrained(
                model_id, torch_dtype=torch_dtype
            ).to(device).eval()
        # min/max pixels keep the vision-token budget bounded on 16 GB.
        self.processor = AutoProcessor.from_pretrained(
            model_id, max_pixels=self.max_image_side * self.max_image_side
        )

    def _load_image(self, image_path):
        from PIL import Image

        img = Image.open(image_path).convert("RGB")
        s = self.max_image_side
        if max(img.size) > s:
            img.thumbnail((s, s), Image.BICUBIC)
        return img

    def generate(self, image_path, prompt_caption, prompt_text, words, max_new_tokens=256):
        import torch

        image = self._load_image(image_path)
        messages = [{
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": prompt_text}],
        }]
        chat = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(text=[chat], images=[image], return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            gen = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        trimmed = gen[:, inputs.input_ids.shape[1]:]
        out = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return out


def make_backend(args):
    if args.backend == "dummy":
        return DummyBackend()
    if args.backend == "qwen":
        return QwenBackend(
            model_id=args.model_id, device=args.device, dtype=args.dtype,
            max_image_side=args.max_image_side, load_in_4bit=args.load_in_4bit,
        )
    raise ValueError(f"unknown backend {args.backend!r}")


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def load_annotations(ann_files, dataset_division=None, limit=None):
    ann = []
    for f in ann_files:
        ann += json.load(open(f, "r", encoding="utf-8"))
    if dataset_division and dataset_division > 1:
        # Mirror DGM4_Dataset's prefix slicing so the cache covers exactly the
        # subset training will use.
        ann = ann[: int(len(ann) / dataset_division)]
    if limit:
        ann = ann[: int(limit)]
    return ann


def read_existing_keys(out_path):
    keys = set()
    if os.path.isfile(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    keys.add(f"{rec.get('image')}::{rec.get('text_sha1')}")
                except json.JSONDecodeError:
                    continue
    return keys


def main():
    ap = argparse.ArgumentParser(description="Build offline VLM evidence cache for HAMMER.")
    ap.add_argument("--ann", nargs="+", required=True, help="DGM4 metadata json file(s).")
    ap.add_argument("--image-root", required=True,
                    help="Root that ann['image'] paths are relative to (e.g. ../../datasets).")
    ap.add_argument("--out", required=True, help="Output JSONL path.")
    ap.add_argument("--split", default="train", help="Stored in each record's 'split' field.")
    ap.add_argument("--backend", choices=["qwen", "dummy"], default="qwen")
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--max-image-side", type=int, default=896)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--max-words", type=int, default=50,
                    help="MUST match config['max_words'] used in training.")
    lc = ap.add_mutually_exclusive_group()
    lc.add_argument("--lowercase", dest="lowercase", action="store_true",
                    help="Lowercase captions (bert-base-uncased). MUST match training.")
    lc.add_argument("--no-lowercase", dest="lowercase", action="store_false",
                    help="Preserve case (deberta-v3).")
    ap.set_defaults(lowercase=True)
    ap.add_argument("--dataset-division", type=int, default=None,
                    help="Mirror DGM4_Dataset prefix slicing to cover the training subset.")
    ap.add_argument("--limit", type=int, default=None, help="Cap number of samples (debug).")
    ap.add_argument("--resume", action="store_true", help="Skip samples already in --out.")
    ap.add_argument("--flush-every", type=int, default=50)
    args = ap.parse_args()

    ann = load_annotations(args.ann, args.dataset_division, args.limit)
    print(f"[build] {len(ann)} samples from {args.ann} "
          f"(division={args.dataset_division}, limit={args.limit})")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    existing = read_existing_keys(args.out) if args.resume else set()
    if args.resume:
        print(f"[build] resume: {len(existing)} records already present, will skip them")

    fail_path = args.out + ".failures.jsonl"
    backend = make_backend(args)
    print(f"[build] backend={args.backend} model_id={backend.model_id} "
          f"max_words={args.max_words} lowercase={args.lowercase}")

    n_ok = n_fail = n_skip = 0
    t0 = time.time()
    with open(args.out, "a", encoding="utf-8") as out_f, \
            open(fail_path, "a", encoding="utf-8") as fail_f:
        for n, a in enumerate(ann):
            image_rel = a["image"]
            prompt_caption = pre_caption(a["text"], args.max_words, lowercase=args.lowercase)
            text_sha1 = hash_caption(prompt_caption)
            key = f"{image_rel}::{text_sha1}"
            if key in existing:
                n_skip += 1
                continue

            prompt_text, words = build_prompt(prompt_caption)
            image_path = os.path.join(args.image_root, image_rel)

            status = "error"
            fields = {}
            raw = ""
            for attempt in range(2):  # one retry on parse failure
                try:
                    raw = backend.generate(image_path, prompt_caption, prompt_text, words,
                                           max_new_tokens=args.max_new_tokens)
                except FileNotFoundError:
                    status = "image_missing"
                    break
                except Exception as e:  # noqa: BLE001 - log and continue the run
                    status = f"backend_error:{type(e).__name__}"
                    raw = str(e)
                    break
                fields, status = parse_vlm_json(raw, len(words))
                if status == "ok":
                    break

            rec = {
                "id": a.get("id"),
                "image": image_rel,
                "text_sha1": text_sha1,
                "split": args.split,
                "model_id": backend.model_id,
                "prompt_version": PROMPT_VERSION,
                "parse_status": status,
            }
            if status == "ok":
                rec.update(fields)
                n_ok += 1
            else:
                # Keep a placeholder so --resume doesn't retry forever, and log raw.
                rec.update({
                    "fake_probability": 0.0,
                    "manipulation_probs": {c: 0.0 for c in MANIP_CLASSES},
                    "unsupported_word_indices": [],
                    "word_scores": {},
                })
                n_fail += 1
                fail_f.write(json.dumps({"key": key, "status": status,
                                         "prompt_caption": prompt_caption,
                                         "raw": raw[:2000]}) + "\n")

            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if (n + 1) % args.flush_every == 0:
                out_f.flush()
                rate = (n + 1) / max(1e-6, time.time() - t0)
                print(f"[build] {n+1}/{len(ann)} ok={n_ok} fail={n_fail} skip={n_skip} "
                      f"{rate:.2f} samples/s", flush=True)

    dt = time.time() - t0
    print(f"[build] DONE ok={n_ok} fail={n_fail} skip={n_skip} in {dt/60:.1f} min "
          f"-> {args.out}")
    if n_fail:
        print(f"[build] {n_fail} failures logged to {fail_path}")


if __name__ == "__main__":
    main()
