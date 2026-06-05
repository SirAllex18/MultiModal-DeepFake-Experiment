"""Offline VLM evidence cache for HAMMER distillation.

This module is the single source of truth for the cache *contract* shared by
the offline builder (``tools/build_vlm_cache.py``) and the training-time
consumer (``dataset/dataset.py``):

* the cache is a JSONL file, one VLM-evidence record per sample;
* a record is addressed by ``image + "::" + sha1(prompt_caption)`` where
  ``prompt_caption`` is the *exact* string the text encoder sees, i.e. the
  output of ``dataset.utils.pre_caption(ann["text"], max_words, lowercase)``.

Keeping the key derivation here (``make_key`` / ``hash_caption``) means the
builder and the consumer can never drift out of sync: both import the same
function. DGM4 ids repeat across manipulated variants, so we deliberately key
on image-path + caption-hash rather than ``id`` alone.

The consumer never needs ``transformers`` or any VLM dependency: it only reads
JSON and emits tensors, so it stays importable inside the (old-transformers)
HAMMER environment.
"""

import hashlib
import json

import torch


# Canonical DGM4 manipulation-class order used everywhere in HAMMER
# (cls_head output, multilabel metrics, fake_cls parsing).
MANIP_CLASSES = ("face_swap", "face_attribute", "text_swap", "text_attribute")

# Bump when the prompt or the JSON schema changes in a way that invalidates
# previously generated records. Stored per-record so a cache built with an old
# prompt can be detected and regenerated.
PROMPT_VERSION = "v3"


def hash_caption(prompt_caption: str) -> str:
    """SHA1 of the *exact* pre-captioned string, UTF-8 encoded.

    The builder and the consumer must pass byte-identical strings here. Do not
    lowercase / strip / normalise separately on either side: feed both the
    output of ``pre_caption(...)`` with the same ``max_words`` and ``lowercase``
    that training uses.
    """
    return hashlib.sha1(prompt_caption.encode("utf-8")).hexdigest()


def make_key(image: str, prompt_caption: str) -> str:
    """Cache key for a sample: ``"<image-path>::<sha1(caption)>"``."""
    return f"{image}::{hash_caption(prompt_caption)}"


def _empty_target(max_words: int) -> dict:
    """A fully-formed but inert target (``valid=0``).

    Returned for missing / failed / disabled records. Every field has a fixed
    shape so the default PyTorch collate batches a list of these cleanly, and
    the masked losses contribute exactly zero.
    """
    return {
        "valid": torch.zeros((), dtype=torch.float),
        "fake_prob": torch.zeros((), dtype=torch.float),
        "multicls_probs": torch.zeros(len(MANIP_CLASSES), dtype=torch.float),
        "word_scores": torch.zeros(max_words, dtype=torch.float),
        "word_mask": torch.zeros(max_words, dtype=torch.float),
    }


class VLMCache:
    """Read-only view over a JSONL VLM-evidence cache.

    Parameters
    ----------
    path:
        JSONL file produced by ``tools/build_vlm_cache.py``. If ``None`` or the
        file does not exist, the cache behaves as fully empty: every lookup
        returns an inert ``valid=0`` target. That keeps partial-cache (and
        no-cache) training working without special-casing the call sites.
    max_words:
        Width of the per-word target vectors. Must match ``config['max_words']``
        used by the dataset / training so word indices line up.
    verbose:
        Print a one-line load summary (record count) on construction.
    """

    def __init__(self, path=None, max_words: int = 50, verbose: bool = True):
        self.path = path
        self.max_words = int(max_words)
        self.records = {}
        self._n_lines = 0
        self._n_bad_lines = 0

        if path:
            self._load(path, verbose=verbose)
        elif verbose:
            print("[VLMCache] no cache file provided -> all targets valid=0")

    def _load(self, path, verbose=True):
        import os

        if not os.path.isfile(path):
            if verbose:
                print(f"[VLMCache] cache file not found: {path} -> all targets valid=0")
            return

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self._n_lines += 1
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    self._n_bad_lines += 1
                    continue
                image = rec.get("image")
                text_sha1 = rec.get("text_sha1")
                if not image or not text_sha1:
                    self._n_bad_lines += 1
                    continue
                # Last record wins on duplicate keys (resumed/overwritten runs).
                self.records[f"{image}::{text_sha1}"] = rec

        if verbose:
            print(
                f"[VLMCache] loaded {len(self.records)} records from {path} "
                f"({self._n_lines} lines, {self._n_bad_lines} skipped)"
            )

    def __len__(self):
        return len(self.records)

    def lookup(self, image: str, prompt_caption: str):
        """Return the raw record for a sample, or ``None`` if absent."""
        return self.records.get(make_key(image, prompt_caption))

    def build_target(self, image: str, prompt_caption: str) -> dict:
        """Build the per-sample ``vlm_target`` tensors.

        ``prompt_caption`` must be the already-``pre_caption``-ed string the
        dataset returns as ``caption`` (so the hash matches the builder).

        Returns a dict with fixed-shape tensors:
            valid          scalar  1.0 if a usable record was found
            fake_prob      scalar  VLM P(fake)
            multicls_probs [4]     VLM P(manip) in MANIP_CLASSES order
            word_scores    [max_words]  per-word suspicion in [0,1]
            word_mask      [max_words]  1.0 for real caption-word positions

        Missing record, ``parse_status != "ok"``, or any structural problem
        yields the inert ``valid=0`` target.
        """
        rec = self.lookup(image, prompt_caption)
        if rec is None or rec.get("parse_status") != "ok":
            return _empty_target(self.max_words)

        target = _empty_target(self.max_words)

        # --- binary ---
        try:
            target["fake_prob"] = torch.tensor(
                float(rec.get("fake_probability", 0.0)), dtype=torch.float
            ).clamp_(0.0, 1.0)
        except (TypeError, ValueError):
            return _empty_target(self.max_words)

        # --- multi-label (FS, FA, TS, TA) ---
        mp = rec.get("manipulation_probs", {}) or {}
        probs = torch.tensor(
            [float(mp.get(c, 0.0) or 0.0) for c in MANIP_CLASSES], dtype=torch.float
        ).clamp_(0.0, 1.0)
        target["multicls_probs"] = probs

        # --- per-word suspicion ---
        # word_mask marks the real caption words (0..num_words-1, capped at
        # max_words) so the token loss supervises supported words toward 0 and
        # flagged words toward their score. Indices the VLM emits outside the
        # caption range are dropped (and surfaced by the validator).
        num_words = min(len(prompt_caption.split(" ")), self.max_words)
        word_scores = torch.zeros(self.max_words, dtype=torch.float)
        word_mask = torch.zeros(self.max_words, dtype=torch.float)
        word_mask[:num_words] = 1.0

        scores_map = rec.get("word_scores", {}) or {}
        unsupported = rec.get("unsupported_word_indices", []) or []
        for idx in unsupported:
            try:
                i = int(idx)
            except (TypeError, ValueError):
                continue
            if 0 <= i < num_words:
                # explicit per-word score if present, else treat flag as 1.0
                s = scores_map.get(str(i), scores_map.get(i, 1.0))
                try:
                    word_scores[i] = float(s)
                except (TypeError, ValueError):
                    word_scores[i] = 1.0
        # Also honour any standalone word_scores entries not in `unsupported`.
        for k, v in scores_map.items():
            try:
                i = int(k)
                if 0 <= i < num_words:
                    word_scores[i] = max(float(word_scores[i]), float(v))
            except (TypeError, ValueError):
                continue
        word_scores.clamp_(0.0, 1.0)

        target["word_scores"] = word_scores
        target["word_mask"] = word_mask
        target["valid"] = torch.ones((), dtype=torch.float)
        return target


def empty_target(max_words: int) -> dict:
    """Public helper: an inert ``valid=0`` target (for the disabled path)."""
    return _empty_target(max_words)
