# HAMMER + VLM Implementation Plan for DGM4

## 1. Goal

Implement a VLM-enhanced version of HAMMER for the DGM4 dataset.

The VLM will be used as an offline semantic teacher. It will read each image-caption pair and generate structured JSON evidence. HAMMER will then consume this cached evidence during training through auxiliary distillation losses.

Target dissertation framing:

```text
VLM-assisted semantic evidence distillation for multimodal media manipulation detection and grounding.
```

Important scope decisions:

- Do not replace HAMMER with the VLM.
- Do not run the VLM inside HAMMER training.
- Do not enable VLM bbox distillation in the first implementation.
- Evaluate using the existing DGM4/HAMMER metrics.

Canonical implementation file:

```text
D:\Projects\MultiModal\HAMMER+VLM\plan_codex.md
```

## 2. Related Work Context

### HAMMER / DGM4

HAMMER is the official DGM4 baseline. It jointly predicts:

- binary real/fake classification
- multi-label manipulation type: face swap, face attribute, text swap, text attribute
- manipulated image bbox
- manipulated text tokens

This project keeps HAMMER as the main model and adds VLM-generated semantic supervision.

Reference:

- Shao et al., "Detecting and Grounding Multi-Modal Media Manipulation", CVPR 2023 / TPAMI 2024.
- https://arxiv.org/abs/2304.02556

### ASAP

ASAP is the closest related work. It uses off-the-shelf large models during training to generate auxiliary captions/explanations for better semantic alignment on DGM4.

How this project differs:

- ASAP uses generated auxiliary text for alignment.
- This project uses structured Qwen JSON evidence as soft targets for HAMMER's existing heads.

Reference:

- Zhang et al., "ASAP: Advancing Semantic Alignment Promotes Multi-Modal Manipulation Detecting and Grounding", CVPR 2025.
- https://openaccess.thecvf.com/content/CVPR2025/html/Zhang_ASAP_Advancing_Semantic_Alignment_Promotes_Multi-Modal_Manipulation_Detecting_and_Grounding_CVPR_2025_paper.html

### FKA-Owl / FakeNewsGPT4

FKA-Owl uses an LVLM for fake news detection and augments it with forgery-specific and semantic-correlation knowledge.

Relevant takeaway:

- Raw LVLMs are not enough for DGM4-style manipulation detection.
- VLM semantic knowledge is useful, but task-specific grounding and manipulation heads are still needed.

Reference:

- Liu et al., "FKA-Owl: Advancing Multimodal Fake News Detection through Knowledge-Augmented LVLMs", ACM MM 2024.
- https://arxiv.org/abs/2403.01988

### MFC-Bench and MMFakeBench

These works benchmark LVLMs directly on multimodal fact-checking and misinformation tasks.

Relevant takeaway:

- Direct VLM prompting is a useful comparison point, but it is not the strongest way to solve DGM4.
- A specialized detector such as HAMMER should remain the trainable model.

References:

- MFC-Bench: https://arxiv.org/abs/2406.11288
- MMFakeBench: https://proceedings.iclr.cc/paper_files/paper/2025/hash/d6c53fe062716387ff0df73cc53de60c-Abstract-Conference.html

## 3. Hardware and Environment

Training environment:

```text
OS: Linux VM
GPU: 1 x NVIDIA RTX 5080
VRAM: 16 GB
Package manager: conda
```

### 3.1 Important RTX 5080 Constraint

The RTX 5080 is a Blackwell GPU. Do not use the original HAMMER README environment unchanged.

The original HAMMER setup recommends old PyTorch/CUDA versions:

```text
torch 1.10
torchvision 0.11.1
CUDA 11.3
```

That is likely incompatible with RTX 5080. Blackwell GPUs need CUDA 12.8+ support in the PyTorch build.

Implementation requirement:

- use a PyTorch build with CUDA 12.8+ support
- verify `torch.cuda.is_available()`
- verify a small tensor operation runs on the RTX 5080 before training

References:

- NVIDIA Blackwell compatibility guide: https://docs.nvidia.com/cuda/blackwell-compatibility-guide/
- NVIDIA driver/toolkit matrix: https://docs.nvidia.com/datacenter/tesla/drivers/latest/cuda-toolkit-driver-and-architecture-matrix.html
- PyTorch CUDA 12.8 wheels: https://pytorch.org/get-started/previous-versions/

### 3.2 Use Two Conda Environments

Use two environments on the same Linux VM.

#### Environment A: HAMMER training

Purpose:

- run HAMMER training and evaluation
- read cached VLM JSONL evidence

Requirements:

- PyTorch with CUDA 12.8+ support
- torchvision matching the selected PyTorch version
- old or compatible Transformers stack for HAMMER's vendored BERT code
- `ruamel_yaml`
- `timm`
- `scikit-learn`
- `scipy`
- `tensorboard`

Key warning:

- Keep `transformers` close to the original HAMMER requirement if possible.
- If `transformers==4.8.1` is difficult with the selected Python/PyTorch stack, solve this in the HAMMER environment only. Do not merge the Qwen environment into this one.

#### Environment B: VLM cache generation

Purpose:

- run Qwen on image-caption pairs
- write JSONL evidence cache

Recommended default VLM:

```text
Qwen/Qwen2.5-VL-7B-Instruct
```

RTX 5080 16 GB settings:

- use 4-bit quantization if needed
- batch size 1
- cap image token budget
- use `max_new_tokens <= 256`

Fallback VLM if 7B is unstable or too slow:

```text
Qwen/Qwen2.5-VL-3B-Instruct
```

Qwen reference:

- https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct
- https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct

### 3.3 Transformers Version Mismatch

This project must treat PyTorch/CUDA compatibility and Transformers compatibility as separate problems.

For RTX 5080 support, the HAMMER environment needs a modern PyTorch build with CUDA 12.8+ support. The official PyTorch install page provides CUDA 12.8 wheels for recent PyTorch versions.

However, that does not mean HAMMER should use modern Transformers.

HAMMER's `models/xbert.py` is a vendored ALBEF/Hugging Face BERT fork. It imports old Transformers internals such as:

```python
from transformers.file_utils import ModelOutput
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers.modeling_utils import PreTrainedModel
```

The repository also pins:

```text
transformers==4.8.1
```

Qwen2.5-VL, on the other hand, requires a modern Transformers stack that contains Qwen2.5-VL model classes and processors. Qwen cannot run under `transformers==4.8.1`.

Decision:

- HAMMER environment: modern PyTorch for RTX 5080, but keep Transformers close to the HAMMER-compatible version.
- VLM environment: modern Transformers for Qwen.
- Communication between environments: JSONL cache only.

Do not try to make one conda environment serve both HAMMER and Qwen unless there is a clear reason to spend time porting and retesting HAMMER's custom BERT fork.

Would HAMMER work on newer Transformers?

- It might import on some intermediate versions.
- It is not guaranteed on current Transformers.
- Even if imports succeed, subtle `PreTrainedModel`, output dataclass, tokenizer, and `from_pretrained` behavior changes can break training or checkpoint loading.
- The safer engineering decision is to keep HAMMER's Transformers dependency isolated and tested.

Practical setup recommendation:

1. Create the HAMMER env first.
2. Install modern PyTorch/CUDA for RTX 5080.
3. Install HAMMER's Python dependencies.
4. Prefer Python 3.10 for the HAMMER env, because it gives better odds of combining modern PyTorch wheels with older research-code dependencies.
5. Try `transformers==4.8.1`.
6. If old Transformers/tokenizers installation fails, do not keep chasing exact pins indefinitely. Either:
   - use the nearest old Transformers version that imports `models.xbert`, or
   - patch `models/xbert.py` imports for a newer Transformers version and then retest carefully.
7. Required smoke tests before any VLM work:
   - import `models.xbert`
   - instantiate `HAMMER`
   - load `bert-base-uncased`
   - run one dataloader batch
   - run one forward/backward pass
8. Keep the selected working HAMMER Transformers version pinned in an environment file.

Do not assume HAMMER works on current Transformers just because the imports can be patched. Checkpoint loading, tokenizer behavior, and output dataclass behavior must be verified with a one-step forward/backward run.

## 4. Implementation Summary

The implementation is a two-stage pipeline.

### Stage 1: Offline VLM Evidence Generation

Input:

- DGM4 metadata JSON
- DGM4 image files
- caption text

Output:

- JSONL cache with one VLM evidence record per sample

The VLM cache is generated once and reused during HAMMER training.

### Stage 2: HAMMER Distillation Training

HAMMER reads the VLM cache and adds auxiliary losses:

- optional VLM binary soft distillation
- VLM manipulation-type soft distillation for text-semantic labels first
- VLM token-level soft distillation

Do not use VLM bbox distillation initially.

Initial scientific restriction:

- Use token distillation.
- Use multi-label distillation on `text_swap` and `text_attribute`.
- Mask or heavily down-weight `face_swap` and `face_attribute` teacher columns, because Qwen is not a forensic artifact detector.
- Treat binary distillation as optional and low-weight. If enabled, mask image-only fake samples or use a very small weight so Qwen's semantic consistency score does not fight DGM4's forensic labels.

## 5. VLM Cache Format

Use JSONL. One line per sample.

Cache key:

```text
image + "::" + sha1(caption)
```

Reason:

- DGM4 ids may repeat across manipulated variants.
- Image path plus caption hash is safer than `id` alone.

Caption hashing rule:

- Use the exact caption string that HAMMER training sees after `pre_caption(ann["text"], max_words)`.
- Use the same UTF-8 encoding in cache generation and training.
- Do not lowercase, strip, normalize Unicode, or re-tokenize differently for the hash.
- Store both `text_sha1` and `prompt_caption` in debug logs for failed records.

Prompt safety rule:

- The cache builder may use the image path to load pixels.
- The VLM prompt must receive only the decoded image and caption.
- Never include image filename, directory, manipulation method name, `fake_cls`, `fake_image_box`, or `fake_text_pos` in the prompt.

Example record:

```json
{
  "id": 768092,
  "image": "DGM4/manipulation/HFGI/768092-HFGI.jpg",
  "text_sha1": "sha1-of-caption",
  "split": "train",
  "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
  "prompt_version": "v1",
  "parse_status": "ok",
  "fake_probability": 0.82,
  "manipulation_probs": {
    "face_swap": 0.15,
    "face_attribute": 0.25,
    "text_swap": 0.70,
    "text_attribute": 0.55
  },
  "unsupported_word_indices": [8, 13, 17],
  "word_scores": {
    "8": 0.90,
    "13": 0.80,
    "17": 0.85
  },
  "rationale": "The image does not clearly support the claimed action in the caption."
}
```

Required fields:

- `id`
- `image`
- `text_sha1`
- `split`
- `model_id`
- `prompt_version`
- `parse_status`
- `fake_probability`
- `manipulation_probs`
- `unsupported_word_indices`
- `word_scores`

Optional fields:

- `rationale`
- `image_caption`
- `expected_image_from_text`
- `suspicious_bbox_xyxy`
- `bbox_confidence`

Even if bbox fields are generated, they are not used for training at first.

Missing or failed records:

- `parse_status != "ok"` should produce `valid = 0`.
- missing cache records should produce `valid = 0`.
- masked losses must skip the term or use a safe denominator when all records in a batch are invalid.

## 6. VLM Prompt

The prompt must ask for strict JSON only.

Rules:

- Provide the image.
- Provide the caption.
- Build the caption with `dataset.utils.pre_caption(ann["text"], max_words)`.
- Provide a numbered caption word list from that pre-captioned string.
- Do not reveal DGM4 ground-truth labels.
- Do not expose image paths containing manipulation method names.
- Ask for semantic consistency and suspicious caption words.

Prompt template:

```text
You are analyzing a news image and its accompanying caption.

Caption:
"{caption}"

Numbered caption words:
[0] ...
[1] ...

Task:
Determine whether the image and caption are semantically consistent.
Look for unsupported people, actions, attributes, objects, places, events, or sentiment.
Return word indices for caption words that are not visually supported.

Return strict JSON only:
{
  "fake_probability": float between 0 and 1,
  "manipulation_probs": {
    "face_swap": float between 0 and 1,
    "face_attribute": float between 0 and 1,
    "text_swap": float between 0 and 1,
    "text_attribute": float between 0 and 1
  },
  "unsupported_word_indices": [integer word indices],
  "word_scores": {"word_index": float between 0 and 1},
  "rationale": "one concise sentence"
}
```

Generation settings:

- `do_sample=False`
- leave `temperature` unset when greedy decoding is used
- `max_new_tokens=256`
- retry once if JSON parsing fails
- log failed raw outputs separately

## 7. Files to Add

Add:

```text
configs/train_vlm_distill.yaml
configs/test_vlm_distill.yaml
dataset/vlm_cache.py
tools/build_vlm_cache.py
tools/validate_vlm_cache.py
tools/inspect_vlm_cache.py
scripts/run_vlm_cache_sample.sh
scripts/run_train_vlm_distill.sh
scripts/run_eval_vlm_distill.sh
```

Optional:

```text
explain.py
```

## 8. Files to Modify

Modify:

```text
dataset/dataset.py
models/HAMMER.py
train.py
test.py
```

Implementation rule:

- Baseline behavior must remain unchanged when `vlm_distill: false`.

## 9. Dataset Integration

Current dataset return:

```python
image, label, caption, fake_image_box, fake_text_pos_list, W, H
```

When VLM distillation is enabled, return:

```python
image, label, caption, fake_image_box, fake_text_pos_list, W, H, vlm_target
```

`vlm_target` should contain tensors:

```python
{
    "valid": scalar,
    "fake_prob": scalar,
    "multicls_probs": shape [4],
    "word_scores": shape [max_words],
    "word_mask": shape [max_words]
}
```

Default for missing cache records:

- `valid = 0`
- all probabilities = 0
- all masks = 0

This allows partial-cache training.

## 10. Token Mapping

DGM4 labels use word indices. HAMMER trains on BERT subword tokens.

Use the existing token mapping logic in `text_input_adjust()` as the model.

Required implementation:

1. VLM outputs `word_scores` over the pre-captioned caption words.
2. Generate the VLM prompt word list from `pre_caption(ann["text"], max_words).split(" ")`.
3. Do not generate the prompt word list from raw `ann["text"]`, because punctuation, casing, `<person>`, truncation, hyphens, and slashes are normalized by `pre_caption`.
4. Dataset returns `word_scores` with shape `[max_words]`.
5. `train.py` maps VLM word scores to BERT subword scores using `text_input.word_ids(i)`.
6. Use the same CLS/SEP convention as `text_input_adjust()`:
   - HAMMER keeps CLS in `input_ids`.
   - HAMMER removes the final SEP.
   - token labels and logits exclude CLS via `sequence_output[:, 1:]`.
   - the relevant `word_ids()` slice is `subword_idx[1:-1]`.
7. Token distillation loss is applied only to valid non-padding tokens.
8. Add a debug assertion or validation report that the largest VLM word index is less than the number of words in the pre-captioned caption.

Silent failure to avoid:

- If VLM token targets are shifted by one because CLS is included in one path and excluded in another, training will compile but `F1_tok` can degrade.

## 11. HAMMER Loss Additions

Add config keys:

```yaml
vlm_distill: true
vlm_cache_file: "vlm_cache/qwen25_train.jsonl"

loss_vlm_bic_wgt: 0.00
loss_vlm_mlc_wgt: 0.05
loss_vlm_tmg_wgt: 0.10
loss_vlm_bbox_wgt: 0.00
vlm_teacher_temperature: 1.0
vlm_mlc_use_classes: ["text_swap", "text_attribute"]
```

Initial setting:

- keep binary VLM distillation disabled at first (`loss_vlm_bic_wgt: 0.00`)
- enable TS/TA multi-label distillation
- enable token distillation
- keep bbox distillation disabled

### 11.1 Binary Distillation

Use `fake_probability` as a soft target for HAMMER's binary head.

Target:

```python
teacher = torch.stack([1.0 - p_fake, p_fake], dim=1)
```

Loss:

```python
loss_vlm_bic = F.kl_div(
    F.log_softmax(logits_real_fake / T, dim=1),
    teacher,
    reduction="batchmean"
) * (T * T)
```

Mask by `vlm_target["valid"]`.

Binary distillation is optional because Qwen's fake probability is mostly semantic. It can mark face-swap or face-attribute samples as visually consistent even when DGM4 labels them fake. If binary distillation is enabled later, use a very small weight such as `0.01` or `0.02`, or mask image-only manipulation classes.

### 11.2 Multi-Label Distillation

Use `manipulation_probs` as soft targets for:

- face swap
- face attribute
- text swap
- text attribute

Loss:

```python
loss_vlm_mlc = BCEWithLogits(logits_multicls, vlm_multicls_probs)
```

Mask by `vlm_target["valid"]`.

Initial class mask:

```python
# class order: FS, FA, TS, TA
vlm_mlc_class_weight = torch.tensor([0.0, 0.0, 1.0, 1.0], device=device)
```

Rationale:

- Qwen is more useful for text-semantic mismatch than for forensic face artifacts.
- Distilling FS/FA probabilities from Qwen can inject noise.

### 11.3 Token Distillation

Use VLM word scores after mapping them to BERT subword positions.

HAMMER's token grounding head is a 2-way classifier, not a single sigmoid head. `logits_tok` has shape:

```text
[batch, seq_without_cls, 2]
```

Class convention:

- class 0 = normal token
- class 1 = manipulated token

Derive a scalar fake logit before BCE:

```python
token_fake_logit = logits_tok[..., 1] - logits_tok[..., 0]
```

Loss:

```python
loss_vlm_tmg = BCEWithLogits(token_fake_logit, vlm_token_scores)
```

Mask by:

- attention mask
- VLM word/token mask
- cache validity flag

Implementation guard:

- If the mask has no valid elements in a batch, skip this loss or divide by `mask.sum().clamp_min(1.0)`.

### 11.4 Bbox Distillation

Initial setting:

```yaml
loss_vlm_bbox_wgt: 0.00
```

Rationale:

- Qwen can provide semantic localization, but DGM4 image grounding is partly forensic.
- VLM bbox supervision may highlight faces or salient people rather than the manipulated region.
- Keep HAMMER's original bbox losses unchanged.

## 12. Training Script Changes

### train.py

Required:

- support batches with or without `vlm_target`
- move VLM tensors to GPU
- map VLM word scores to token scores
- pass `vlm_target` into `HAMMER.forward`
- log:
  - `loss_vlm_bic` if enabled
  - `loss_vlm_mlc`
  - `loss_vlm_tmg`

Total loss:

```python
loss = original_hammer_loss
     + loss_vlm_bic_wgt * loss_vlm_bic
     + loss_vlm_mlc_wgt * loss_vlm_mlc
     + loss_vlm_tmg_wgt * loss_vlm_tmg
```

Implementation guard:

- If a VLM loss weight is `0.0`, skip computing that loss unless it is needed for logging.
- If a masked VLM loss has no valid elements, return `torch.zeros((), device=image.device)` for that term.

### models/HAMMER.py

Required:

```python
def forward(..., vlm_target=None, alpha=0, is_train=True):
```

Behavior:

- compute original HAMMER losses
- compute VLM losses only during training and only when `vlm_target` exists
- return original losses plus VLM losses
- keep inference unchanged

### test.py

Required:

- ignore `vlm_target` during official evaluation
- keep official metrics comparable to baseline

Optional:

- export predictions plus cached rationale for qualitative examples

## 13. Single-GPU Training Notes

The current scripts assume multi-GPU training.

Update scripts for one RTX 5080:

- `NUM_GPU=1`
- batch size reduced if OOM
- use gradient accumulation if needed
- ensure distributed launch does not expect 8 GPUs

Suggested first training config:

```yaml
batch_size_train: 8
batch_size_val: 16
dataset_division: 10
loss_vlm_bic_wgt: 0.00
loss_vlm_mlc_wgt: 0.05
loss_vlm_tmg_wgt: 0.10
loss_vlm_bbox_wgt: 0.00
vlm_mlc_use_classes: ["text_swap", "text_attribute"]
```

After the smoke test works, remove or increase `dataset_division`.

## 14. Evaluation

Use the existing DGM4 metrics.

Primary metrics:

- `AUC_cls`
- `ACC_cls`
- `EER_cls`
- `MAP`
- `OF1`
- `CF1`
- `F1_FS`
- `F1_FA`
- `F1_TS`
- `F1_TA`
- `F1_tok`
- `IOU_score`
- `IOU_ACC_50`

Expected useful improvements:

- `F1_TS`
- `F1_TA`
- `F1_tok`
- possibly `MAP`
- possibly small `AUC_cls` improvement

Expected unchanged or uncertain metrics:

- `F1_FS`
- `F1_FA`
- `IOU_score`
- `IOU_ACC_50`

## 15. Experiments

Minimum experiments:

| ID | Method |
| --- | --- |
| E0 | HAMMER baseline |
| E1 | HAMMER + VLM TS/TA multi-label distillation |
| E2 | HAMMER + VLM TS/TA multi-label + token distillation |

Optional ablations:

| ID | Method |
| --- | --- |
| A1 | E2 with half VLM loss weights |
| A2 | E2 with token distillation disabled |
| A3 | E2 with Qwen2.5-VL-3B instead of 7B |
| A4 | E2 plus low-weight binary distillation on semantic/text-manipulation samples |

Result table:

| Method | AUC | ACC | EER | MAP | OF1 | CF1 | F1_TS | F1_TA | F1_tok | IoU@0.5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HAMMER baseline | | | | | | | | | | |
| + VLM TS/TA MLC | | | | | | | | | | |
| + VLM TS/TA MLC/TMG | | | | | | | | | | |

## 16. Implementation Milestones

### Milestone 1: Environment and Baseline

Tasks:

- create HAMMER conda environment on Linux VM
- verify RTX 5080 CUDA support
- run one HAMMER dataloader batch
- run one training batch without VLM

Output:

- confirmed environment
- baseline smoke-test command

### Milestone 2: VLM Cache Generation

Tasks:

- create VLM conda environment
- implement `tools/build_vlm_cache.py`
- generate a small train cache
- implement `tools/validate_vlm_cache.py`

Output:

- `vlm_cache/qwen25_train_debug.jsonl`
- cache validation report

### Milestone 3: Dataset and Loss Integration

Tasks:

- implement `dataset/vlm_cache.py`
- extend `DGM4_Dataset`
- add VLM target mapping in `train.py`
- add VLM losses in `HAMMER.forward`

Output:

- single-batch train run with VLM losses logged

### Milestone 4: First Training Run

Tasks:

- generate a larger train cache
- train E1 and E2
- evaluate with official metrics

Output:

- validation/test metric table
- loss logs
- checkpoint paths

### Milestone 5: Qualitative Examples

Tasks:

- export several predictions
- show image, caption, predicted manipulation type, suspicious words, bbox, and VLM rationale

Output:

- examples for dissertation figures

## 17. Main Dissertation Takeaways

Expected positive takeaway:

```text
VLM-generated semantic evidence can improve HAMMER most clearly on text-based and cross-modal manipulation signals, especially text swap, text attribute, and manipulated-token grounding.
```

Important negative or neutral takeaway:

```text
The VLM is not expected to substantially improve forensic image grounding. HAMMER's task-specific visual branch remains responsible for bbox localization.
```

Final claim should be based on measured results:

- If `F1_TS`, `F1_TA`, `F1_tok`, or `MAP` improve, present VLM distillation as useful semantic supervision.
- If only qualitative explanations improve, present the VLM as an interpretability aid.
- If no metric improves, report that direct semantic distillation from Qwen is insufficient under the tested constraints and analyze likely causes.

## 18. Immediate Next Step

Implement in this order:

1. Linux environment setup for RTX 5080.
2. One-batch HAMMER baseline smoke test.
3. VLM cache generator.
4. Cache loader in dataset.
5. VLM losses in HAMMER.
6. One-batch VLM-distillation smoke test.
7. First training run on a reduced split.
