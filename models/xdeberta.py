# coding=utf-8
"""
Multimodal DeBERTa-V2/V3 for the HAMMER pipeline.

This file mirrors the public interface that ``xbert.py`` exposes to
``HAMMER.py`` but replaces BERT with DeBERTa-V3 (built on the V2 architecture).

Key features that the HAMMER training/inference loop depends on are preserved:

* ``mode='text' | 'fusion' | 'multi_modal'`` dispatch on the encoder so that
  the first ``fusion_layer`` blocks can run on text alone and the remaining
  blocks fuse text with image features.
* Cross-attention sublayer at every block ``layer_num >= fusion_layer`` that
  lets the text stream attend to the image stream (``encoder_hidden_states``).
  Cross-attention uses standard scaled-dot-product (no disentangled bias),
  because image patches do not have a relative-position relationship with
  text tokens.
* Self-attention at every layer keeps DeBERTa's disentangled attention with
  shared content/position projections and shared relative-position embeddings
  (DeBERTa-V3 ``share_att_key=True``, ``pos_att_type=['p2c','c2p']``).
* ``encoder_embeds`` argument so that the fusion stack can be re-entered with
  the text representations produced by an earlier ``mode='text'`` pass.
* ``DebertaV2ForTokenClassification`` mirror with ``label_smoothing``,
  ``soft_labels``/``alpha`` distillation, and ``return_logits`` flag, exactly
  as the HAMMER token-grounding head expects.

The class with the trainable model is ``DebertaV2ForTokenClassification`` and
its inner backbone is named ``self.deberta`` (to match the upstream weight
prefix used by ``DebertaV2Model.from_pretrained('microsoft/deberta-v3-base')``).
"""

import inspect as _inspect
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPoolingAndCrossAttentions,
    TokenClassifierOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.models.deberta_v2.configuration_deberta_v2 import DebertaV2Config

# ---------------------------------------------------------------------------
# Reuse DeBERTa-V2 building blocks from the installed transformers package.
# These are the parts of DeBERTa we do not want to re-implement (disentangled
# self-attention, relative-position bucketing, the optional Conv layer, ...).
# ---------------------------------------------------------------------------
from transformers.models.deberta_v2.modeling_deberta_v2 import (
    DebertaV2Embeddings,
    DebertaV2Intermediate,
    DebertaV2Output,
    DebertaV2Attention,
    DebertaV2SelfOutput,
    build_relative_position,
)

try:  # ConvLayer only exists when conv_kernel_size > 0;
    from transformers.models.deberta_v2.modeling_deberta_v2 import ConvLayer
except Exception:  # pragma: no cover - older transformers without ConvLayer.
    ConvLayer = None


# ``build_relative_position`` changed signatures between transformers 4.x
# (integer sizes + device kwarg) and 5.x (tensor query/key layers). Detect
# once at import so the encoder forward stays a single hot-path.
_BUILD_REL_POS_PARAM_NAMES = tuple(
    _inspect.signature(build_relative_position).parameters.keys()
)
_BUILD_REL_POS_TAKES_TENSORS = _BUILD_REL_POS_PARAM_NAMES[:2] == (
    "query_layer",
    "key_layer",
)


def _safe_build_relative_position(query, key, bucket_size, max_position):
    """Version-agnostic shim around HF's ``build_relative_position``.

    Both the 4.x (int sizes) and 5.x (tensors) signatures end up returning a
    ``[1, query_size, key_size]`` tensor on the device of ``query``.
    """
    if _BUILD_REL_POS_TAKES_TENSORS:
        return build_relative_position(
            query, key, bucket_size=bucket_size, max_position=max_position
        )
    return build_relative_position(
        query.size(-2),
        key.size(-2),
        bucket_size=bucket_size,
        max_position=max_position,
        device=query.device,
    )


# ---------------------------------------------------------------------------
# Cross-attention block (text Q, image K/V).
# ---------------------------------------------------------------------------
class CrossAttentionSelf(nn.Module):
    """Standard multi-head cross-attention: text tokens attend to image patches.

    No disentangled / relative-position bias is used here. The encoder side
    (image features) has no positional alignment with text tokens, so adding
    text-relative buckets would inject a meaningless prior.
    """

    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "hidden_size %d not divisible by num_attention_heads %d"
                % (config.hidden_size, config.num_attention_heads)
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        encoder_width = getattr(config, "encoder_width", config.hidden_size)
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(encoder_width, self.all_head_size)
        self.value = nn.Linear(encoder_width, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        return x.view(*new_shape).permute(0, 2, 1, 3)

    def forward(self, hidden_states, encoder_hidden_states, encoder_attention_mask=None):
        q = self.transpose_for_scores(self.query(hidden_states))
        k = self.transpose_for_scores(self.key(encoder_hidden_states))
        v = self.transpose_for_scores(self.value(encoder_hidden_states))

        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.attention_head_size)
        if encoder_attention_mask is not None:
            # encoder_attention_mask is additive ([B,1,1,enc_seq], 0 / -10000).
            scores = scores + encoder_attention_mask
        probs = F.softmax(scores, dim=-1)
        probs = self.dropout(probs)

        ctx = torch.matmul(probs, v)
        ctx = ctx.permute(0, 2, 1, 3).contiguous()
        ctx = ctx.view(ctx.size(0), ctx.size(1), self.all_head_size)
        return ctx


class CrossAttentionBlock(nn.Module):
    """Self-attention output + cross-attention residual + LayerNorm."""

    def __init__(self, config):
        super().__init__()
        self.cross = CrossAttentionSelf(config)
        # Reuse DeBERTa's residual + LayerNorm pattern (matches the rest of
        # the layer for consistent initialisation behaviour).
        self.output = DebertaV2SelfOutput(config)

    def forward(self, hidden_states, encoder_hidden_states, encoder_attention_mask=None):
        ctx = self.cross(hidden_states, encoder_hidden_states, encoder_attention_mask)
        return self.output(ctx, hidden_states)


# ---------------------------------------------------------------------------
# A DeBERTa layer that optionally has a cross-attention sublayer.
# ---------------------------------------------------------------------------
class DebertaV2LayerMM(nn.Module):
    """DeBERTa-V2 transformer layer with an optional cross-attention block.

    Order inside the layer:
        1. self-attention (DeBERTa disentangled)
        2. cross-attention to ``encoder_hidden_states`` if ``has_cross_attention``
        3. intermediate + output (FFN with residual)
    """

    def __init__(self, config, layer_num: int):
        super().__init__()
        self.config = config
        self.layer_num = layer_num
        self.has_cross_attention = layer_num >= config.fusion_layer

        self.attention = DebertaV2Attention(config)
        if self.has_cross_attention:
            self.crossattention = CrossAttentionBlock(config)
        self.intermediate = DebertaV2Intermediate(config)
        self.output = DebertaV2Output(config)

    def forward(
        self,
        hidden_states,
        attention_mask,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        relative_pos=None,
        rel_embeddings=None,
        output_attentions: bool = False,
    ):
        # ``DebertaV2Attention`` has returned either a bare tensor or a tuple
        # across transformers releases. Normalize both forms here so the
        # multimodal wrapper is not tied to that internal detail.
        attn_result = self.attention(
            hidden_states,
            attention_mask,
            output_attentions=output_attentions,
            query_states=None,
            relative_pos=relative_pos,
            rel_embeddings=rel_embeddings,
        )
        if isinstance(attn_result, tuple):
            attn_output = attn_result[0]
            att_matrix = attn_result[1] if len(attn_result) > 1 else None
        else:
            attn_output, att_matrix = attn_result, None

        if self.has_cross_attention and encoder_hidden_states is not None:
            attn_output = self.crossattention(
                attn_output,
                encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
            )

        intermediate = self.intermediate(attn_output)
        layer_output = self.output(intermediate, attn_output)
        return layer_output, att_matrix


# ---------------------------------------------------------------------------
# Encoder with mode dispatch (text / fusion / multi_modal) and shared
# relative-position embeddings.
# ---------------------------------------------------------------------------
class DebertaV2EncoderMM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.layer = nn.ModuleList(
            [DebertaV2LayerMM(config, layer_num=i) for i in range(config.num_hidden_layers)]
        )

        self.relative_attention = getattr(config, "relative_attention", False)
        if self.relative_attention:
            self.max_relative_positions = getattr(config, "max_relative_positions", -1)
            if self.max_relative_positions < 1:
                self.max_relative_positions = config.max_position_embeddings
            self.position_buckets = getattr(config, "position_buckets", -1)
            pos_ebd_size = self.max_relative_positions * 2
            if self.position_buckets > 0:
                pos_ebd_size = self.position_buckets * 2
            self.rel_embeddings = nn.Embedding(pos_ebd_size, config.hidden_size)

        self.norm_rel_ebd = [x.strip() for x in getattr(config, "norm_rel_ebd", "none").lower().split("|")]
        if "layer_norm" in self.norm_rel_ebd:
            self.LayerNorm = nn.LayerNorm(config.hidden_size, config.layer_norm_eps, elementwise_affine=True)

        if ConvLayer is not None and getattr(config, "conv_kernel_size", 0) > 0:
            self.conv = ConvLayer(config)
        else:
            self.conv = None

        self.gradient_checkpointing = False

    # --- helpers (ported / adapted from HF DebertaV2Encoder) ----------------
    def get_rel_embedding(self):
        rel_embeddings = self.rel_embeddings.weight if self.relative_attention else None
        if rel_embeddings is not None and "layer_norm" in self.norm_rel_ebd:
            rel_embeddings = self.LayerNorm(rel_embeddings)
        return rel_embeddings

    def get_attention_mask(self, attention_mask):
        if attention_mask.dim() <= 2:
            extended = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = extended * extended.squeeze(-2).unsqueeze(-1)
        elif attention_mask.dim() == 3:
            attention_mask = attention_mask.unsqueeze(1)
        return attention_mask

    def get_rel_pos(self, hidden_states, query_states=None, relative_pos=None):
        if self.relative_attention and relative_pos is None:
            q = query_states if query_states is not None else hidden_states
            relative_pos = _safe_build_relative_position(
                q,
                hidden_states,
                bucket_size=self.position_buckets,
                max_position=self.max_relative_positions,
            )
        return relative_pos

    @staticmethod
    def _invert_encoder_attention_mask(encoder_attention_mask, dtype):
        """Convert a 2D encoder mask into an additive [B,1,1,enc_seq] mask."""
        if encoder_attention_mask is None:
            return None
        if encoder_attention_mask.dim() == 2:
            mask = encoder_attention_mask[:, None, None, :]
        elif encoder_attention_mask.dim() == 3:
            mask = encoder_attention_mask[:, None, :, :]
        else:
            mask = encoder_attention_mask
        mask = mask.to(dtype=dtype)
        return (1.0 - mask) * torch.finfo(dtype).min

    # --- forward ------------------------------------------------------------
    def forward(
        self,
        hidden_states,
        attention_mask,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        return_dict: bool = True,
        mode: str = "multi_modal",
    ):
        if mode == "text":
            start_layer, output_layer = 0, self.config.fusion_layer
        elif mode == "fusion":
            start_layer, output_layer = self.config.fusion_layer, self.config.num_hidden_layers
        else:  # 'multi_modal'
            start_layer, output_layer = 0, self.config.num_hidden_layers

        # 2D (or 3D) -> bool/expanded mask used by DisentangledSelfAttention.
        if attention_mask.dim() <= 2:
            input_mask = attention_mask
        else:
            input_mask = attention_mask.sum(-2) > 0
        attention_mask_ext = self.get_attention_mask(attention_mask)
        relative_pos = self.get_rel_pos(hidden_states, query_states=None, relative_pos=None)

        # Encoder (image) attention mask for cross-attention. Convert from the
        # 0/1 mask the caller supplies into the additive form used inside our
        # CrossAttentionSelf module.
        encoder_attn_mask_additive = self._invert_encoder_attention_mask(
            encoder_attention_mask, dtype=hidden_states.dtype
        )

        all_hidden_states = (hidden_states,) if output_hidden_states else None
        all_attentions = () if output_attentions else None

        rel_embeddings = self.get_rel_embedding()

        next_kv = hidden_states
        # The conv layer in DeBERTa-V2/V3 is applied right after layer 0 to mix
        # local context. We only run it in modes where layer 0 is part of the
        # forward; in 'fusion' mode the input already had the conv applied.
        run_conv_after_first = mode != "fusion" and self.conv is not None

        for i in range(start_layer, output_layer):
            layer_module = self.layer[i]
            layer_output, attn_weights = layer_module(
                next_kv,
                attention_mask_ext,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attn_mask_additive,
                relative_pos=relative_pos,
                rel_embeddings=rel_embeddings,
                output_attentions=output_attentions,
            )

            if i == 0 and run_conv_after_first:
                layer_output = self.conv(hidden_states, layer_output, input_mask)

            if output_attentions:
                all_attentions = all_attentions + (attn_weights,)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (layer_output,)

            next_kv = layer_output

        if not return_dict:
            return tuple(v for v in [next_kv, all_hidden_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=next_kv,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )


# ---------------------------------------------------------------------------
# Pooler (CLS pooling, matches HF DebertaV2 ContextPooler shape).
# ---------------------------------------------------------------------------
class DebertaV2PoolerMM(nn.Module):
    def __init__(self, config):
        super().__init__()
        pooler_hidden_size = getattr(config, "pooler_hidden_size", config.hidden_size)
        self.dense = nn.Linear(config.hidden_size, pooler_hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token = hidden_states[:, 0]
        return self.activation(self.dense(first_token))


# ---------------------------------------------------------------------------
# PreTrainedModel base.
# ---------------------------------------------------------------------------
class DebertaV2PreTrainedModelMM(PreTrainedModel):
    config_class = DebertaV2Config
    base_model_prefix = "deberta"
    supports_gradient_checkpointing = False
    _keys_to_ignore_on_load_missing = [r"position_ids"]
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


# ---------------------------------------------------------------------------
# Backbone with multimodal forward signature.
# ---------------------------------------------------------------------------
class DebertaV2ModelMM(DebertaV2PreTrainedModelMM):
    """DeBERTa-V2 backbone exposing the same multimodal forward interface as
    ``BertModel`` in ``xbert.py``: ``encoder_embeds``, ``encoder_hidden_states``,
    ``encoder_attention_mask``, and ``mode``.
    """

    def __init__(self, config, add_pooling_layer: bool = False):
        super().__init__(config)
        self.config = config
        self.embeddings = DebertaV2Embeddings(config)
        self.encoder = DebertaV2EncoderMM(config)
        self.pooler = DebertaV2PoolerMM(config) if add_pooling_layer else None
        # In Hugging Face's PreTrainedModel API ``post_init`` does both weight
        # init and any submodule tying. ``init_weights`` is the older name and
        # is still available; using post_init keeps us aligned with modern HF.
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        encoder_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        mode: str = "multi_modal",
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Specify either input_ids or inputs_embeds, not both.")
        if encoder_embeds is not None:
            input_shape = encoder_embeds.size()[:-1]
            device = encoder_embeds.device
        elif input_ids is not None:
            input_shape = input_ids.size()
            device = input_ids.device
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            device = inputs_embeds.device
        else:
            raise ValueError("You must provide input_ids, inputs_embeds, or encoder_embeds.")

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        if encoder_embeds is None:
            embedding_output = self.embeddings(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                mask=attention_mask,
                inputs_embeds=inputs_embeds,
            )
        else:
            # Caller hands us already-embedded representations (e.g. text
            # tokens that have been processed by mode='text' earlier).
            embedding_output = encoder_embeds

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=return_dict,
            mode=mode,
        )

        sequence_output = encoder_outputs[0] if not return_dict else encoder_outputs.last_hidden_state
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + tuple(encoder_outputs[1:])

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


# ---------------------------------------------------------------------------
# Token-classification head used by HAMMER for the TMG (Text-grounding) loss.
# Mirrors xbert.BertForTokenClassification one-for-one, with the inner model
# attribute renamed from ``bert`` to ``deberta`` to match the upstream weight
# prefix.
# ---------------------------------------------------------------------------
class DebertaV2ForTokenClassification(DebertaV2PreTrainedModelMM):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config, label_smoothing: float = 0.0):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.deberta = DebertaV2ModelMM(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = self._build_mlp(config.hidden_size, config.num_labels)
        self.label_smoothing = label_smoothing
        self.post_init()

    @staticmethod
    def _build_mlp(input_dim: int, output_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.LayerNorm(input_dim * 2),
            nn.GELU(),
            nn.Linear(input_dim * 2, input_dim * 2),
            nn.LayerNorm(input_dim * 2),
            nn.GELU(),
            nn.Linear(input_dim * 2, output_dim),
        )

    # The ``bert`` attribute is what the HAMMER model accesses on the
    # text encoder for raw forward passes (``self.text_encoder.bert(...)``).
    # We expose ``deberta`` under both names so HAMMER.py stays clean.
    @property
    def bert(self):  # backwards-compatible alias
        return self.deberta

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        encoder_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        mode: str = "multi_modal",
        soft_labels=None,
        alpha: float = 0.0,
        return_logits: bool = False,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            encoder_embeds=encoder_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            mode=mode,
        )

        sequence_output = outputs.last_hidden_state
        # Drop the CLS token (matches xbert.BertForTokenClassification).
        sequence_output = self.dropout(sequence_output[:, 1:])
        logits = self.classifier(sequence_output)

        if return_logits:
            return logits

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(label_smoothing=self.label_smoothing)
            if attention_mask is not None:
                attn_no_cls = attention_mask[:, 1:]
                active = attn_no_cls.reshape(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                active_logits = logits.view(-1, self.num_labels)
                loss = loss_fct(active_logits, labels.view(-1))

        if soft_labels is not None:
            loss_distill = -torch.sum(F.log_softmax(active_logits, dim=-1) * soft_labels, dim=-1)
            labels_flat = labels.view(-1)
            loss_distill = loss_distill[labels_flat != -100].mean()
            loss = (1 - alpha) * loss + alpha * loss_distill

        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# Public re-exports so HAMMER.py can ``from models.xdeberta import ...``
__all__ = [
    "DebertaV2Config",
    "DebertaV2ModelMM",
    "DebertaV2ForTokenClassification",
]
