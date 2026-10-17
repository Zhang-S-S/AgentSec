from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .config import ModelConfig
from .ops import MILPool, MaskedAttentionPool, masked_softmax


def projection(input_dim: int, hidden_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)
    )


class ContentModalityEncoder(nn.Module):
    """Sparse MIL evidence and set-level behavior branches for one content type."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        dim = config.hidden_dim
        self.instance_risk = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim // 2), nn.GELU(),
                                           nn.Linear(dim // 2, 1))
        self.mil = MILPool(config.mil_pooling, config.mil_temperature, config.top_k)
        self.attention_temperature = config.attention_temperature
        self.sparse_fusion = nn.Sequential(nn.Linear(dim + 1, dim), nn.GELU(), nn.LayerNorm(dim))
        self.behavior_enabled = config.behavior_encoder
        if self.behavior_enabled:
            layer = nn.TransformerEncoderLayer(
                d_model=dim, nhead=config.transformer_heads,
                dim_feedforward=config.transformer_ffn_dim, dropout=config.dropout,
                activation="gelu", batch_first=True, norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(layer, config.transformer_layers,
                                                     enable_nested_tensor=False)
            self.behavior_pool = MaskedAttentionPool(dim)
            self.branch_gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.GELU(), nn.Linear(dim, dim))
        self.output_norm = nn.LayerNorm(dim)

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        batch, length, dim = values.shape
        present = mask.any(dim=1)
        instance_logits = self.instance_risk(values).squeeze(-1) if length else values.new_zeros((batch, 0))
        instance_weights = masked_softmax(instance_logits / self.attention_temperature, mask)
        sparse_vector = torch.sum(instance_weights.unsqueeze(-1) * values, dim=1)
        mil_logit = self.mil(instance_logits, mask)
        sparse = self.sparse_fusion(torch.cat([sparse_vector, mil_logit.unsqueeze(-1)], dim=-1))
        sparse = sparse * present.unsqueeze(-1)

        if self.behavior_enabled:
            behavior = values.new_zeros((batch, dim))
            behavior_weights = values.new_zeros((batch, length))
            # TransformerEncoder can produce NaN for all-masked rows, so only run
            # the subset having at least one valid instance.
            if length and present.any():
                selected = present.nonzero(as_tuple=False).squeeze(-1)
                selected_values = values.index_select(0, selected)
                selected_mask = mask.index_select(0, selected)
                selected_encoded = self.transformer(
                    selected_values, src_key_padding_mask=~selected_mask
                )
                selected_behavior, selected_weights = self.behavior_pool(selected_encoded, selected_mask)
                behavior = behavior.index_copy(0, selected, selected_behavior)
                behavior_weights = behavior_weights.index_copy(0, selected, selected_weights)
            gate = torch.sigmoid(self.branch_gate(torch.cat([sparse, behavior], dim=-1)))
            combined = self.output_norm(gate * sparse + (1 - gate) * behavior)
            combined = combined * present.unsqueeze(-1)
        else:
            behavior = values.new_zeros((batch, dim))
            behavior_weights = values.new_zeros((batch, length))
            gate = values.new_ones((batch, dim)) * present.unsqueeze(-1)
            combined = sparse
        return {
            "representation": combined,
            "sparse_representation": sparse,
            "behavior_representation": behavior,
            "instance_logits": instance_logits.masked_fill(~mask, 0.0),
            "evidence_weights": instance_weights,
            "behavior_weights": behavior_weights,
            "branch_gate": gate,
            "mil_logit": mil_logit,
            "present": present,
        }


class URLEncoder(nn.Module):
    """Numerical lexical features + byte CNN + source embedding."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        dim = config.hidden_dim
        self.byte_embedding = nn.Embedding(257, 32, padding_idx=0)
        self.convolutions = nn.ModuleList([nn.Conv1d(32, 48, width) for width in (3, 4, 5)])
        self.byte_projection = nn.Linear(48 * 3, dim // 2)
        self.numeric = nn.Sequential(nn.Linear(config.url_feature_dim, dim // 2), nn.GELU(),
                                     nn.LayerNorm(dim // 2))
        self.source_embedding = nn.Embedding(4, 32)
        self.fusion = nn.Sequential(nn.Linear(dim + 32, dim), nn.GELU(), nn.LayerNorm(dim))
        self.pooling = config.url_pooling
        self.attention_pool = MaskedAttentionPool(dim)

    def forward(self, byte_ids: torch.Tensor, numeric: torch.Tensor,
                source: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, count, length = byte_ids.shape
        dim = self.fusion[-1].normalized_shape[0]
        if count == 0:
            return numeric.new_zeros((batch, dim)), numeric.new_zeros((batch, 0)), numeric.new_zeros((batch, 0, dim))
        flat = byte_ids.reshape(batch * count, length)
        embedded = self.byte_embedding(flat).transpose(1, 2)
        byte_parts = []
        for convolution in self.convolutions:
            # Collate always pads byte strings to at least the largest kernel.
            byte_parts.append(torch.amax(F.gelu(convolution(embedded)), dim=-1))
        byte_features = self.byte_projection(torch.cat(byte_parts, dim=-1)).reshape(batch, count, -1)
        numeric_features = self.numeric(numeric)
        source_features = self.source_embedding(source)
        items = self.fusion(torch.cat([numeric_features, byte_features, source_features], dim=-1))
        items = items * mask.unsqueeze(-1)
        if self.pooling == "attention":
            pooled, weights = self.attention_pool(items, mask)
        else:
            denominator = mask.sum(dim=1, keepdim=True).clamp_min(1).to(items.dtype)
            pooled = items.sum(dim=1) / denominator
            weights = mask.to(items.dtype) / denominator
        return pooled, weights, items


class MultiSourceRiskModel(nn.Module):
    """Moltbook risk detector whose prediction always includes the content trunk."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        self.config.validate()
        c, dim = self.config, self.config.hidden_dim
        self.content_projection = projection(c.embedding_dim, dim, c.dropout)
        self.post_type = nn.Parameter(torch.empty(dim).normal_(std=0.02))
        self.comment_type = nn.Parameter(torch.empty(dim).normal_(std=0.02))
        self.post_encoder = ContentModalityEncoder(c)
        self.comment_encoder = ContentModalityEncoder(c)
        self.content_score = nn.Sequential(nn.Linear(dim, dim // 2), nn.Tanh(), nn.Linear(dim // 2, 1))
        self.moltbook_profile = projection(c.embedding_dim, dim, c.dropout)
        self.twitter_profile = projection(c.embedding_dim, dim, c.dropout)
        self.alignment = nn.Sequential(nn.Linear(dim * 4 + 1, dim), nn.GELU(), nn.LayerNorm(dim))
        self.url_encoder = URLEncoder(c)
        self.aux_score = nn.Sequential(nn.Linear(dim * 2, dim // 2), nn.Tanh(), nn.Linear(dim // 2, 1))
        self.aux_gate = nn.Sequential(nn.Linear(dim * 2 + 4, dim), nn.GELU(), nn.Linear(dim, 1))
        self.account_norm = nn.LayerNorm(dim)
        self.direct_fusion = nn.Sequential(nn.Linear(dim * 6 + 6, dim), nn.GELU(), nn.LayerNorm(dim))
        self.classifier = nn.Sequential(nn.Linear(dim, 128), nn.GELU(), nn.Dropout(c.dropout), nn.Linear(128, 1))

    @staticmethod
    def _apply_dropout(values: torch.Tensor, mask: torch.Tensor, probability: float,
                       training: bool) -> tuple[torch.Tensor, torch.Tensor]:
        if not training or probability == 0:
            return values * mask.unsqueeze(-1), mask
        keep = torch.rand(mask.shape, device=mask.device) >= probability
        effective = mask & keep
        return values * effective.unsqueeze(-1), effective

    def _profile(self, values: torch.Tensor, mask: torch.Tensor,
                 encoder: nn.Module) -> torch.Tensor:
        return encoder(values) * mask.unsqueeze(-1)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, Any]:
        post_mask, comment_mask = batch["post_mask"].bool(), batch["comment_mask"].bool()
        original_post_present, original_comment_present = post_mask.any(1), comment_mask.any(1)
        if self.config.content_mode == "post_only":
            if not original_post_present.all():
                raise ValueError("post_only received an account without a valid Post")
            comment_mask = torch.zeros_like(comment_mask)
        elif self.config.content_mode == "comment_only":
            if not original_comment_present.all():
                raise ValueError("comment_only received an account without a valid Comment")
            post_mask = torch.zeros_like(post_mask)
        elif not (original_post_present | original_comment_present).all():
            raise ValueError("Every account must have at least one valid Post or Comment")

        post_values = self.content_projection(batch["posts"]) + self.post_type
        comment_values = self.content_projection(batch["comments"]) + self.comment_type
        post_values = post_values * post_mask.unsqueeze(-1)
        comment_values = comment_values * comment_mask.unsqueeze(-1)
        post = self.post_encoder(post_values, post_mask)
        comment = self.comment_encoder(comment_values, comment_mask)
        content_stack = torch.stack([post["representation"], comment["representation"]], dim=1)
        content_masks = torch.stack([post["present"], comment["present"]], dim=1)
        content_weights = masked_softmax(self.content_score(content_stack).squeeze(-1), content_masks)
        z_content = torch.sum(content_weights.unsqueeze(-1) * content_stack, dim=1)

        m_mask = batch["moltbook_profile_mask"].bool() & self.config.use_moltbook_profile
        t_mask = batch["twitter_profile_mask"].bool() & self.config.use_twitter_profile
        z_m = self._profile(batch["moltbook_profile"], m_mask, self.moltbook_profile)
        z_t = self._profile(batch["twitter_profile"], t_mask, self.twitter_profile)
        z_m, m_mask = self._apply_dropout(z_m, m_mask, self.config.moltbook_profile_dropout, self.training)
        z_t, t_mask = self._apply_dropout(z_t, t_mask, self.config.twitter_profile_dropout, self.training)
        align_mask = m_mask & t_mask & self.config.use_alignment
        cosine = F.cosine_similarity(z_m, z_t, dim=-1, eps=1e-8).unsqueeze(-1)
        z_align = self.alignment(torch.cat([z_m, z_t, (z_m - z_t).abs(), z_m * z_t, cosine], dim=-1))
        z_align = z_align * align_mask.unsqueeze(-1)

        url_mask = batch["url_mask"].bool() & self.config.use_url
        z_url, url_weights, url_items = self.url_encoder(
            batch["url_bytes"], batch["url_features"], batch["url_source"], url_mask
        )
        url_present = url_mask.any(1)
        z_url, url_present = self._apply_dropout(z_url, url_present, self.config.url_dropout, self.training)
        effective_url_mask = url_mask & url_present.unsqueeze(-1)
        url_weights = url_weights * url_present.unsqueeze(-1)
        url_items = url_items * url_present.unsqueeze(-1).unsqueeze(-1)

        aux_stack = torch.stack([z_m, z_t, z_align, z_url], dim=1)
        aux_masks = torch.stack([m_mask, t_mask, align_mask, url_present], dim=1)
        conditioned = torch.cat([aux_stack, z_content.unsqueeze(1).expand(-1, 4, -1)], dim=-1)
        aux_weights = masked_softmax(self.aux_score(conditioned).squeeze(-1), aux_masks)
        z_aux = torch.sum(aux_weights.unsqueeze(-1) * aux_stack, dim=1)
        any_aux = aux_masks.any(1)
        g_aux = torch.sigmoid(self.aux_gate(torch.cat([z_content, z_aux, aux_masks.float()], dim=-1)))
        g_aux = g_aux * any_aux.unsqueeze(-1)
        if self.config.fusion == "hierarchical":
            z_account = self.account_norm(z_content + g_aux * z_aux)
        else:
            representations = torch.cat([
                post["representation"], comment["representation"], z_m, z_t, z_align, z_url,
                torch.cat([content_masks, aux_masks], dim=1).float(),
            ], dim=-1)
            z_account = self.direct_fusion(representations)
        account_logit = self.classifier(z_account).squeeze(-1)
        return {
            "account_logit": account_logit,
            "account_representation": z_account,
            "post_instance_logits": post["instance_logits"],
            "comment_instance_logits": comment["instance_logits"],
            "post_evidence_weights": post["evidence_weights"],
            "comment_evidence_weights": comment["evidence_weights"],
            "post_behavior_weights": post["behavior_weights"],
            "comment_behavior_weights": comment["behavior_weights"],
            "url_attention_weights": url_weights,
            "content_fusion_weights": content_weights,
            "post_branch_gate": post["branch_gate"],
            "comment_branch_gate": comment["branch_gate"],
            "auxiliary_fusion_weights": aux_weights,
            "g_aux": g_aux,
            "post_mil_logit": post["mil_logit"],
            "comment_mil_logit": comment["mil_logit"],
            "modality_masks": {
                "post": post["present"], "comment": comment["present"],
                "moltbook_profile": m_mask, "twitter_profile": t_mask,
                "alignment": align_mask, "url": url_present,
                "url_instances": effective_url_mask,
            },
            "url_item_representations": url_items,
        }
