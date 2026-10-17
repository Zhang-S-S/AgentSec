"""Risk-aware residual MIL for Moltbook Post/Comment bags.

The mean representation is kept as a stable residual path.  A learned risk
branch can only move the representation toward instances that receive high
risk scores.  Optional record-level duplicate/burst features affect the risk
scores, not the semantic embeddings themselves.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from .model_fine_ablation import FineAblationModelConfig, FineAblationRiskModel
from .ops import MILPool, masked_softmax


@dataclass
class RiskResidualMILConfig(FineAblationModelConfig):
    post_top_k: int = 3
    comment_top_k: int = 5
    residual_gate_bias: float = -2.0
    instance_behavior_dim: int = 0

    def validate(self) -> None:
        super().validate()
        if not self.use_mil_branch or self.behavior_encoder:
            raise ValueError("RiskResidualMIL requires MIL on and Transformer off")
        if self.post_top_k < 1 or self.comment_top_k < 1:
            raise ValueError("Top-k values must be positive")
        if self.instance_behavior_dim not in {0, 5}:
            raise ValueError("instance_behavior_dim must be 0 or 5")


class ResidualMILContentEncoder(nn.Module):
    """Mean-residual, risk-attentive encoder for one content modality."""

    def __init__(self, config: RiskResidualMILConfig, top_k: int) -> None:
        super().__init__()
        dim = config.hidden_dim
        self.top_k = int(top_k)
        self.attention_temperature = float(config.attention_temperature)
        self.instance_behavior_dim = int(config.instance_behavior_dim)
        self.instance_risk = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim // 2), nn.GELU(),
            nn.Linear(dim // 2, 1),
        )
        if self.instance_behavior_dim:
            self.behavior_risk = nn.Sequential(
                nn.LayerNorm(self.instance_behavior_dim),
                nn.Linear(self.instance_behavior_dim, 32), nn.GELU(),
                nn.Linear(32, 1, bias=False),
            )
        else:
            self.behavior_risk = None
        self.residual_gate = nn.Sequential(
            nn.Linear(dim * 2 + 1, dim // 2), nn.GELU(),
            nn.Linear(dim // 2, 1),
        )
        nn.init.zeros_(self.residual_gate[-1].weight)
        nn.init.constant_(self.residual_gate[-1].bias, config.residual_gate_bias)
        self.output_norm = nn.LayerNorm(dim)
        self._pending_behavior: torch.Tensor | None = None
        self.last_representation: torch.Tensor | None = None
        self.last_gate: torch.Tensor | None = None

    def set_instance_behavior(self, value: torch.Tensor | None) -> None:
        self._pending_behavior = value

    def _topk_mean(self, scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        pooled: list[torch.Tensor] = []
        for row in range(scores.shape[0]):
            valid = scores[row][mask[row]]
            if valid.numel() == 0:
                pooled.append(scores.new_zeros(()))
            else:
                pooled.append(valid.topk(min(self.top_k, valid.numel())).values.mean())
        return torch.stack(pooled) if pooled else scores.new_zeros((0,))

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        mask = mask.bool()
        batch, length, dim = values.shape
        present = mask.any(dim=1)
        instance_logits = (
            self.instance_risk(values).squeeze(-1)
            if length else values.new_zeros((batch, 0))
        )
        behavior = self._pending_behavior
        self._pending_behavior = None
        if self.instance_behavior_dim:
            if behavior is None:
                raise ValueError("Instance behavior features are enabled but missing")
            if behavior.shape[:2] != values.shape[:2] or behavior.shape[-1] != self.instance_behavior_dim:
                raise ValueError("Instance behavior features do not align with content")
            if length:
                assert self.behavior_risk is not None
                instance_logits = instance_logits + self.behavior_risk(behavior).squeeze(-1)

        denominator = mask.sum(dim=1, keepdim=True).clamp_min(1).to(values.dtype)
        mean_vector = (values * mask.unsqueeze(-1)).sum(dim=1) / denominator
        risk_weights = masked_softmax(
            instance_logits / self.attention_temperature, mask
        )
        risk_vector = torch.sum(risk_weights.unsqueeze(-1) * values, dim=1)
        mil_logit = self._topk_mean(instance_logits, mask)
        gate = torch.sigmoid(self.residual_gate(torch.cat([
            mean_vector, risk_vector, mil_logit.unsqueeze(-1)
        ], dim=-1)))
        combined = self.output_norm(
            mean_vector + gate * (risk_vector - mean_vector)
        ) * present.unsqueeze(-1)
        self.last_representation = combined
        self.last_gate = gate.squeeze(-1)
        zeros = values.new_zeros((batch, length))
        return {
            "representation": combined,
            "sparse_representation": risk_vector * present.unsqueeze(-1),
            "behavior_representation": values.new_zeros((batch, dim)),
            "instance_logits": instance_logits.masked_fill(~mask, 0.0),
            "evidence_weights": risk_weights,
            "behavior_weights": zeros,
            "branch_gate": gate.expand(-1, dim) * present.unsqueeze(-1),
            "mil_logit": mil_logit,
            "present": present,
            "mean_representation": mean_vector * present.unsqueeze(-1),
            "risk_representation": risk_vector * present.unsqueeze(-1),
            "residual_gate": gate.squeeze(-1),
        }


class RiskResidualMILModel(FineAblationRiskModel):
    """Account encoder with residual MIL and a pure Post/Comment text head."""

    def __init__(self, config: RiskResidualMILConfig | None = None) -> None:
        super().__init__(config or RiskResidualMILConfig())
        self.config: RiskResidualMILConfig
        dim = self.config.hidden_dim
        self.post_encoder = ResidualMILContentEncoder(self.config, self.config.post_top_k)
        self.comment_encoder = ResidualMILContentEncoder(self.config, self.config.comment_top_k)
        self.text_classifier = nn.Sequential(
            nn.Linear(dim, 128), nn.GELU(), nn.Dropout(self.config.dropout),
            nn.Linear(128, 1),
        )
        self.url_instance_risk = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim // 2), nn.GELU(),
            nn.Linear(dim // 2, 1),
        )
        self.url_mil = MILPool(
            self.config.mil_pooling,
            self.config.mil_temperature,
            self.config.top_k,
        )
        self._capture_enabled = False
        self._captured_outputs: list[dict[str, Any]] = []

    def begin_capture(self) -> None:
        self._captured_outputs = []
        self._capture_enabled = True

    def end_capture(self) -> list[dict[str, Any]]:
        result = self._captured_outputs
        self._captured_outputs = []
        self._capture_enabled = False
        return result

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, Any]:
        self.post_encoder.set_instance_behavior(batch.get("post_instance_behavior"))
        self.comment_encoder.set_instance_behavior(batch.get("comment_instance_behavior"))
        output = super().forward(batch)
        post_representation = self.post_encoder.last_representation
        comment_representation = self.comment_encoder.last_representation
        if post_representation is None or comment_representation is None:
            raise RuntimeError("Residual MIL encoders did not produce representations")
        content_stack = torch.stack([
            post_representation, comment_representation
        ], dim=1)
        content_weights = output["content_fusion_weights"]
        text_representation = torch.sum(
            content_weights.unsqueeze(-1) * content_stack, dim=1
        )
        text_logit = self.text_classifier(text_representation).squeeze(-1)

        url_items = output["url_item_representations"]
        url_mask = output["modality_masks"]["url_instances"].bool()
        batch_size, item_count, _ = url_items.shape
        if item_count:
            url_instance_logits = self.url_instance_risk(url_items).squeeze(-1)
            url_mil_logit = self.url_mil(url_instance_logits, url_mask)
        else:
            url_instance_logits = url_items.new_zeros((batch_size, 0))
            url_mil_logit = url_items.new_zeros(batch_size)
        output.update({
            "text_representation": text_representation,
            "text_logit": text_logit,
            # JointR3H treats account_logit as its auxiliary/baseline logit.
            # Making it text-only gives direct supervision to the text bag.
            "account_logit": text_logit,
            "url_instance_logits": url_instance_logits.masked_fill(~url_mask, 0.0),
            "url_mil_logit": url_mil_logit,
            "post_residual_gate": self.post_encoder.last_gate,
            "comment_residual_gate": self.comment_encoder.last_gate,
        })
        if self._capture_enabled:
            self._captured_outputs.append(output)
        return output
