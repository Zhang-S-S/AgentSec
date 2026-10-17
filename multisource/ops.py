from __future__ import annotations

import torch
from torch import nn


def masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Softmax with exact zero output for an all-masked row."""
    mask = mask.to(dtype=torch.bool, device=logits.device)
    if logits.shape != mask.shape:
        mask = torch.broadcast_to(mask, logits.shape)
    if logits.size(dim) == 0:
        return torch.zeros_like(logits)
    safe = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
    weights = torch.softmax(safe, dim=dim) * mask.to(logits.dtype)
    denominator = weights.sum(dim=dim, keepdim=True)
    return torch.where(denominator > 0, weights / denominator.clamp_min(1e-12), torch.zeros_like(weights))


class MILPool(nn.Module):
    """Account-level pooling over instance logits; empty rows map to zero."""

    def __init__(self, method: str = "normalized_lse", temperature: float = 1.0, top_k: int = 5) -> None:
        super().__init__()
        self.method, self.temperature, self.top_k = method, temperature, top_k

    def forward(self, scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.bool()
        batch = scores.shape[0]
        pooled = []
        for row in range(batch):
            valid = scores[row][mask[row]]
            if valid.numel() == 0:
                pooled.append(scores.new_zeros(()))
                continue
            if self.method in {"normalized_lse", "lse"}:
                value = self.temperature * torch.logsumexp(valid / self.temperature, dim=0)
                if self.method == "normalized_lse":
                    value = value - self.temperature * torch.log(
                        valid.new_tensor(float(valid.numel()))
                    )
            elif self.method == "topk":
                value = valid.topk(min(self.top_k, valid.numel())).values.mean()
            elif self.method == "max":
                value = valid.max()
            elif self.method == "noisy_or":
                # log(1 - p) = logsigmoid(-s); aggregate in log space.
                log_not_risk = torch.nn.functional.logsigmoid(-valid).sum()
                probability = -torch.expm1(log_not_risk)
                probability = probability.clamp(1e-7, 1 - 1e-7)
                value = torch.logit(probability)
            else:
                raise ValueError(f"Unsupported MIL method: {self.method}")
            pooled.append(value)
        return torch.stack(pooled) if pooled else scores.new_zeros((0,))


class MaskedAttentionPool(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.score = nn.Sequential(nn.Linear(dim, dim // 2), nn.Tanh(), nn.Linear(dim // 2, 1, bias=False))

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.score(values).squeeze(-1)
        weights = masked_softmax(logits, mask)
        return torch.sum(weights.unsqueeze(-1) * values, dim=1), weights
