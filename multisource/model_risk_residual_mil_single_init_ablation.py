"""Single-initialization ablation of the residual top-k content encoder.

The positive-control path is intentionally identical to RiskResidualMILModel:
the original encoders are constructed exactly once and no replacement module
is instantiated after model construction.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

import multisource.model_risk_residual_mil as residual_model
from multisource.model_risk_residual_mil import (
    ResidualMILContentEncoder,
    RiskResidualMILConfig,
    RiskResidualMILModel,
)


@dataclass
class SingleInitRiskResidualMILConfig(RiskResidualMILConfig):
    use_residual_topk_mil: bool = True


class SingleInitResidualMILContentEncoder(ResidualMILContentEncoder):
    """Original encoder with a forward-only mean-path switch."""

    def __init__(
        self, config: SingleInitRiskResidualMILConfig, top_k: int
    ) -> None:
        self.use_residual_topk_mil = bool(config.use_residual_topk_mil)
        super().__init__(config, top_k)

    def forward(
        self, values: torch.Tensor, mask: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        output = super().forward(values, mask)
        if self.use_residual_topk_mil:
            return output

        present = output["present"]
        mean_representation = output["mean_representation"]
        representation = self.output_norm(mean_representation)
        representation = representation * present.unsqueeze(-1)
        zero_gate = output["residual_gate"].new_zeros(
            output["residual_gate"].shape
        )
        self.last_representation = representation
        self.last_gate = zero_gate
        output.update(
            {
                "representation": representation,
                "sparse_representation": torch.zeros_like(representation),
                "risk_representation": torch.zeros_like(representation),
                "branch_gate": torch.zeros_like(output["branch_gate"]),
                "residual_gate": zero_gate,
            }
        )
        return output


class SingleInitRiskResidualMILModel(RiskResidualMILModel):
    """Construct the selectable encoders once, at the original construction site."""

    def __init__(
        self, config: SingleInitRiskResidualMILConfig | None = None
    ) -> None:
        resolved = config or SingleInitRiskResidualMILConfig()
        original_encoder = residual_model.ResidualMILContentEncoder
        residual_model.ResidualMILContentEncoder = (
            SingleInitResidualMILContentEncoder
        )
        try:
            super().__init__(resolved)
        finally:
            residual_model.ResidualMILContentEncoder = original_encoder
        self.config: SingleInitRiskResidualMILConfig
