"""Pure attention-MIL control for the residual top-k content encoder."""
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
class PureMILConfig(RiskResidualMILConfig):
    aggregation_mode: str = "pure_attention_mil"

    def validate(self) -> None:
        super().validate()
        if self.aggregation_mode != "pure_attention_mil":
            raise ValueError("PureMILConfig requires pure_attention_mil")


class PureMILContentEncoder(ResidualMILContentEncoder):
    """Use the learned instance-risk attention directly as the bag encoder.

    The full residual/top-k modules are still instantiated in the same order as
    the Full Model, but q, the mean-residual shift, and its gate are excluded
    from the returned representation and therefore receive no task gradient.
    """

    def forward(
        self, values: torch.Tensor, mask: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        output = super().forward(values, mask)
        present = output["present"]
        risk_representation = output["risk_representation"]
        representation = self.output_norm(risk_representation)
        representation = representation * present.unsqueeze(-1)
        zero_gate = torch.zeros_like(output["residual_gate"])
        self.last_representation = representation
        self.last_gate = zero_gate
        output.update(
            {
                "representation": representation,
                "sparse_representation": risk_representation,
                "branch_gate": torch.zeros_like(output["branch_gate"]),
                "mil_logit": torch.zeros_like(output["mil_logit"]),
                "residual_gate": zero_gate,
            }
        )
        return output


class PureMILModel(RiskResidualMILModel):
    """Construct pure-MIL encoders once at the original construction site."""

    def __init__(self, config: PureMILConfig | None = None) -> None:
        resolved = config or PureMILConfig()
        original_encoder = residual_model.ResidualMILContentEncoder
        residual_model.ResidualMILContentEncoder = PureMILContentEncoder
        try:
            super().__init__(resolved)
        finally:
            residual_model.ResidualMILContentEncoder = original_encoder
        self.config: PureMILConfig
