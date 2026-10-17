"""Training adapter for strict single-initialization MoltRisk ablations."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from comment_network import joint_fine_ablation as fine_base
from comment_network import joint_risk_residual_mil as residual
from comment_network.gnn_undirected_ablation import UndirectedHeterophilyLayer
from comment_network.joint_risk_residual_mil import JointRiskResidualMILModel
from multisource.model_risk_residual_mil_single_init_ablation import (
    SingleInitRiskResidualMILConfig,
    SingleInitRiskResidualMILModel,
)


@dataclass
class SingleInitRiskResidualMILTrainConfig(
    residual.RiskResidualMILTrainConfig
):
    use_directionality: bool = True
    disable_directed_contribution: bool = False

    def validate(self) -> None:
        fine_base.JointTrainConfig.validate(self)
        if self.text_loss_weight < 0 or self.instance_loss_weight < 0:
            raise ValueError("Auxiliary loss weights must be non-negative")
        if self.positive_top_k < 1:
            raise ValueError("positive_top_k must be positive")
        if not self.use_content:
            raise ValueError("Residual MIL ablations require content")


class StrictGraphAblationJointModel(JointRiskResidualMILModel):
    """Change only the requested graph operation after shared initialization."""

    disable_directed_contribution: bool = False
    replace_directionality: bool = False

    def __init__(self, account_encoder: torch.nn.Module, config: Any = None) -> None:
        # Construct the exact full joint model first. This fixes every shared
        # parameter at the same seed-dependent value as the Full Model.
        super().__init__(account_encoder, config)
        if self.replace_directionality:
            # Replacing the target graph operator is intentional. Preserve the
            # RNG state so this replacement cannot shift later training masks.
            cpu_state = torch.get_rng_state()
            cuda_states = (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            )
            self.directed = UndirectedHeterophilyLayer(
                self.config.content_dim,
                self.config.edge_dim,
                self.config.dropout,
                directed_mode=self.config.directed_mode,
                use_edge_features=self.config.use_edge_features,
            )
            torch.set_rng_state(cpu_state)
            if cuda_states is not None:
                torch.cuda.set_rng_state_all(cuda_states)

    def _directed_presence(
        self,
        edge_index: torch.Tensor,
        size: int,
        device: torch.device,
        mode: str,
    ) -> torch.Tensor:
        if self.disable_directed_contribution:
            return torch.zeros(size, dtype=torch.bool, device=device)
        return super()._directed_presence(edge_index, size, device, mode)


def train_single_init_risk_residual_mil_model(
    embedding_dir: Path,
    url_cache: Path | None,
    splits_path: Path,
    graph_arrays: Path,
    output_dir: Path,
    model_config: SingleInitRiskResidualMILConfig,
    train_config: SingleInitRiskResidualMILTrainConfig,
) -> dict[str, Any]:
    if bool(model_config.instance_behavior_dim) != train_config.use_instance_behavior:
        raise ValueError("Model/data instance behavior switches disagree")
    original_encoder = fine_base.FineAblationRiskModel
    original_joint_model = fine_base.JointR3HFusionModel
    original_run_loader = fine_base._run_loader
    original_dataset = fine_base.MultiSourceDataset
    original_collator = fine_base.JointBatchCollator
    fine_base.FineAblationRiskModel = SingleInitRiskResidualMILModel
    if (
        train_config.disable_directed_contribution
        or not train_config.use_directionality
    ):
        StrictGraphAblationJointModel.disable_directed_contribution = bool(
            train_config.disable_directed_contribution
        )
        StrictGraphAblationJointModel.replace_directionality = not bool(
            train_config.use_directionality
        )
        fine_base.JointR3HFusionModel = StrictGraphAblationJointModel
    else:
        fine_base.JointR3HFusionModel = JointRiskResidualMILModel
    fine_base._run_loader = residual._run_loader
    try:
        return fine_base.train_joint_model(
            embedding_dir=embedding_dir,
            url_cache=url_cache,
            splits_path=splits_path,
            graph_arrays=graph_arrays,
            output_dir=output_dir,
            model_config=model_config,
            train_config=train_config,
        )
    finally:
        fine_base.FineAblationRiskModel = original_encoder
        fine_base.JointR3HFusionModel = original_joint_model
        fine_base._run_loader = original_run_loader
        fine_base.MultiSourceDataset = original_dataset
        fine_base.JointBatchCollator = original_collator
        StrictGraphAblationJointModel.disable_directed_contribution = False
        StrictGraphAblationJointModel.replace_directionality = False
