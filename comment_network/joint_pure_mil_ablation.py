"""Training adapter for the pure attention-MIL ablation."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from comment_network import joint_fine_ablation as fine_base
from comment_network import joint_risk_residual_mil as residual
from comment_network.joint_risk_residual_mil import (
    JointRiskResidualMILModel,
    RiskResidualMILTrainConfig,
)
from multisource.model_pure_mil_ablation import PureMILConfig, PureMILModel


def train_joint_pure_mil_model(
    embedding_dir: Path,
    url_cache: Path | None,
    splits_path: Path,
    graph_arrays: Path,
    output_dir: Path,
    model_config: PureMILConfig,
    train_config: RiskResidualMILTrainConfig,
) -> dict[str, Any]:
    if model_config.instance_behavior_dim != 0:
        raise ValueError("The pure-MIL control does not use behavior features")
    if train_config.use_instance_behavior:
        raise ValueError("The pure-MIL control does not use behavior features")
    original_encoder = fine_base.FineAblationRiskModel
    original_joint_model = fine_base.JointR3HFusionModel
    original_run_loader = fine_base._run_loader
    original_dataset = fine_base.MultiSourceDataset
    original_collator = fine_base.JointBatchCollator
    fine_base.FineAblationRiskModel = PureMILModel
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
