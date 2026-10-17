"""Training adapter for risk-aware residual MIL experiments."""
from __future__ import annotations

import random
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from multisource import data as multisource_data
from multisource.model_risk_residual_mil import (
    RiskResidualMILConfig,
    RiskResidualMILModel,
)

from . import joint_fine_ablation as base
from .gnn_evidence_loss import JointEvidenceFusionModel


INSTANCE_BEHAVIOR_CACHE = Path(
    os.environ.get("SARA_INSTANCE_BEHAVIOR_CACHE", "data/instance_behavior.npy")
)


class BehaviorMultiSourceDataset(multisource_data.MultiSourceDataset):
    """Original embedding dataset plus aligned duplicate/burst features."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if not INSTANCE_BEHAVIOR_CACHE.is_file():
            raise FileNotFoundError(INSTANCE_BEHAVIOR_CACHE)
        self._behavior_cache: np.ndarray | None = None

    def _behavior(self) -> np.ndarray:
        if self._behavior_cache is None:
            self._behavior_cache = np.load(INSTANCE_BEHAVIOR_CACHE, mmap_mode="r")
            if self._behavior_cache.ndim != 2 or self._behavior_cache.shape[1] != 5:
                raise ValueError("Invalid instance behavior cache")
        return self._behavior_cache

    def _content_vectors(
        self, aid: str, source: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        db, shards = self._resources()
        refs = db.execute(
            "SELECT record_id,shard,row_index,valid_mask FROM instances "
            "WHERE account_id=? AND source_type=? ORDER BY record_id",
            (aid, source),
        ).fetchall()
        refs = [row for row in refs if row[3]]
        limit = self.limits[source]
        if limit > 0 and len(refs) > limit:
            if self.training:
                refs = random.sample(refs, limit)
            else:
                rng = random.Random(f"{self.seed}:{aid}:{source}")
                refs = rng.sample(refs, limit)
            refs.sort(key=lambda row: (row[1], row[2]))
        if not refs:
            return (
                np.empty((0, self.dimension), np.float32),
                np.empty(0, np.bool_),
                np.empty((0, 5), np.float32),
            )
        values = np.empty((len(refs), self.dimension), np.float32)
        record_ids = np.empty(len(refs), np.int64)
        for output_row, (record_id, shard, row, _valid) in enumerate(refs):
            values[output_row] = shards.get(int(shard))[int(row)]
            record_ids[output_row] = int(record_id)
        behavior_cache = self._behavior()
        if int(record_ids.max()) >= len(behavior_cache):
            raise ValueError("Instance behavior cache does not cover record IDs")
        behavior = np.asarray(behavior_cache[record_ids], dtype=np.float32)
        return values, np.ones(len(refs), dtype=np.bool_), behavior

    def __getitem__(self, index: int) -> dict[str, Any]:
        account = self.accounts[index]
        aid = str(account["account_id"])
        posts, post_mask, post_behavior = self._content_vectors(aid, "post")
        comments, comment_mask, comment_behavior = self._content_vectors(aid, "comment")
        if not (post_mask.any() or comment_mask.any()):
            raise ValueError(f"Account {aid!r} has neither valid Post nor Comment")
        moltbook, moltbook_mask = self._profile(aid, "moltbook_profile")
        twitter, twitter_mask = self._profile(aid, "twitter_profile")
        urls = list(self.urls.get(aid, []))
        if self.max_urls > 0 and len(urls) > self.max_urls:
            if self.training:
                urls = random.sample(urls, self.max_urls)
            else:
                rng = random.Random(f"{self.seed}:{aid}:url")
                urls = rng.sample(urls, self.max_urls)
        return {
            "account_id": aid,
            "label": float(account["label"]),
            "posts": posts,
            "post_mask": post_mask,
            "post_instance_behavior": post_behavior,
            "comments": comments,
            "comment_mask": comment_mask,
            "comment_instance_behavior": comment_behavior,
            "moltbook_profile": moltbook,
            "moltbook_profile_mask": moltbook_mask,
            "twitter_profile": twitter,
            "twitter_profile_mask": twitter_mask,
            "urls": urls,
        }


def _pad_behavior(rows: list[dict[str, Any]], key: str, length: int) -> torch.Tensor:
    result = np.zeros((len(rows), length, 5), dtype=np.float32)
    for index, row in enumerate(rows):
        size = len(row[key])
        result[index, :size] = row[key]
    return torch.from_numpy(result)


def behavior_collate_accounts(
    rows: list[dict[str, Any]], url_max_bytes: int = 256
) -> dict[str, Any]:
    result = multisource_data.collate_accounts(rows, url_max_bytes)
    result["post_instance_behavior"] = _pad_behavior(
        rows, "post_instance_behavior", result["posts"].shape[1]
    )
    result["comment_instance_behavior"] = _pad_behavior(
        rows, "comment_instance_behavior", result["comments"].shape[1]
    )
    return result


class BehaviorJointBatchCollator(base.JointBatchCollator):
    def __call__(self, target_view_nodes: list[int]) -> dict[str, Any]:
        sample = self.sampler.sample(target_view_nodes)
        global_nodes = sample.pop("global_nodes")
        target_count = int(sample.pop("target_count"))
        if not self.use_content:
            raise ValueError("Behavior-aware residual MIL requires content")
        assert self.target_content is not None and self.context_content is not None
        target_rows = [
            self.target_content[int(index)] for index in global_nodes[:target_count]
        ]
        context_rows = [
            self.context_content[int(index)] for index in global_nodes[target_count:]
        ]
        target_content = behavior_collate_accounts(target_rows, self.url_max_bytes)
        context_content = (
            behavior_collate_accounts(context_rows, self.url_max_bytes)
            if context_rows else None
        )
        result: dict[str, Any] = {
            "target_content": target_content,
            "context_content": context_content,
            "target_count": target_count,
            "target_account_ids": [row["account_id"] for row in target_rows],
            "target_labels": torch.tensor(
                [row["label"] for row in target_rows], dtype=torch.float32
            ),
        }
        for key, value in sample.items():
            dtype = (
                torch.long
                if key in {"view_nodes", "edge_index", "incidence_index"}
                else torch.float32
            )
            result[key] = torch.as_tensor(value, dtype=dtype)
        return result


class JointRiskResidualMILModel(JointEvidenceFusionModel):
    """Expose target-instance logits while retaining the established graph model."""

    def forward(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        encoder = self.account_encoder
        if not isinstance(encoder, RiskResidualMILModel):
            raise TypeError("JointRiskResidualMILModel requires RiskResidualMILModel")
        encoder.begin_capture()
        try:
            output = super().forward(*args, **kwargs)
            captured = encoder.end_capture()
        except Exception:
            encoder.end_capture()
            raise
        if not captured:
            raise RuntimeError("Account encoder capture is empty")
        target = captured[0]
        target_batch = args[0] if args else kwargs["target_batch"]
        output.update({
            "text_logit": target["text_logit"],
            "post_instance_logits": target["post_instance_logits"],
            "comment_instance_logits": target["comment_instance_logits"],
            "post_instance_mask": target_batch["post_mask"].bool(),
            "comment_instance_mask": target_batch["comment_mask"].bool(),
            "post_residual_gate": target["post_residual_gate"],
            "comment_residual_gate": target["comment_residual_gate"],
        })
        return output


@dataclass
class RiskResidualMILTrainConfig(base.JointTrainConfig):
    text_loss_weight: float = 0.1
    instance_loss_weight: float = 0.0
    positive_top_k: int = 5
    use_instance_behavior: bool = False

    def validate(self) -> None:
        super().validate()
        if self.text_loss_weight < 0 or self.instance_loss_weight < 0:
            raise ValueError("Auxiliary loss weights must be non-negative")
        if self.positive_top_k < 1:
            raise ValueError("positive_top_k must be positive")
        if not self.use_content or not self.use_directed:
            raise ValueError("Residual MIL suite requires content and directed graph")


def asymmetric_instance_loss(
    post_logits: torch.Tensor,
    post_mask: torch.Tensor,
    comment_logits: torch.Tensor,
    comment_mask: torch.Tensor,
    labels: torch.Tensor,
    positive_top_k: int,
) -> torch.Tensor:
    """Positive bags supervise top-k evidence; negative bags supervise all items."""
    scores = torch.cat([post_logits, comment_logits], dim=1)
    masks = torch.cat([post_mask, comment_mask], dim=1).bool()
    positive_terms: list[torch.Tensor] = []
    negative_terms: list[torch.Tensor] = []
    for row in range(scores.shape[0]):
        valid = scores[row][masks[row]]
        if not valid.numel():
            continue
        if labels[row] > 0.5:
            bag_logit = valid.topk(min(positive_top_k, valid.numel())).values.mean()
            positive_terms.append(nn.functional.softplus(-bag_logit))
        else:
            negative_terms.append(nn.functional.softplus(valid).mean())
    group_terms: list[torch.Tensor] = []
    if positive_terms:
        group_terms.append(torch.stack(positive_terms).mean())
    if negative_terms:
        group_terms.append(torch.stack(negative_terms).mean())
    if not group_terms:
        return scores.sum() * 0.0
    return torch.stack(group_terms).mean()


def _run_loader(
    model: JointRiskResidualMILModel,
    loader: DataLoader,
    device: torch.device,
    config: RiskResidualMILTrainConfig,
    pos_weight: torch.Tensor | None,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: Any | None = None,
) -> tuple[float, list[int], list[float], list[str], dict[str, Any]]:
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "main": 0.0, "text": 0.0, "instance": 0.0}
    total_items = 0
    truth: list[int] = []
    probabilities: list[float] = []
    account_ids: list[str] = []
    gate_sum = np.zeros(3, dtype=np.float64)
    present_sum = np.zeros(3, dtype=np.float64)
    residual_gate_sum = np.zeros(2, dtype=np.float64)
    topology_gate_sum = 0.0
    context = torch.enable_grad if training else torch.no_grad
    with context():
        for batch in loader:
            batch = base._move(batch, device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            target_count = int(batch["target_count"])
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=config.amp and device.type == "cuda",
            ):
                output = model(
                    batch["target_content"], batch["context_content"],
                    batch["topology_features"], batch["repetition_features"],
                    batch["edge_index"], batch["edge_features"],
                    batch["thread_features"], batch["incidence_index"],
                    batch["incidence_features"],
                )
                labels = batch["target_labels"]
                logits = output["account_logit"][:target_count]
                text_logits = output["text_logit"][:target_count]
                main_loss = nn.functional.binary_cross_entropy_with_logits(
                    logits, labels, pos_weight=pos_weight
                )
                text_loss = nn.functional.binary_cross_entropy_with_logits(
                    text_logits, labels, pos_weight=pos_weight
                )
                instance_loss = asymmetric_instance_loss(
                    output["post_instance_logits"][:target_count],
                    output["post_instance_mask"][:target_count],
                    output["comment_instance_logits"][:target_count],
                    output["comment_instance_mask"][:target_count],
                    labels,
                    config.positive_top_k,
                )
                loss = (
                    main_loss
                    + config.text_loss_weight * text_loss
                    + config.instance_loss_weight * instance_loss
                )
            if training:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    if config.grad_clip:
                        nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if config.grad_clip:
                        nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                    optimizer.step()

            count = len(labels)
            totals["loss"] += float(loss.detach().cpu()) * count
            totals["main"] += float(main_loss.detach().cpu()) * count
            totals["text"] += float(text_loss.detach().cpu()) * count
            totals["instance"] += float(instance_loss.detach().cpu()) * count
            total_items += count
            truth.extend(labels.long().detach().cpu().tolist())
            probabilities.extend(torch.sigmoid(logits.float()).detach().cpu().tolist())
            account_ids.extend(batch["target_account_ids"])
            gate_sum += output["modality_gates"][:target_count].detach().float().sum(0).cpu().numpy()
            present = output["branch_present"]
            present_sum += torch.stack([
                present["repetition"][:target_count],
                present["directed"][:target_count],
                present["hypergraph"][:target_count],
            ], dim=-1).float().sum(0).cpu().numpy()
            topology_gate_sum += float(
                output["topology_gate"][:target_count].detach().float().sum().cpu()
            )
            residual_gate_sum[0] += float(
                output["post_residual_gate"][:target_count].detach().float().sum().cpu()
            )
            residual_gate_sum[1] += float(
                output["comment_residual_gate"][:target_count].detach().float().sum().cpu()
            )

    denominator = max(1, total_items)
    diagnostics = {
        "main_loss": totals["main"] / denominator,
        "text_loss": totals["text"] / denominator,
        "instance_loss": totals["instance"] / denominator,
        "text_loss_weight": config.text_loss_weight,
        "instance_loss_weight": config.instance_loss_weight,
        "mean_residual_gate": {
            "post": residual_gate_sum[0] / denominator,
            "comment": residual_gate_sum[1] / denominator,
        },
        "mean_gate": dict(zip(
            ("repetition", "directed", "hypergraph"),
            (gate_sum / denominator).tolist(),
        )),
        "present_rate": dict(zip(
            ("repetition", "directed", "hypergraph"),
            (present_sum / denominator).tolist(),
        )),
        "mean_topology_gate": topology_gate_sum / denominator,
    }
    return totals["loss"] / denominator, truth, probabilities, account_ids, diagnostics


def train_joint_risk_residual_mil_model(
    embedding_dir: Path,
    url_cache: Path | None,
    splits_path: Path,
    graph_arrays: Path,
    output_dir: Path,
    model_config: RiskResidualMILConfig,
    train_config: RiskResidualMILTrainConfig,
) -> dict[str, Any]:
    if bool(model_config.instance_behavior_dim) != train_config.use_instance_behavior:
        raise ValueError("Model/data instance behavior switches disagree")
    original_encoder = base.FineAblationRiskModel
    original_joint_model = base.JointR3HFusionModel
    original_run_loader = base._run_loader
    original_dataset = base.MultiSourceDataset
    original_collator = base.JointBatchCollator
    base.FineAblationRiskModel = RiskResidualMILModel
    base.JointR3HFusionModel = JointRiskResidualMILModel
    base._run_loader = _run_loader
    if train_config.use_instance_behavior:
        base.MultiSourceDataset = BehaviorMultiSourceDataset
        base.JointBatchCollator = BehaviorJointBatchCollator
    try:
        return base.train_joint_model(
            embedding_dir=embedding_dir,
            url_cache=url_cache,
            splits_path=splits_path,
            graph_arrays=graph_arrays,
            output_dir=output_dir,
            model_config=model_config,
            train_config=train_config,
        )
    finally:
        base.FineAblationRiskModel = original_encoder
        base.JointR3HFusionModel = original_joint_model
        base._run_loader = original_run_loader
        base.MultiSourceDataset = original_dataset
        base.JointBatchCollator = original_collator
