"""Joint content/directed-graph model exposing five risk-evidence logits."""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .gnn_fine_ablation import JointR3HConfig, JointR3HFusionModel


EVIDENCE_NAMES = ("post", "comment", "url", "incoming", "outgoing")


class JointEvidenceFusionModel(JointR3HFusionModel):
    """Original joint fusion model plus incoming/outgoing risk heads.

    Evidence heads provide auxiliary supervision only. The final prediction is
    still produced by the unchanged fused account classifier.
    """

    def __init__(self, account_encoder: nn.Module,
                 config: JointR3HConfig | None = None) -> None:
        super().__init__(account_encoder, config)
        dim = self.config.content_dim

        def risk_head() -> nn.Sequential:
            return nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim // 2),
                nn.GELU(),
                nn.Linear(dim // 2, 1),
            )

        self.incoming_risk = risk_head()
        self.outgoing_risk = risk_head()

    @staticmethod
    def _content_evidence(output: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        logits = torch.stack([
            output["post_mil_logit"],
            output["comment_mil_logit"],
            output["url_mil_logit"],
        ], dim=-1)
        masks = output["modality_masks"]
        present = torch.stack([
            masks["post"].bool(),
            masks["comment"].bool(),
            masks["url"].bool(),
        ], dim=-1)
        return logits, present

    @staticmethod
    def _edge_presence(edge_index: torch.Tensor, size: int,
                       endpoint: int, device: torch.device) -> torch.Tensor:
        degree = torch.zeros(size, device=device)
        if edge_index.numel():
            indices = edge_index[endpoint].long()
            degree.index_add_(0, indices, torch.ones(len(indices), device=device))
        return degree > 0

    def forward(self, target_batch: dict[str, Any], context_batch: dict[str, Any] | None,
                topology_features: torch.Tensor, repetition_features: torch.Tensor,
                edge_index: torch.Tensor, edge_features: torch.Tensor,
                thread_features: torch.Tensor, incidence_index: torch.Tensor,
                incidence_features: torch.Tensor) -> dict[str, Any]:
        topology = self.topology_encoder(torch.nan_to_num(topology_features))
        if not self.config.use_content:
            raise ValueError("The risk-evidence design requires the content branch")

        target_output = self.account_encoder(target_batch)
        content_parts = [target_output["account_representation"]]
        baseline_parts = [target_output["account_logit"]]
        target_evidence, target_evidence_mask = self._content_evidence(target_output)
        content_evidence_parts = [target_evidence]
        content_evidence_mask_parts = [target_evidence_mask]
        if context_batch is not None:
            context_output = self.account_encoder(context_batch)
            content_parts.append(context_output["account_representation"])
            baseline_parts.append(context_output["account_logit"])
            context_evidence, context_evidence_mask = self._content_evidence(context_output)
            content_evidence_parts.append(context_evidence)
            content_evidence_mask_parts.append(context_evidence_mask)

        content = torch.cat(content_parts, dim=0)
        baseline_logit = torch.cat(baseline_parts, dim=0)
        content_evidence = torch.cat(content_evidence_parts, dim=0)
        content_evidence_mask = torch.cat(content_evidence_mask_parts, dim=0)
        if content.shape[1] != self.config.content_dim:
            raise ValueError(
                f"Account encoder returned {content.shape[1]} dimensions; "
                f"expected {self.config.content_dim}"
            )

        if self.config.use_topology:
            topology_weight = torch.sigmoid(
                self.topology_gate(torch.cat([content, topology], dim=-1))
            )
            graph_seed = content + topology_weight * topology
        else:
            topology_weight = content.new_zeros((len(content), 1))
            graph_seed = content
        base_representation = content

        repetition = self.repetition_encoder(torch.nan_to_num(repetition_features))
        repetition_present = torch.isfinite(repetition_features).any(dim=-1)
        repetition_present = repetition_present & self.config.use_repetition
        repetition = repetition * repetition_present.unsqueeze(-1)

        if self.config.use_directed:
            directed, directed_aux = self.directed(graph_seed, edge_index, edge_features)
            directed_delta = directed - graph_seed
            directed_present = self._directed_presence(
                edge_index, len(content), content.device, self.config.directed_mode
            )
            directed_delta, directed_present = self._drop_branch(
                directed_delta, directed_present
            )
            incoming_context = directed_aux["incoming_context"]
            outgoing_context = directed_aux["outgoing_context"]
            incoming_present = self._edge_presence(
                edge_index, len(content), 1, content.device
            ) & (self.config.directed_mode in {"both", "incoming"})
            outgoing_present = self._edge_presence(
                edge_index, len(content), 0, content.device
            ) & (self.config.directed_mode in {"both", "outgoing"})
            incoming_logit = self.incoming_risk(incoming_context).squeeze(-1)
            outgoing_logit = self.outgoing_risk(outgoing_context).squeeze(-1)
        else:
            directed_delta, directed_aux = torch.zeros_like(content), {}
            directed_present = torch.zeros(len(content), dtype=torch.bool,
                                             device=content.device)
            incoming_present = torch.zeros_like(directed_present)
            outgoing_present = torch.zeros_like(directed_present)
            incoming_logit = content.new_zeros(len(content))
            outgoing_logit = content.new_zeros(len(content))

        if self.config.use_hypergraph:
            hypergraph, hypergraph_aux = self.hypergraph(
                graph_seed, thread_features, incidence_index, incidence_features
            )
            hypergraph_delta = hypergraph - graph_seed
            hypergraph_present = self._hypergraph_presence(
                incidence_index, len(content), content.device
            )
            hypergraph_delta, hypergraph_present = self._drop_branch(
                hypergraph_delta, hypergraph_present
            )
        else:
            hypergraph_delta, hypergraph_aux = torch.zeros_like(content), {}
            hypergraph_present = torch.zeros(len(content), dtype=torch.bool,
                                               device=content.device)

        availability = torch.stack([
            repetition_present, directed_present, hypergraph_present,
        ], dim=-1).to(content.dtype)
        gates = torch.sigmoid(self.fusion_gate(torch.cat([
            base_representation, repetition, directed_delta, hypergraph_delta,
            availability,
        ], dim=-1))) * availability
        fused = (
            base_representation
            + gates[:, 0:1] * repetition
            + gates[:, 1:2] * directed_delta
            + gates[:, 2:3] * hypergraph_delta
        )

        evidence_logits = torch.cat([
            content_evidence,
            incoming_logit.unsqueeze(-1),
            outgoing_logit.unsqueeze(-1),
        ], dim=-1)
        evidence_mask = torch.cat([
            content_evidence_mask,
            incoming_present.unsqueeze(-1),
            outgoing_present.unsqueeze(-1),
        ], dim=-1)
        return {
            "account_logit": self.classifier(fused).squeeze(-1),
            "baseline_logit": baseline_logit,
            "account_representation": fused,
            "content_representation": content,
            "base_representation": base_representation,
            "topology_gate": topology_weight.squeeze(-1),
            "modality_gates": gates,
            "branch_present": {
                "repetition": repetition_present,
                "directed": directed_present,
                "hypergraph": hypergraph_present,
            },
            "evidence_logits": evidence_logits,
            "evidence_mask": evidence_mask,
            "evidence_names": EVIDENCE_NAMES,
            "directed": directed_aux,
            "hypergraph": hypergraph_aux,
        }
