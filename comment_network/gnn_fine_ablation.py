"""Repetition-, role-, and reach-aware heterophilic network (R3H-Net).

This module deliberately does not modify ``MultiSourceRiskModel``.  It is an
opt-in experimental head that consumes the existing account representation,
the directed comment graph, and the account-thread incidence hypergraph.  It
uses only PyTorch tensor operations and therefore has no PyG dependency.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass
class R3HConfig:
    content_dim: int = 256
    hidden_dim: int = 128
    repetition_dim: int = 5
    edge_dim: int = 9
    thread_dim: int = 7
    incidence_dim: int = 5
    dropout: float = 0.2
    graph_dropout: float = 0.15
    use_repetition: bool = True
    use_directed: bool = True
    use_hypergraph: bool = True

    def validate(self) -> None:
        for name in ("content_dim", "hidden_dim", "repetition_dim", "edge_dim",
                     "thread_dim", "incidence_dim"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        for name in ("dropout", "graph_dropout"):
            if not 0 <= getattr(self, name) < 1:
                raise ValueError(f"{name} must be in [0, 1)")


def _mlp(input_dim: int, output_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, output_dim), nn.GELU(), nn.Dropout(dropout),
        nn.Linear(output_dim, output_dim),
    )


def _segment_softmax(logits: torch.Tensor, index: torch.Tensor, size: int) -> torch.Tensor:
    """Softmax independently over entries sharing a destination index.

    Reductions are deliberately evaluated in float32. Under CUDA autocast,
    ``exp`` may return float32 for float16 logits; accumulating that result into
    ``logits.new_zeros`` would otherwise make ``index_add_`` fail with a mixed
    scalar-type error. The normalized weights are cast back for message fusion.
    """
    if logits.numel() == 0:
        return logits
    work_logits = logits.float()
    maxima = work_logits.new_full((size,), -torch.inf)
    maxima.scatter_reduce_(0, index, work_logits, reduce="amax", include_self=True)
    unnormalized = torch.exp(work_logits - maxima.index_select(0, index))
    denominator = unnormalized.new_zeros(size).index_add_(0, index, unnormalized)
    weights = unnormalized / denominator.index_select(0, index).clamp_min(1e-12)
    return weights.to(dtype=logits.dtype)


def _aggregate(messages: torch.Tensor, weights: torch.Tensor, index: torch.Tensor,
               size: int) -> torch.Tensor:
    output = messages.new_zeros((size, messages.shape[-1]))
    if messages.numel():
        output.index_add_(0, index, messages * weights.unsqueeze(-1))
    return output


class DirectedHeterophilyLayer(nn.Module):
    """Relation-aware directed messages that retain neighbor disagreement.

    Plain GCN averaging is inappropriate here because risk-risk ties are
    depleted in the observed network.  Incoming and outgoing contexts are kept
    separate, and absolute differences/products make disassortative evidence
    available to the update function instead of smoothing it away.
    """

    def __init__(self, hidden_dim: int, edge_dim: int, dropout: float,
                 directed_mode: str = "both",
                 use_edge_features: bool = True) -> None:
        super().__init__()
        if directed_mode not in {"both", "incoming", "outgoing"}:
            raise ValueError(f"Unsupported directed_mode: {directed_mode}")
        self.directed_mode = directed_mode
        self.use_edge_features = use_edge_features
        self.edge_encoder = _mlp(edge_dim, hidden_dim, dropout)
        self.in_message = _mlp(hidden_dim * 2, hidden_dim, dropout)
        self.out_message = _mlp(hidden_dim * 2, hidden_dim, dropout)
        self.in_attention = nn.Linear(hidden_dim * 3, 1)
        self.out_attention = nn.Linear(hidden_dim * 3, 1)
        self.update = _mlp(hidden_dim * 7, hidden_dim, dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, nodes: torch.Tensor, edge_index: torch.Tensor,
                edge_features: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        source, target = edge_index[0].long(), edge_index[1].long()
        edge_input = edge_features if self.use_edge_features else torch.zeros_like(edge_features)
        encoded_edge = self.edge_encoder(edge_input)
        source_state = nodes.index_select(0, source)
        target_state = nodes.index_select(0, target)

        if self.directed_mode in {"both", "incoming"}:
            incoming_message = self.in_message(
                torch.cat([source_state, encoded_edge], dim=-1)
            )
            incoming_logit = self.in_attention(
                torch.cat([target_state, source_state, encoded_edge], dim=-1)
            ).squeeze(-1)
            incoming_weight = _segment_softmax(incoming_logit, target, len(nodes))
            incoming = _aggregate(
                incoming_message, incoming_weight, target, len(nodes)
            )
        else:
            incoming_weight = edge_features.new_zeros(edge_features.shape[0])
            incoming = torch.zeros_like(nodes)

        if self.directed_mode in {"both", "outgoing"}:
            outgoing_message = self.out_message(
                torch.cat([target_state, encoded_edge], dim=-1)
            )
            outgoing_logit = self.out_attention(
                torch.cat([source_state, target_state, encoded_edge], dim=-1)
            ).squeeze(-1)
            outgoing_weight = _segment_softmax(outgoing_logit, source, len(nodes))
            outgoing = _aggregate(
                outgoing_message, outgoing_weight, source, len(nodes)
            )
        else:
            outgoing_weight = edge_features.new_zeros(edge_features.shape[0])
            outgoing = torch.zeros_like(nodes)

        contrast = torch.cat([
            nodes, incoming, outgoing, (nodes - incoming).abs(),
            (nodes - outgoing).abs(), nodes * incoming, nodes * outgoing,
        ], dim=-1)
        updated = self.norm(nodes + self.update(contrast))
        return updated, {
            "incoming_attention": incoming_weight,
            "outgoing_attention": outgoing_weight,
            "incoming_context": incoming,
            "outgoing_context": outgoing,
        }


class RoleAwareThreadHypergraphLayer(nn.Module):
    """Account -> thread -> account attention with author/commenter roles."""

    def __init__(self, hidden_dim: int, thread_dim: int, incidence_dim: int,
                 dropout: float) -> None:
        super().__init__()
        self.thread_encoder = _mlp(thread_dim, hidden_dim, dropout)
        self.incidence_encoder = _mlp(incidence_dim, hidden_dim, dropout)
        self.node_to_thread_message = _mlp(hidden_dim * 2, hidden_dim, dropout)
        self.node_to_thread_attention = nn.Linear(hidden_dim * 3, 1)
        self.thread_norm = nn.LayerNorm(hidden_dim)
        self.thread_to_node_message = _mlp(hidden_dim * 2, hidden_dim, dropout)
        self.thread_to_node_attention = nn.Linear(hidden_dim * 3, 1)
        self.node_update = _mlp(hidden_dim * 4, hidden_dim, dropout)
        self.node_norm = nn.LayerNorm(hidden_dim)

    def forward(self, nodes: torch.Tensor, thread_features: torch.Tensor,
                incidence_index: torch.Tensor, incidence_features: torch.Tensor
                ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        node_index, thread_index = incidence_index[0].long(), incidence_index[1].long()
        threads = self.thread_encoder(thread_features)
        incidence = self.incidence_encoder(incidence_features)
        incident_nodes = nodes.index_select(0, node_index)
        incident_threads = threads.index_select(0, thread_index)

        n2t_message = self.node_to_thread_message(torch.cat([incident_nodes, incidence], dim=-1))
        n2t_logit = self.node_to_thread_attention(
            torch.cat([incident_threads, incident_nodes, incidence], dim=-1)
        ).squeeze(-1)
        n2t_weight = _segment_softmax(n2t_logit, thread_index, len(threads))
        thread_context = _aggregate(n2t_message, n2t_weight, thread_index, len(threads))
        threads = self.thread_norm(threads + thread_context)

        updated_incident_threads = threads.index_select(0, thread_index)
        t2n_message = self.thread_to_node_message(
            torch.cat([updated_incident_threads, incidence], dim=-1)
        )
        t2n_logit = self.thread_to_node_attention(
            torch.cat([incident_nodes, updated_incident_threads, incidence], dim=-1)
        ).squeeze(-1)
        t2n_weight = _segment_softmax(t2n_logit, node_index, len(nodes))
        node_context = _aggregate(t2n_message, t2n_weight, node_index, len(nodes))
        contrast = torch.cat([
            nodes, node_context, (nodes - node_context).abs(), nodes * node_context,
        ], dim=-1)
        updated_nodes = self.node_norm(nodes + self.node_update(contrast))
        return updated_nodes, {
            "node_to_thread_attention": n2t_weight,
            "thread_to_node_attention": t2n_weight,
            "thread_representation": threads,
            "thread_context": node_context,
        }


class R3HNetwork(nn.Module):
    """Opt-in R3H-Net head for account-level binary risk detection.

    ``content_representation`` should normally be the 256-D representation
    returned by ``MultiSourceRiskModel`` before its classifier. Repetition
    features are explicit URL-insensitive account statistics, recommended in
    this order: exact duplicate ratio, normalized duplicate ratio, maximum
    duplicate-cluster share, cross-target reuse, and cross-post reuse.
    """

    def __init__(self, config: R3HConfig | None = None) -> None:
        super().__init__()
        self.config = config or R3HConfig()
        self.config.validate()
        c, dim = self.config, self.config.hidden_dim
        self.content_projection = _mlp(c.content_dim, dim, c.dropout)
        self.repetition_encoder = nn.Sequential(
            nn.LayerNorm(c.repetition_dim), _mlp(c.repetition_dim, dim, c.dropout),
            nn.LayerNorm(dim),
        )
        self.directed = DirectedHeterophilyLayer(dim, c.edge_dim, c.dropout)
        self.hypergraph = RoleAwareThreadHypergraphLayer(
            dim, c.thread_dim, c.incidence_dim, c.dropout
        )
        self.graph_fusion = nn.Sequential(
            _mlp(dim * 4, dim, c.dropout), nn.LayerNorm(dim),
        )
        self.modality_gate = nn.Sequential(
            nn.Linear(dim * 3 + 2, dim), nn.GELU(), nn.Linear(dim, 2),
        )
        self.output_norm = nn.LayerNorm(dim)
        self.classifier = nn.Sequential(
            nn.Linear(dim, dim // 2), nn.GELU(), nn.Dropout(c.dropout), nn.Linear(dim // 2, 1),
        )

    def _graph_dropout(self, value: torch.Tensor, present: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.training or self.config.graph_dropout == 0:
            return value * present.unsqueeze(-1), present
        keep = torch.rand(len(value), device=value.device) >= self.config.graph_dropout
        effective = present & keep
        return value * effective.unsqueeze(-1), effective

    def forward(self, content_representation: torch.Tensor, repetition_features: torch.Tensor,
                edge_index: torch.Tensor, edge_features: torch.Tensor,
                thread_features: torch.Tensor, incidence_index: torch.Tensor,
                incidence_features: torch.Tensor) -> dict[str, Any]:
        content = self.content_projection(content_representation)
        repetition_input = torch.nan_to_num(repetition_features)
        repetition = self.repetition_encoder(repetition_input)
        if self.config.use_directed:
            directed, directed_aux = self.directed(content, edge_index, edge_features)
        else:
            directed, directed_aux = content, {}
        if self.config.use_hypergraph:
            hypergraph, hypergraph_aux = self.hypergraph(
                content, thread_features, incidence_index, incidence_features
            )
        else:
            hypergraph, hypergraph_aux = content, {}
        if self.config.use_directed and self.config.use_hypergraph:
            graph = self.graph_fusion(torch.cat([
                directed, hypergraph, (directed - hypergraph).abs(), directed * hypergraph,
            ], dim=-1))
        elif self.config.use_directed:
            graph = directed
        elif self.config.use_hypergraph:
            graph = hypergraph
        else:
            graph = torch.zeros_like(content)

        graph_degree = content.new_zeros(len(content))
        if self.config.use_directed and edge_index.numel():
            graph_degree.index_add_(0, edge_index[0].long(), torch.ones_like(edge_index[0], dtype=content.dtype))
            graph_degree.index_add_(0, edge_index[1].long(), torch.ones_like(edge_index[1], dtype=content.dtype))
        if self.config.use_hypergraph and incidence_index.numel():
            graph_degree.index_add_(0, incidence_index[0].long(),
                                    torch.ones_like(incidence_index[0], dtype=content.dtype))
        graph_present = graph_degree > 0
        graph, graph_present = self._graph_dropout(graph, graph_present)
        repetition_present = torch.isfinite(repetition_features).any(dim=-1)
        repetition_present = repetition_present & self.config.use_repetition
        repetition = repetition * repetition_present.unsqueeze(-1)

        availability = torch.stack([repetition_present, graph_present], dim=-1).to(content.dtype)
        gates = torch.sigmoid(self.modality_gate(
            torch.cat([content, repetition, graph, availability], dim=-1)
        )) * availability
        representation = self.output_norm(
            content + gates[:, 0:1] * repetition + gates[:, 1:2] * graph
        )
        return {
            "account_logit": self.classifier(representation).squeeze(-1),
            "account_representation": representation,
            "repetition_representation": repetition,
            "graph_representation": graph,
            "modality_gates": gates,
            "graph_present": graph_present,
            "directed": directed_aux,
            "hypergraph": hypergraph_aux,
        }


@dataclass
class JointR3HConfig:
    """Configuration for the end-to-end parallel R3H fusion model."""

    content_dim: int = 256
    topology_dim: int = 10
    repetition_dim: int = 5
    edge_dim: int = 9
    thread_dim: int = 4
    incidence_dim: int = 5
    dropout: float = 0.2
    graph_dropout: float = 0.15
    initial_gate_bias: float = -2.5
    use_content: bool = True
    use_repetition: bool = True
    use_directed: bool = True
    use_hypergraph: bool = True
    directed_mode: str = "both"
    use_topology: bool = True
    use_edge_features: bool = True

    def validate(self) -> None:
        for name in ("content_dim", "topology_dim", "repetition_dim", "edge_dim",
                     "thread_dim", "incidence_dim"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        for name in ("dropout", "graph_dropout"):
            if not 0 <= getattr(self, name) < 1:
                raise ValueError(f"{name} must be in [0, 1)")
        if not (self.use_content or self.use_repetition
                or self.use_directed or self.use_hypergraph):
            raise ValueError("At least one content or relational branch must be enabled")
        if self.directed_mode not in {"both", "incoming", "outgoing"}:
            raise ValueError(f"Unsupported directed_mode: {self.directed_mode}")


class JointR3HFusionModel(nn.Module):
    """End-to-end content, repetition, directed-graph, and hypergraph model.

    The supplied account encoder is the original ``MultiSourceRiskModel``.
    Target and sampled context accounts are encoded with shared weights. The
    original content representation stays on a direct residual path, while
    graph branches contribute gated corrections before one final classifier.
    """

    def __init__(self, account_encoder: nn.Module,
                 config: JointR3HConfig | None = None) -> None:
        super().__init__()
        self.account_encoder = account_encoder
        self.config = config or JointR3HConfig()
        self.config.validate()
        c, dim = self.config, self.config.content_dim
        self.topology_encoder = nn.Sequential(
            nn.LayerNorm(c.topology_dim), _mlp(c.topology_dim, dim, c.dropout),
            nn.LayerNorm(dim),
        )
        self.topology_gate = nn.Sequential(
            nn.Linear(dim * 2, dim), nn.GELU(), nn.Linear(dim, 1),
        )
        self.repetition_encoder = nn.Sequential(
            nn.LayerNorm(c.repetition_dim), _mlp(c.repetition_dim, dim, c.dropout),
            nn.LayerNorm(dim),
        )
        self.directed = DirectedHeterophilyLayer(
            dim, c.edge_dim, c.dropout,
            directed_mode=c.directed_mode,
            use_edge_features=c.use_edge_features,
        )
        self.hypergraph = RoleAwareThreadHypergraphLayer(
            dim, c.thread_dim, c.incidence_dim, c.dropout
        )
        self.fusion_gate = nn.Sequential(
            nn.Linear(dim * 4 + 3, dim), nn.GELU(), nn.Dropout(c.dropout),
            nn.Linear(dim, 3),
        )
        nn.init.zeros_(self.fusion_gate[-1].weight)
        nn.init.constant_(self.fusion_gate[-1].bias, c.initial_gate_bias)
        classifier = getattr(account_encoder, "classifier", None)
        if classifier is None:
            raise ValueError("account_encoder must expose its classifier")
        # Start the final decision surface from the original content classifier.
        self.classifier = copy.deepcopy(classifier)

    def _drop_branch(self, value: torch.Tensor,
                     present: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.training or self.config.graph_dropout == 0:
            return value * present.unsqueeze(-1), present
        keep = torch.rand(len(value), device=value.device) >= self.config.graph_dropout
        effective = present & keep
        return value * effective.unsqueeze(-1), effective

    @staticmethod
    def _directed_presence(edge_index: torch.Tensor, size: int,
                           device: torch.device, mode: str) -> torch.Tensor:
        degree = torch.zeros(size, device=device)
        if edge_index.numel():
            ones = torch.ones(edge_index.shape[1], device=device)
            if mode in {"both", "outgoing"}:
                degree.index_add_(0, edge_index[0].long(), ones)
            if mode in {"both", "incoming"}:
                degree.index_add_(0, edge_index[1].long(), ones)
        return degree > 0

    @staticmethod
    def _hypergraph_presence(incidence_index: torch.Tensor, size: int,
                             device: torch.device) -> torch.Tensor:
        degree = torch.zeros(size, device=device)
        if incidence_index.numel():
            degree.index_add_(
                0, incidence_index[0].long(),
                torch.ones(incidence_index.shape[1], device=device),
            )
        return degree > 0

    def forward(self, target_batch: dict[str, Any], context_batch: dict[str, Any] | None,
                topology_features: torch.Tensor, repetition_features: torch.Tensor,
                edge_index: torch.Tensor, edge_features: torch.Tensor,
                thread_features: torch.Tensor, incidence_index: torch.Tensor,
                incidence_features: torch.Tensor) -> dict[str, Any]:
        topology = self.topology_encoder(torch.nan_to_num(topology_features))
        if self.config.use_content:
            target_output = self.account_encoder(target_batch)
            content_parts = [target_output["account_representation"]]
            baseline_parts = [target_output["account_logit"]]
            if context_batch is not None:
                context_output = self.account_encoder(context_batch)
                content_parts.append(context_output["account_representation"])
                baseline_parts.append(context_output["account_logit"])
            content = torch.cat(content_parts, dim=0)
            baseline_logit = torch.cat(baseline_parts, dim=0)
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
        else:
            # Content-free ablations use label-independent topology only as the
            # node seed. The account encoder is deliberately not evaluated.
            content = topology.new_zeros((len(topology), self.config.content_dim))
            baseline_logit = topology.new_zeros(len(topology))
            topology_weight = topology.new_ones((len(topology), 1))
            graph_seed = topology
            base_representation = topology

        repetition = self.repetition_encoder(torch.nan_to_num(repetition_features))
        repetition_present = torch.isfinite(repetition_features).any(dim=-1)
        repetition_present = repetition_present & self.config.use_repetition
        repetition = repetition * repetition_present.unsqueeze(-1)

        if self.config.use_directed:
            directed, directed_aux = self.directed(graph_seed, edge_index, edge_features)
            directed_delta = directed - graph_seed
            directed_present = self._directed_presence(
                edge_index, len(content), content.device,
                self.config.directed_mode,
            )
            directed_delta, directed_present = self._drop_branch(
                directed_delta, directed_present
            )
        else:
            directed_delta, directed_aux = torch.zeros_like(content), {}
            directed_present = torch.zeros(len(content), dtype=torch.bool,
                                             device=content.device)

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
            base_representation, repetition, directed_delta, hypergraph_delta, availability,
        ], dim=-1))) * availability
        fused = (
            base_representation
            + gates[:, 0:1] * repetition
            + gates[:, 1:2] * directed_delta
            + gates[:, 2:3] * hypergraph_delta
        )
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
            "directed": directed_aux,
            "hypergraph": hypergraph_aux,
        }
