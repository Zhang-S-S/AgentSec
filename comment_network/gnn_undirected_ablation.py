"""Undirected control layer for the directionality ablation."""
from __future__ import annotations

import torch
from torch import nn

from comment_network.gnn_fine_ablation import _aggregate, _mlp, _segment_softmax


class UndirectedHeterophilyLayer(nn.Module):
    """Shared message passing over a symmetrized version of the same graph.

    It intentionally accepts the constructor signature of DirectedHeterophilyLayer
    so it can be substituted only for the directionality control experiment.
    """

    def __init__(
        self,
        hidden_dim: int,
        edge_dim: int,
        dropout: float,
        directed_mode: str = "both",
        use_edge_features: bool = True,
    ) -> None:
        super().__init__()
        if directed_mode != "both":
            raise ValueError("Undirected control requires the same incoming+outgoing sample")
        self.directed_mode = directed_mode
        self.use_edge_features = use_edge_features
        self.edge_encoder = _mlp(edge_dim, hidden_dim, dropout)
        self.message = _mlp(hidden_dim * 2, hidden_dim, dropout)
        self.attention = nn.Linear(hidden_dim * 3, 1)
        self.update = _mlp(hidden_dim * 4, hidden_dim, dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        nodes: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        source, target = edge_index[0].long(), edge_index[1].long()
        # Every sampled directed edge is exposed in both directions, while the
        # same message/attention parameters are shared for all neighbors.
        undirected_source = torch.cat([source, target], dim=0)
        undirected_target = torch.cat([target, source], dim=0)
        doubled_features = torch.cat([edge_features, edge_features], dim=0)
        edge_input = (
            doubled_features
            if self.use_edge_features
            else torch.zeros_like(doubled_features)
        )
        encoded_edge = self.edge_encoder(edge_input)
        neighbor_state = nodes.index_select(0, undirected_source)
        center_state = nodes.index_select(0, undirected_target)
        messages = self.message(torch.cat([neighbor_state, encoded_edge], dim=-1))
        logits = self.attention(
            torch.cat([center_state, neighbor_state, encoded_edge], dim=-1)
        ).squeeze(-1)
        weights = _segment_softmax(logits, undirected_target, len(nodes))
        context = _aggregate(messages, weights, undirected_target, len(nodes))
        contrast = torch.cat(
            [nodes, context, (nodes - context).abs(), nodes * context], dim=-1
        )
        updated = self.norm(nodes + self.update(contrast))
        # The evidence heads are inactive in the selected model (weight=0).
        # Returning the shared context in both slots preserves the established
        # joint-model interface without reintroducing direction-specific paths.
        return updated, {
            "incoming_attention": weights,
            "outgoing_attention": weights,
            "incoming_context": context,
            "outgoing_context": context,
        }
