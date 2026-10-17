from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class ModelConfig:
    embedding_dim: int = 4096
    hidden_dim: int = 256
    transformer_layers: int = 2
    transformer_heads: int = 8
    transformer_ffn_dim: int = 1024
    dropout: float = 0.2
    mil_pooling: str = "normalized_lse"
    mil_temperature: float = 1.0
    attention_temperature: float = 1.0
    top_k: int = 5
    behavior_encoder: bool = True
    content_mode: str = "post_comment"  # post_only/comment_only/post_comment
    use_moltbook_profile: bool = True
    use_twitter_profile: bool = True
    use_alignment: bool = True
    use_url: bool = True
    fusion: str = "hierarchical"  # hierarchical/direct_concat
    url_feature_dim: int = 18
    url_max_bytes: int = 256
    url_pooling: str = "attention"  # attention/deepsets
    moltbook_profile_dropout: float = 0.1
    twitter_profile_dropout: float = 0.1
    url_dropout: float = 0.1

    def validate(self) -> None:
        if self.embedding_dim < 1 or self.hidden_dim < 2:
            raise ValueError("embedding_dim must be positive and hidden_dim must be at least 2")
        if self.transformer_heads < 1:
            raise ValueError("transformer_heads must be positive")
        if self.hidden_dim % self.transformer_heads:
            raise ValueError("hidden_dim must be divisible by transformer_heads")
        if self.transformer_layers < 1 or self.transformer_ffn_dim < 1:
            raise ValueError("transformer layer count and FFN dimension must be positive")
        if self.url_feature_dim < 1:
            raise ValueError("url_feature_dim must be positive")
        if self.mil_pooling not in {"normalized_lse", "lse", "topk", "max", "noisy_or"}:
            raise ValueError(f"Unsupported MIL pooling: {self.mil_pooling}")
        if self.mil_temperature <= 0 or self.attention_temperature <= 0:
            raise ValueError("MIL/attention temperatures must be positive")
        if self.top_k < 1:
            raise ValueError("top_k must be positive")
        if self.content_mode not in {"post_only", "comment_only", "post_comment"}:
            raise ValueError(f"Unsupported content_mode: {self.content_mode}")
        if self.fusion not in {"hierarchical", "direct_concat"}:
            raise ValueError(f"Unsupported fusion: {self.fusion}")
        if self.url_pooling not in {"attention", "deepsets"}:
            raise ValueError(f"Unsupported URL pooling: {self.url_pooling}")
        if self.url_max_bytes < 5:
            raise ValueError("url_max_bytes must be at least 5 for the Byte-CNN kernels")
        for name in ("dropout", "moltbook_profile_dropout", "twitter_profile_dropout", "url_dropout"):
            value = float(getattr(self, name))
            if not 0 <= value < 1:
                raise ValueError(f"{name} must be in [0, 1)")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TrainConfig:
    seed: int = 42
    batch_size: int = 16
    max_posts: int = 128
    max_comments: int = 128
    max_urls: int = 64
    epochs: int = 30
    patience: int = 5
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    device: str = "cuda"
    num_workers: int = 4
    amp: bool = True
    pos_weight: bool = True
    oversample: bool = False
    rho_fn: float = 0.0
    mil_aux_weight: float = 0.0
    threshold_strategy: str = "max_f1"
    target_recall: float = 0.80

    def validate(self) -> None:
        if self.pos_weight and self.oversample:
            raise ValueError("oversampling and pos_weight are mutually exclusive")
        if not 0 <= self.rho_fn < 1:
            raise ValueError("rho_fn must satisfy 0 <= rho_fn < 1")
        if self.mil_aux_weight < 0:
            raise ValueError("mil_aux_weight must be non-negative")
        if self.threshold_strategy not in {"max_f1", "max_precision_at_recall"}:
            raise ValueError(f"Unsupported threshold strategy: {self.threshold_strategy}")
        if not 0 <= self.target_recall <= 1:
            raise ValueError("target_recall must be in [0, 1]")
        if min(self.batch_size, self.epochs, self.patience) < 1:
            raise ValueError("batch_size, epochs, and patience must be positive")
        if min(self.max_posts, self.max_comments, self.max_urls) < 0:
            raise ValueError("instance limits must be non-negative (zero means unlimited)")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
