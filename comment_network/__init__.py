"""Reply-aware Moltbook comment-network construction and R3H training."""

from typing import Any

__all__ = ["AnalysisConfig", "build_and_analyze"]


def __getattr__(name: str) -> Any:
    """Load analysis dependencies only when the construction API is requested."""
    if name in __all__:
        from .analysis import AnalysisConfig, build_and_analyze

        return {"AnalysisConfig": AnalysisConfig, "build_and_analyze": build_and_analyze}[name]
    raise AttributeError(name)
