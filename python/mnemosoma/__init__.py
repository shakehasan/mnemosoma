"""Mnemosoma: memory for multi-agent, multi-user systems."""

from ._core import __version__
from .consolidation import ConsolidationReport, Consolidator, Fact
from .store import MemoryStore
from .types import Hit, Identity, Memory, Scope

__all__ = [
    "ConsolidationReport",
    "Consolidator",
    "Fact",
    "Hit",
    "Identity",
    "Memory",
    "MemoryStore",
    "Scope",
    "__version__",
]
