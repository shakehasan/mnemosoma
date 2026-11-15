"""Defaults used by the Python layer, named in one place.

Ranking and indexing parameters live in the engine's ``Tuning`` (see
``MemoryStore(tuning=...)``). These are the remaining defaults: how much
to fetch for steps that happen in Python, and the sizes the integrations
start from. Each can be overridden where it is used.
"""

from __future__ import annotations

DEFAULT_RECALL_K = 8
"""Memories returned by ``recall`` when the caller does not say."""

DEFAULT_CONTEXT_TOKENS = 800
"""Token budget for a rendered prompt block and for the agent ``recall`` tool."""

DEFAULT_TOOL_RECALL_K = 5
"""Memories an agent's ``recall`` tool returns by default."""

MAX_TOOL_RECALL_K = 20
"""Most memories a model may ask a ``recall`` tool for in one call."""

RERANK_POOL_FACTOR = 4
"""With a reranker, the engine fetches ``k`` times this many candidates."""

RERANK_MIN_POOL = 20
"""...and never fewer than this."""

REVIEW_SCAN_FACTOR = 4
"""``pending_review`` scans ``limit`` times this many recent facts for unconfirmed ones."""

REVIEW_MIN_SCAN = 200
"""...and never fewer than this."""

DEFAULT_REVIEW_LIMIT = 50
"""Items ``pending_review`` returns by default."""

SCAN_LIMIT = 10_000
"""Most memories read in one pass by steps that walk a whole scope: the
consolidator and the LangGraph store's search and namespace listing."""

DEFAULT_CONSOLIDATION_BATCH = 20
"""Episodes handed to an extractor in one call."""

DUPLICATE_SEARCH_K = 5
"""Existing facts the consolidator compares a new fact against."""

SECONDS_PER_MINUTE = 60
"""LangGraph expresses time-to-live in minutes; the engine takes seconds."""
