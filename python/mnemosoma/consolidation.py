"""Consolidation: turning raw episodes into durable facts, off the write path.

Writes never call a model. This pass does, later and in the background. It
reads new episodes from one scope, asks an extractor for the facts in them,
and writes each fact back linked to the episodes it came from. Because of
that link, a fact is never more trusted than its sources, and it disappears
for any reader who loses access to them.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from . import limits
from .store import MemoryStore
from .types import Identity, Memory, Moment, Scope


@dataclass(frozen=True)
class Fact:
    """One fact found in a batch of episodes."""

    content: str
    subject: str | None = None
    """A stable key such as ``"ana/employer"``. Facts sharing a subject supersede each other."""
    entities: tuple[str, ...] = ()
    source_ids: tuple[int, ...] = ()
    """Episodes the fact came from. Empty means the whole batch."""


Extractor = Callable[[Sequence[Memory]], Sequence[Fact]]


@dataclass
class ConsolidationReport:
    episodes: int = 0
    written: list[int] = field(default_factory=list)
    corroborated: list[int] = field(default_factory=list)
    unchanged: int = 0


def _same(a: str, b: str) -> bool:
    return " ".join(a.lower().split()) == " ".join(b.lower().split())


class Consolidator:
    """Runs the consolidation pass for one caller and scope at a time.

    The consolidator acts with the permissions of the identity it is given,
    so it can only read and write what that agent could.

    Args:
        store: The memory store.
        extractor: Turns a batch of episodes into facts. Use
            :func:`mnemosoma.langchain.llm_extractor` for a model-backed one.
        batch_size: Most episodes handed to the extractor in one call.
    """

    def __init__(
        self, store: MemoryStore, extractor: Extractor, *, batch_size: int = limits.DEFAULT_CONSOLIDATION_BATCH
    ) -> None:
        self._store = store
        self._extract = extractor
        self._batch_size = batch_size
        self._cursor: dict[tuple[str, Scope], int] = {}

    def run(self, who: Identity, scope: Scope, *, now: Moment | None = None) -> ConsolidationReport:
        """Consolidate episodes in ``scope`` written since the last run.

        The position is held in memory, so a new process starts from the
        beginning; rerunning is safe because repeated facts are recognised.
        """
        key = (who.tenant, scope)
        after = self._cursor.get(key, 0)
        hits = self._store.recall(
            who, kinds=["episodic"], scopes=[scope.kind], k=limits.SCAN_LIMIT, min_trust="untrusted", now=now
        )
        episodes = sorted(
            (h.memory for h in hits if h.memory.scope == scope and h.memory.id > after),
            key=lambda m: m.id,
        )
        report = ConsolidationReport(episodes=len(episodes))
        for start in range(0, len(episodes), self._batch_size):
            batch = episodes[start : start + self._batch_size]
            batch_ids = {m.id for m in batch}
            for fact in self._extract(batch):
                sources = tuple(i for i in fact.source_ids if i in batch_ids) or tuple(sorted(batch_ids))
                self._store_fact(who, scope, fact, sources, report, now)
            self._cursor[key] = batch[-1].id
        return report

    def _store_fact(
        self,
        who: Identity,
        scope: Scope,
        fact: Fact,
        sources: tuple[int, ...],
        report: ConsolidationReport,
        now: Moment | None,
    ) -> None:
        existing = self._find_existing(who, scope, fact, now)
        if existing is not None:
            # Already known. A second agent reaching the same fact is evidence for it.
            vouched = {name.removeprefix("approved:") for name in existing.corroborated_by}
            if existing.author_agent != who.agent and who.agent not in vouched:
                self._store.corroborate(who, existing.id, now=now)
                report.corroborated.append(existing.id)
            else:
                report.unchanged += 1
            return
        report.written.append(
            self._store.write(
                who,
                fact.content,
                scope=scope,
                kind="semantic",
                subject=fact.subject,
                entities=fact.entities,
                assertion="inferred",
                source="consolidation",
                derived_from=sources,
                now=now,
            )
        )

    def _find_existing(self, who: Identity, scope: Scope, fact: Fact, now: Moment | None) -> Memory | None:
        if fact.subject is not None:
            versions = self._store.history(who, scope, fact.subject, now=now)
            current = versions[-1] if versions else None
            return current if current is not None and _same(current.content, fact.content) else None
        hits = self._store.recall(
            who,
            fact.content,
            kinds=["semantic"],
            scopes=[scope.kind],
            k=limits.DUPLICATE_SEARCH_K,
            min_trust="untrusted",
            now=now,
        )
        return next((h.memory for h in hits if h.memory.scope == scope and _same(h.memory.content, fact.content)), None)
