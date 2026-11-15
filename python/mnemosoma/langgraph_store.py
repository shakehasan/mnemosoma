"""A LangGraph ``BaseStore`` backed by Mnemosoma.

LangGraph agents that already use ``store.put`` / ``store.get`` /
``store.search`` get Mnemosoma's guarantees without changing their code:
every read is scoped to one identity, every value keeps its history, and
search uses the hybrid, graph-expanded recall.

    from langgraph.graph import StateGraph
    from mnemosoma.langgraph_store import MnemosomaStore

    store = MnemosomaStore(memory, identity=me, scope=Scope.user(me.user))
    graph = builder.compile(store=store)

Requires the ``langchain`` extra: ``pip install "mnemosoma[langchain]"``.

How LangGraph's model maps onto Mnemosoma:

- A ``(namespace, key)`` pair is one fact. Putting it again supersedes the
  old value, which stays readable through :meth:`MnemosomaStore.history`.
- Deleting erases every version, as :meth:`MemoryStore.forget` does.
- The store acts as one identity in one scope, fixed when it is created.
  Two users' agents each get their own ``MnemosomaStore`` and cannot read
  each other's values, whatever namespace they ask for.
- Search leaves out values a write-time guard marked untrusted.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import Iterable
from datetime import datetime
from typing import Any

from langgraph.store.base import (
    BaseStore,
    GetOp,
    Item,
    ListNamespacesOp,
    MatchCondition,
    Op,
    PutOp,
    Result,
    SearchItem,
    SearchOp,
)

from . import limits
from .store import MemoryStore
from .types import Identity, Memory, Scope

_PREFIX_TAG = "lg-prefix:"


def _subject(namespace: tuple[str, ...], key: str) -> str:
    return "lg:" + json.dumps([*namespace, key], ensure_ascii=False)


def _parse_subject(subject: str | None) -> tuple[tuple[str, ...], str] | None:
    if not subject or not subject.startswith("lg:"):
        return None
    *namespace, key = json.loads(subject[3:])
    return tuple(namespace), key


def _prefix_tags(namespace: tuple[str, ...]) -> list[str]:
    return [
        _PREFIX_TAG + json.dumps(list(namespace[:depth]), ensure_ascii=False) for depth in range(len(namespace) + 1)
    ]


def _matches(value: dict[str, Any], conditions: dict[str, Any] | None) -> bool:
    return all(value.get(field) == expected for field, expected in (conditions or {}).items())


def _path_matches(namespace: tuple[str, ...], condition: MatchCondition) -> bool:
    path = tuple(condition.path)
    if len(path) > len(namespace):
        return False
    part = namespace[: len(path)] if condition.match_type == "prefix" else namespace[len(namespace) - len(path) :]
    return all(want in ("*", got) for want, got in zip(path, part, strict=True))


class MnemosomaStore(BaseStore):
    """LangGraph's ``BaseStore`` interface over a :class:`MemoryStore`.

    Args:
        memory: The Mnemosoma store.
        identity: Who this store acts as. Fixed for its lifetime.
        scope: Where values are written and read. A user scope gives each
            user their own values; a team scope gives a team a shared one.
    """

    supports_ttl = True

    def __init__(self, memory: MemoryStore, *, identity: Identity, scope: Scope) -> None:
        self.memory = memory
        self.identity = identity
        self.scope = scope

    # ------------------------------------------------------------------

    def _current(self, namespace: tuple[str, ...], key: str) -> tuple[Memory, Memory] | None:
        versions = self.memory.history(self.identity, self.scope, _subject(namespace, key))
        live = [m for m in versions if m.valid_to is None]
        return (versions[0], live[-1]) if live else None

    @staticmethod
    def _value(m: Memory) -> dict[str, Any]:
        return json.loads(m.content)

    def _item(self, namespace: tuple[str, ...], key: str, first: Memory, current: Memory) -> Item:
        return Item(
            value=self._value(current),
            key=key,
            namespace=namespace,
            created_at=first.recorded_at,
            updated_at=current.recorded_at,
        )

    def _get(self, op: GetOp) -> Item | None:
        found = self._current(tuple(op.namespace), op.key)
        return None if found is None else self._item(tuple(op.namespace), op.key, *found)

    def _put(self, op: PutOp) -> None:
        namespace = tuple(op.namespace)
        if op.value is None:
            for version in self.memory.history(self.identity, self.scope, _subject(namespace, op.key)):
                # A version may already be gone, erased along with an earlier one.
                with contextlib.suppress(KeyError):
                    self.memory.forget(self.identity, version.id)
            return
        self.memory.write(
            self.identity,
            json.dumps(op.value, ensure_ascii=False, sort_keys=True),
            scope=self.scope,
            subject=_subject(namespace, op.key),
            tags=_prefix_tags(namespace),
            source="langgraph-store",
            ttl=None if op.ttl is None else op.ttl * limits.SECONDS_PER_MINUTE,
        )

    def _search(self, op: SearchOp) -> list[SearchItem]:
        prefix_tag = _prefix_tags(tuple(op.namespace_prefix))[-1]
        hits = self.memory.recall(
            self.identity,
            op.query or "",
            k=limits.SCAN_LIMIT,
            kinds=["semantic"],
            scopes=[self.scope.kind],
        )
        results: list[SearchItem] = []
        for hit in hits:
            m = hit.memory
            parsed = _parse_subject(m.subject)
            if parsed is None or m.scope != self.scope or prefix_tag not in m.tags:
                continue
            value = self._value(m)
            if not _matches(value, op.filter):
                continue
            namespace, key = parsed
            found = self._current(namespace, key)
            created = found[0].recorded_at if found else m.recorded_at
            results.append(
                SearchItem(
                    namespace=namespace,
                    key=key,
                    value=value,
                    created_at=created,
                    updated_at=m.recorded_at,
                    score=hit.score if op.query else None,
                )
            )
        return results[op.offset : op.offset + op.limit]

    def _list_namespaces(self, op: ListNamespacesOp) -> list[tuple[str, ...]]:
        hits = self.memory.recall(self.identity, k=limits.SCAN_LIMIT, kinds=["semantic"], scopes=[self.scope.kind])
        seen: set[tuple[str, ...]] = set()
        for hit in hits:
            parsed = _parse_subject(hit.memory.subject)
            if parsed is None or hit.memory.scope != self.scope:
                continue
            namespace = parsed[0]
            if all(_path_matches(namespace, c) for c in op.match_conditions or ()):
                seen.add(namespace if op.max_depth is None else namespace[: op.max_depth])
        return sorted(seen)[op.offset : op.offset + op.limit]

    # ------------------------------------------------------------------

    def batch(self, ops: Iterable[Op]) -> list[Result]:
        results: list[Result] = []
        for op in ops:
            if isinstance(op, GetOp):
                results.append(self._get(op))
            elif isinstance(op, PutOp):
                self._put(op)
                results.append(None)
            elif isinstance(op, SearchOp):
                results.append(self._search(op))
            elif isinstance(op, ListNamespacesOp):
                results.append(self._list_namespaces(op))
            else:
                raise TypeError(f"unsupported store operation: {type(op).__name__}")
        return results

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        return await asyncio.to_thread(self.batch, list(ops))

    def history(self, namespace: tuple[str, ...], key: str) -> list[tuple[datetime, dict[str, Any]]]:
        """Every value this key has held, oldest first, with when it was written."""
        versions = self.memory.history(self.identity, self.scope, _subject(tuple(namespace), key))
        return [(m.recorded_at, self._value(m)) for m in versions]
