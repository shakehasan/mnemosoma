"""The Python face of the Rust engine."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from . import _core
from .context import render_context
from .guards import Draft, Guard
from .tracing import traced
from .types import (
    Assertion,
    Hit,
    Identity,
    Kind,
    Memory,
    Moment,
    Scope,
    ScopeKind,
    Trust,
    duration_ms,
    to_ms,
)

Embedder = Callable[[str], Sequence[float]]
Reranker = Callable[[str, Sequence[str]], Sequence[float]]


def _token_cost(text: str) -> int:
    return len(text.encode()) // 4 + 1


class MemoryStore:
    """Shared memory for many agents and users.

    Every method takes the caller's :class:`Identity` first. The engine decides
    what that caller may read or write; nothing here trusts the caller to
    filter its own results.

    Args:
        signing_key: Secret used to sign each memory when it is written. Keep it
            out of the log file's directory.
        path: Log file to persist to. ``None`` keeps everything in memory.
        half_life_days: Salience halves after this long without a recall.
        archive_below: ``sweep`` archives episodes whose salience decays below this.
        policies: Replace the default role policies. Maps a role name to
            ``{"read": [...], "write": [...], "can_promote": bool}``.
        embedder: Turns text into a vector, for semantic recall. Any function
            works, for example ``OllamaEmbeddings(...).embed_query``. Without
            one, the engine uses its built-in hashing embedder.
        reranker: Scores ``(query, texts)`` after the engine's own ranking, for
            example a cross-encoder. Higher is better. The engine still decides
            what is eligible; the reranker only reorders.
        guards: Checks run on every write, in order. See :mod:`mnemosoma.guards`.
    """

    def __init__(
        self,
        signing_key: str | bytes,
        path: str | Path | None = None,
        *,
        half_life_days: float = 30.0,
        archive_below: float = 0.05,
        policies: dict[str, dict[str, Any]] | None = None,
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
        guards: Sequence[Guard] = (),
    ) -> None:
        key = signing_key.encode() if isinstance(signing_key, str) else bytes(signing_key)
        self._core = _core.Store(
            key,
            None if path is None else str(path),
            half_life_days,
            archive_below,
            None if policies is None else json.dumps(policies),
        )
        self._embedder = embedder
        self._reranker = reranker
        self._guards = tuple(guards)

    def _embed(self, text: str, given: Sequence[float] | None) -> list[float] | None:
        if given is not None:
            return list(given)
        if self._embedder is not None and text.strip():
            return list(self._embedder(text))
        return None

    @traced("mnemosoma.write", "tool")
    def write(
        self,
        who: Identity,
        content: str,
        *,
        scope: Scope,
        kind: Kind = "semantic",
        subject: str | None = None,
        entities: Iterable[str] = (),
        tags: Iterable[str] = (),
        assertion: Assertion = "observed",
        source: str = "agent",
        trust: Trust = "standard",
        derived_from: Iterable[int] = (),
        valid_from: Moment | None = None,
        ttl: timedelta | float | None = None,
        salience: float | None = None,
        embedding: Sequence[float] | None = None,
        now: Moment | None = None,
    ) -> int:
        """Write one memory and return its id.

        Give a fact a ``subject`` (for example ``"ana/employer"``) and a later
        write with the same subject in the same scope supersedes it. The old
        version stays readable through :meth:`history` and ``as_of`` recall.
        Name the people and things it mentions in ``entities`` and recall can
        follow them to related memories.

        Raises:
            PermissionError: The caller's role may not write to ``scope``, or
                ``derived_from`` names a memory the caller cannot read.
            GuardBlocked: A guard refused the write.
        """
        draft = Draft(content=content, trust=trust, tags=list(tags), source=source)
        for guard in self._guards:
            guard(draft)
        memory = {
            "scope": scope.to_dict(),
            "kind": kind,
            "content": draft.content,
            "subject": subject,
            "entities": list(entities),
            "tags": draft.tags,
            "embedding": self._embed(draft.content, embedding),
            "assertion": assertion,
            "source": draft.source,
            "trust": draft.trust,
            "derived_from": list(derived_from),
            "valid_from": to_ms(valid_from),
            "ttl_ms": duration_ms(ttl),
            "salience": salience,
        }
        return self._core.write(who.to_json(), json.dumps(memory), to_ms(now))

    @traced("mnemosoma.recall", "retriever")
    def recall(
        self,
        who: Identity,
        text: str = "",
        *,
        k: int = 8,
        token_budget: int | None = None,
        kinds: Iterable[Kind] = (),
        scopes: Iterable[ScopeKind] = (),
        as_of: Moment | None = None,
        min_trust: Trust = "standard",
        include_archived: bool = False,
        graph_hops: int = 1,
        exact: bool = False,
        embedding: Sequence[float] | None = None,
        now: Moment | None = None,
    ) -> list[Hit]:
        """Return the memories most relevant to ``text`` that the caller may read.

        With empty ``text`` the newest matching memories come back, which is how
        an agent reads a team blackboard or picks up a handoff. ``as_of`` answers
        from the facts that were true at that moment. Untrusted content is left
        out unless ``min_trust="untrusted"``. ``graph_hops`` is how far to follow
        shared entities from the best matches; ``0`` turns that off. Large
        scopes use an approximate vector index; ``exact=True`` scores every
        eligible memory instead.

        Each hit's ``why`` says which rankers chose it and who shared it.
        """
        rerank = self._reranker is not None and bool(text.strip())
        query = {
            "text": text,
            "embedding": self._embed(text, embedding),
            "k": max(k * 4, 20) if rerank else k,
            "token_budget": None if rerank else token_budget,
            "kinds": list(kinds),
            "scopes": list(scopes),
            "as_of": to_ms(as_of),
            "min_trust": min_trust,
            "include_archived": include_archived,
            "graph_hops": graph_hops,
            "exact": exact,
        }
        raw = self._core.recall(who.to_json(), json.dumps(query), to_ms(now))
        hits = [Hit.from_dict(h) for h in json.loads(raw)]
        if not rerank or not hits:
            return hits

        scores = list(self._reranker(text, [h.memory.content for h in hits]))
        if len(scores) != len(hits):
            raise ValueError(f"reranker returned {len(scores)} scores for {len(hits)} texts")
        order = sorted(range(len(hits)), key=lambda i: -scores[i])
        picked: list[Hit] = []
        used = 0
        for i in order:
            if len(picked) >= k:
                break
            cost = _token_cost(hits[i].memory.content)
            if token_budget is not None and used + cost > token_budget:
                continue
            used += cost
            picked.append(replace(hits[i], score=float(scores[i])))
        return picked

    def context(self, who: Identity, text: str, *, token_budget: int = 800, **recall_options: Any) -> str:
        """Recall for ``text`` and render the result as a block of prompt context.

        The block groups memories into established, unconfirmed and untrusted,
        says where each came from, and fits ``token_budget``. Extra keyword
        arguments go to :meth:`recall`. Returns ``""`` when nothing is found.
        """
        return render_context(self.recall(who, text, token_budget=token_budget, **recall_options))

    def pending_review(self, who: Identity, *, limit: int = 50, now: Moment | None = None) -> list[Hit]:
        """Inferences the caller can read that nobody has confirmed yet, newest first.

        This is the queue for a human or a curator agent: confirm an item with
        :meth:`corroborate`, or remove it with :meth:`forget`.
        """
        hits = self.recall(who, kinds=["semantic"], k=max(limit * 4, 200), now=now)
        return [h for h in hits if not h.established][:limit]

    def get(self, who: Identity, memory_id: int, *, now: Moment | None = None) -> Memory:
        """Fetch one memory. Raises ``KeyError`` if it is missing or hidden from the caller."""
        return Memory.from_dict(json.loads(self._core.get(who.to_json(), memory_id, to_ms(now))))

    def history(self, who: Identity, scope: Scope, subject: str, *, now: Moment | None = None) -> list[Memory]:
        """Every version of a fact the caller may read, oldest first."""
        raw = self._core.history(who.to_json(), json.dumps(scope.to_dict()), subject, to_ms(now))
        return [Memory.from_dict(m) for m in json.loads(raw)]

    def grant(
        self,
        who: Identity,
        grantee: str,
        *,
        tag: str | None = None,
        expires_at: Moment | None = None,
    ) -> int:
        """Let user ``grantee`` read the caller's user-private memories.

        With ``tag``, only memories carrying that tag are shared. Returns the
        grant id, which :meth:`revoke` takes.
        """
        return self._core.grant(who.to_json(), grantee, tag, to_ms(expires_at))

    def revoke(self, who: Identity, grant_id: int, *, now: Moment | None = None) -> None:
        """Revoke a grant. Facts other agents derived from the shared memories are hidden too."""
        self._core.revoke(who.to_json(), grant_id, to_ms(now))

    def corroborate(self, who: Identity, memory_id: int, *, now: Moment | None = None) -> None:
        """Vouch for a memory. Two agents, or one promoting role, establish an inference."""
        self._core.corroborate(who.to_json(), memory_id, to_ms(now))

    def forget(self, who: Identity, memory_id: int, *, now: Moment | None = None) -> int:
        """Erase a memory and everything derived from it, from disk as well. Returns the count."""
        return self._core.forget(who.to_json(), memory_id, to_ms(now))

    def sweep(self, *, now: Moment | None = None) -> dict[str, int]:
        """Drop expired memories and archive faded episodes. Returns ``{"expired", "archived"}``."""
        return json.loads(self._core.sweep(to_ms(now)))

    def build_index(self) -> int:
        """Index every memory not yet in the approximate vector indexes, in parallel.

        Recall does this by itself when it needs to. Call it after a bulk load
        so the first queries do not pay for it. Returns how many were added.
        """
        return self._core.build_index()

    def stats(self) -> dict[str, int]:
        return json.loads(self._core.stats())


def _in_thread(name: str):
    async def method(self: AsyncMemoryStore, *args: Any, **kwargs: Any) -> Any:
        return await asyncio.to_thread(getattr(self.store, name), *args, **kwargs)

    method.__name__ = name
    method.__doc__ = f"Awaitable form of :meth:`MemoryStore.{name}`."
    return method


class AsyncMemoryStore:
    """The same API as :class:`MemoryStore`, with every method awaitable.

    Calls run on worker threads. The engine releases Python's interpreter lock
    while it works, so many agents can await memory at once without blocking
    the event loop.

    Example:
        ``hits = await AsyncMemoryStore(store).recall(me, "venue contract")``
    """

    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    write = _in_thread("write")
    recall = _in_thread("recall")
    context = _in_thread("context")
    pending_review = _in_thread("pending_review")
    get = _in_thread("get")
    history = _in_thread("history")
    grant = _in_thread("grant")
    revoke = _in_thread("revoke")
    corroborate = _in_thread("corroborate")
    forget = _in_thread("forget")
    sweep = _in_thread("sweep")
    build_index = _in_thread("build_index")
