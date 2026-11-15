"""The Python face of the Rust engine."""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from . import _core
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

        Raises:
            PermissionError: The caller's role may not write to ``scope``, or
                ``derived_from`` names a memory the caller cannot read.
        """
        memory = {
            "scope": scope.to_dict(),
            "kind": kind,
            "content": content,
            "subject": subject,
            "entities": list(entities),
            "tags": list(tags),
            "embedding": self._embed(content, embedding),
            "assertion": assertion,
            "source": source,
            "trust": trust,
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
        embedding: Sequence[float] | None = None,
        now: Moment | None = None,
    ) -> list[Hit]:
        """Return the memories most relevant to ``text`` that the caller may read.

        With empty ``text`` the newest matching memories come back, which is how
        an agent reads a team blackboard or picks up a handoff. ``as_of`` answers
        from the facts that were true at that moment. Untrusted content is left
        out unless ``min_trust="untrusted"``.
        """
        query = {
            "text": text,
            "embedding": self._embed(text, embedding),
            "k": k,
            "token_budget": token_budget,
            "kinds": list(kinds),
            "scopes": list(scopes),
            "as_of": to_ms(as_of),
            "min_trust": min_trust,
            "include_archived": include_archived,
        }
        raw = self._core.recall(who.to_json(), json.dumps(query), to_ms(now))
        return [Hit.from_dict(h) for h in json.loads(raw)]

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

    def stats(self) -> dict[str, int]:
        return json.loads(self._core.stats())
