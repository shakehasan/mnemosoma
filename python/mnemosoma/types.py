"""Typed values shared by the SDK. The engine itself lives in Rust."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timedelta, timezone
from typing import Any, Literal

Kind = Literal["working", "episodic", "semantic", "procedural", "handoff"]
ScopeKind = Literal["agent_private", "role", "team", "user_private", "tenant"]
Assertion = Literal["observed", "told_by_user", "inferred"]
Trust = Literal["untrusted", "standard", "trusted"]

Moment = datetime | int
"""A point in time: an aware ``datetime`` or integer milliseconds since the epoch."""


def to_ms(moment: Moment | None) -> int | None:
    if moment is None or isinstance(moment, int):
        return moment
    if moment.tzinfo is None:
        raise ValueError("datetimes must be timezone-aware")
    return int(moment.timestamp() * 1000)


def from_ms(ms: int | None) -> datetime | None:
    return None if ms is None else datetime.fromtimestamp(ms / 1000, tz=timezone.utc)


def duration_ms(duration: timedelta | float | None) -> int | None:
    """A ``timedelta`` or a number of seconds, as milliseconds."""
    if duration is None:
        return None
    seconds = duration.total_seconds() if isinstance(duration, timedelta) else float(duration)
    return int(seconds * 1000)


@dataclass(frozen=True)
class Identity:
    """Who is calling. Build this from your own authentication, never from model output."""

    tenant: str
    user: str
    agent: str
    role: str
    team: str | None = None
    session: str | None = None

    def acting_as(self, **changes: Any) -> Identity:
        """A copy with some fields replaced, e.g. ``me.acting_as(agent="reviewer-1", role="reviewer")``."""
        return replace(self, **changes)

    def to_json(self) -> str:
        return json.dumps(asdict(self))


@dataclass(frozen=True)
class Scope:
    """Where a memory lives. Use the constructors: ``Scope.team("task-1")``."""

    kind: ScopeKind
    id: str | None = None

    @classmethod
    def agent(cls, agent: str) -> Scope:
        return cls("agent_private", agent)

    @classmethod
    def role(cls, role: str) -> Scope:
        return cls("role", role)

    @classmethod
    def team(cls, team: str) -> Scope:
        return cls("team", team)

    @classmethod
    def user(cls, user: str) -> Scope:
        return cls("user_private", user)

    @classmethod
    def tenant(cls) -> Scope:
        return cls("tenant")

    def to_dict(self) -> dict[str, str]:
        return {"kind": self.kind} if self.id is None else {"kind": self.kind, "id": self.id}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Scope:
        return cls(data["kind"], data.get("id"))


@dataclass(frozen=True)
class Memory:
    id: int
    tenant: str
    scope: Scope
    kind: Kind
    content: str
    subject: str | None
    entities: tuple[str, ...]
    tags: tuple[str, ...]
    author_user: str
    author_agent: str
    author_role: str
    assertion: Assertion
    source: str
    trust: Trust
    derived_from: tuple[int, ...]
    corroborated_by: tuple[str, ...]
    valid_from: datetime
    valid_to: datetime | None
    recorded_at: datetime
    superseded_by: int | None
    expires_at: datetime | None
    salience: float
    access_count: int
    archived: bool
    signature: str = field(repr=False)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Memory:
        return cls(
            id=d["id"],
            tenant=d["tenant"],
            scope=Scope.from_dict(d["scope"]),
            kind=d["kind"],
            content=d["content"],
            subject=d.get("subject"),
            entities=tuple(d.get("entities", ())),
            tags=tuple(d.get("tags", ())),
            author_user=d["author_user"],
            author_agent=d["author_agent"],
            author_role=d["author_role"],
            assertion=d["assertion"],
            source=d["source"],
            trust=d["trust"],
            derived_from=tuple(d.get("derived_from", ())),
            corroborated_by=tuple(d.get("corroborated_by", ())),
            valid_from=from_ms(d["valid_from"]),
            valid_to=from_ms(d.get("valid_to")),
            recorded_at=from_ms(d["recorded_at"]),
            superseded_by=d.get("superseded_by"),
            expires_at=from_ms(d.get("expires_at")),
            salience=d["salience"],
            access_count=d.get("access_count", 0),
            archived=d.get("archived", False),
            signature=d["signature"],
        )


@dataclass(frozen=True)
class Hit:
    memory: Memory
    score: float
    established: bool
    """False for an agent's inference that nobody has corroborated yet."""
    why: dict[str, Any] = field(default_factory=dict)
    """Why it was returned: ``keyword_rank``, ``vector_rank``, ``entity_rank`` and
    ``graph_rank`` (1 is best, ``None`` if that ranker did not pick it), and
    ``shared_by`` when another user shared it."""

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Hit:
        return cls(Memory.from_dict(d["memory"]), d["score"], d["established"], d.get("why", {}))
