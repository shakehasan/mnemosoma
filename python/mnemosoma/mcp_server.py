"""A Model Context Protocol server for Mnemosoma.

Any MCP client (a desktop assistant, an IDE agent, another framework) can
use the store through four tools: ``remember``, ``recall``, ``history`` and
``forget``.

Requires the ``mcp`` extra: ``pip install "mnemosoma[mcp]"``.

Run it over stdio, configured by environment variables::

    MNEMOSOMA_KEY=...           signing key (required)
    MNEMOSOMA_PATH=memory.log   log file (default: in memory)
    MNEMOSOMA_TENANT=acme       the identity this server acts as (required)
    MNEMOSOMA_USER=ana
    MNEMOSOMA_AGENT=desktop-assistant
    MNEMOSOMA_ROLE=researcher
    MNEMOSOMA_TEAM=launch       optional
    MNEMOSOMA_SCOPE=user        where remember writes: user, team or agent

    mnemosoma-mcp

One server process is one identity. The client's model chooses what to
remember and look up; it has no way to choose whose memory it sees.
"""

from __future__ import annotations

import os

try:  # mcp 2.x
    from mcp.server.mcpserver import MCPServer
except ImportError:  # mcp 1.x, where the same class is called FastMCP
    from mcp.server.fastmcp import FastMCP as MCPServer

from .guards import injection_guard
from .store import MemoryStore
from .types import Identity, Scope


def build_server(store: MemoryStore, identity: Identity, *, write_scope: Scope, name: str = "mnemosoma") -> MCPServer:
    """Create an MCP server that acts as ``identity`` on ``store``."""
    server = MCPServer(name)

    @server.tool()
    def remember(content: str, subject: str | None = None, entities: list[str] | None = None) -> str:
        """Save a fact to long-term memory.

        Args:
            content: The fact, as one self-contained sentence.
            subject: Optional stable key for a fact that can change, such as
                "ana/employer". A later memory with the same subject replaces it.
            entities: People, organisations and things the fact mentions.
        """
        try:
            memory_id = store.write(
                identity,
                content,
                scope=write_scope,
                subject=subject,
                entities=entities or (),
                assertion="inferred",
                source="mcp",
            )
        except (PermissionError, ValueError) as refused:
            return f"Not saved: {refused}"
        return f"Saved as memory {memory_id}."

    @server.tool()
    def recall(query: str, k: int = 5) -> str:
        """Search long-term memory and return what is relevant, with its source and reliability."""
        return store.context(identity, query, k=max(1, min(k, 20))) or "No relevant memories."

    @server.tool()
    def history(subject: str) -> str:
        """Show every past and current version of a fact, by its subject key."""
        versions = store.history(identity, write_scope, subject)
        if not versions:
            return f"No memory with subject {subject!r}."
        return "\n".join(
            f"- {m.content} (from {m.valid_from:%Y-%m-%d}"
            f"{f' to {m.valid_to:%Y-%m-%d}' if m.valid_to else ', current'})"
            for m in versions
        )

    @server.tool()
    def forget(memory_id: int) -> str:
        """Permanently erase a memory and everything derived from it."""
        try:
            removed = store.forget(identity, memory_id)
        except (PermissionError, KeyError) as refused:
            return f"Not erased: {refused}"
        return f"Erased {removed} memor{'y' if removed == 1 else 'ies'}."

    return server


def _from_env() -> MCPServer:
    def need(name: str) -> str:
        value = os.environ.get(name)
        if not value:
            raise SystemExit(f"{name} is not set. See `python -m mnemosoma.mcp_server --help`.")
        return value

    identity = Identity(
        tenant=need("MNEMOSOMA_TENANT"),
        user=need("MNEMOSOMA_USER"),
        agent=need("MNEMOSOMA_AGENT"),
        role=need("MNEMOSOMA_ROLE"),
        team=os.environ.get("MNEMOSOMA_TEAM") or None,
    )
    where = os.environ.get("MNEMOSOMA_SCOPE", "user")
    if where == "user":
        scope = Scope.user(identity.user)
    elif where == "agent":
        scope = Scope.agent(identity.agent)
    elif where == "team" and identity.team:
        scope = Scope.team(identity.team)
    else:
        raise SystemExit("MNEMOSOMA_SCOPE must be user, agent, or team (team needs MNEMOSOMA_TEAM).")
    store = MemoryStore(need("MNEMOSOMA_KEY"), os.environ.get("MNEMOSOMA_PATH") or None, guards=[injection_guard])
    return build_server(store, identity, write_scope=scope)


def main() -> None:
    import sys

    if any(arg in ("-h", "--help") for arg in sys.argv[1:]):
        print(__doc__)
        return
    _from_env().run()


if __name__ == "__main__":
    main()
