# mnemosoma-core

The Rust engine behind [Mnemosoma](https://github.com/shakehasan/mnemosoma): shared memory for teams of AI agents and the people they work for.

- Six scopes (agent, role, team, user, user-to-user grant, tenant) with per-role read and write policies, enforced before ranking
- Bi-temporal facts with supersession chains and point-in-time recall
- HMAC-signed append-only log, trust tiers and taint through derivation
- Hybrid recall: BM25 over posting lists, vectors (exact or HNSW), entity matches and graph expansion, fused with reciprocal rank fusion
- Every hit explains which rankers chose it and whose grant made it visible
- Salience decay, TTLs, archival and cascading erasure

```rust
use mnemosoma_core::*;

let mut store = MemoryStore::in_memory(StoreConfig::new("change-me"));
let ana = Identity {
    tenant: "acme".into(), user: "ana".into(), agent: "researcher-1".into(),
    role: "researcher".into(), team: Some("launch".into()), session: None,
};
let now = now_ms();
store.write(&ana, NewMemory::new(Scope::UserPrivate("ana".into()), MemoryKind::Semantic, "Ana works at Hooli"), now)?;
let hits = store.recall(&ana, &Query::text("where does ana work"), now);
# Ok::<(), Error>(())
```

Most people use it through the Python package, which adds LangChain, LangGraph and MCP integration. See the [main README](https://github.com/shakehasan/mnemosoma) for the design, measurements and evaluation.

License: MIT
