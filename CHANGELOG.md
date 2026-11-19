# Changelog

Notable changes to Mnemosoma. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

First public version, not yet tagged.

### Engine

- Six memory scopes with per-role read and write policies, enforced before ranking.
- User-to-user grants with tags, expiry and revocation; derived memories are hidden with their sources.
- Bi-temporal facts with supersession chains and point-in-time recall.
- HMAC-signed append-only log, trust tiers and taint through derivation.
- Hybrid recall: BM25 over posting lists, vector similarity, entity matches and graph expansion, combined with reciprocal rank fusion and packed to a token budget.
- Explainable hits: each result reports its rank in every ranker and who shared it.
- Columnar segments per scope, and a filtered HNSW vector index with parallel build and an exact fallback.
- Salience decay, time-to-live, archival and cascading erasure.
- Crash recovery for the log, atomic multi-step writes and an optional sync on every write.
- One documented, validated `Tuning` structure for every ranking and indexing parameter.
- Stemming (eighteen languages) and English stopword removal, on by default and configurable.
- Optional name finding on write (`extract_entities`): by capitalisation, with no model, learned per scope.
- The name ranker matches whole words through an index, weighs rare names more, and splits credit between ties.

### Python

- Typed SDK with type stubs, an async variant, pluggable embedder and reranker.
- Write-time guards for prompt injection and personal data.
- Consolidation of episodes into linked facts, with a model-backed extractor.
- Prompt-context rendering grouped by reliability, and a review queue for unconfirmed inferences.

### Integrations

- LangChain retriever and agent tools.
- LangGraph `BaseStore` implementation and a multi-agent example.
- Model Context Protocol server with `remember`, `recall`, `history` and `forget`.
- Optional LangSmith tracing.

### Evaluation and tooling

- Behaviour evaluation suite, gated in CI.
- Load benchmark with a permission-leak check.
- LongMemEval retrieval benchmark with a published protocol.
- CI on Linux, macOS and Windows with formatting, lint and type checks.
- Release pipeline building wheels for five platforms.
