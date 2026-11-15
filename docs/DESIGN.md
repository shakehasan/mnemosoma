# Design decisions

This document records the choices that shape Mnemosoma, what each one costs, and what was considered instead. The [README](../README.md) describes what the system does; this explains why it is built this way.

## Contents

1. [Permissions are checked before ranking](#1-permissions-are-checked-before-ranking)
2. [The caller's identity comes from the host, never the model](#2-the-callers-identity-comes-from-the-host-never-the-model)
3. [One scope per memory, one segment per scope](#3-one-scope-per-memory-one-segment-per-scope)
4. [The log is the source of truth; indexes are derived](#4-the-log-is-the-source-of-truth-indexes-are-derived)
5. [Facts are superseded, not overwritten](#5-facts-are-superseded-not-overwritten)
6. [No model on the write path](#6-no-model-on-the-write-path)
7. [Rank fusion, with priors kept small](#7-rank-fusion-with-priors-kept-small)
8. [Graph expansion over caller-supplied entities](#8-graph-expansion-over-caller-supplied-entities)
9. [Approximate search per scope, exact when small](#9-approximate-search-per-scope-exact-when-small)
10. [Trust is recorded at write time and inherited](#10-trust-is-recorded-at-write-time-and-inherited)
11. [A Rust engine behind a Python API](#11-a-rust-engine-behind-a-python-api)
12. [A built-in embedder that needs no model](#12-a-built-in-embedder-that-needs-no-model)
13. [Every parameter is a named setting](#13-every-parameter-is-a-named-setting)
14. [Known limits](#known-limits)

## 1. Permissions are checked before ranking

**Decision.** Recall first works out which memories the caller may read, then ranks only those.

**Alternative.** Rank everything, then drop what the caller may not see. This is simpler, and it is what a vector database with a metadata filter bolted on tends to do.

**Why not.** Post-filtering fails in three ways. It can return fewer than `k` results, or none, when the best matches belong to someone else. Scores computed over everything leak information: BM25 weights a term by how rare it is, so a term's weight changes with documents the caller cannot read. And it puts the security boundary in the last step of a pipeline, where one missed code path is a data leak.

**Cost.** The engine has to know about identity, scopes and grants, so it cannot be a thin wrapper over a general-purpose index. Keyword statistics are computed per query over the caller's eligible rows, which is more work than using global statistics.

## 2. The caller's identity comes from the host, never the model

**Decision.** Every call takes an `Identity` built by the application from its own authentication. Agent tools and the MCP server are bound to one identity when they are created; the model has no argument that changes it.

**Why.** A model's output can be steered by anything it reads. If "which user am I acting for" were a tool argument, a prompt injection could ask for another user's memory. Fixing identity in code removes that path entirely instead of relying on the model to refuse.

**Cost.** One tool set or server per identity. A deployment serving many users creates these per request or per session.

## 3. One scope per memory, one segment per scope

**Decision.** A memory lives in exactly one of six scopes: agent-private, role, team, user-private, shared by grant, or tenant. The engine stores each (tenant, scope) as its own segment, with its own keyword postings, vectors and approximate index.

**Alternative.** Access-control lists on each memory, as document systems do.

**Why not.** Per-memory lists are flexible and slow to evaluate: every candidate needs its own check. With scopes, "what can this caller read" becomes "which segments can this caller open", a handful of lookups. Recall cost then follows the size of what the caller may read. In the load benchmark a store that grows tenfold with the readable set held constant shows almost no change in recall time.

**Cost.** Sharing is coarser. A memory cannot be shared with an arbitrary list of people; it is shared by putting it in a wider scope or by a grant from one user to another, optionally limited by tag. Grants limited by tag and memories derived from others still need a per-memory check, which is the slow path.

## 4. The log is the source of truth; indexes are derived

**Decision.** Writes append signed events to a log. Every index is rebuilt from it when the store opens.

**Why.** It keeps the durable format small and stable while the indexes change freely: the move to columnar segments, the keyword index and the approximate index needed no migration. It also makes tampering detectable, since each record carries an HMAC over its immutable fields and an edited entry is quarantined on load.

**Cost.** Opening a large store means replaying the log and rebuilding indexes, so start-up time grows with size. Erasure has to rewrite the log, because an append-only file cannot forget by appending. There is no snapshot format yet.

## 5. Facts are superseded, not overwritten

**Decision.** Facts carry two times: when they were true (valid time) and when the store learned them (transaction time). A new fact with the same subject closes the old one's validity and links to it. The old fact stays, readable through history and point-in-time recall.

**Alternative.** Update in place, or let a model decide what to delete, as some memory systems do.

**Why not.** Overwriting answers "what is true now" and loses "what did we believe in March", which matters for audit and for any question about the past. Deleting on a model's judgement makes a wrong extraction permanent. Supersession is deterministic: no model decides which fact wins; later valid time does.

**Cost.** The caller, or the consolidation pass, has to supply a subject key. Without one, two contradicting facts both stay current. Storage grows with history.

## 6. No model on the write path

**Decision.** Writing a memory checks permission, signs, appends and indexes. Extraction of facts from raw episodes happens later, in a separate consolidation pass.

**Why.** A model call takes hundreds of milliseconds and can fail. Putting it in the write path would make every agent step wait on it and would lose data when it fails. Keeping the raw episode first means extraction can be rerun with a better model.

**Cost.** Facts lag behind episodes until consolidation runs. Until then recall works from the raw text.

## 7. Rank fusion, with priors kept small

**Decision.** Keyword, vector and entity rankers are combined with reciprocal rank fusion. Trust, confirmation and salience then adjust the score by about one percent each.

**Why fusion.** The rankers' scores are on different scales and cannot be added. Fusing ranks needs no calibration and no training data.

**Why small priors.** This was learned from a failure. Fusion packs scores tightly: the best match in two rankers leads the next by under two percent. An early version let salience scale scores between 0.5 and 1.0, and the evaluation suite showed recently used memories outranking clearly better matches. The priors are now tie-breakers, and a regression test covers the case.

**Cost.** Fusion ignores how much better the top result is than the second. A learned reranker would use that; a hook for one exists, and none is shipped.

## 8. Graph expansion over caller-supplied entities

**Decision.** Memories may name the entities they mention. Recall passes a share of each top match's score along shared entities to other memories, for a configurable number of hops, with common entities passing on less.

**Alternative.** Build a full knowledge graph with typed relations, extracted by a model at write time.

**Why not.** That puts a model back on the write path and makes recall quality depend on extraction quality. Entity co-mention is cruder and cheap, and it handles the common multi-hop case: the answer shares no words with the question but mentions the same person or thing as a memory that does.

**Cost.** Entities must be supplied by the caller or the consolidation pass; the engine does not find them. Co-mention carries no relation type, so it cannot tell "works for" from "met once".

## 9. Approximate search per scope, exact when small

**Decision.** Each scope has its own HNSW graph. Search walks past rows the caller may not read and returns only eligible ones. Scopes below a size threshold, or where the caller can read only a small share, are scanned exactly. The graph is built lazily, off the write path.

**Why per scope.** A single global graph would spend most of its time stepping over other tenants' vectors. Per-scope graphs are what the caller can mostly read.

**Why the exact fallback.** Below a few thousand rows an exact scan is faster than a graph walk and has perfect recall. When a caller can read only a sliver of a scope, a filtered graph walk degrades, so scanning the sliver is better.

**Cost.** Approximate results differ slightly from exact ones; the tests require at least 90% agreement on the top ten. Building the graph for a very large scope takes time (about 70 seconds for one million vectors in a single tenant on a 12-core laptop). Erased vectors leave a zeroed node in the graph until the store is reopened.

## 10. Trust is recorded at write time and inherited

**Decision.** Each memory records who wrote it, how they came to hold it (observed, told by a user, or inferred) and a trust tier. A derived memory is never more trusted than its least trusted source. An inference stays marked until other agents corroborate it or a curator approves it. Untrusted content is left out of recall unless asked for.

**Why.** Content-based detection of poisoned memories is a losing game against an adaptive attacker. Where a memory came from is a fact the system controls. Recording it at write time and propagating it through derivation means a claim planted by a web page cannot be laundered into a trusted fact by an agent restating it.

**Cost.** It depends on callers labelling outside content honestly. The pattern-based injection guard catches unlabelled content only some of the time (7 of 10 phrasings in the evaluation) and exists as a second line, not the first.

## 11. A Rust engine behind a Python API

**Decision.** Storage, filtering, ranking and signing are in Rust. The API, integrations and consolidation are in Python. Structured values cross the boundary as JSON.

**Why.** The engine's work is CPU-bound and runs on every agent step. In Rust it releases Python's interpreter lock, so agents on different threads do not queue, and it has no garbage collection pauses. Python is where agent frameworks live, so the surface people touch is Python.

**Why JSON at the boundary.** It keeps the binding thin and the two sides independently testable. The cost is serialisation on each call, which is small next to the work done and would be the first thing to replace if profiling showed otherwise.

**Cost.** Two languages, two toolchains, and a build step. The release pipeline builds wheels so users do not need Rust.

## 12. A built-in embedder that needs no model

**Decision.** The default embedder hashes words and character trigrams into a fixed-size vector. Any embedding function can replace it.

**Why.** It makes the store work with no downloads, no API key and no GPU, which matters for tests, CI and first use. It tolerates small spelling differences.

**Cost.** It matches spelling, not meaning. On LongMemEval it finds an answer-bearing session in the top ten for 93.6% of questions overall and for 73.3% of preference questions, which depend on meaning. Semantic recall needs a real embedding model.

## 13. Every parameter is a named setting

**Decision.** Numbers that shape ranking and indexing live in one `Tuning` structure, each documented with what it trades off, validated when the store is created, and overridable individually.

**Why.** Constants scattered through ranking code cannot be reasoned about or changed safely. Naming them makes the published measurements reproducible: the benchmark output records the settings used.

**Cost.** More surface to document and keep stable.

## Known limits

- **One process.** There is no server mode, replication or sharding. Writers share one lock.
- **Start-up replays the log.** No snapshots.
- **Durability.** Appends are not fsynced individually.
- **Retrieval only.** End-to-end answer accuracy with a reader model has not been measured.
- **English-centric defaults.** The stopword list is English; stemming supports other languages by setting.
- **Entities are not extracted.** Graph expansion needs them supplied.
