# Security

Mnemosoma stores what agents remember about people, so its security properties are part of what it is for. This page says what it is designed to guarantee, what it does not cover, and how to report a problem.

## Reporting a vulnerability

Please report privately, not in a public issue. On GitHub, open the repository's **Security** tab and choose **Report a vulnerability**. Include what you did, what you expected, and what happened; a failing test is ideal.

The cases treated as most serious are any way for a caller to read, rank, or infer the content of a memory outside its scopes, and any way to make a memory appear more trusted than its sources.

## What the engine is designed to guarantee

- **Isolation.** A recall never returns a memory the caller's identity may not read: not from another tenant, another user, another agent's private scope or another team. Filters run before ranking, and keyword statistics are computed only over the caller's eligible memories.
- **Revocation.** Revoking a grant hides the shared memories and anything derived from them.
- **Tamper evidence.** Each memory is signed with HMAC-SHA256 when written. A log entry edited on disk fails verification on load and is quarantined.
- **Trust inheritance.** A derived memory is never more trusted than its least trusted source, and an inference cannot claim trusted status.
- **Erasure.** Forgetting a memory removes it, everything derived from it, their index entries, and their records in the log file.

These are covered by tests in `crates/mnemosoma-core/tests/engine.rs` and by the isolation, sharing and poisoning suites in `evals/run.py`.

## What it relies on you for

- **Identity.** The engine trusts the `Identity` it is given. Build it from your own authentication. Never take it from model output or from an untrusted request.
- **The signing key.** Keep it out of the directory that holds the log. Anyone with the key and write access to the file can forge entries.
- **Labelling outside content.** Write web pages, tool output and uploads with `trust="untrusted"`. The injection guard is pattern based and catches only some unlabelled content.
- **Transport and storage.** The log file is not encrypted. Use disk encryption and file permissions suited to the data.

## Out of scope

- Denial of service by a caller who is allowed to write.
- Side channels through timing.
- What a model does with memories it was correctly allowed to read.
