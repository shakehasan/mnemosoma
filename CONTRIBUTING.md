# Contributing

Thanks for looking at Mnemosoma. This page covers how to build it, what CI checks, and the few conventions the code follows.

## Set up

You need Rust (stable) and Python 3.10 or newer.

```bash
python -m venv .venv
```

```bash
pip install maturin pytest ruff mypy "langchain-core>=1.0" "langgraph>=1.0" "langsmith>=0.4" "mcp>=1.2"
```

```bash
maturin develop --release
```

Activate the virtual environment before the `pip` and `maturin` steps. On Windows with the GNU Rust toolchain, an LLVM `dlltool` must be on `PATH`; see the README.

## Before you open a pull request

CI runs all of these and fails on any of them:

```bash
cargo fmt --all -- --check
```

```bash
cargo clippy --workspace --all-targets -- -D warnings
```

```bash
cargo test -p mnemosoma-core
```

```bash
ruff format --check python tests evals examples benchmarks
```

```bash
ruff check python tests evals examples benchmarks
```

```bash
mypy
```

```bash
pytest
```

```bash
python evals/run.py --check
```

## Conventions

- **No bare numbers in ranking or indexing code.** A number that shapes behaviour is a field of `Tuning` in `crates/mnemosoma-core/src/config.rs`, with a comment saying what it trades off and a range check in `validate`. Python-side defaults go in `python/mnemosoma/limits.py`.
- **Permissions before ranking.** Anything that reads memories for a caller goes through `eligible` in the store. A ranker must never see a row that is not in the caller's view. A change to recall needs a test showing that a memory outside the caller's scopes cannot be returned through it.
- **Tests are named for the failure they guard.** `revoking_a_grant_hides_what_was_derived_from_it`, not `test_revoke_2`.
- **Numbers in the README are measured.** If you change one, say how you measured it in the pull request. The load benchmark and `benchmarks/longmemeval.py` print what they used.
- **Behaviour changes update the evals.** If `python evals/run.py` reports a different number after your change, explain why in the pull request and update the README table.
- **The log format is stable.** New fields need a serde default so older logs still load.

## Where things are

- `crates/mnemosoma-core/src/store.rs`: the write path, the read path and maintenance.
- `crates/mnemosoma-core/src/config.rs`: every tunable parameter.
- `crates/mnemosoma-core/src/ann.rs`: the approximate vector index.
- `python/mnemosoma/`: the SDK and integrations.
- `docs/DESIGN.md`: why it is built this way.

## Reporting security problems

See [SECURITY.md](SECURITY.md). Please do not open a public issue for a vulnerability.
