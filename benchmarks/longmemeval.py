"""LongMemEval retrieval benchmark for Mnemosoma.

LongMemEval (Wu et al., ICLR 2025) asks 500 questions, each about a long
chat history of timestamped sessions. Every question is labelled with the
sessions that contain its answer. This script measures retrieval: given the
question, does Mnemosoma return those sessions?

    python benchmarks/longmemeval.py --download     # fetch the data, then run
    python benchmarks/longmemeval.py                # run on cached data
    python benchmarks/longmemeval.py --json results.json

Protocol, so the numbers can be reproduced and compared:

- Data: ``longmemeval_s_cleaned.json`` from the authors' Hugging Face
  dataset ``xiaowu0162/longmemeval-cleaned`` (MIT licence). It is downloaded
  to a local cache and never copied into this repository.
- For each question a fresh store is filled with that question's history
  and queried once with the question text. Nothing is tuned per question.
- Two indexing granularities. ``session``: one memory per session, holding
  the whole conversation. ``round``: one memory per user message and the
  reply to it; a session's rank is the rank of its best round.
- Abstention questions (ids ending ``_abs``) have no evidence session and
  are left out of retrieval metrics, as in the paper.
- Metrics at k sessions: ``recall_any`` (at least one evidence session
  found), ``recall_all`` (every evidence session found) and ``ndcg``.
- No language model is used anywhere. With the default built-in embedder
  the result reflects keyword and spelling overlap, not meaning. Pass
  ``--embedder fastembed`` to add a small local embedding model
  (``BAAI/bge-small-en-v1.5``, MIT licence, runs on a CPU).

This measures retrieval only. End-to-end answer accuracy also needs a
reader model and a judge, and is not measured here.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import urllib.request
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from mnemosoma import Identity, MemoryStore, Scope

DATASET_URL = "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json"
DEFAULT_DATA = Path.home() / ".cache" / "mnemosoma" / "longmemeval" / "longmemeval_s_cleaned.json"
ABSTENTION_SUFFIX = "_abs"
CUTOFFS = (1, 3, 5, 10)
SESSIONS_WANTED = max(CUTOFFS)
ROUNDS_PER_QUERY = 100
"""Rounds fetched per question at round granularity, enough to cover ten distinct sessions."""
DOWNLOAD_CHUNK = 1 << 20
FASTEMBED_MODEL = "BAAI/bge-small-en-v1.5"

READER = Identity(tenant="benchmark", user="user", agent="assistant", role="researcher")
SCOPE = Scope.user(READER.user)

EMBED_BATCH = 64


class LocalEmbedder:
    """A local embedding model with a cache.

    Histories in LongMemEval share many sessions, so each distinct text is
    embedded once and reused across questions.
    """

    def __init__(self, model_name: str) -> None:
        try:
            from fastembed import TextEmbedding
        except ImportError as missing:
            raise SystemExit("--embedder fastembed needs `pip install fastembed`") from missing
        self.name = model_name
        self._model = TextEmbedding(model_name)
        self._cache: dict[str, list[float]] = {}

    def passages(self, texts: Sequence[str]) -> list[list[float]]:
        new = list(dict.fromkeys(text for text in texts if text not in self._cache))
        for text, vector in zip(new, self._model.embed(new, batch_size=EMBED_BATCH), strict=True):
            self._cache[text] = vector.tolist()
        return [self._cache[text] for text in texts]

    def query(self, text: str) -> list[float]:
        return next(iter(self._model.query_embed(text))).tolist()


def download(destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(".part")
    print(f"Downloading {DATASET_URL}", file=sys.stderr)
    with urllib.request.urlopen(DATASET_URL) as response, partial.open("wb") as out:
        total = int(response.headers.get("Content-Length", 0))
        done = 0
        while chunk := response.read(DOWNLOAD_CHUNK):
            out.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / total:6.1%} of {total / 1e6:.0f} MB", end="", file=sys.stderr)
    print(file=sys.stderr)
    partial.replace(destination)


def session_text(session: list[dict[str, Any]]) -> str:
    return "\n".join(f"{turn['role']}: {turn['content']}" for turn in session)


def rounds(session: list[dict[str, Any]]) -> list[str]:
    """Split a session into user message + reply pairs."""
    out: list[list[dict[str, Any]]] = []
    for turn in session:
        if turn["role"] == "user" or not out:
            out.append([turn])
        else:
            out[-1].append(turn)
    return [session_text(group) for group in out]


def ranked_sessions(item: dict[str, Any], granularity: str, embedder: LocalEmbedder | None) -> list[str]:
    """Session ids for one question, best first, as Mnemosoma retrieves them."""
    texts: list[str] = []
    owners: list[str] = []
    for session_id, session in zip(item["haystack_session_ids"], item["haystack_sessions"], strict=True):
        parts = [session_text(session)] if granularity == "session" else rounds(session)
        for text in parts:
            if text.strip():
                texts.append(text)
                owners.append(session_id)
    vectors: Sequence[list[float] | None] = embedder.passages(texts) if embedder else [None] * len(texts)

    store = MemoryStore("benchmark")
    session_of: dict[int, str] = {}
    for text, session_id, vector in zip(texts, owners, vectors, strict=True):
        session_of[store.write(READER, text, scope=SCOPE, kind="episodic", embedding=vector)] = session_id

    wanted = SESSIONS_WANTED if granularity == "session" else ROUNDS_PER_QUERY
    question_vector = embedder.query(item["question"]) if embedder else None
    ranked: list[str] = []
    for hit in store.recall(READER, item["question"], k=wanted, embedding=question_vector):
        session_id = session_of[hit.memory.id]
        if session_id not in ranked:
            ranked.append(session_id)
    return ranked[:SESSIONS_WANTED]


def score(ranked: list[str], evidence: set[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for k in CUTOFFS:
        top = ranked[:k]
        found = [session_id in evidence for session_id in top]
        out[f"recall_any@{k}"] = float(any(found))
        out[f"recall_all@{k}"] = float(evidence.issubset(top))
        gain = sum(1 / math.log2(rank + 2) for rank, hit in enumerate(found) if hit)
        ideal = sum(1 / math.log2(rank + 2) for rank in range(min(k, len(evidence))))
        out[f"ndcg@{k}"] = gain / ideal
    return out


def run(items: list[dict[str, Any]], granularity: str, embedder: LocalEmbedder | None) -> dict[str, Any]:
    totals: dict[str, float] = defaultdict(float)
    by_type: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    counts: dict[str, int] = defaultdict(int)
    started = time.perf_counter()
    for number, item in enumerate(items, 1):
        scores = score(ranked_sessions(item, granularity, embedder), set(item["answer_session_ids"]))
        counts[item["question_type"]] += 1
        for name, value in scores.items():
            totals[name] += value
            by_type[item["question_type"]][name] += value
        print(f"\r  {granularity}: {number}/{len(items)}", end="", file=sys.stderr)
    print(file=sys.stderr)
    return {
        "granularity": granularity,
        "questions": len(items),
        "seconds": round(time.perf_counter() - started, 1),
        "overall": {name: value / len(items) for name, value in totals.items()},
        "by_type": {
            kind: {"questions": counts[kind], **{name: value / counts[kind] for name, value in scores.items()}}
            for kind, scores in sorted(by_type.items())
        },
    }


def show(result: dict[str, Any]) -> None:
    print(f"\n{result['granularity']} granularity, {result['questions']} questions, {result['seconds']} s")
    header = "".join(f"{f'@{k}':>8}" for k in CUTOFFS)
    print(f"  {'':<28}{header}")
    for metric in ("recall_any", "recall_all", "ndcg"):
        row = "".join(f"{result['overall'][f'{metric}@{k}']:>8.1%}" for k in CUTOFFS)
        print(f"  {metric:<28}{row}")
    print("  recall_any by question type:")
    for kind, scores in result["by_type"].items():
        row = "".join(f"{scores[f'recall_any@{k}']:>8.1%}" for k in CUTOFFS)
        print(f"    {kind + ' (' + str(scores['questions']) + ')':<26}{row}")


def main() -> int:
    parser = argparse.ArgumentParser(description="LongMemEval retrieval benchmark for Mnemosoma.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="path to longmemeval_s_cleaned.json")
    parser.add_argument("--download", action="store_true", help="download the dataset if it is not cached")
    parser.add_argument("--granularity", choices=["session", "round", "both"], default="both")
    parser.add_argument("--embedder", choices=["builtin", "fastembed"], default="builtin")
    parser.add_argument("--limit", type=int, help="only the first N questions, for a quick check")
    parser.add_argument("--json", type=Path, help="also write the results to this file")
    args = parser.parse_args()

    if not args.data.exists():
        if not args.download:
            print(f"{args.data} not found. Run with --download to fetch it (277 MB).", file=sys.stderr)
            return 2
        download(args.data)

    with args.data.open(encoding="utf-8") as source:
        data = json.load(source)
    items = [item for item in data if not item["question_id"].endswith(ABSTENTION_SUFFIX)]
    skipped = len(data) - len(items)
    if args.limit:
        items = items[: args.limit]
    print(f"{len(items)} questions ({skipped} abstention questions left out), embedder: {args.embedder}")

    embedder = LocalEmbedder(FASTEMBED_MODEL) if args.embedder == "fastembed" else None
    granularities = ["session", "round"] if args.granularity == "both" else [args.granularity]
    results = [run(items, granularity, embedder) for granularity in granularities]
    for result in results:
        show(result)
    if args.json:
        payload = {
            "dataset": DATASET_URL,
            "embedder": args.embedder if embedder is None else FASTEMBED_MODEL,
            "tuning": MemoryStore.default_tuning(),
            "results": results,
        }
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
