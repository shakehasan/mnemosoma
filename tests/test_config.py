from datetime import datetime, timezone

import pytest
from mnemosoma import Identity, MemoryStore, Scope, limits

T0 = datetime(2026, 11, 1, tzinfo=timezone.utc)
ANA = Identity(tenant="acme", user="ana", agent="r1", role="researcher", team="task-1")
BO = Identity(tenant="acme", user="bo", agent="r2", role="researcher", team="task-1")
TEAM = Scope.team("task-1")


def test_default_tuning_lists_every_parameter():
    tuning = MemoryStore.default_tuning()
    assert tuning["rrf_k"] == 60.0
    assert tuning["hnsw"] == {"m": 16, "ef_construction": 100}
    for name in ("bm25_k1", "bm25_b", "graph_carry", "ann_min_rows", "chars_per_token", "weight_unconfirmed"):
        assert name in tuning


def test_partial_tuning_keeps_other_defaults_and_changes_behaviour():
    # With no graph carry, entity links pass on nothing.
    quiet = MemoryStore("k", tuning={"graph_carry": 0.0})
    default = MemoryStore("k")
    for store in (quiet, default):
        store.write(ANA, "The venue contract was signed by Dana", scope=TEAM, entities=["Dana"], now=T0)
        store.write(ANA, "Her direct line is extension 4417", scope=TEAM, entities=["Dana"], now=T0)

    def graph_rank(store):
        hits = store.recall(ANA, "who signed the venue contract", now=T0)
        return next((h.why["graph_rank"] for h in hits if h.memory.content.startswith("Her direct")), "not found")

    assert graph_rank(default) == 1
    # It shares no words with the question, so without the graph it is not found.
    assert graph_rank(quiet) == "not found"


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"tuning": {"bm25_b": 1.5}}, "bm25_b"),
        ({"tuning": {"rrf_k": 0}}, "rrf_k"),
        ({"tuning": {"no_such_setting": 1}}, "unknown field"),
        ({"tuning": {"hnsw": {"m": 0}}}, "hnsw.m"),
        ({"half_life_days": 0}, "half_life_ms"),
        ({"archive_below": 2.0}, "archive_below"),
        ({"required_corroborations": 0}, "required_corroborations"),
    ],
)
def test_invalid_settings_are_refused_with_the_setting_named(options, message):
    with pytest.raises(ValueError, match=message):
        MemoryStore("k", **options)


def test_empty_signing_key_is_refused():
    with pytest.raises(ValueError, match="signing_key"):
        MemoryStore("")


def test_required_corroborations_is_configurable():
    strict = MemoryStore("k", required_corroborations=1)
    guess = strict.write(ANA, "The client probably prefers email", scope=TEAM, assertion="inferred", now=T0)
    assert not strict.recall(ANA, "client prefers", now=T0)[0].established
    strict.corroborate(BO, guess, now=T0)
    assert strict.recall(ANA, "client prefers", now=T0)[0].established


def test_chars_per_token_drives_the_budget():
    text = "x" * 40
    roomy = MemoryStore("k", tuning={"chars_per_token": 40})
    tight = MemoryStore("k", tuning={"chars_per_token": 1})
    for store in (roomy, tight):
        store.write(ANA, f"note {text}", scope=TEAM, now=T0)
    assert len(roomy.recall(ANA, "note", token_budget=5, now=T0)) == 1
    assert tight.recall(ANA, "note", token_budget=5, now=T0) == []


def test_limits_are_named():
    assert limits.DEFAULT_RECALL_K > 0
    assert limits.RERANK_MIN_POOL >= limits.DEFAULT_RECALL_K
