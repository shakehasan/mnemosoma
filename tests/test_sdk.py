from datetime import datetime, timedelta, timezone

import pytest

from mnemosoma import Consolidator, Fact, Identity, MemoryStore, Scope

T0 = datetime(2026, 11, 1, tzinfo=timezone.utc)
ANA = Identity(tenant="acme", user="ana", agent="r1", role="researcher", team="task-1")
BO = Identity(tenant="acme", user="bo", agent="r2", role="researcher", team="task-1")


@pytest.fixture
def store():
    return MemoryStore("test-key")


def texts(hits):
    return [h.memory.content for h in hits]


def test_write_and_recall_round_trip(store):
    memory_id = store.write(ANA, "Ana prefers morning meetings", scope=Scope.user("ana"), now=T0)
    (hit,) = store.recall(ANA, "when does ana like meetings", now=T0)
    assert hit.memory.id == memory_id
    assert hit.memory.scope == Scope.user("ana")
    assert hit.memory.author_agent == "r1"
    assert hit.memory.recorded_at == T0
    assert hit.established


def test_other_users_and_tenants_see_nothing(store):
    store.write(ANA, "Ana's salary review is in May", scope=Scope.user("ana"), now=T0)
    assert store.recall(BO, "salary review", now=T0) == []
    outsider = Identity(tenant="globex", user="ana", agent="r1", role="researcher", team="task-1")
    assert store.recall(outsider, "salary review", now=T0) == []


def test_denied_write_raises_permission_error(store):
    with pytest.raises(PermissionError):
        store.write(ANA, "company policy", scope=Scope.tenant(), now=T0)
    with pytest.raises(PermissionError):
        store.write(ANA, "planted", scope=Scope.user("bo"), now=T0)


def test_point_in_time_recall_and_history(store):
    scope = Scope.user("ana")
    store.write(ANA, "Ana works at Initech", scope=scope, subject="ana/employer", now=T0)
    store.write(ANA, "Ana works at Hooli", scope=scope, subject="ana/employer", now=T0 + timedelta(days=10))
    later = T0 + timedelta(days=20)

    assert texts(store.recall(ANA, "where does ana work", now=later)) == ["Ana works at Hooli"]
    past = store.recall(ANA, "where does ana work", as_of=T0 + timedelta(days=5), now=later)
    assert texts(past) == ["Ana works at Initech"]

    old, new = store.history(ANA, scope, "ana/employer", now=later)
    assert old.valid_to == T0 + timedelta(days=10)
    assert old.superseded_by == new.id


def test_grant_then_revoke(store):
    store.write(ANA, "Falcon ships in March", scope=Scope.user("ana"), tags=["falcon"], now=T0)
    assert store.recall(BO, "falcon", now=T0) == []
    grant = store.grant(ANA, "bo", tag="falcon")
    assert texts(store.recall(BO, "falcon", now=T0)) == ["Falcon ships in March"]
    store.revoke(ANA, grant, now=T0)
    assert store.recall(BO, "falcon", now=T0) == []


def test_ttl_forget_and_stats(store):
    store.write(ANA, "scratch", scope=Scope.agent("r1"), kind="working", ttl=timedelta(minutes=5), now=T0)
    keep = store.write(ANA, "Ana's passport is X123", scope=Scope.user("ana"), now=T0)
    assert store.sweep(now=T0 + timedelta(minutes=6)) == {"expired": 1, "archived": 0}
    assert store.forget(ANA, keep, now=T0) == 1
    with pytest.raises(KeyError):
        store.get(ANA, keep, now=T0)
    assert store.stats()["memories"] == 0


def test_persists_across_reopen(tmp_path):
    path = tmp_path / "memory.log"
    first = MemoryStore("k", path)
    first.write(ANA, "the door code is 7788", scope=Scope.user("ana"), now=T0)
    del first
    second = MemoryStore("k", path)
    assert texts(second.recall(ANA, "door code", now=T0)) == ["the door code is 7788"]
    assert MemoryStore("wrong-key", path).stats() == {"memories": 0, "tenants": 0, "grants": 0, "quarantined": 1}


def test_custom_embedder_is_used_for_both_sides():
    vectors = {"alpha": [1.0, 0.0], "beta": [0.0, 1.0], "second thing": [0.1, 0.9]}
    store = MemoryStore("k", embedder=lambda text: vectors[text])
    store.write(ANA, "alpha", scope=Scope.team("task-1"), now=T0)
    store.write(ANA, "beta", scope=Scope.team("task-1"), now=T0)
    assert texts(store.recall(ANA, "second thing", now=T0))[0] == "beta"


def test_naive_datetimes_are_rejected(store):
    with pytest.raises(ValueError):
        store.write(ANA, "x", scope=Scope.agent("r1"), now=datetime(2026, 1, 1))


def test_consolidation_links_facts_to_their_episodes(store):
    team = Scope.team("task-1")
    e1 = store.write(ANA, "Call with Dana: she confirmed the venue holds 80 people", scope=team, kind="episodic", now=T0)
    store.write(ANA, "Small talk about the weather", scope=team, kind="episodic", now=T0)

    def extractor(episodes):
        assert [m.kind for m in episodes] == ["episodic", "episodic"]
        return [Fact("The venue holds 80 people", subject="venue/capacity", entities=("Dana",), source_ids=(e1,))]

    report = Consolidator(store, extractor).run(ANA, team, now=T0)
    assert report.episodes == 2 and len(report.written) == 1

    fact = store.get(ANA, report.written[0], now=T0)
    assert fact.assertion == "inferred"
    assert fact.derived_from == (e1,)
    assert not store.recall(ANA, "venue capacity", kinds=["semantic"], now=T0)[0].established

    # Nothing new: a second run finds no episodes.
    assert Consolidator(store, extractor).run(ANA, team, now=T0).unchanged == 1

    # A second agent reaching the same fact corroborates it instead of duplicating it.
    report = Consolidator(store, extractor).run(BO, team, now=T0)
    assert report.corroborated == [fact.id] and report.written == []

    # Erasing the episode erases the fact built on it.
    assert store.forget(ANA, e1, now=T0) == 2
