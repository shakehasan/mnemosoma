"""Three agents, three roles, one shared memory, wired together with LangGraph.

A planner, a researcher and a reviewer organise a team dinner for a user. No
model is called, so this runs offline and costs nothing; each node is plain
Python standing in for an agent. What it shows is the memory behaviour:

- the planner hands work to the researcher through the team blackboard;
- the researcher uses what the user said, and keeps private scratch notes;
- the reviewer cannot see those notes, and confirms the researcher's finding;
- nobody but a curator can publish to the whole organisation.

Recall here uses the built-in embedder, which matches words and spelling, not
meaning. Pass ``embedder=`` to ``MemoryStore`` for semantic recall.

Run it:

    pip install "mnemosoma[langchain]"
    python examples/team_graph.py
"""

from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from mnemosoma import Identity, MemoryStore, Scope

TEAM = Scope.team("dinner")


class State(TypedDict):
    request: str
    log: list[str]


def build(store: MemoryStore, ana: Identity):
    planner = ana.acting_as(agent="planner-1", role="planner")
    researcher = ana.acting_as(agent="researcher-1", role="researcher")
    reviewer = ana.acting_as(agent="reviewer-1", role="reviewer")

    def plan(state: State) -> dict:
        store.write(
            planner,
            f"Request: {state['request']}. Next: researcher picks a restaurant that suits everyone.",
            scope=TEAM,
            kind="handoff",
            ttl=3600,
        )
        return {"log": state["log"] + ["planner: wrote a handoff to the team blackboard"]}

    def research(state: State) -> dict:
        log = list(state["log"])
        handoff = store.recall(researcher, kinds=["handoff"], k=1)[0].memory
        log.append(f"researcher: picked up handoff from {handoff.author_agent}")

        diet = store.recall(researcher, "is Ana vegetarian", scopes=["user_private"], k=1)[0].memory
        log.append(f"researcher: recalled what the user said -> {diet.content!r}")

        store.write(
            researcher,
            "Shortlist so far: Verde Kitchen, Il Forno. Il Forno has only one vegetarian main.",
            scope=Scope.agent(researcher.agent),
            kind="working",
            ttl=3600,
        )
        finding = store.write(
            researcher,
            "Restaurant choice: Verde Kitchen. Fully vegetarian menu, seats twelve.",
            scope=TEAM,
            subject="dinner/restaurant",
            entities=["Verde Kitchen"],
            assertion="inferred",
            derived_from=[diet.id],
        )
        log.append(f"researcher: wrote finding {finding} to the team, marked as an inference")
        return {"log": log}

    def review(state: State) -> dict:
        log = list(state["log"])
        hit = store.recall(reviewer, "which restaurant", kinds=["semantic"], k=1)[0]
        log.append(f"reviewer: sees {hit.memory.content!r} (established={hit.established})")

        notes = store.recall(reviewer, "shortlist", scopes=["agent_private"])
        log.append(f"reviewer: researcher's private notes visible -> {len(notes)}")

        store.corroborate(reviewer, hit.memory.id)
        try:
            store.write(reviewer, "Verde Kitchen is the company's default venue.", scope=Scope.tenant())
        except PermissionError as denied:
            log.append(f"reviewer: tried to publish company-wide -> {denied}")
        return {"log": log}

    graph = StateGraph(State)
    graph.add_node("plan", plan)
    graph.add_node("research", research)
    graph.add_node("review", review)
    graph.add_edge(START, "plan")
    graph.add_edge("plan", "research")
    graph.add_edge("research", "review")
    graph.add_edge("review", END)
    return graph.compile()


def main() -> list[str]:
    store = MemoryStore(signing_key="demo-key-change-me")
    ana = Identity(tenant="acme", user="ana", agent="assistant", role="researcher", team="dinner")

    # Something the user said in an earlier conversation.
    store.write(ana, "Ana is vegetarian.", scope=Scope.user("ana"), subject="ana/diet", assertion="told_by_user")

    app = build(store, ana)
    result = app.invoke({"request": "team dinner for twelve on Friday", "log": []})
    log = result["log"]

    # The curator role can approve a finding outright and publish it.
    curator = ana.acting_as(agent="curator-1", role="curator")
    finding = store.recall(curator, "which restaurant", kinds=["semantic"], k=1)[0]
    store.corroborate(curator, finding.memory.id)
    approved = store.get(curator, finding.memory.id)
    log.append(f"curator: approved the finding -> corroborated_by={list(approved.corroborated_by)}")

    # A different user in the same company sees none of Ana's memory.
    bo = Identity(tenant="acme", user="bo", agent="assistant", role="researcher", team="offsite")
    log.append(f"bo's agent: memories visible about the dinner -> {len(store.recall(bo, 'vegetarian restaurant dinner'))}")
    return log


if __name__ == "__main__":
    for line in main():
        print(line)
