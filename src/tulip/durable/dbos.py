# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""Run an agent or a graph on DBOS: durable waits, recovered from a database.

DBOS keeps workflow state in Postgres (SQLite for development), inside your own
process: there is no separate server. A run is a DBOS workflow; the agent's work
runs in steps, in segments, the first run until the agent finishes or pauses on a
held call, then a resume after each decision. Between segments the workflow
waits for a decision message for as long as it takes, and a process that
restarts meanwhile picks the wait up again::

    DBOS(config={"name": "agents", "system_database_url": "postgresql://..."})
    register_agents({"refunds": make_agent})
    DBOS.launch()

    handle = await start_agent_run(
        agent="refunds", prompt="refund order 4821", thread_id="t1"
    )
    # ... a person approves in the ApprovalStore, then:
    await signal_decided(handle.workflow_id)
    outcome = await handle.get_result()

Register the agents before ``DBOS.launch()`` in every process, since launching
recovers unfinished runs. The agent must have a checkpointer every process
reaches: the conversation lives there between segments, and approval decisions
live in the approval store, exactly as without DBOS.

Each segment runs at most once. It records that it began before it starts; if
the process dies inside it, DBOS recovers the run but the segment is not run
again, because its tool calls may already have happened. The run fails with
:class:`InterruptedSegmentError` and the checkpoint shows where it stopped. Needs
``pip install "tulip-agents[dbos]"``.

**Graphs.** A :class:`~tulip.multiagent.graph.StateGraph` runs the same way. A
node that calls :func:`~tulip.core.interrupt.interrupt` pauses the run, and the
workflow waits, durably, for the value to resume it with: a person's decision, a
job's result from another machine, anything::

    # make_graph() returns a graph with a checkpointer
    register_graphs({"forge": make_graph})
    DBOS.launch()

    handle = await start_graph_run(
        graph="forge", inputs={"request": 12}, thread_id="t-12"
    )
    # .interrupt is the payload the pausing node passed to interrupt()
    paused = await pending_interrupt(handle.workflow_id)
    await resume_graph(handle.workflow_id, {"approved": True})
    outcome = await handle.get_result()  # .status "done", .state

The graph keeps its state in its checkpointer between segments, and each
segment runs with its own copy of the graph's config, so one registered graph
serves any number of runs at once. A segment runs at most once, as an agent's
does: a node interrupted by a dying process is not run a second time.

**A run nothing runs.** DBOS recovers, at launch, the runs left pending under
its executor id by putting them back on its internal queue, and any live
process of the application may take them from there. Two processes with the
same executor id at once (the default, ``local``, in every process: a rolling
deploy's old and new pods) can leave a paused run held by the one that then
exits: the run stays pending, its resume or decision is kept unread, and
nothing wakes it until the next launch. :func:`redispatch_stalled_runs` finds
the runs paused at their wait with a message unread for ``older_than`` seconds
and dispatches each again (DBOS's ``resume_workflow``): taken by a live
process, the run replays its recorded steps to the wait, reads the message and
goes on. Call it now and then (every half minute), or
:func:`redispatch_run` a few seconds after sending one run its value. A run
inside a segment has no pause on record and is never dispatched again, since a
segment runs at most once::

    await resume_graph(workflow_id, {"approved": True})
    await asyncio.sleep(5)
    await redispatch_run(workflow_id)  # False: the run took it

**Where the state lives.** The local default needs nothing but the machine:
DBOS on SQLite and a file checkpointer. In production both live in Postgres.
Neither DBOS's tables nor a checkpointer's carry a tenant column, so a
multi-tenant caller keeps its tenants' data in its own tables under row-level
security, puts only ids in graph state and interrupt payloads, and keys thread
and workflow ids per tenant.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any

from dbos import DBOS, SetWorkflowID

from tulip.durable.segments import (
    GraphSegmentOutcome,
    SegmentError,
    SegmentOutcome,
    run_agent_segment,
    run_graph,
)


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from dbos import WorkflowHandleAsync

    from tulip.agent import Agent
    from tulip.multiagent.graph import StateGraph


#: The topic decisions are sent on.
DECISION_TOPIC = "tulip.decision"

#: The workflow event holding the pause a run is waiting on.
PENDING_EVENT = "tulip.pending"

#: The topic a paused graph run's resume value is sent on.
GRAPH_RESUME_TOPIC = "tulip.graph.resume"

#: The workflow event holding the pause a graph run is waiting on.
GRAPH_PENDING_EVENT = "tulip.graph.pending"

_DECISION_WAIT_SECONDS = 86_400.0

_AGENTS: dict[str, Callable[[], Agent]] = {}
_GRAPHS: dict[str, Callable[[], StateGraph]] = {}


class InterruptedSegmentError(SegmentError):
    """A segment began in a process that died, so it is not run a second time."""


def register_agents(agents: Mapping[str, Callable[[], Agent]]) -> None:
    """Make agents available to runs in this process, by name."""
    _AGENTS.update(agents)


@DBOS.step(name="tulip_run_segment")
async def run_segment(
    agent: str,
    thread_id: str,
    kind: str,
    index: int,
    prompt: str = "",
    answer: str = "",
    perform: bool = True,
) -> SegmentOutcome:
    """Run or resume a registered agent until it finishes or pauses, at most once."""
    workflow_id = DBOS.workflow_id
    if workflow_id is not None:
        began = f"tulip.segment.{index}.began"
        if await DBOS.get_event_async(workflow_id, began, timeout_seconds=0):
            raise InterruptedSegmentError(
                f"segment {index} of run {workflow_id} was interrupted and may already "
                "have called tools, so it is not run again; the checkpoint for thread "
                f"{thread_id!r} shows where it stopped"
            )
        await DBOS.set_event_async(began, value=True)
    return await run_agent_segment(
        _AGENTS,
        agent,
        thread_id=thread_id,
        kind=kind,
        prompt=prompt,
        answer=answer,
        perform=perform,
    )


@DBOS.workflow(name="TulipAgentRun")
async def agent_workflow(agent: str, prompt: str, thread_id: str) -> SegmentOutcome:
    """Run segments until the agent is done, waiting for a decision at each pause."""
    outcome = await run_segment(agent, thread_id, "run", 0, prompt=prompt)
    index = 0
    while outcome.status == "paused":
        await DBOS.set_event_async(PENDING_EVENT, outcome)
        decision: Any = None
        while decision is None:
            decision = await DBOS.recv_async(DECISION_TOPIC, timeout_seconds=_DECISION_WAIT_SECONDS)
        await DBOS.set_event_async(PENDING_EVENT, None)
        index += 1
        outcome = await run_segment(
            agent,
            thread_id,
            "resume",
            index,
            answer=str(decision.get("answer", "approved")),
            perform=bool(decision.get("perform", True)),
        )
    return outcome


async def start_agent_run(
    *,
    agent: str,
    prompt: str,
    thread_id: str,
    workflow_id: str | None = None,
) -> WorkflowHandleAsync[SegmentOutcome]:
    """Start a durable run of a registered agent; the workflow id defaults to the thread."""
    with SetWorkflowID(workflow_id or f"tulip-{thread_id}"):
        return await DBOS.start_workflow_async(agent_workflow, agent, prompt, thread_id)


async def pending_decision(workflow_id: str) -> SegmentOutcome | None:
    """The pause a run is waiting on, or ``None``."""
    value = await DBOS.get_event_async(workflow_id, PENDING_EVENT, timeout_seconds=0)
    return value if isinstance(value, SegmentOutcome) else None


async def signal_decided(
    workflow_id: str, *, answer: str = "approved", perform: bool = True
) -> None:
    """Tell a paused run that its approval has been decided, so it resumes."""
    await DBOS.send_async(workflow_id, {"answer": answer, "perform": perform}, topic=DECISION_TOPIC)


# --- graphs ------------------------------------------------------------------


def register_graphs(graphs: Mapping[str, Callable[[], StateGraph]]) -> None:
    """Make graphs available to runs in this process, by name.

    Each factory returns a graph whose config has a checkpointer every process
    reaches. Register before ``DBOS.launch()`` in every process, since launching
    recovers unfinished runs.
    """
    _GRAPHS.update(graphs)


@DBOS.step(name="tulip_run_graph_segment")
async def run_graph_segment(
    graph: str,
    thread_id: str,
    index: int,
    inputs: dict[str, Any] | None = None,
    resume: Any = None,
    resuming: bool = False,
) -> GraphSegmentOutcome:
    """Run a registered graph, or resume it with ``resume``, until it ends or pauses, at most once."""
    workflow_id = DBOS.workflow_id
    if workflow_id is not None:
        began = f"tulip.graph.segment.{index}.began"
        if await DBOS.get_event_async(workflow_id, began, timeout_seconds=0):
            raise InterruptedSegmentError(
                f"segment {index} of graph run {workflow_id} was interrupted and its nodes "
                "may already have had their effects, so it is not run again; the checkpoint "
                f"for thread {thread_id!r} shows where it stopped"
            )
        await DBOS.set_event_async(began, value=True)
    return await run_graph(
        _GRAPHS, graph, thread_id=thread_id, inputs=inputs, resume=resume, resuming=resuming
    )


@DBOS.workflow(name="TulipGraphRun")
async def graph_workflow(graph: str, inputs: dict[str, Any], thread_id: str) -> GraphSegmentOutcome:
    """Run segments until the graph ends, waiting for a resume value at each pause."""
    outcome = await run_graph_segment(graph, thread_id, 0, inputs=inputs)
    index = 0
    while outcome.status == "paused":
        await DBOS.set_event_async(GRAPH_PENDING_EVENT, outcome)
        envelope: Any = None
        while envelope is None:
            envelope = await DBOS.recv_async(
                GRAPH_RESUME_TOPIC, timeout_seconds=_DECISION_WAIT_SECONDS
            )
        await DBOS.set_event_async(GRAPH_PENDING_EVENT, None)
        index += 1
        outcome = await run_graph_segment(
            graph, thread_id, index, resume=envelope.get("value"), resuming=True
        )
    return outcome


async def start_graph_run(
    *,
    graph: str,
    inputs: dict[str, Any],
    thread_id: str,
    workflow_id: str | None = None,
) -> WorkflowHandleAsync[GraphSegmentOutcome]:
    """Start a durable run of a registered graph; the workflow id defaults to the thread's."""
    with SetWorkflowID(workflow_id or f"tulip-graph-{thread_id}"):
        return await DBOS.start_workflow_async(graph_workflow, graph, dict(inputs), thread_id)


async def pending_interrupt(workflow_id: str) -> GraphSegmentOutcome | None:
    """The pause a graph run is waiting on (``.interrupt`` is the payload), or ``None``."""
    value = await DBOS.get_event_async(workflow_id, GRAPH_PENDING_EVENT, timeout_seconds=0)
    return value if isinstance(value, GraphSegmentOutcome) else None


async def resume_graph(workflow_id: str, value: Any) -> None:
    """Resume a paused graph run: the pausing node's ``interrupt()`` returns ``value``.

    The value travels wrapped, so ``None`` is a value like any other. A resume
    sent while the run is not yet paused is kept, and taken at its next pause.
    """
    await DBOS.send_async(workflow_id, {"value": value}, topic=GRAPH_RESUME_TOPIC)


# --- runs nothing runs ----------------------------------------------------------

#: Each kind of run, by DBOS's workflow name: the topic it waits on, the event its pause is in.
_WAITS = {
    "TulipGraphRun": (GRAPH_RESUME_TOPIC, GRAPH_PENDING_EVENT),
    "TulipAgentRun": (DECISION_TOPIC, PENDING_EVENT),
}


def _unread(older_than: float, workflow_id: str | None = None) -> list[tuple[str, str]]:
    """Pending runs with a message on their topic unread for ``older_than`` seconds."""
    import sqlalchemy as sa
    from dbos._dbos import _get_dbos_instance
    from dbos._schemas.system_database import SystemSchema

    ws, n = SystemSchema.workflow_status, SystemSchema.notifications
    cutoff = int((time.time() - older_than) * 1000)
    query = (
        sa.select(ws.c.workflow_uuid, ws.c.name)
        .distinct()
        .select_from(ws.join(n, n.c.destination_uuid == ws.c.workflow_uuid))
        .where(
            ws.c.status == "PENDING",
            n.c.consumed == False,  # noqa: E712 - SQL, not Python
            n.c.created_at_epoch_ms <= cutoff,
            sa.or_(
                *(
                    sa.and_(ws.c.name == name, n.c.topic == topic)
                    for name, (topic, _) in _WAITS.items()
                )
            ),
        )
    )
    if workflow_id is not None:
        query = query.where(ws.c.workflow_uuid == workflow_id)
    # DBOS has no public read of its messages; its system tables are the record.
    with _get_dbos_instance()._sys_db.engine.begin() as c:  # noqa: SLF001
        return [(str(row[0]), str(row[1])) for row in c.execute(query)]


async def _find(older_than: float, workflow_id: str | None = None) -> list[tuple[str, str]]:
    # On DBOS's own pool, as its async calls run theirs: a relaunch shuts the old one down.
    from dbos._dbos import _get_dbos_instance

    pool = _get_dbos_instance()._executor  # noqa: SLF001
    return await asyncio.get_running_loop().run_in_executor(pool, _unread, older_than, workflow_id)


async def _paused(workflow_id: str, name: str) -> bool:
    event = _WAITS[name][1]
    return await DBOS.get_event_async(workflow_id, event, timeout_seconds=0) is not None


async def stalled_runs(*, older_than: float = 30.0) -> list[str]:
    """Runs paused at their wait with a resume or decision unread for ``older_than`` seconds."""
    return [w for w, name in await _find(older_than) if await _paused(w, name)]


async def redispatch_run(workflow_id: str, *, older_than: float = 0.0) -> bool:
    """Dispatch a paused run again when a message to it has lain unread ``older_than`` seconds.

    Returns whether it was. A run that took its message, is inside a segment, or
    has ended is left alone.
    """
    found = await _find(older_than, workflow_id)
    if not found or not await _paused(workflow_id, found[0][1]):
        return False
    DBOS.logger.warning(
        f"tulip: run {workflow_id} is paused with a message nobody has read; dispatching it again"
    )
    await DBOS.resume_workflow_async(workflow_id)
    return True


async def redispatch_stalled_runs(*, older_than: float = 30.0) -> list[str]:
    """Every run :func:`stalled_runs` finds, dispatched again; their workflow ids."""
    return [
        w
        for w in await stalled_runs(older_than=older_than)
        if await redispatch_run(w, older_than=older_than)
    ]


__all__ = [
    "DECISION_TOPIC",
    "GRAPH_PENDING_EVENT",
    "GRAPH_RESUME_TOPIC",
    "PENDING_EVENT",
    "GraphSegmentOutcome",
    "InterruptedSegmentError",
    "agent_workflow",
    "graph_workflow",
    "pending_decision",
    "pending_interrupt",
    "redispatch_run",
    "redispatch_stalled_runs",
    "register_agents",
    "register_graphs",
    "resume_graph",
    "run_graph_segment",
    "run_segment",
    "signal_decided",
    "stalled_runs",
    "start_agent_run",
    "start_graph_run",
]
