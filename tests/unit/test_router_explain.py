# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``router.explain()`` — the decision record.

Two properties matter and neither is "it returns something". First, the
explanation must agree with what the compiler actually does: an explanation
that drifts from the real selection is worse than none, because it will be
believed. Second, it must cost nothing — no model call, no execution — or
developers will not reach for it.

Everything here is deterministic: no model is contacted at any point.
"""

from __future__ import annotations

import pytest

from tulip.router import (
    CapabilityIndex,
    CognitiveCompiler,
    GoalFrame,
    PolicyGate,
    ProtocolRegistry,
    TaskType,
    builtin_protocols,
)
from tulip.router.explain import RankedProtocol, RejectedProtocol, RoutingExplanation
from tulip.router.goal_frame import Complexity, Risk
from tulip.tools.registry import create_registry


def _compiler(**kwargs) -> CognitiveCompiler:
    registry = ProtocolRegistry()
    registry.register_many(builtin_protocols())
    return CognitiveCompiler(
        protocols=registry,
        capabilities=CapabilityIndex(create_registry()),
        policy=kwargs.pop("policy", PolicyGate()),
        model="openai:gpt-4o-mini",
        **kwargs,
    )


def _frame(
    goal: TaskType = TaskType.ANSWER,
    complexity: Complexity = Complexity.LOW,
    risk: Risk = Risk.LOW,
) -> GoalFrame:
    return GoalFrame(
        primary_goal=goal,
        domain="ops",
        complexity=complexity,
        risk=risk,
        success_criteria=["done"],
    )


# ------------------------------------------------------- agrees with compile --


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("goal", "complexity", "risk"),
    [
        (TaskType.ANSWER, Complexity.LOW, Risk.LOW),
        (TaskType.DIAGNOSE, Complexity.HIGH, Risk.MEDIUM),
        (TaskType.COMPARE, Complexity.HIGH, Risk.LOW),
        (TaskType.GENERATE_CODE, Complexity.MEDIUM, Risk.LOW),
        (TaskType.REMEDIATE, Complexity.MEDIUM, Risk.HIGH),
        (TaskType.COORDINATE, Complexity.MEDIUM, Risk.LOW),
        (TaskType.PLAN, Complexity.MEDIUM, Risk.MEDIUM),
    ],
)
async def test_explain_names_the_protocol_compile_actually_builds(
    goal: TaskType, complexity: Complexity, risk: Risk
) -> None:
    """The whole value proposition: the explanation must not drift from reality."""
    compiler = _compiler()
    frame = _frame(goal, complexity, risk)

    explanation = compiler.explain(frame)
    runnable = await compiler.compile(frame)

    built = getattr(runnable, "protocol_id", None) or getattr(
        getattr(runnable, "inner", None), "protocol_id", None
    )
    assert explanation.selected_protocol == built


# ------------------------------------------------------------- the evidence --


def test_rejected_names_the_gate_that_stopped_each_protocol() -> None:
    compiler = _compiler()
    # HIGH risk remediation: one survivor, everything else gated.
    explanation = compiler.explain(_frame(TaskType.REMEDIATE, Complexity.MEDIUM, Risk.HIGH))

    by_id = {r.protocol_id: r for r in explanation.rejected}
    assert by_id["debate"].gate == "goal"
    assert "does not handle" in by_id["debate"].reason
    # plan_execute_validate handles REMEDIATE but caps risk at medium
    assert by_id["plan_execute_validate"].gate == "risk"
    assert "caps risk" in by_id["plan_execute_validate"].reason


def test_every_protocol_is_either_a_candidate_or_rejected() -> None:
    """No protocol may silently vanish — that would hide a routing bug."""
    compiler = _compiler()
    explanation = compiler.explain(_frame(TaskType.DIAGNOSE, Complexity.HIGH, Risk.MEDIUM))

    accounted = {c.protocol_id for c in explanation.candidates} | {
        r.protocol_id for r in explanation.rejected
    }
    assert accounted == {p.id for p in compiler.protocols.all()}


def test_candidates_are_ordered_best_first_and_mark_the_winner() -> None:
    compiler = _compiler()
    explanation = compiler.explain(_frame(TaskType.RESEARCH, Complexity.HIGH, Risk.LOW))

    ranks = [c.rank for c in explanation.candidates]
    assert ranks == sorted(ranks), "candidates must be ordered by rank, best first"
    assert explanation.candidates[0].selected is True
    assert explanation.candidates[0].protocol_id == explanation.selected_protocol
    assert sum(1 for c in explanation.candidates if c.selected) == 1


def test_a2a_is_rejected_on_capabilities_when_no_endpoint_is_configured() -> None:
    """The synthetic peer capability is the opt-in; without it, say so."""
    compiler = _compiler()
    explanation = compiler.explain(_frame(TaskType.COORDINATE, Complexity.MEDIUM, Risk.LOW))

    a2a = next(r for r in explanation.rejected if r.protocol_id == "a2a_delegate")
    assert a2a.gate == "capabilities"
    assert "not available" in a2a.reason


def test_configuring_an_endpoint_moves_a2a_into_the_candidates() -> None:
    compiler = _compiler(a2a_endpoint="http://peer.invalid/a2a")
    explanation = compiler.explain(_frame(TaskType.COORDINATE, Complexity.MEDIUM, Risk.LOW))

    assert "a2a_delegate" in {c.protocol_id for c in explanation.candidates}


# ---------------------------------------------------------------- the policy --


def test_high_risk_reports_that_a_human_must_approve() -> None:
    compiler = _compiler()
    explanation = compiler.explain(_frame(TaskType.REMEDIATE, Complexity.MEDIUM, Risk.HIGH))

    assert explanation.requires_approval is True
    assert explanation.allowed is True  # allowed, but held
    assert explanation.policy is not None
    assert explanation.policy.reason


def test_low_risk_needs_no_approval() -> None:
    compiler = _compiler()
    explanation = compiler.explain(_frame(TaskType.ANSWER, Complexity.LOW, Risk.LOW))

    assert explanation.requires_approval is False
    assert explanation.allowed is True


def test_no_matching_protocol_yields_an_empty_selection_not_an_exception() -> None:
    """``explain`` is diagnostic — it must describe a dead end, not raise into one."""
    registry = ProtocolRegistry()
    registry.register_many([p for p in builtin_protocols() if p.id == "debate"])
    compiler = CognitiveCompiler(
        protocols=registry,
        capabilities=CapabilityIndex(create_registry()),
        policy=PolicyGate(),
        model="openai:gpt-4o-mini",
    )
    explanation = compiler.explain(_frame(TaskType.REMEDIATE, Complexity.LOW, Risk.LOW))

    assert explanation.selected_protocol is None
    assert explanation.candidates == ()
    assert explanation.rejected  # and it says why
    assert explanation.policy is None


# ---------------------------------------------------------------- rendering --


def test_str_shows_frame_selection_rejections_and_policy() -> None:
    compiler = _compiler()
    rendered = str(
        compiler.explain(
            _frame(TaskType.REMEDIATE, Complexity.MEDIUM, Risk.HIGH),
            goal="Delete every record older than seven years",
        )
    )

    assert "Delete every record older than seven years" in rendered
    assert "GoalFrame" in rendered
    assert "remediate" in rendered
    assert "approval_gated_execution" in rendered
    assert "Rejected" in rendered
    assert "human approval required" in rendered


def test_no_chain_of_thought_is_exposed() -> None:
    """Structured evidence only — the rationale field stays empty without a picker."""
    compiler = _compiler()
    explanation = compiler.explain(_frame())

    assert explanation.picker_rationale is None


def test_types_are_frozen_so_a_record_cannot_be_edited_after_the_fact() -> None:
    compiler = _compiler()
    explanation = compiler.explain(_frame())

    for obj in (explanation, explanation.candidates[0]):
        with pytest.raises(Exception):  # noqa: B017,PT011 — FrozenInstanceError
            obj.protocol_id = "tampered"  # type: ignore[misc]


def test_dataclasses_render_standalone() -> None:
    rejected = RejectedProtocol("debate", "goal", "does not handle 'remediate'")
    ranked = RankedProtocol("debate", (0, 1, 2, 2), 0, False, "high", 2, selected=True)

    assert "debate" in str(rejected)
    assert "does not handle" in str(rejected)
    assert "→" in str(ranked)
    assert "not canonical" in str(ranked)


# ------------------------------------------------------------------- costs ----


def test_explain_contacts_no_model() -> None:
    """A model string that would explode if instantiated proves nothing is called."""
    registry = ProtocolRegistry()
    registry.register_many(builtin_protocols())
    compiler = CognitiveCompiler(
        protocols=registry,
        capabilities=CapabilityIndex(create_registry()),
        policy=PolicyGate(),
        model="nonexistent-provider:does-not-exist",
    )

    explanation = compiler.explain(_frame(TaskType.DIAGNOSE, Complexity.HIGH, Risk.MEDIUM))

    assert isinstance(explanation, RoutingExplanation)
    assert explanation.selected_protocol == "specialist_fanout"
