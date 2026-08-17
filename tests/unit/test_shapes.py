# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Tests for the multi-agent shapes.

These replace the router's shape tests. What changed is *when* a shape is
chosen, not what it builds, so the properties worth pinning are the ones the
router could not have: that a shape is an ordinary tool the loop may decline,
that hooks reach the leaves inside it, and that a shape refuses work it cannot
do rather than doing it badly.

Everything here is deterministic — no model is contacted.
"""

from __future__ import annotations

import pytest

from tulip.shapes import MAX_BRANCHES, MAX_CODE_LOOPS, shape_tools
from tulip.testing import ScriptedModel, text


def _model(reply: str = "a finding") -> ScriptedModel:
    return ScriptedModel([text(reply)], repeat_last=True)


def _tools(model: ScriptedModel, **kw) -> dict:
    return {t.name: t for t in shape_tools(model=model, **kw)}


# ----------------------------------------------------------------- surface --


def test_all_four_shapes_are_returned() -> None:
    assert set(_tools(_model())) == {
        "fan_out",
        "debate",
        "plan_and_verify",
        "code_until_tests_pass",
    }


def test_shapes_are_ordinary_tools_with_schemas() -> None:
    """The whole point: a shape is a tool the loop may call, or not."""
    for shape in shape_tools(model=_model()):
        assert shape.name
        assert shape.description, f"{shape.name} needs a description the model can judge"
        schema = shape.to_schema() if hasattr(shape, "to_schema") else None
        assert schema is None or isinstance(schema, dict)


def test_descriptions_say_when_not_to_use_the_shape() -> None:
    """A tool that only advertises its upside gets over-used.

    Over-routing was the router's most expensive failure — 18 of 26 trap
    prompts. The tools inherit the same temptation, so each description has to
    push back explicitly.
    """
    by_name = {t.name: (t.description or "").lower() for t in shape_tools(model=_model())}
    assert "do not use it" in by_name["fan_out"]
    assert "not for questions that have an answer" in by_name["debate"]
    assert "overkill" in by_name["plan_and_verify"]


# ------------------------------------------------------------------ fan out --


@pytest.mark.asyncio
async def test_fan_out_runs_one_branch_per_aspect() -> None:
    model = _model()
    out = await _tools(model)["fan_out"].execute(task="why slow?", aspects=["logs", "metrics"])

    assert model.call_count == 2, "one agent per aspect"
    assert out.count("a finding") == 2, "every branch's answer comes back"


@pytest.mark.asyncio
async def test_fan_out_refuses_a_single_aspect_instead_of_fanning_out() -> None:
    """One angle is a direct answer. Spending a pipeline on it is the bug."""
    model = _model()
    out = await _tools(model)["fan_out"].execute(task="x", aspects=["only one"])

    assert "at least two" in out
    assert model.call_count == 0, "refusing must cost nothing"


@pytest.mark.asyncio
async def test_fan_out_ignores_blank_aspects() -> None:
    model = _model()
    out = await _tools(model)["fan_out"].execute(task="x", aspects=["logs", "  ", "", "metrics"])

    assert model.call_count == 2
    assert "at least two" not in out


@pytest.mark.asyncio
async def test_fan_out_caps_the_branch_count() -> None:
    """A model asked for angles will produce fifteen; each is a token bill."""
    model = _model()
    await _tools(model)["fan_out"].execute(
        task="x", aspects=[f"aspect {i}" for i in range(MAX_BRANCHES + 5)]
    )

    assert model.call_count == MAX_BRANCHES


@pytest.mark.asyncio
async def test_fan_out_briefs_each_branch_with_its_own_assignment() -> None:
    model = _model()
    await _tools(model)["fan_out"].execute(task="why slow?", aspects=["the logs", "the metrics"])

    prompts = " ".join(
        m.content or "" for turn in model.received_messages for m in turn if m.content
    )
    assert "the logs" in prompts
    assert "the metrics" in prompts
    assert "why slow?" in prompts, "every branch is told the overall question"


# ------------------------------------------------------------------- debate --


@pytest.mark.asyncio
async def test_debate_runs_two_debaters_and_a_judge() -> None:
    model = _model()
    out = await _tools(model)["debate"].execute(question="gRPC or REST?")

    assert model.call_count == 3
    assert out.count("a finding") == 3, "both transcripts and the verdict come back"


@pytest.mark.asyncio
async def test_debaters_are_told_to_take_opposing_sides() -> None:
    model = _model()
    await _tools(model)["debate"].execute(question="gRPC or REST?")

    prompts = " ".join(
        m.content or "" for turn in model.received_messages for m in turn if m.content
    )
    assert "*for*" in prompts
    assert "*against*" in prompts
    assert "impartial judge" in prompts


# ---------------------------------------------------------- plan and verify --


@pytest.mark.asyncio
async def test_plan_and_verify_runs_three_stages_in_order() -> None:
    model = _model()
    await _tools(model)["plan_and_verify"].execute(task="ship it", success_criteria="it ships")

    assert model.call_count == 3
    prompts = " ".join(
        m.content or "" for turn in model.received_messages for m in turn if m.content
    )
    assert "planner stage" in prompts
    assert "executor stage" in prompts
    assert "validator stage" in prompts


@pytest.mark.asyncio
async def test_the_validator_is_given_the_success_criteria() -> None:
    model = _model()
    await _tools(model)["plan_and_verify"].execute(
        task="ship it", success_criteria="every client still works"
    )

    prompts = " ".join(
        m.content or "" for turn in model.received_messages for m in turn if m.content
    )
    assert "every client still works" in prompts


@pytest.mark.asyncio
async def test_missing_criteria_gets_a_default_rather_than_an_empty_check() -> None:
    model = _model()
    await _tools(model)["plan_and_verify"].execute(task="ship it")

    prompts = " ".join(
        m.content or "" for turn in model.received_messages for m in turn if m.content
    )
    assert "completed as asked" in prompts


# --------------------------------------------------------------- code loop --


@pytest.mark.asyncio
async def test_code_loop_stops_as_soon_as_the_output_says_pass() -> None:
    model = ScriptedModel([text("PASS all green")], repeat_last=True)
    await _tools(model)["code_until_tests_pass"].execute(task="write add()")

    assert model.call_count == 1, "PASS on the first iteration ends the loop"


@pytest.mark.asyncio
async def test_code_loop_is_bounded_when_pass_never_arrives() -> None:
    """A model that cannot reach PASS must stop, not spend the budget."""
    model = ScriptedModel([text("FAIL: still broken")], repeat_last=True)
    await _tools(model)["code_until_tests_pass"].execute(task="write add()")

    assert model.call_count <= MAX_CODE_LOOPS


# ----------------------------------------------------------- the gate seam --


@pytest.mark.asyncio
async def test_hooks_reach_the_leaves_inside_a_shape() -> None:
    """Governance arrives through hooks, and the leaves make the tool calls.

    A shape that built its own agents without the caller's hooks would leave
    every branch of a fan-out ungoverned — and the branches are the part that
    touches the world.
    """
    seen: list[str] = []

    class RecordingHook:
        async def on_before_model_call(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
            seen.append("before_model_call")

    model = _model()
    tools = _tools(model, hooks=[RecordingHook()])
    await tools["fan_out"].execute(task="why slow?", aspects=["logs", "metrics"])

    assert len(seen) >= 2, f"the hook must fire inside every branch, saw {seen}"


@pytest.mark.asyncio
async def test_inner_agents_get_the_tools_they_were_given() -> None:
    from tulip.tools.decorator import tool

    @tool
    def lookup(query: str) -> str:
        """Look something up."""
        return "result"

    model = _model()
    tools = _tools(model, tools=[lookup])
    await tools["fan_out"].execute(task="x", aspects=["a", "b"])

    offered = [name for turn in model.offered_tools for name in turn]
    assert "lookup" in offered


@pytest.mark.asyncio
async def test_debaters_get_no_tools_by_design() -> None:
    """A debate is an argument, not an investigation — tools would invite one."""
    from tulip.tools.decorator import tool

    @tool
    def lookup(query: str) -> str:
        """Look something up."""
        return "result"

    model = _model()
    await _tools(model, tools=[lookup])["debate"].execute(question="a or b?")

    assert all(not turn for turn in model.offered_tools)
