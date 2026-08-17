# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Multi-agent shapes, as tools the loop may call.

A fan-out, a debate, a plan-then-verify pipeline and a code-until-tests-pass
loop — the shapes that used to be *topologies a router compiled before the run*,
rebound as *tools an agent calls during it*.

    from tulip import Agent
    from tulip.shapes import shape_tools

    agent = Agent(
        model="openai:gpt-5.5",
        tools=[*my_tools, *shape_tools(model="openai:gpt-5.5")],
    )

Why the change is worth the churn
---------------------------------
The router chose a shape from the *verb* in the request, before any evidence
existed. Measured over 180 labelled prompts on two models, that is where it
lost: given "Diagnose across logs, metrics and traces: what does the ``-v``
flag mean?", the extractor correctly reported ``complexity=low`` and the goal
gate then barred the cheapest shape from consideration, because
``direct_response`` did not declare ``diagnose``. Three parallel agents
answered a one-line question.

Every advantage the router had also shrank as models improved — on a weak model
it beat free-form protocol choice by 2.8 points, on a strong one it lost by
1.7; its boundedness guarantee mattered on the weak model (100% vs 50% under
prompt injection) and was indistinguishable on the strong one (100% vs 100%).
A component whose value is inversely proportional to model quality is a
component with a shrinking future.

Late binding fixes the timing. The agent reads the request, tries something,
and reaches for machinery when it has a reason to. It can fan out, read the
results, and fan out again — two decisions, the second informed by the first.
A compiled topology gets one.

What this is not
----------------
It is not a guarantee. A weak model will still over-reach for an expensive
shape: on gpt-4o-mini these tools were chosen well on 1 of 3 trap prompts, on
gpt-5.5 on 3 of 4. This removes the *structural* barrier that made good
behaviour impossible; it does not remove the temptation. Real guarantees come
from :func:`tulip.security.admit`, which runs inside the tool body where no
model reaches — and which governs these tools exactly like any other, because
they are ordinary tools.
"""

from __future__ import annotations

from typing import Any

from tulip.tools.decorator import tool


#: Cap on parallel branches in one ``fan_out``. A model asked for angles will
#: happily produce fifteen; each is a full agent with its own token bill, and
#: past a handful the synthesis gets worse rather than better.
MAX_BRANCHES = 6

#: Iteration cap for :func:`code_until_tests_pass`. Bounded so a model that
#: cannot reach PASS stops rather than spending the budget discovering that.
MAX_CODE_LOOPS = 4


def shape_tools(
    model: Any,
    *,
    tools: list[Any] | None = None,
    hooks: list[Any] | None = None,
    plugins: list[Any] | None = None,
) -> list[Any]:
    """Return the multi-agent shapes as tools, bound to ``model``.

    Args:
        model: Model instance or string every agent inside a shape uses.
        tools: Tools the inner agents may call. A fanned-out branch with no
            tools reasons from the model alone, which is rarely what you want.
        hooks: Lifecycle hooks attached to **every** agent a shape creates,
            leaves included. This is the seam governance arrives through: a
            hook that can cancel a tool call must reach the agents that make
            the calls, and inside a shape those are constructed here.
        plugins: Agent plugins (skills, and anything else) for inner agents.

    Returns:
        Four tools: ``fan_out``, ``debate``, ``plan_and_verify``,
        ``code_until_tests_pass``.
    """
    from tulip.agent.agent import Agent

    inner_tools = list(tools or [])
    inner_hooks = list(hooks or [])
    inner_plugins = list(plugins or [])

    def _agent(system_prompt: str, *, with_tools: bool = True) -> Any:
        """One inner agent. The single place hooks and plugins get attached.

        Concentrating construction is deliberate: a caller governs a run with a
        hook, but the agents that make the tool calls are built in here. A
        builder that constructed its own agents would leave every leaf inside a
        fan-out ungoverned, and the leaves are the dangerous part.
        """
        return Agent(
            model=model,
            tools=inner_tools if with_tools else [],
            system_prompt=system_prompt,
            hooks=inner_hooks,
            plugins=inner_plugins,
        )

    async def _text(agent: Any, prompt: str) -> str:
        result = await agent.arun(prompt)
        return str(getattr(result, "message", "") or "")

    # -- fan out ------------------------------------------------------------

    @tool
    async def fan_out(task: str, aspects: list[str]) -> str:
        """Investigate several independent angles at once, then read them together.

        Use when a question genuinely splits into strands that do not depend on
        each other — the logs *and* the metrics *and* the recent deploys. Each
        aspect becomes its own agent, run in parallel, and every answer comes
        back to you.

        Do not use it to make one straightforward question look thorough. It
        costs several times as much as answering directly and the joined output
        reads worse.

        Args:
            task: The overall question every branch is serving.
            aspects: One short brief per branch — what it should look into and
                report. Two to six.
        """
        from tulip.agent.composition import ParallelPipeline

        chosen = [a.strip() for a in aspects if a.strip()][:MAX_BRANCHES]
        if len(chosen) < 2:
            return (
                "fan_out needs at least two distinct aspects. One angle is a "
                "direct answer — just answer it."
            )
        branches = [
            _agent(
                f"You are one branch of a parallel investigation into:\n"
                f"  {task}\n\n"
                f"Your assignment, and only yours:\n"
                f"  {aspect}\n\n"
                "Report what you found on your assignment in a few sentences. "
                "Do not attempt the other branches' work and do not summarise "
                "the whole question — the caller joins the branches."
            )
            for aspect in chosen
        ]
        pipeline = ParallelPipeline(agents=branches, merge_strategy="concatenate")
        result = await pipeline.run(task)
        return str(getattr(result, "final_output", "") or "")

    # -- debate -------------------------------------------------------------

    @tool
    async def debate(question: str) -> str:
        """Argue a question both ways, then have a judge decide on the merits.

        Use for genuinely contested trade-offs where the strongest case each way
        is itself the useful output. Not for questions that have an answer — a
        debate about whether HTTP is stateless is theatre, and costs three
        agents to produce it.

        Args:
            question: The proposition to argue.
        """
        from tulip.agent.composition import ParallelPipeline

        pro = _agent(
            "You are Debater A. Argue strongly *for* the proposition implied by "
            "the user's question. Cite at least two concrete reasons. Label your "
            "answer 'A:' on the first line.",
            with_tools=False,
        )
        con = _agent(
            "You are Debater B. Argue strongly *against* the proposition implied "
            "by the user's question. Cite at least two concrete reasons. Label "
            "your answer 'B:' on the first line.",
            with_tools=False,
        )
        judge = _agent(
            "You are an impartial judge. Read both debater transcripts and pick "
            "the stronger argument on the merits. Be terse and end with: "
            "'WINNER: A | B | inconclusive'.",
            with_tools=False,
        )
        floor = ParallelPipeline(agents=[pro, con], merge_strategy="concatenate")
        transcript = await floor.run(question)
        joined = str(getattr(transcript, "final_output", "") or "")
        verdict = await _text(judge, f"Question: {question}\n\nTranscripts:\n{joined}")
        return f"{joined}\n\n{verdict}"

    # -- plan, execute, verify ----------------------------------------------

    @tool
    async def plan_and_verify(task: str, success_criteria: str = "") -> str:
        """Plan the work, carry it out, then check the result against the plan.

        Use for staged work where a missed step is expensive and worth a
        separate verification pass. Overkill for anything you could simply do —
        this runs three agents in sequence.

        Args:
            task: The work to plan, execute and verify.
            success_criteria: What "done" means. The validator checks against
                this, so a vague criterion buys a vague check.
        """
        from tulip.agent.composition import SequentialPipeline

        criteria = success_criteria.strip() or "the task is completed as asked"
        planner = _agent(
            f"You are the planner stage. Produce a concrete numbered plan for "
            f"the user's task. Success criteria: {criteria}."
        )
        executor = _agent(
            "You are the executor stage. Carry out the plan above using the "
            "available tools. Be decisive."
        )
        validator = _agent(
            f"You are the validator stage. Compare the executor's output against "
            f"the success criteria: {criteria}. State 'PASS' or 'FAIL: <reason>' "
            f"on the first line, then summarise."
        )
        pipeline = SequentialPipeline(agents=[planner, executor, validator])
        result = await pipeline.run(task)
        return str(getattr(result, "final_output", "") or "")

    # -- code until the tests pass ------------------------------------------

    @tool
    async def code_until_tests_pass(task: str) -> str:
        """Write code and keep revising it until it reports PASS.

        Use when correctness is checkable — there are tests, or an assertion the
        output must satisfy. The loop is bounded at four iterations; it stops and
        reports rather than spinning.

        Args:
            task: What to implement, including how success is checked.
        """
        from tulip.agent.composition import LoopAgent

        coder = _agent(
            "You are a code-generate-and-test loop. Each iteration:\n"
            "1. Produce or revise the code to satisfy the request.\n"
            "2. Use the available tools to run it and its tests.\n"
            "3. On the very first line, write 'PASS' if all tests passed and "
            "the spec is met, else 'FAIL: <one-line reason>'.\n"
            "Keep iterating until you can write PASS."
        )
        # ``condition`` signals *stop*, not continue — it is checked as
        # ``stopped = self.condition(output)``.
        loop = LoopAgent(
            agent=coder,
            max_loops=MAX_CODE_LOOPS,
            condition=lambda output: str(output).lstrip().startswith("PASS"),
        )
        result = await loop.run(task)
        return str(getattr(result, "final_output", None) or getattr(result, "message", "") or "")

    return [fan_out, debate, plan_and_verify, code_until_tests_pass]
