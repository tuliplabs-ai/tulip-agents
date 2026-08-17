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


def _find_runner(tools: list[Any]) -> Any | None:
    """The caller's shell tool, if they gave us one.

    A shape that wants to *check* something has to run something, and running
    it straight from here would step around whatever gate the host put in front
    of its own tools — which is the one thing this library exists not to do. So
    a check goes through the host's runner, or it does not happen and the shape
    says so.
    """
    for candidate in tools:
        if getattr(candidate, "name", "") in {"bash", "shell", "run_command"}:
            return candidate
    return None


#: Appended to a check so its exit status is readable whatever the runner
#: prints. Deciding "did it work" from arbitrary tool output is guesswork; an
#: explicit marker is not.
_EXIT_MARKER = "__tulip_check_exit__"


async def _run_check(runner: Any, command: str) -> tuple[bool, str]:
    """Run ``command`` through the host's tool. Returns ``(passed, output)``."""
    wrapped = f'{command}; echo "{_EXIT_MARKER}:$?"'
    try:
        out = str(await runner.execute(command=wrapped))
    except Exception as exc:  # noqa: BLE001 — refused or broken, either way it failed
        return False, f"the check could not be run: {type(exc).__name__}: {exc}"
    return f"{_EXIT_MARKER}:0" in out, out.replace(f"{_EXIT_MARKER}:", "exit status ")


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
    async def plan_and_verify(task: str, success_criteria: str = "", check: str = "") -> str:
        """Plan the work, carry it out, then confirm it actually landed.

        Use for staged work where a missed step is expensive. Overkill for
        anything you could simply do — this runs three agents in sequence.

        If ``check`` is given, the confirmation is that command's exit status.
        Without one, the last stage is an agent reading the result and
        reporting — a second opinion, not a verification, and labelled as such.
        That distinction is not pedantry: asked to create a file, the reading
        version reported "Verified: file exists" about a file that did not.

        Args:
            task: The work to plan, execute and verify.
            success_criteria: What "done" means. A vague criterion buys a vague
                check.
            check: A shell command that exits 0 when the work is right. This is
                what turns the last stage from an opinion into a verdict.
        """
        from tulip.agent.composition import SequentialPipeline

        planner = _agent(
            "The planner stage. You break work into ordered steps. List them plainly, no preamble.",
            with_tools=False,
        )
        doer = _agent(
            "The executor stage. You carry out a plan using your tools. Make "
            "real changes to real files; never describe a change you have not "
            "made."
        )
        criteria = success_criteria or "the task is completed as asked"
        await SequentialPipeline(agents=[planner, doer]).run(f"{task}\n\nDone means: {criteria}")

        runner = _find_runner(inner_tools)
        if check and runner is not None:
            passed, out = await _run_check(runner, check)
            verdict = "VERIFIED" if passed else "FAILED VERIFICATION"
            return f"{verdict} — `{check}`:\n{out[:1500]}"

        reviewer = _agent(
            "The validator stage. You inspect what was actually done and report "
            "it. Use your tools to look; do not take anyone's word for it. Say "
            "plainly what you could and could not confirm."
        )
        read = await _text(reviewer, f"Did this happen?\n{task}\n\nDone means: {criteria}")
        return f"UNVERIFIED (no check given) — a reading, not a verdict:\n\n{read}"

    # -- code until the tests pass ------------------------------------------

    @tool
    async def code_until_tests_pass(task: str, check: str = "") -> str:
        """Write code, run a command that decides whether it is right, repeat.

        Use when correctness is *checkable* — a test command, or an assertion
        that either exits zero or does not. Bounded at four rounds; it stops
        and reports rather than spinning.

        The check is the whole point. An earlier version ended a round when the
        model wrote the word PASS, which is a self-declaration and not a check:
        asked three times to write a file and prove it worked, it returned the
        code as prose, said PASS, and wrote nothing at all — three times out of
        three. What makes this worth more than asking the agent directly is
        that something other than the model decides when to stop.

        Args:
            task: What to implement, and where it should live.
            check: A shell command that exits 0 when the work is right — a test
                run, a script, an assertion. Without one this cannot verify
                anything and says so, rather than claiming a pass it cannot see.
        """
        runner = _find_runner(inner_tools)
        coder = _agent(
            "You write and fix code. You have tools — use them to create and "
            "edit real files rather than printing code in your reply. After a "
            "change, say briefly what you changed and where."
        )

        if not check or runner is None:
            missing = "no check command was given" if not check else "no shell tool is available"
            said = await _text(coder, task)
            return f"UNVERIFIED — {missing}, so nothing confirmed this.\n\n{said}"

        attempt = task
        last = ""
        for round_number in range(1, MAX_CODE_LOOPS + 1):
            await _text(coder, attempt)
            passed, last = await _run_check(runner, check)
            if passed:
                return f"PASS on round {round_number}.\n\n`{check}`:\n{last[:1500]}"
            attempt = (
                f"{task}\n\nThe check `{check}` is still failing. Its output:\n"
                f"{last[:2000]}\n\nFix the actual files and try again."
            )
        return (
            f"FAIL after {MAX_CODE_LOOPS} rounds — `{check}` never passed.\n\n"
            f"Last output:\n{last[:1500]}"
        )

    return [fan_out, debate, plan_and_verify, code_until_tests_pass]
