# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit: a call the gate held is not one of its step's calls.

A call is counted against its step's ``max_tool_calls`` when it is admitted. A held call
never ran; ``release`` takes its admission back, and the call is counted once, when it is
admitted again after the person's decision.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from tulip.playbooks.v2.engine import PlaybookRuntime, parse_playbook_v2


EXAMPLE = Path(__file__).parent / "fixtures" / "refund-dispute.v2.yaml"


def runtime(**triage: Any) -> tuple[PlaybookRuntime, list[dict[str, Any]]]:
    body: dict[str, Any] = yaml.safe_load(EXAMPLE.read_text("utf-8"))
    for group in body["step_groups"]:
        for step in group["steps"]:
            if step["id"] == "triage":
                step.update(triage)
    events: list[dict[str, Any]] = []
    rt = PlaybookRuntime(parse_playbook_v2(body), emit=events.append, enforce=True)
    rt.start()
    return rt, events


def _too_many(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [e for e in events if e.get("violation") == "too_many_calls"]


def test_a_released_call_is_admitted_again_once() -> None:
    rt, events = runtime(max_tool_calls=1)
    assert rt._admit("lookup_order") == ""
    # Held: it never ran.
    assert rt.release("lookup_order") is True
    # Approved and performed: admitted, and credited as the step's one call.
    assert rt._admit("lookup_order") == ""
    rt._credit("lookup_order", '{"ok": true}')
    assert _too_many(events) == []
    assert rt._work["triage"].attempts == 1
    assert rt._work["triage"].executed == ["lookup_order"]
    # The limit still holds for the next one.
    assert "at most 1" in rt._admit("lookup_order")
    assert len(_too_many(events)) == 1


def test_without_a_release_the_held_call_used_the_steps_only_call() -> None:
    rt, events = runtime(max_tool_calls=1)
    rt._admit("lookup_order")
    assert rt._admit("lookup_order") != ""
    assert len(_too_many(events)) == 1


def test_nothing_to_release() -> None:
    rt, _ = runtime()
    assert rt.release("lookup_order") is False
    rt._admit("lookup_order")
    rt._credit("lookup_order", '{"ok": true}')  # it finished: nothing waits
    assert rt.release("lookup_order") is False
    assert rt._work["triage"].attempts == 1


def test_a_released_call_is_not_credited_and_does_not_meet_a_minimum() -> None:
    rt, _ = runtime(min_tool_calls=1)
    rt._admit("lookup_order")
    rt.release("lookup_order")
    # A result arriving for it afterwards finds no admission to credit.
    rt._credit("lookup_order", '{"ok": true}')
    assert rt._work["triage"].executed == []
    assert rt._work["triage"].attempts == 0


def test_releasing_a_question_drops_only_its_attribution() -> None:
    rt, _ = runtime()
    rt._admit("lookup_order")
    rt._credit("lookup_order", '{"ok": true}')
    rt._admit("ask_user")
    assert rt.release("ask_user") is True
    assert rt._work["triage"].attempts == 1
    assert rt._owners["ask_user"] == []
