"""The evergreen brief's two cost-and-safety properties: it refreshes once per
CONVERSATION, and a bad model turn never erases what is already written.

Both were wrong when the feature was first built — it refreshed after every message, and an
empty reply blanked the section a reader looks at first.
"""

from __future__ import annotations

import asyncio

import pytest

from coworker.projects import brief as brief_mod
from coworker.projects.store import ProjectStore, parse_project_markdown


class Model:
    """Answers the refresh call with whatever JSON the test asks for."""

    def __init__(self, reply: str):
        self.reply = reply
        self.calls = 0

    def complete(self, **kw):
        self.calls += 1
        from coworker.providers.base import AssistantTurn

        return AssistantTurn(text=self.reply)


def _store(tmp_path) -> tuple[ProjectStore, str]:
    store = ProjectStore(tmp_path)
    project = store.create("Reporting migration", purpose="Move off Looker.")
    return store, project.id


CONVERSATION = [
    {"role": "user", "content": "Did the Metabase spike work?"},
    {"role": "assistant", "content": "Yes, but SSO is unresolved."},
]


# -- a bad model turn must not destroy the brief ---------------------------------------


def test_an_empty_summary_leaves_the_existing_one_alone(tmp_path):
    """"" is a valid JSON string and passes the parser, so a hiccuped call would erase the
    section a reader looks at first — and leave the next refresh nothing to build on."""
    store, pid = _store(tmp_path)
    brief_mod.refresh_brief(
        store, pid, CONVERSATION,
        provider=Model('{"where_it_stands": "Spike done, SSO is the blocker."}'),
        model="m",
    )
    assert "SSO is the blocker" in store.read_markdown(pid)

    ok = brief_mod.refresh_brief(
        store, pid, CONVERSATION, provider=Model('{"where_it_stands": "   "}'), model="m"
    )
    assert ok is True  # the rest of the update still applied
    assert "SSO is the blocker" in store.read_markdown(pid)


def test_a_model_that_raises_leaves_the_file_byte_for_byte(tmp_path):
    """A failed refresh must cost one refresh, never a conversation."""

    class Broken:
        def complete(self, **kw):
            raise RuntimeError("rate limited")

    store, pid = _store(tmp_path)
    before = store.read_markdown(pid)
    assert brief_mod.refresh_brief(store, pid, CONVERSATION, provider=Broken(), model="m") is False
    assert store.read_markdown(pid) == before


def test_junk_output_does_not_corrupt_the_file(tmp_path):
    store, pid = _store(tmp_path)
    before = store.read_markdown(pid)
    for junk in ("", "I'm sorry, I can't help with that.", "[1, 2, 3]", "{}"):
        assert brief_mod.refresh_brief(store, pid, CONVERSATION, provider=Model(junk), model="m") is False
    assert store.read_markdown(pid) == before


def test_the_purpose_is_never_model_maintained(tmp_path):
    """It is the user's paragraph. A refresh that rewrote it would quietly replace the one
    thing they authored, and they would have no way to tell."""
    store, pid = _store(tmp_path)
    brief_mod.refresh_brief(
        store, pid, CONVERSATION,
        provider=Model('{"where_it_stands": "Auth is the blocker."}'), model="m",
    )
    assert parse_project_markdown(store.read_markdown(pid)).purpose == "Move off Looker."


# -- once per conversation, not once per message ---------------------------------------


class _Manager:
    """Just enough SessionManager to exercise the debounce."""

    def __init__(self, monkeypatch, quiet=0.05):
        from coworker.server import manager as mgr_mod

        monkeypatch.setattr(mgr_mod, "PROJECT_REFRESH_QUIET_S", quiet)
        self._mgr_mod = mgr_mod
        self.refreshes = 0
        self._project_tasks = set()
        self._project_pending = {}
        self._engines = {"s1": type("E", (), {"project_id": "p", "model": "m", "messages": []})()}

    async def _refresh_project_after_turn(self, project_id, messages, model, engine, session_id=""):
        try:
            await asyncio.sleep(self._mgr_mod.PROJECT_REFRESH_QUIET_S)
        except asyncio.CancelledError:
            return
        finally:
            if self._project_pending.get(session_id) is asyncio.current_task():
                self._project_pending.pop(session_id, None)
        self.refreshes += 1

    def turn(self):
        from coworker.server.manager import SessionManager

        SessionManager._maybe_refresh_project(self, "s1")


@pytest.mark.asyncio
async def test_a_burst_of_turns_produces_one_refresh(monkeypatch):
    """This fires from `mark_idle`, which runs after EVERY message. One model call per
    message is roughly fifteen times the cost of one per chat on a normal back-and-forth."""
    m = _Manager(monkeypatch)
    for _ in range(20):
        m.turn()
        await asyncio.sleep(0)  # let the loop schedule, as a real burst would
    await asyncio.sleep(0.3)
    assert m.refreshes == 1


@pytest.mark.asyncio
async def test_a_second_conversation_later_refreshes_again(monkeypatch):
    """Debouncing must not mean "once ever" — a chat resumed after a break earns its own."""
    m = _Manager(monkeypatch)
    m.turn()
    await asyncio.sleep(0.3)
    m.turn()
    await asyncio.sleep(0.3)
    assert m.refreshes == 2


@pytest.mark.asyncio
async def test_a_cancelled_refresh_does_not_leak_its_slot(monkeypatch):
    """The pending map is keyed by session and would otherwise hold a dead task forever."""
    m = _Manager(monkeypatch)
    m.turn()
    m.turn()
    await asyncio.sleep(0.3)
    assert m._project_pending == {}
