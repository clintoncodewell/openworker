"""Projects: the folder store, the evergreen brief updater, and their wiring.

The reason a project exists instead of "just scroll the chat history" is the brief:
after each conversation a model call folds what happened into `project.md`, and the
user can correct it by hand at any time. The failures these tests exist for are the
ones that would break that contract silently — an update that overwrites what the
user wrote, a misbehaving model that corrupts or truncates the file, storage that
escapes its directory, or a refresh failure that takes the conversation down with it.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3

import pytest
from fastapi.testclient import TestClient

from coworker.conversations import ConversationStore
from coworker.projects import ProjectStore, refresh_brief
from coworker.projects.brief import MAX_DECISIONS, MAX_THREADS
from coworker.projects.store import parse_project_markdown, render_project_markdown
from coworker.providers import AssistantTurn, ModelCapabilities, ProviderClient
from coworker.server import SessionManager, create_app
from coworker.sessions import SessionRecord


class BriefModel(ProviderClient):
    """A provider whose one job is the brief-maintaining call. `reply` is the text to
    return, an exception to raise, or a callable invoked at call time — the callable
    is how tests edit files while the "model call" is in flight."""

    def __init__(self, reply):
        self.reply = reply
        self.calls = 0

    def complete(self, *, model, messages, tools=None, **settings):
        self.calls += 1
        reply = self.reply() if callable(self.reply) else self.reply
        if isinstance(reply, Exception):
            raise reply
        return AssistantTurn(text=reply)

    def capabilities(self, model):
        return ModelCapabilities()


def _update(where="nothing changed", decisions=(), new_threads=(), resolved=()):
    return json.dumps(
        {
            "where_it_stands": where,
            "decisions": [{"decision": d, "reason": ""} for d in decisions],
            "new_open_threads": list(new_threads),
            "resolved_open_threads": list(resolved),
        }
    )


def _session(session_id, workspace, project_id=None):
    return SessionRecord(
        session_id=session_id,
        workspace=str(workspace),
        model="fake",
        mode="interactive",
        messages=[{"role": "user", "content": "hello"}],
        project_id=project_id,
    )


# -- the store -----------------------------------------------------------------------


def test_create_list_get_delete_round_trip(tmp_path):
    """A project is a plain folder the user can find and back up. If the folder and
    the store ever disagree, the GUI shows projects that don't exist (or hides ones
    that do) and there is no way to reconcile by hand."""
    store = ProjectStore(tmp_path / "projects")
    project = store.create("Cottage", purpose="Fix up the cottage")

    assert project.path.is_dir()
    assert project.markdown_path.is_file()
    assert project.files_path.is_dir()
    assert [p.name for p in store.list()] == ["Cottage"]
    assert store.get(project.id).id == project.id

    assert store.delete(project.id) is True
    assert not project.path.exists()
    assert store.list() == []
    assert store.get(project.id) is None
    assert store.delete(project.id) is False


def test_hostile_names_cannot_escape_the_projects_directory(tmp_path):
    """The project name becomes a folder name on the user's disk. A crafted name that
    walked out of `<state>/projects/` would write (and `delete` would rmtree) wherever
    it landed, so containment is checked on the RESOLVED path, not the slug string."""
    store = ProjectStore(tmp_path / "projects")
    root = store.root.resolve()
    for name in ("../../etc/passwd", "!!!", "", "Ａ" * 300):
        project = store.create(name)
        assert project.path.resolve().is_relative_to(root), name


def test_two_projects_with_the_same_name_do_not_collide(tmp_path):
    """A repeated name is exactly what a user will type ("Trip planning"). The second
    project must not silently absorb — or rmtree on delete — the first one's brief."""
    store = ProjectStore(tmp_path / "projects")
    first = store.create("Trip planning")
    second = store.create("Trip planning")

    assert first.id != second.id
    assert first.path != second.path
    assert store.get(first.id).path == first.path

    store.delete(first.id)
    survivor = store.get(second.id)
    assert survivor is not None
    assert survivor.markdown_path.is_file()


# -- the updater ---------------------------------------------------------------------


def test_refresh_rewrites_the_summary_appends_a_decision_and_ticks_a_thread(tmp_path):
    """The fold-in IS the feature: a stale "Where it stands" or a decision that never
    got recorded is the exact complaint about the commercial equivalents."""
    store = ProjectStore(tmp_path / "projects")
    project = store.create("Cottage", purpose="Fix up the cottage")

    model = BriefModel(_update(where="Roofers quoted", new_threads=["Choose a roofer"]))
    assert refresh_brief(store, project.id, [], provider=model, model="fake") is True
    doc = parse_project_markdown(store.read_markdown(project.id))
    assert doc.purpose == "Fix up the cottage"
    assert doc.where_it_stands == "Roofers quoted"
    assert doc.open_threads == [(False, "Choose a roofer")]

    model.reply = _update(
        where="Roofer booked",
        decisions=["Hired Acme Roofing"],
        resolved=["Choose a roofer"],
    )
    assert refresh_brief(store, project.id, [], provider=model, model="fake") is True
    doc = parse_project_markdown(store.read_markdown(project.id))
    assert doc.where_it_stands == "Roofer booked"
    assert doc.open_threads == [(True, "Choose a roofer")]
    assert len(doc.decisions) == 1
    assert "Hired Acme Roofing" in doc.decisions[0]


def test_refresh_never_overwrites_a_purpose_edited_while_it_ran(tmp_path):
    """Purpose is the user's own words and the updater never maintains it. The model
    call takes minutes, so the file is re-read after it — an edit that lands mid-call
    must survive the write that follows, or correcting the memory costs the correction."""
    store = ProjectStore(tmp_path / "projects")
    project = store.create("Garden", purpose="Original purpose")

    def edit_during_the_call():
        text = project.markdown_path.read_text(encoding="utf-8")
        project.markdown_path.write_text(
            text.replace("Original purpose", "Hand-edited while the model ran"),
            encoding="utf-8",
        )
        return _update(where="Shrubs pruned")

    assert (
        refresh_brief(store, project.id, [], provider=BriefModel(edit_during_the_call), model="fake")
        is True
    )
    doc = parse_project_markdown(store.read_markdown(project.id))
    assert doc.purpose == "Hand-edited while the model ran"
    assert doc.where_it_stands == "Shrubs pruned"


def test_a_failed_model_call_leaves_the_brief_byte_for_byte_unchanged(tmp_path):
    """A refresh runs after every turn; providers fail routinely. A failure that
    half-wrote the brief — or propagated into the turn path — would make the feature
    a way to LOSE conversations."""
    store = ProjectStore(tmp_path / "projects")
    project = store.create("Fragile", purpose="p")
    before = store.read_markdown(project.id).encode("utf-8")

    ok = refresh_brief(
        store, project.id, [], provider=BriefModel(RuntimeError("provider down")), model="fake"
    )
    assert ok is False
    assert store.read_markdown(project.id).encode("utf-8") == before


@pytest.mark.parametrize(
    "reply",
    [
        "",
        "Sounds good — updated the brief.",
        '["a", "list"]',
        '{"decisions": []}',
        "```json\nnot json at all\n```",
    ],
)
def test_malformed_model_output_does_not_corrupt_the_brief(tmp_path, reply):
    """`_parse_output` demands a JSON object with a string `where_it_stands`; models
    drift off format constantly. Garbage in must mean "no refresh", never a mangled
    or emptied project.md."""
    store = ProjectStore(tmp_path / "projects")
    project = store.create("Drifty", purpose="p")
    before = store.read_markdown(project.id).encode("utf-8")

    ok = refresh_brief(store, project.id, [], provider=BriefModel(reply), model="fake")
    assert ok is False
    assert store.read_markdown(project.id).encode("utf-8") == before


def test_decisions_are_capped_at_the_most_recent_40(tmp_path):
    """The brief must stay a brief. Without a cap every dated decision accumulates
    forever and the file becomes a second transcript nobody reads."""
    store = ProjectStore(tmp_path / "projects")
    project = store.create("Long-running")
    doc = parse_project_markdown(store.read_markdown(project.id))
    doc.decisions = [f"- 2026-01-01 — old decision {i}" for i in range(MAX_DECISIONS)]
    store.write_document(project.id, doc)

    assert (
        refresh_brief(
            store, project.id, [], provider=BriefModel(_update(decisions=["fresh decision"])), model="fake"
        )
        is True
    )
    out = parse_project_markdown(store.read_markdown(project.id))
    assert len(out.decisions) == MAX_DECISIONS
    assert out.decisions[-1].endswith("fresh decision")
    assert "old decision 0" not in " ".join(out.decisions)


def test_the_thread_cap_drops_resolved_threads_before_unresolved(tmp_path):
    """Resolved threads are finished work — the cheapest history to lose when the cap
    bites. Dropping unresolved ones instead would silently delete live to-dos."""
    store = ProjectStore(tmp_path / "projects")
    project = store.create("Busy")
    doc = parse_project_markdown(store.read_markdown(project.id))
    doc.open_threads = [(False, f"live thread {i}") for i in range(20)] + [
        (True, f"done thread {i}") for i in range(10)
    ]
    store.write_document(project.id, doc)

    assert (
        refresh_brief(
            store, project.id, [], provider=BriefModel(_update(new_threads=["brand new"])), model="fake"
        )
        is True
    )
    out = parse_project_markdown(store.read_markdown(project.id))
    assert len(out.open_threads) == MAX_THREADS
    assert sum(1 for done, _ in out.open_threads if not done) == 21  # 20 live + the new one
    assert (False, "brand new") in out.open_threads


# -- persistence and wiring ----------------------------------------------------------


def test_a_hand_deleted_brief_does_not_take_the_project_list_down(tmp_path):
    """`project.md` is advertised as user-editable, and deleting a file is the edge
    case of editing one. One such folder must not make every project listing (and
    every refresh of it) raise — the next refresh recreates the brief instead."""
    store = ProjectStore(tmp_path / "projects")
    project = store.create("Half-deleted", purpose="p")
    project.markdown_path.unlink()

    assert [p.name for p in store.list()] == ["Half-deleted"]
    assert store.read_markdown(project.id) == ""

    ok = refresh_brief(
        store, project.id, [], provider=BriefModel(_update(where="recreated")), model="fake"
    )
    assert ok is True
    doc = parse_project_markdown(store.read_markdown(project.id))
    assert doc.where_it_stands == "recreated"





def test_a_session_record_from_before_projects_still_loads(tmp_path):
    """The column arrived with this feature; every existing install has a sessions
    table without it. A load that crashed on the old schema would orphan every prior
    conversation on first run after upgrade."""
    base = tmp_path / "state"
    base.mkdir()
    conn = sqlite3.connect(base / "coworker.db")
    # The pre-projects schema: everything the loader reads unguarded, minus project_id.
    conn.executescript(
        """
        CREATE TABLE sessions (
            session_id TEXT PRIMARY KEY, workspace TEXT, model TEXT, mode TEXT,
            title TEXT, agent TEXT DEFAULT 'code', n_msgs INTEGER DEFAULT 0, messages TEXT,
            extra_roots TEXT, pinned INTEGER DEFAULT 0, archived INTEGER DEFAULT 0,
            origin TEXT, origin_label TEXT,
            auto_title TEXT, renamed INTEGER DEFAULT 0,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        """
    )
    conn.execute(
        "INSERT INTO sessions (session_id, workspace, model, mode, title, messages)"
        " VALUES ('old-session', '/tmp/x', 'm', 'auto', 'Old', ?)",
        (json.dumps([{"role": "user", "content": "hi"}]),),
    )
    conn.commit()
    conn.close()

    record = ConversationStore(base).load("old-session")
    assert record is not None
    assert record.project_id is None
    assert record.messages[0]["content"] == "hi"


def test_a_turn_without_a_project_touches_nothing(tmp_path):
    """The post-turn hook runs after EVERY turn. A session that never opted into a
    project must not grow folders, background tasks, or model calls from it."""
    manager = SessionManager(workspace=tmp_path, provider=BriefModel(_update()))
    engine = manager.get_engine("plain-session")
    engine.project_id = None

    async def finish_the_turn():
        manager._maybe_refresh_project("plain-session")
        await asyncio.gather(*manager._project_tasks)

    asyncio.run(finish_the_turn())
    assert list(manager.project_store.root.iterdir()) == []


def test_the_post_turn_hook_refreshes_the_attached_project(tmp_path):
    """The fold-in is supposed to happen on its own after each conversation. A hook
    that only fired on manual refresh would reintroduce the staleness the feature
    exists to fix."""
    model = BriefModel(_update(where="brief updated after the turn"))
    manager = SessionManager(workspace=tmp_path, provider=model)
    project = manager.project_store.create("Hooked", purpose="p")
    engine = manager.get_engine("hooked-session")
    engine.project_id = project.id

    async def finish_the_turn():
        manager._maybe_refresh_project("hooked-session")
        await asyncio.gather(*manager._project_tasks)

    asyncio.run(finish_the_turn())
    doc = parse_project_markdown(manager.project_store.read_markdown(project.id))
    assert doc.where_it_stands == "brief updated after the turn"


def test_the_project_block_reaches_the_system_prompt(tmp_path):
    """The brief only helps if the model actually sees it. A session rebuilt from
    disk must get the block (and a rebuilt one must not keep a stale block from a
    project it no longer belongs to)."""
    manager = SessionManager(workspace=tmp_path, provider=BriefModel(_update()))
    project = manager.project_store.create("Briefed", purpose="Ship the ledger rewrite")
    manager.session_store.save(_session("in-project", tmp_path, project_id=project.id))

    attached = manager.get_engine("in-project")
    system = attached.messages[0]["content"]
    assert "<openworker-project>" in system
    assert "Ship the ledger rewrite" in system
    assert attached.project_id == project.id

    plain = manager.get_engine("plain-session")
    assert "<openworker-project>" not in plain.messages[0]["content"]


def test_attach_and_refresh_return_the_full_updated_project(tmp_path):
    manager = SessionManager(workspace=tmp_path, provider=BriefModel(_update(where="Fresh")))
    project = manager.project_store.create("Launch", purpose="Ship it")
    manager.session_store.save(_session("launch-chat", tmp_path))

    attached = manager.attach_project_session(project.id, "launch-chat")
    assert attached["ok"] is True
    assert attached["session_ids"] == ["launch-chat"]
    assert "Ship it" in attached["project_md"]

    refreshed = manager.refresh_project(project.id)
    assert refreshed["ok"] is True
    assert refreshed["session_ids"] == ["launch-chat"]
    assert "Fresh" in refreshed["project_md"]


# -- HTTP ----------------------------------------------------------------------------


def _client(tmp_path, provider):
    manager = SessionManager(workspace=tmp_path, provider=provider)
    return manager, TestClient(create_app(manager))


def test_projects_http_round_trip(tmp_path):
    """The GUI talks to these routes only. A route that 500s on a missing project or
    forgets to clear session links on delete leaves the desktop shell showing a
    project that no longer exists."""
    manager, client = _client(tmp_path, BriefModel(_update()))
    created = client.post(
        "/v1/projects", json={"name": "Kitchen", "purpose": "Renovate the kitchen"}
    )
    assert created.status_code == 200
    body = created.json()
    assert body["ok"] is True
    assert "Renovate the kitchen" in body["project_md"]
    project_id = body["id"]

    listed = client.get("/v1/projects").json()["projects"]
    assert [p["name"] for p in listed] == ["Kitchen"]

    fetched = client.get(f"/v1/projects/{project_id}").json()
    assert fetched["ok"] is True
    assert fetched["session_ids"] == []

    manager.session_store.save(_session("reno-chat", tmp_path))
    attach = client.post(
        f"/v1/projects/{project_id}/sessions", json={"session_id": "reno-chat"}
    )
    assert attach.status_code == 200
    attached = attach.json()
    assert attached["ok"] is True
    assert attached["session_ids"] == ["reno-chat"]
    assert "Renovate the kitchen" in attached["project_md"]
    assert manager.session_store.load("reno-chat").project_id == project_id
    assert client.get(f"/v1/projects/{project_id}").json()["session_ids"] == ["reno-chat"]

    refreshed = client.post(f"/v1/projects/{project_id}/refresh").json()
    assert refreshed["ok"] is True
    assert refreshed["session_ids"] == ["reno-chat"]
    assert "project_md" in refreshed

    deleted = client.delete(f"/v1/projects/{project_id}")
    assert deleted.status_code == 200
    assert deleted.json()["ok"] is True
    assert list(manager.project_store.root.iterdir()) == []
    assert manager.project_store.get(project_id) is None
    # deleting the project releases its sessions, so a rebuilt engine has no stale block
    assert manager.session_store.load("reno-chat").project_id is None


def test_a_new_websocket_session_can_start_inside_a_project(tmp_path):
    manager, client = _client(tmp_path, BriefModel(_update()))
    project = manager.project_store.create("Launch")

    with client.websocket_connect(
        f"/ws/session/project-chat?agent=cowork&project_id={project.id}"
    ) as ws:
        assert ws.receive_json()["type"] == "ready"

    assert manager.session_store.load("project-chat").project_id == project.id
    assert manager.project_store.get(project.id).session_ids == ["project-chat"]


def test_starting_a_session_in_a_dead_project_leaves_no_orphan(tmp_path):
    """The save used to run BEFORE the project was checked, so a stale project id left an
    untitled empty conversation stranded in the sidebar with no way to explain it."""
    manager, client = _client(tmp_path, BriefModel(_update()))

    with client.websocket_connect(
        "/ws/session/ghost-chat?agent=cowork&project_id=no-such-project"
    ) as ws:
        message = ws.receive_json()

    assert message["type"] == "error"
    assert message["data"]["ok"] is False
    assert manager.session_store.load("ghost-chat") is None


def test_unknown_project_requests_return_a_clean_error(tmp_path):
    """A stale GUI (or a hand-typed curl) will name a project that is gone. The answer
    must be a structured error, never a traceback page."""
    _, client = _client(tmp_path, BriefModel(_update()))
    assert client.get("/v1/projects/nope").json() == {"ok": False, "error": "no such project"}
    assert client.post("/v1/projects/nope", json={"name": "x"}).json()["ok"] is False
    assert client.delete("/v1/projects/nope").json()["ok"] is False
    assert (
        client.post("/v1/projects/nope/sessions", json={"session_id": "s"}).json()["ok"]
        is False
    )
    assert client.post("/v1/projects/nope/refresh").json()["ok"] is False
