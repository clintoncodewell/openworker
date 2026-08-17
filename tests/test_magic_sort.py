import asyncio
import json

from coworker.providers import AssistantTurn, ModelCapabilities, ProviderClient
from coworker.server.manager import SessionManager
from coworker.sessions import SessionRecord


class SortProvider(ProviderClient):
    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def complete(self, *, model, messages, tools=None, **settings):
        self.calls.append({"model": model, "messages": messages, **settings})
        if isinstance(self.reply, Exception):
            raise self.reply
        return AssistantTurn(text=self.reply)

    def capabilities(self, model):
        return ModelCapabilities()


def _session(session_id, workspace, *, title=None, pinned=False, archived=False):
    return SessionRecord(
        session_id=session_id,
        workspace=str(workspace),
        model="fake",
        mode="interactive",
        messages=[{"role": "user", "content": title or session_id}],
        title=title,
        pinned=pinned,
        archived=archived,
    )


def _run_inline(monkeypatch, manager):
    async def inline(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr("coworker.server.manager.asyncio.to_thread", inline)
    return asyncio.run(manager.propose_magic_sort())


def test_propose_magic_sort_parses_good_json_and_keeps_leave(tmp_path, monkeypatch):
    monkeypatch.setenv("COWORKER_STATE_DIR", str(tmp_path / "state"))
    provider = SortProvider("")
    manager = SessionManager(data_dir=tmp_path / "data", provider=provider, model="fast-default")
    for sid, title in (("one", "Plan Japan flights"), ("two", "Japan hotel shortlist"), ("three", "Loose thought")):
        manager.session_store.save(_session(sid, tmp_path, title=title))
    folder = manager.create_folder("Travel")["folder"]
    project = manager.project_store.create("Japan trip", purpose="Plan the Japan holiday")
    provider.reply = "Here you go:\n```json\n" + json.dumps(
        {
            "proposals": [
                {"session_id": "one", "action": "project", "project_id": project.id},
                {"session_id": "two", "action": "existing_folder", "folder_id": folder["id"]},
                {"session_id": "three", "action": "leave"},
            ]
        }
    ) + "\n```"

    result = _run_inline(monkeypatch, manager)

    assert result == {
        "ok": True,
        "proposals": [
            {"session_id": "one", "title": "Plan Japan flights", "action": "project", "project_id": project.id, "target_name": "Japan trip"},
            {"session_id": "two", "title": "Japan hotel shortlist", "action": "existing_folder", "folder_id": folder["id"], "target_name": "Travel"},
            {"session_id": "three", "title": "Loose thought", "action": "leave", "target_name": "Leave where it is"},
        ],
        "considered": 3,
        "skipped": 0,
    }
    assert provider.calls[0]["model"] == "fast-default"
    assert provider.calls[0]["reasoning_effort"] == "none"


def test_propose_magic_sort_garbage_is_a_clean_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("COWORKER_STATE_DIR", str(tmp_path / "state"))
    manager = SessionManager(
        data_dir=tmp_path / "data", provider=SortProvider("not JSON at all"), model="fake"
    )
    manager.session_store.save(_session("one", tmp_path))

    assert _run_inline(monkeypatch, manager) == {
        "ok": False,
        "error": "Could not sort right now, try again",
    }


def test_apply_magic_sort_reuses_new_folder_and_skips_stale_target(tmp_path, monkeypatch):
    monkeypatch.setenv("COWORKER_STATE_DIR", str(tmp_path / "state"))
    manager = SessionManager(data_dir=tmp_path / "data", provider=SortProvider(""))
    for sid in ("one", "two", "three"):
        manager.session_store.save(_session(sid, tmp_path))
    stale = manager.create_folder("Stale")["folder"]
    manager.delete_folder(stale["id"])

    result = manager.apply_magic_sort(
        [
            {"session_id": "one", "action": "new_folder", "target_name": "Japan"},
            {"session_id": "two", "action": "new_folder", "target_name": "Japan"},
            {"session_id": "three", "action": "existing_folder", "folder_id": stale["id"], "target_name": "Stale"},
        ]
    )

    assert result == {"ok": True, "moved": 2, "folders_created": 1, "skipped": 1, "unchanged": 0}
    folders = manager.chat_folders.list()
    assert [folder["name"] for folder in folders] == ["Japan"]
    assert manager.session_store.load("one").folder_id == folders[0]["id"]
    assert manager.session_store.load("two").folder_id == folders[0]["id"]
    assert manager.session_store.load("three").folder_id is None
