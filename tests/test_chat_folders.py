from coworker.chat_folders import ChatFolderStore
from coworker.server.app import create_app
from coworker.server.manager import SessionManager
from coworker.sessions import SessionRecord


def _session(session_id: str, workspace: str) -> SessionRecord:
    return SessionRecord(
        session_id=session_id,
        workspace=workspace,
        model="gpt-5.5",
        mode="interactive",
        messages=[{"role": "user", "content": session_id}],
        agent="cowork",
    )


def _endpoint(app, path: str, method: str):
    return next(
        route.endpoint
        for route in app.routes
        if getattr(route, "path", None) == path and method in getattr(route, "methods", set())
    )


def test_chat_folder_store_crud_persists(tmp_path):
    path = tmp_path / "chat_folders.json"
    store = ChatFolderStore(path)

    created = store.create("  Customer   work  ")
    assert created["name"] == "Customer work"
    assert len(created["id"]) == 12
    assert ChatFolderStore(path).list() == [created]

    renamed = store.rename(created["id"], "Priority")
    assert renamed["name"] == "Priority"
    assert store.delete(created["id"]) is True
    assert store.delete(created["id"]) is False
    assert store.list() == []


def test_folder_nav_layout_persists(tmp_path, monkeypatch):
    monkeypatch.setenv("COWORKER_STATE_DIR", str(tmp_path / "state"))
    data_dir = tmp_path / "data"
    manager = SessionManager(data_dir=data_dir)

    assert manager.set_nav_layout("folder") == {"ok": True, "nav_layout": "folder"}
    assert SessionManager(data_dir=data_dir).get_settings()["nav_layout"] == "folder"


def test_folder_routes_assign_rename_delete_and_clear_sessions(tmp_path, monkeypatch):
    monkeypatch.setenv("COWORKER_STATE_DIR", str(tmp_path / "state"))
    manager = SessionManager(data_dir=tmp_path / "data")
    manager.session_store.save(_session("one", str(tmp_path)))
    app = create_app(manager)

    created = _endpoint(app, "/v1/folders", "POST")({"name": "Work"})
    assert created["ok"] is True
    folder = created["folder"]
    assert _endpoint(app, "/v1/folders", "GET")()["folders"] == [folder]

    assigned = _endpoint(app, "/v1/sessions/{session_id}/folder", "POST")(
        "one", {"folder_id": folder["id"]}
    )
    assert assigned == {"ok": True, "session_id": "one", "folder_id": folder["id"]}
    assert manager.list_sessions()[0]["folder_id"] == folder["id"]

    renamed = _endpoint(app, "/v1/folders/{folder_id}/rename", "POST")(
        folder["id"], {"name": "Important"}
    )
    assert renamed["folder"]["name"] == "Important"

    deleted = _endpoint(app, "/v1/folders/{folder_id}", "DELETE")(folder["id"])
    assert deleted["ok"] is True and deleted["cleared_sessions"] == 1
    assert manager.list_sessions()[0]["folder_id"] is None


def test_delete_folder_clears_every_assigned_session_only(tmp_path, monkeypatch):
    monkeypatch.setenv("COWORKER_STATE_DIR", str(tmp_path / "state"))
    manager = SessionManager(data_dir=tmp_path / "data")
    for session_id in ("one", "two", "three", "other"):
        manager.session_store.save(_session(session_id, str(tmp_path)))
    target = manager.create_folder("Target")["folder"]
    untouched = manager.create_folder("Untouched")["folder"]
    for session_id in ("one", "two", "three"):
        assert manager.set_session_folder(session_id, target["id"])["ok"] is True
    assert manager.set_session_folder("other", untouched["id"])["ok"] is True

    deleted = manager.delete_folder(target["id"])

    assert deleted["cleared_sessions"] == 3
    assert all(manager.session_store.load(sid).folder_id is None for sid in ("one", "two", "three"))
    assert manager.session_store.load("other").folder_id == untouched["id"]


def test_set_session_folder_validation_and_archive_all(tmp_path, monkeypatch):
    monkeypatch.setenv("COWORKER_STATE_DIR", str(tmp_path / "state"))
    manager = SessionManager(data_dir=tmp_path / "data")
    for session_id in ("one", "two", "pinned", "__internal"):
        manager.session_store.save(_session(session_id, str(tmp_path)))
    manager.session_store.set_flags("two", archived=True)
    manager.session_store.set_flags("pinned", pinned=True)
    app = create_app(manager)
    set_folder = _endpoint(app, "/v1/sessions/{session_id}/folder", "POST")

    assert set_folder("missing", {"folder_id": None}) == {
        "ok": False,
        "error": "no such session",
    }
    assert set_folder("one", {"folder_id": "missing"}) == {
        "ok": False,
        "error": "no such folder",
    }
    assert set_folder("__internal", {"folder_id": None}) == {
        "ok": False,
        "error": "internal sessions cannot be modified here",
    }

    swept = _endpoint(app, "/v1/sessions/archive-all", "POST")()
    assert swept == {"ok": True, "archived_sessions": 1}
    assert manager.session_store.load("one").archived is True
    assert manager.session_store.load("two").archived is True
    assert manager.session_store.load("pinned").archived is False
    assert manager.session_store.load("__internal").archived is False


def test_folder_routes_return_clean_validation_errors(tmp_path, monkeypatch):
    monkeypatch.setenv("COWORKER_STATE_DIR", str(tmp_path / "state"))
    manager = SessionManager(data_dir=tmp_path / "data")
    manager.session_store.save(_session("one", str(tmp_path)))
    app = create_app(manager)
    invalid_id = _endpoint(app, "/v1/sessions/{session_id}/folder", "POST")(
        "one", {"folder_id": 123}
    )
    assert invalid_id == {"ok": False, "error": "no such folder"}

    empty_name = _endpoint(app, "/v1/folders", "POST")({"name": "   "})
    assert empty_name == {"ok": False, "error": "folder name required"}
