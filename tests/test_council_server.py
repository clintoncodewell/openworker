"""The council's REST surface: read/edit config, test a source, browse past runs."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from coworker.council import config_path, runs_dir
from coworker.server.app import create_app
from coworker.server.manager import SessionManager


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("COWORKER_STATE_DIR", str(tmp_path / "state"))
    return TestClient(create_app(SessionManager(workspace=str(tmp_path))))


def test_config_ships_defaults_and_source_kinds(client):
    body = client.get("/v1/council/config").json()
    assert body["preset"] == "analysis"
    # The GUI needs the shipped text to render "reset to default".
    assert body["defaults"]["analysis"]["chair"].startswith("You are the chair")
    assert "folder" in body["source_kinds"] and "mcp" in body["source_kinds"]
    assert [r["name"] for r in body["default_roles"]][:2] == ["Advocate", "Skeptic"]


def test_editing_a_prompt_persists_and_reads_back(client):
    resp = client.post(
        "/v1/council/config", json={"prompts": {"analysis": {"chair": "MY CHAIR"}}}
    ).json()
    assert resp["ok"] is True
    assert resp["prompts"]["analysis"]["chair"] == "MY CHAIR"
    assert client.get("/v1/council/config").json()["prompts"]["analysis"]["chair"] == "MY CHAIR"
    # …and the shipped default is still offered alongside it, for the reset button.
    assert resp["defaults"]["analysis"]["chair"] != "MY CHAIR"


def test_a_partial_edit_leaves_the_rest_alone(client):
    """The GUI saves one pane at a time; saving prompts must not wipe sources."""
    client.post(
        "/v1/council/config",
        json={"sources": [{"kind": "folder", "target": "/tmp", "label": "Notes"}]},
    )
    client.post("/v1/council/config", json={"preset": "decision"})
    body = client.get("/v1/council/config").json()
    assert body["preset"] == "decision"
    assert body["sources"][0]["label"] == "Notes"


def test_derived_fields_are_never_written_to_disk(client):
    client.post("/v1/council/config", json={"preset": "decision"})
    stored = json.loads(config_path().read_text())
    assert "defaults" not in stored and "source_kinds" not in stored


def test_config_reports_the_panel_that_would_actually_run(client):
    body = client.post("/v1/council/config", json={"panel": ["a:one", "b:two"]}).json()
    assert body["resolved_panel"] == [
        {"model": "a:one", "role": "Advocate"},
        {"model": "b:two", "role": "Skeptic"},
    ]


def test_source_test_previews_a_folder(client, tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("THE BUDGET IS 40k")
    resp = client.post(
        "/v1/council/sources/test", json={"kind": "folder", "target": str(docs)}
    ).json()
    assert resp["ok"] is True
    assert "THE BUDGET IS 40k" in resp["preview"]
    assert resp["chars"] > 0


def test_source_test_reports_a_bad_path_instead_of_500(client):
    resp = client.post(
        "/v1/council/sources/test", json={"kind": "folder", "target": "/does/not/exist"}
    ).json()
    assert resp["ok"] is False and "FileNotFoundError" in resp["error"]


def test_runs_are_listed_newest_first_and_readable(client):
    base = runs_dir()
    for name in ("20260101-000000-older", "20260202-000000-newer"):
        d = base / name
        d.mkdir(parents=True)
        (d / "finding.md").write_text(f"finding for {name}")
    listed = client.get("/v1/council/runs").json()
    assert [r["id"] for r in listed] == ["20260202-000000-newer", "20260101-000000-older"]
    one = client.get("/v1/council/runs/20260202-000000-newer").json()
    assert one["ok"] is True
    assert "finding for" in one["files"]["finding.md"]


def test_no_runs_yet_is_an_empty_list_not_an_error(client):
    assert client.get("/v1/council/runs").json() == []


@pytest.mark.parametrize("run_id", ["../secrets.json", "..", "nope", "a/../../etc"])
def test_a_run_id_cannot_escape_the_runs_directory(client, run_id):
    """`run_id` is a directory name. Without the containment check it is a path, and the
    endpoint reads any file on the box."""
    resp = client.get(f"/v1/council/runs/{run_id}")
    assert resp.status_code == 404 or resp.json().get("ok") is False
