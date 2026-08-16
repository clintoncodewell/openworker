"""Prompt queue state-machine, durability, and ownership regression tests."""

from __future__ import annotations

import asyncio
import json

import pytest

from coworker.prompt_queue import PromptQueue, PromptQueueError
from coworker.providers import AssistantTurn, ModelCapabilities, ProviderClient
from coworker.server.manager import SessionManager


def _queue(tmp_path, *, now=100.0):
    ids = iter(["q-1", "q-2", "q-3"])
    return PromptQueue(
        tmp_path / "queue.json", clock=lambda: now, id_factory=lambda: next(ids)
    )


def test_queue_is_strict_ordered_and_returns_deep_snapshots(tmp_path):
    queue = _queue(tmp_path)
    queue.enqueue("s", "later", [{"kind": "text", "text": "a"}], 200)
    queue.enqueue("s", "behind")

    assert queue.claim_ready("s") is None
    snapshot = queue.snapshot("s")
    snapshot["items"][0]["attachments"][0]["text"] = "mutated"
    assert queue.snapshot("s")["items"][0]["attachments"][0]["text"] == "a"


def test_claim_is_recovered_and_paused_after_restart(tmp_path):
    path = tmp_path / "queue.json"
    queue = _queue(tmp_path)
    queued = queue.enqueue("s", "do not lose me")
    claim = queue.claim_ready("s")
    assert claim and queue.snapshot("s")["items"] == []

    restored = PromptQueue(path)
    assert restored.snapshot("s")["items"][0]["id"] == queued["id"]
    assert restored.snapshot("s")["paused"] is True
    assert restored.claim_ready("s") is None


def test_release_restores_claim_to_head_and_pauses(tmp_path):
    queue = _queue(tmp_path)
    first = queue.enqueue("s", "first")
    queue.enqueue("s", "second")
    claim = queue.claim_ready("s")
    assert claim
    queue.release(claim["claim_id"])
    snapshot = queue.snapshot("s")
    assert [item["id"] for item in snapshot["items"]] == [first["id"], "q-2"]
    assert snapshot["paused"] is True


@pytest.mark.parametrize("when", [float("nan"), float("inf"), -float("inf")])
def test_invalid_schedule_is_rejected(tmp_path, when):
    with pytest.raises(PromptQueueError):
        _queue(tmp_path).enqueue("s", "bad", not_before=when)


def test_empty_edit_and_stale_or_duplicate_reorder_are_rejected(tmp_path):
    queue = _queue(tmp_path)
    item = queue.enqueue("s", "keep")
    with pytest.raises(PromptQueueError):
        queue.edit("s", item["id"], "   ")
    with pytest.raises(PromptQueueError):
        queue.reorder("s", [item["id"], item["id"]])
    with pytest.raises(PromptQueueError):
        queue.reorder("s", ["stale"])
    assert queue.snapshot("s")["items"][0]["text"] == "keep"


def test_empty_session_state_is_removed_from_disk(tmp_path):
    queue = _queue(tmp_path)
    item = queue.enqueue("s", "temporary")
    queue.remove("s", item["id"])
    raw = json.loads((tmp_path / "queue.json").read_text(encoding="utf-8"))
    assert raw["queues"] == {}
    assert raw["paused"] == []


def test_non_list_attachments_are_rejected(tmp_path):
    with pytest.raises(PromptQueueError):
        _queue(tmp_path).enqueue("s", "bad", {"kind": "text"})


def test_corrupt_and_malformed_persistence_is_ignored(tmp_path):
    path = tmp_path / "queue.json"
    path.write_text("{broken", encoding="utf-8")
    assert PromptQueue(path).sessions() == []
    path.write_text(
        json.dumps({"queues": {"s": [None, {"id": "x", "text": 42}]}}),
        encoding="utf-8",
    )
    assert PromptQueue(path).sessions() == []


def test_failed_persistence_rolls_mutation_back(tmp_path, monkeypatch):
    queue = _queue(tmp_path)

    def fail():
        raise OSError("disk full")

    monkeypatch.setattr(queue, "_persist", fail)
    with pytest.raises(OSError):
        queue.enqueue("s", "must not appear in memory only")
    assert queue.snapshot("s")["items"] == []


class _Provider(ProviderClient):
    def complete(self, *, model, messages, tools=None, **settings):
        return AssistantTurn(text="ok", finish_reason="stop")

    def capabilities(self, model):
        return ModelCapabilities()


def test_only_one_turn_owner_can_claim_a_session(tmp_path):
    manager = SessionManager(workspace=tmp_path, provider=_Provider())
    assert manager.try_mark_running("s") is True
    assert manager.try_mark_running("s") is False
    manager.mark_idle("s")
    assert manager.try_mark_running("s") is True


def test_scheduler_preserves_queued_attachments(tmp_path, monkeypatch):
    manager = SessionManager(workspace=tmp_path, provider=_Provider())
    attachment = {"kind": "text", "name": "note.txt", "text": "hello"}
    manager.enqueue_prompt("s", "use this", [attachment], not_before=0)
    delivered = []

    async def capture(session_id, message, **kwargs):
        delivered.append((session_id, message, kwargs.get("attachments")))
        manager.ack_queued_claim(kwargs["_queue_claim_id"])
        manager.mark_idle(session_id)

    monkeypatch.setattr(manager, "deliver_to_session", capture)
    asyncio.run(manager.advance_due_queues())
    assert delivered == [("s", "use this", [attachment])]
