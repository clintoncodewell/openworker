"""Durable, server-owned prompt queue state machine.

Queue mutations are synchronous so one event-loop turn observes them atomically.  A claimed
item is persisted separately until its consumer acknowledges that the turn coroutine started;
after a crash, unacknowledged claims are restored to the front and the queue is paused.
"""

from __future__ import annotations

import copy
import json
import math
import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Optional


class PromptQueueError(ValueError):
    pass


class PromptQueue:
    def __init__(
        self,
        path: Path,
        *,
        clock: Callable[[], float] = time.time,
        id_factory: Callable[[], str] = lambda: "q-" + uuid.uuid4().hex[:8],
    ) -> None:
        self.path = path
        self._clock = clock
        self._id_factory = id_factory
        self._queues: dict[str, list[dict[str, Any]]] = {}
        self._paused: set[str] = set()
        self._claims: dict[str, tuple[str, dict[str, Any]]] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.is_file():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            queues = raw.get("queues") if isinstance(raw, dict) else None
            claims = raw.get("claims", {}) if isinstance(raw, dict) else {}
            if not isinstance(queues, dict) or not isinstance(claims, dict):
                return
            clean: dict[str, list[dict[str, Any]]] = {}
            seen: set[str] = set()
            for sid, items in queues.items():
                if not isinstance(sid, str) or not isinstance(items, list):
                    continue
                valid = []
                for value in items:
                    item = self._valid_item(value)
                    if item is not None and item["id"] not in seen:
                        valid.append(item)
                        seen.add(item["id"])
                if valid:
                    clean[sid] = valid
            # A process may die after claim but before turn startup. Restore those items.
            for claim in claims.values():
                if not isinstance(claim, dict):
                    continue
                sid = claim.get("session_id")
                item = self._valid_item(claim.get("item"))
                if isinstance(sid, str) and item is not None and item["id"] not in seen:
                    clean.setdefault(sid, []).insert(0, item)
                    seen.add(item["id"])
            self._queues = clean
            # Restart is deliberately safe: queued work never runs without explicit resume.
            self._paused = set(clean)
        except (OSError, ValueError, TypeError):
            self._queues, self._paused, self._claims = {}, set(), {}

    @staticmethod
    def _valid_item(value: Any) -> Optional[dict[str, Any]]:
        if not isinstance(value, dict) or not isinstance(value.get("id"), str):
            return None
        text = value.get("text")
        attachments = value.get("attachments", [])
        if not isinstance(text, str) or not isinstance(attachments, list):
            return None
        if not text.strip() and not attachments:
            return None
        try:
            created = float(value.get("created_at"))
        except (TypeError, ValueError):
            return None
        if not math.isfinite(created):
            return None
        not_before = value.get("not_before")
        if not_before is not None:
            try:
                not_before = float(not_before)
            except (TypeError, ValueError):
                return None
            if not math.isfinite(not_before):
                return None
        return {
            "id": value["id"],
            "text": text.strip(),
            "attachments": copy.deepcopy(attachments),
            "created_at": created,
            "not_before": not_before,
        }

    def _persist(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "queues": self._queues,
            "paused": sorted(self._paused),
            "claims": {
                cid: {"session_id": sid, "item": item}
                for cid, (sid, item) in self._claims.items()
            },
        }
        fd, name = tempfile.mkstemp(prefix=f".{self.path.name}.", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, separators=(",", ":"))
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(name, self.path)
        except BaseException:
            try:
                os.unlink(name)
            except OSError:
                pass
            raise

    def snapshot(self, session_id: str) -> dict[str, Any]:
        return {
            "items": copy.deepcopy(self._queues.get(session_id, [])),
            "paused": session_id in self._paused,
        }

    def sessions(self) -> list[str]:
        return list(self._queues)

    def enqueue(
        self,
        session_id: str,
        text: str,
        attachments: Optional[list] = None,
        not_before: Optional[float] = None,
    ) -> dict[str, Any]:
        text = (text or "").strip()
        if attachments is not None and not isinstance(attachments, list):
            raise PromptQueueError("attachments must be a list")
        attachments = copy.deepcopy((attachments or [])[:8])
        if not text and not attachments:
            raise PromptQueueError("a queued prompt needs text or an attachment")
        if not_before is not None:
            try:
                not_before = float(not_before)
            except (TypeError, ValueError) as exc:
                raise PromptQueueError("not_before must be a finite timestamp") from exc
            if not math.isfinite(not_before):
                raise PromptQueueError("not_before must be a finite timestamp")
        item = {
            "id": self._id_factory(),
            "text": text,
            "attachments": attachments,
            "created_at": self._clock(),
            "not_before": not_before,
        }
        self._queues.setdefault(session_id, []).append(item)
        try:
            self._persist()
        except BaseException:
            self._queues[session_id].pop()
            self._cleanup(session_id)
            raise
        return copy.deepcopy(item)

    def remove(self, session_id: str, item_id: str) -> Optional[dict[str, Any]]:
        items = self._queues.get(session_id, [])
        for index, item in enumerate(items):
            if item["id"] == item_id:
                found = items.pop(index)
                was_paused = session_id in self._paused
                self._cleanup(session_id)
                try:
                    self._persist()
                except BaseException:
                    self._queues[session_id] = items
                    items.insert(index, found)
                    if was_paused:
                        self._paused.add(session_id)
                    raise
                return copy.deepcopy(found)
        return None

    def claim_ready(self, session_id: str) -> Optional[dict[str, Any]]:
        if session_id in self._paused:
            return None
        items = self._queues.get(session_id, [])
        if not items:
            return None
        not_before = items[0].get("not_before")
        if not_before is not None and self._clock() < not_before:
            return None
        return self._claim_index(session_id, 0)

    def claim(self, session_id: str, item_id: str) -> Optional[dict[str, Any]]:
        """Claim a specific item for an explicit run-now action, ignoring gate/pause."""
        for index, item in enumerate(self._queues.get(session_id, [])):
            if item["id"] == item_id:
                return self._claim_index(session_id, index)
        return None

    def _claim_index(self, session_id: str, index: int) -> dict[str, Any]:
        items = self._queues[session_id]
        item = items.pop(index)
        was_paused = session_id in self._paused
        claim_id = "claim-" + uuid.uuid4().hex
        self._claims[claim_id] = (session_id, item)
        self._cleanup(session_id)
        try:
            self._persist()
        except BaseException:
            self._claims.pop(claim_id, None)
            self._queues[session_id] = items
            items.insert(index, item)
            if was_paused:
                self._paused.add(session_id)
            raise
        return {"claim_id": claim_id, "item": copy.deepcopy(item)}

    def ack(self, claim_id: str) -> bool:
        claim = self._claims.pop(claim_id, None)
        if claim is None:
            return False
        try:
            self._persist()
        except BaseException:
            self._claims[claim_id] = claim
            raise
        return True

    def release(self, claim_id: str, *, pause: bool = True) -> bool:
        claim = self._claims.pop(claim_id, None)
        if claim is None:
            return False
        sid, item = claim
        was_paused = sid in self._paused
        self._queues.setdefault(sid, []).insert(0, item)
        if pause:
            self._paused.add(sid)
        try:
            self._persist()
        except BaseException:
            self._queues[sid].pop(0)
            self._cleanup(sid)
            self._claims[claim_id] = claim
            if pause and not was_paused:
                self._paused.discard(sid)
            raise
        return True

    def edit(self, session_id: str, item_id: str, text: str) -> Optional[dict[str, Any]]:
        text = (text or "").strip()
        for item in self._queues.get(session_id, []):
            if item["id"] == item_id:
                if not text and not item.get("attachments"):
                    raise PromptQueueError("a queued prompt needs text or an attachment")
                old = item["text"]
                item["text"] = text
                try:
                    self._persist()
                except BaseException:
                    item["text"] = old
                    raise
                return copy.deepcopy(item)
        return None

    def reorder(self, session_id: str, order: list[str]) -> list[dict[str, Any]]:
        if len(order) != len(set(order)):
            raise PromptQueueError("queue order contains duplicate ids")
        items = self._queues.get(session_id, [])
        by_id = {item["id"]: item for item in items}
        if set(order) != set(by_id):
            raise PromptQueueError("queue order is stale")
        old = list(items)
        self._queues[session_id] = [by_id[item_id] for item_id in order]
        try:
            self._persist()
        except BaseException:
            self._queues[session_id] = old
            raise
        return self.snapshot(session_id)["items"]

    def pause(self, session_id: str) -> None:
        if self._queues.get(session_id):
            was_paused = session_id in self._paused
            self._paused.add(session_id)
            try:
                self._persist()
            except BaseException:
                if not was_paused:
                    self._paused.discard(session_id)
                raise

    def resume(self, session_id: str) -> None:
        if session_id in self._paused:
            self._paused.remove(session_id)
            try:
                self._persist()
            except BaseException:
                self._paused.add(session_id)
                raise

    def _cleanup(self, session_id: str) -> None:
        if not self._queues.get(session_id):
            self._queues.pop(session_id, None)
            self._paused.discard(session_id)
