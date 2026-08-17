"""Small JSON-backed store for user-defined chat folder labels."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .secrets import state_dir


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class ChatFolderStore:
    def __init__(self, path: Optional[str | Path] = None) -> None:
        self.path = Path(path).expanduser() if path is not None else state_dir() / "chat_folders.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _load(self) -> list[dict[str, str]]:
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        if not isinstance(value, list):
            return []
        return [
            {"id": str(row["id"]), "name": str(row["name"]), "created": str(row["created"])}
            for row in value
            if isinstance(row, dict) and all(key in row for key in ("id", "name", "created"))
        ]

    def _save(self, folders: list[dict[str, str]]) -> None:
        temp = self.path.with_name(f".{self.path.name}.{uuid.uuid4().hex}.tmp")
        temp.write_text(json.dumps(folders, indent=2) + "\n", encoding="utf-8")
        temp.replace(self.path)

    @staticmethod
    def _name(name: str) -> str:
        clean = " ".join((name or "").split())[:120]
        if not clean:
            raise ValueError("folder name required")
        return clean

    def list(self) -> list[dict[str, str]]:
        return self._load()

    def get(self, folder_id: str) -> Optional[dict[str, str]]:
        return next((folder for folder in self._load() if folder["id"] == folder_id), None)

    def create(self, name: str) -> dict[str, str]:
        folders = self._load()
        folder = {"id": uuid.uuid4().hex[:12], "name": self._name(name), "created": _now()}
        folders.append(folder)
        self._save(folders)
        return folder

    def rename(self, folder_id: str, name: str) -> dict[str, str]:
        folders = self._load()
        for folder in folders:
            if folder["id"] == folder_id:
                folder["name"] = self._name(name)
                self._save(folders)
                return folder
        raise KeyError(folder_id)

    def delete(self, folder_id: str) -> bool:
        folders = self._load()
        kept = [folder for folder in folders if folder["id"] != folder_id]
        if len(kept) == len(folders):
            return False
        self._save(kept)
        return True
