"""Projects as ordinary folders: inspectable, editable, syncable state."""

from __future__ import annotations

import json
import re
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from ..secrets import state_dir

_SLUG = re.compile(r"[^a-z0-9]+")
_SECTIONS = ("Purpose", "Where it stands", "Decisions", "Open threads")
_PROJECT_START = "<openworker-project>"
_PROJECT_END = "</openworker-project>"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _slug(name: str) -> str:
    return _SLUG.sub("-", (name or "").lower()).strip("-") or "project"


def _atomic_write(path: Path, text: str) -> None:
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temp.write_text(text, encoding="utf-8")
    temp.replace(path)


@dataclass(frozen=True)
class Project:
    id: str
    name: str
    created: str
    session_ids: list[str]
    instructions: str
    path: Path

    @property
    def markdown_path(self) -> Path:
        return self.path / "project.md"

    @property
    def files_path(self) -> Path:
        return self.path / "files"


@dataclass
class ProjectDocument:
    name: str
    purpose: str
    where_it_stands: str
    decisions: list[str]
    open_threads: list[tuple[bool, str]]


def parse_project_markdown(text: str, *, fallback_name: str = "Project") -> ProjectDocument:
    heading = re.search(r"^#\s+(.+?)\s*$", text or "", re.MULTILINE)
    name = heading.group(1).strip() if heading else fallback_name
    found: dict[str, str] = {}
    matches = list(re.finditer(r"^##\s+(.+?)\s*$", text or "", re.MULTILINE))
    for i, match in enumerate(matches):
        title = match.group(1).strip()
        if title not in _SECTIONS:
            continue
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        found[title] = text[match.end() : end].strip()
    decisions = [line.strip() for line in found.get("Decisions", "").splitlines() if line.strip().startswith("-")]
    threads: list[tuple[bool, str]] = []
    for line in found.get("Open threads", "").splitlines():
        match = re.match(r"^\s*-\s*\[([ xX])\]\s*(.+?)\s*$", line)
        if match:
            threads.append((match.group(1).lower() == "x", match.group(2).strip()))
    return ProjectDocument(
        name=name,
        purpose=found.get("Purpose", ""),
        where_it_stands=found.get("Where it stands", ""),
        decisions=decisions,
        open_threads=threads,
    )


def render_project_markdown(document: ProjectDocument) -> str:
    decisions = "\n".join(document.decisions)
    threads = "\n".join(
        f"- [{'x' if resolved else ' '}] {text}"
        for resolved, text in document.open_threads
    )
    return (
        f"# {document.name}\n\n"
        f"## Purpose\n{document.purpose.strip()}\n\n"
        f"## Where it stands\n{document.where_it_stands.strip()}\n\n"
        f"## Decisions\n{decisions}\n\n"
        f"## Open threads\n{threads}\n"
    )


def replace_project_context(system_prompt: str, block: str = "") -> str:
    pattern = re.compile(
        rf"\n*{re.escape(_PROJECT_START)}.*?{re.escape(_PROJECT_END)}\n*",
        re.DOTALL,
    )
    clean = pattern.sub("\n\n", system_prompt or "").strip()
    return f"{clean}\n\n{block}".strip() if block else clean


class ProjectStore:
    def __init__(self, base_dir: Optional[str | Path] = None) -> None:
        self.root = (
            Path(base_dir).expanduser() if base_dir is not None else state_dir() / "projects"
        )
        self.root.mkdir(parents=True, exist_ok=True)

    def _folders(self) -> list[Path]:
        return [p for p in self.root.iterdir() if p.is_dir() and (p / "project.json").is_file()]

    def _path_for_id(self, project_id: str) -> Optional[Path]:
        for folder in self._folders():
            try:
                data = json.loads((folder / "project.json").read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if data.get("id") == project_id:
                return folder
        return None

    def _load(self, path: Path) -> Project:
        data = json.loads((path / "project.json").read_text(encoding="utf-8"))
        return Project(
            id=str(data["id"]),
            name=str(data["name"]),
            created=str(data["created"]),
            session_ids=[str(x) for x in data.get("session_ids") or []],
            instructions=str(data.get("instructions") or ""),
            path=path,
        )

    def _write_metadata(self, project: Project) -> None:
        data = {
            "id": project.id,
            "name": project.name,
            "created": project.created,
            "session_ids": project.session_ids,
            "instructions": project.instructions,
        }
        _atomic_write(project.path / "project.json", json.dumps(data, indent=2) + "\n")

    def _available_path(self, name: str, *, current: Optional[Path] = None) -> Path:
        stem = _slug(name)[:80]
        candidate = self.root / stem
        n = 2
        while candidate.exists() and candidate != current:
            candidate = self.root / f"{stem}-{n}"
            n += 1
        return candidate

    def create(self, name: str, *, purpose: str = "", instructions: str = "") -> Project:
        clean_name = " ".join((name or "").split())[:160] or "Untitled project"
        path = self._available_path(clean_name)
        path.mkdir(parents=False)
        (path / "files").mkdir()
        project = Project(
            id=str(uuid.uuid4()),
            name=clean_name,
            created=_now(),
            session_ids=[],
            instructions=(instructions or "").strip(),
            path=path,
        )
        self._write_metadata(project)
        _atomic_write(
            project.markdown_path,
            render_project_markdown(
                ProjectDocument(clean_name, (purpose or "").strip(), "", [], [])
            ),
        )
        return project

    def list(self) -> list[Project]:
        projects: list[Project] = []
        for path in self._folders():
            try:
                projects.append(self._load(path))
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
        return sorted(projects, key=lambda p: self.updated_at(p), reverse=True)

    def get(self, project_id: str) -> Optional[Project]:
        path = self._path_for_id(project_id)
        if path is None:
            return None
        try:
            return self._load(path)
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None

    def update(
        self,
        project_id: str,
        *,
        name: Optional[str] = None,
        purpose: Optional[str] = None,
        instructions: Optional[str] = None,
    ) -> Project:
        project = self.get(project_id)
        if project is None:
            raise KeyError(project_id)
        clean_name = project.name
        path = project.path
        if name is not None:
            clean_name = " ".join(name.split())[:160] or "Untitled project"
            target = self._available_path(clean_name, current=path)
            if target != path:
                path.rename(target)
                path = target
        document = parse_project_markdown(
            self.read_markdown(project_id) or "", fallback_name=clean_name
        )
        document.name = clean_name
        if purpose is not None:
            document.purpose = purpose.strip()
        updated = Project(
            project.id,
            clean_name,
            project.created,
            project.session_ids,
            project.instructions if instructions is None else instructions.strip(),
            path,
        )
        self._write_metadata(updated)
        _atomic_write(updated.markdown_path, render_project_markdown(document))
        return updated

    def delete(self, project_id: str) -> bool:
        project = self.get(project_id)
        if project is None:
            return False
        resolved = project.path.resolve()
        root = self.root.resolve()
        if resolved.parent != root:
            raise ValueError("project path escaped the projects directory")
        shutil.rmtree(resolved)
        return True

    def attach(self, project_id: str, session_id: str) -> Project:
        project = self.get(project_id)
        if project is None:
            raise KeyError(project_id)
        if session_id not in project.session_ids:
            project.session_ids.append(session_id)
            self._write_metadata(project)
        return project

    def detach(self, project_id: str, session_id: str) -> Project:
        project = self.get(project_id)
        if project is None:
            raise KeyError(project_id)
        if session_id in project.session_ids:
            project.session_ids.remove(session_id)
            self._write_metadata(project)
        return project

    def read_markdown(self, project_id: str) -> str:
        project = self.get(project_id)
        if project is None:
            raise KeyError(project_id)
        # The brief is user-editable, so it can be deleted outright — treat that as an
        # empty brief (the next refresh or edit recreates it) rather than a dead project.
        if not project.markdown_path.exists():
            return ""
        return project.markdown_path.read_text(encoding="utf-8")

    def write_document(self, project_id: str, document: ProjectDocument) -> None:
        project = self.get(project_id)
        if project is None:
            raise KeyError(project_id)
        _atomic_write(project.markdown_path, render_project_markdown(document))

    def updated_at(self, project: Project) -> str:
        stamps = [
            (project.path / name).stat().st_mtime
            if (project.path / name).exists()
            else 0.0
            for name in ("project.md", "project.json")
        ]
        return (
            datetime.fromtimestamp(max(stamps), timezone.utc)
            .isoformat()
            .replace("+00:00", "Z")
        )

    def files(self, project_id: str) -> list[dict[str, Any]]:
        project = self.get(project_id)
        if project is None:
            raise KeyError(project_id)
        out: list[dict[str, Any]] = []
        for path in sorted(p for p in project.files_path.rglob("*") if p.is_file()):
            stat = path.stat()
            out.append(
                {
                    "name": path.relative_to(project.files_path).as_posix(),
                    "size": stat.st_size,
                    "updated_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc)
                    .isoformat()
                    .replace("+00:00", "Z"),
                }
            )
        return out

    def prompt_block(self, project_id: str) -> str:
        project = self.get(project_id)
        if project is None:
            return ""
        markdown = project.markdown_path.read_text(encoding="utf-8").strip()
        instructions = project.instructions.strip()
        return (
            f"{_PROJECT_START}\n"
            "Project context for this session:\n"
            f"{markdown}\n\n"
            "Standing project instructions:\n"
            f"{instructions or '(none)'}\n"
            f"{_PROJECT_END}"
        )
