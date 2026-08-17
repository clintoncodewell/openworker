"""Fast text search — ripgrep when available, a Python walk otherwise.

ripgrep respects `.gitignore`, so it skips `node_modules`/`target`/`dist` automatically; the
fallback skips a hardcoded set of heavy dirs. Read-only and root-scoped. Returns file:line:text.
"""

from __future__ import annotations

import fnmatch
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Optional

import aisuite as ai

_IGNORE_DIRS = {
    ".git",
    "node_modules",
    "target",
    "dist",
    "build",
    ".venv",
    "venv",
    "__pycache__",
    ".next",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".idea",
}

_SCHEMA = {
    "type": "function",
    "function": {
        "name": "grep",
        "description": (
            "Search the workspace for a regular-expression pattern and return matching lines as "
            "file:line:text. Fast and .gitignore-aware (skips node_modules, build dirs, etc.). "
            "Prefer this over reading files blindly to locate code. Read-only."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regular expression to search for.",
                },
                "path": {
                    "type": "string",
                    "description": "Subdirectory to search (default: whole workspace).",
                },
                "glob": {
                    "type": "string",
                    "description": "Optional filename glob filter, e.g. '*.py'.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Max matches (default 100, max 1000).",
                },
            },
            "required": ["pattern"],
        },
    },
}

_KNOWLEDGE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_knowledge_folder",
        "description": (
            "Search the user's personal Knowledge folder for a regular-expression pattern and "
            "return matching lines as file:line:text. This searches the globally configured "
            "Knowledge folder, NOT the current session workspace. Nothing is loaded unless you "
            "search for it. Read-only."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regular expression to search for in the Knowledge folder.",
                },
                "path": {
                    "type": "string",
                    "description": (
                        "Subdirectory within the Knowledge folder (default: whole folder)."
                    ),
                },
                "glob": {
                    "type": "string",
                    "description": "Optional filename glob filter, e.g. '*.md'.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Max matches (default 100, max 1000).",
                },
            },
            "required": ["pattern"],
        },
    },
}


def search_tools(workspace: str) -> list:
    return _search_tools(Path(workspace).resolve(), _SCHEMA, "workspace")


def knowledge_search_tools(folder: str) -> list:
    """Build the read-only search tool for the globally configured Knowledge folder."""
    return _search_tools(Path(folder).resolve(), _KNOWLEDGE_SCHEMA, "Knowledge folder")


def _search_tools(root: Path, schema: dict[str, Any], scope: str) -> list:
    def grep(
        pattern: str,
        path: str = ".",
        glob: Optional[str] = None,
        max_results: int = 100,
    ) -> dict[str, Any]:
        n = max_results if isinstance(max_results, int) and max_results > 0 else 100
        n = min(n, 1000)
        base = (root / (path or ".")).resolve()
        try:
            base.relative_to(root)  # keep searches inside the configured root
        except ValueError:
            return {"error": f"path escapes the {scope}"}

        rg = shutil.which("rg")
        if rg:
            cmd = [
                rg,
                "--line-number",
                "--no-heading",
                "--color=never",
                "--max-count",
                str(n),
                "-e",
                pattern,
            ]
            if glob:
                cmd += ["--glob", glob]
            # Do not rely solely on a workspace's .gitignore: the Python fallback
            # always omits these generated/dependency directories too. Exclusions come
            # last because ripgrep resolves conflicting globs with the later one winning.
            for ignored in sorted(_IGNORE_DIRS):
                cmd += ["--glob", f"!**/{ignored}/**"]
            cmd.append(str(base))
            try:
                out = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            except Exception as exc:
                return {"error": f"grep failed: {exc}"}
            if out.returncode not in (0, 1):  # 1 = no matches
                return {"error": (out.stderr or "ripgrep error").strip()[:300]}
            return {"engine": "ripgrep", **_parse_rg(out.stdout, root, n)}

        return {"engine": "python", **_py_grep(root, base, pattern, glob, n)}

    tool_name = schema["function"]["name"]
    grep.__name__ = tool_name
    grep.__doc__ = schema["function"]["description"]
    grep.__aisuite_tool_metadata__ = ai.ToolMetadata(
        name=tool_name,
        category="search",
        risk_level="low",
        capabilities=["search"],
        requires_approval=False,
    )
    grep.__coworker_schema__ = schema
    return [grep]


def _rel(path: str, root: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(root))
    except (ValueError, OSError):
        return path


def _parse_rg(stdout: str, root: Path, n: int) -> dict[str, Any]:
    matches: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        parts = line.split(":", 2)
        if len(parts) == 3:
            f, ln, txt = parts
            try:
                rel = str(Path(f).resolve().relative_to(root))
            except (ValueError, OSError):
                # rg followed a symlink out of the search root — drop the match rather
                # than leak an absolute out-of-root path (the entry-point escape check
                # only guards the search root, not what rg then traverses into).
                continue
            matches.append(
                {
                    "file": rel,
                    "line": int(ln) if ln.isdigit() else 0,
                    "text": txt[:300],
                }
            )
        if len(matches) >= n:
            break
    return {"count": len(matches), "matches": matches}


def _py_grep(
    root: Path, base: Path, pattern: str, glob: Optional[str], n: int
) -> dict[str, Any]:
    try:
        rx = re.compile(pattern)
    except re.error as exc:
        return {"error": f"invalid regex: {exc}", "count": 0, "matches": []}
    matches: list[dict[str, Any]] = []
    for dirpath, dirs, files in os.walk(base):
        dirs[:] = [d for d in dirs if d not in _IGNORE_DIRS]
        for fn in files:
            if glob and not fnmatch.fnmatch(fn, glob):
                continue
            fp = Path(dirpath) / fn
            try:
                with open(fp, "r", encoding="utf-8", errors="ignore") as fh:
                    for i, line in enumerate(fh, 1):
                        if rx.search(line):
                            matches.append(
                                {
                                    "file": _rel(str(fp), root),
                                    "line": i,
                                    "text": line.rstrip()[:300],
                                }
                            )
                            if len(matches) >= n:
                                return {"count": len(matches), "matches": matches}
            except OSError:
                continue
    return {"count": len(matches), "matches": matches}
