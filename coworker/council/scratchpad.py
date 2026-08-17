"""The panel's shared scratchpad — a blackboard, and a file Clinton can open.

Blackboard architecture is the oldest answer in multi-agent AI to the problem the council
has: members cannot talk to each other directly, and passing whole transcripts around
fragments what anyone actually learned. Instead each member posts one line worth keeping
(`NOTE:` in its answer), everything posted is visible to everyone in the next round, and
the accumulated notes are the panel's shared working state.

It is deliberately a plain markdown file rather than a database row. The point is that it
is inspectable: a council run leaves `scratchpad.md` and `transcript.md` on disk, so the
reasoning behind a decision is still there in a year when the session is long gone.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from ..secrets import state_dir

# One note per member per round. A model told to write "one line" sometimes writes six;
# the cap is what keeps the scratchpad a scratchpad instead of a second transcript.
MAX_NOTE_CHARS = 400
_NOTE = re.compile(r"^\s*NOTE\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
_SLUG = re.compile(r"[^a-z0-9]+")


def runs_dir() -> Path:
    return state_dir() / "council"


def live_path() -> Path:
    """Where a running council leaves its progress for the GUI to poll."""
    return runs_dir() / "live.json"


def read_live() -> dict[str, Any]:
    """The running council's progress, or `{}`. Never raises: this is polled every couple
    of seconds and a torn or missing file just means there is nothing to show yet."""
    import json

    try:
        return json.loads(live_path().read_text(encoding="utf-8")) or {}
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return {}


def slug(question: str, stamp: str) -> str:
    """A stable, readable directory name: the date plus the first few words."""
    words = _SLUG.sub("-", (question or "council").lower()).strip("-").split("-")
    tail = "-".join([w for w in words if w][:6]) or "council"
    return f"{stamp}-{tail}"[:80]


def extract_note(text: str) -> Optional[str]:
    """The `NOTE:` line from a member's answer, or None.

    Takes the LAST match: the prompt puts NOTE at the end, and an earlier hit is usually
    the model quoting the instruction back rather than answering it.
    """
    matches = _NOTE.findall(text or "")
    if not matches:
        return None
    note = matches[-1].strip()
    if not note or note.lower().rstrip(".") in ("none", "n/a", "nothing"):
        return None
    return note[:MAX_NOTE_CHARS]


class Scratchpad:
    """The blackboard for one council run. Append-only, rendered for prompts, saved to disk."""

    def __init__(self, question: str, *, directory: Optional[Path] = None) -> None:
        self.question = question
        self.entries: list[dict[str, str]] = []
        # Microseconds, not seconds: two councils on the same question within one second
        # would otherwise share a directory and silently overwrite each other's record.
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
        self.dir = directory or (runs_dir() / slug(question, stamp))

    def post(self, model: str, role: str, note: str, round_no: int) -> None:
        self.entries.append(
            {"model": model, "role": role, "note": note, "round": str(round_no)}
        )

    def collect(self, answers: list[dict[str, Any]], round_no: int) -> None:
        """Harvest every member's NOTE from one round."""
        for answer in answers:
            note = extract_note(answer.get("text") or "")
            if note:
                self.post(
                    answer.get("model", "?"), answer.get("role", ""), note, round_no
                )

    def render(self, alias: Optional[dict[str, str]] = None) -> str:
        """The scratchpad as members see it. Empty until someone posts.

        `alias` renames each model to its Member letter — the chair reads the notes with the
        same names withheld as the transcript, or the anonymity leaks straight back out here.
        """
        if not self.entries:
            return ""
        lines = []
        for e in self.entries:
            who = (alias or {}).get(e["model"], e["model"])
            lines.append(f"- [{who} · {e['role']}] {e['note']}" if e["role"] else f"- [{who}] {e['note']}")
        return "PANEL SCRATCHPAD (shared notes posted by members so far):\n" + "\n".join(lines)

    def to_markdown(self) -> str:
        out = [f"# Council scratchpad\n\n**Question:** {self.question}\n"]
        if not self.entries:
            out.append("_No notes were posted._\n")
        current = None
        for e in self.entries:
            if e["round"] != current:
                current = e["round"]
                out.append(f"\n## Round {current}\n")
            who = f"{e['model']} · {e['role']}" if e["role"] else e["model"]
            out.append(f"- **{who}** — {e['note']}")
        return "\n".join(out) + "\n"

    def publish(self, status: str, **extra: Any) -> None:
        """Write the run's current state where the GUI can read it.

        A council takes minutes and returns nothing until it is done, which reads as a hung
        app. The engine has no channel to the UI — it runs inside a tool call on a worker
        thread — so it leaves its progress in a file and the GUI polls it. Best-effort by
        design: losing a progress update must never cost the run.
        """
        import json

        payload = {
            "run": self.dir.name,
            "question": self.question,
            "status": status,
            "updated": datetime.now(timezone.utc).timestamp(),
            "notes": [dict(e) for e in self.entries],
            **extra,
        }
        try:
            path = live_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            # The temp name carries the run id. Two councils in different sessions writing
            # `live.json.tmp` at once would interleave into one file and then rename it, so
            # the reader gets a valid-looking mixture of two runs.
            tmp = path.with_name(f"{path.name}.{self.dir.name}.tmp")
            tmp.write_text(json.dumps(payload), encoding="utf-8")
            tmp.replace(path)  # atomic: a poll must never read half a file
        except OSError:
            pass

    def save(self, transcript: str, result: dict[str, Any]) -> dict[str, str]:
        """Write the run to disk. Best-effort: a read-only disk must not lose the answer
        that is already in memory and on its way back to the caller."""
        written: dict[str, str] = {}
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            for name, body in (
                ("scratchpad.md", self.to_markdown()),
                ("transcript.md", transcript),
                ("finding.md", _finding_markdown(self.question, result)),
            ):
                path = self.dir / name
                path.write_text(body, encoding="utf-8")
                written[name] = str(path)
        except OSError:
            return written
        return written


def _finding_markdown(question: str, result: dict[str, Any]) -> str:
    panel = ", ".join(
        f"{m['model']} ({m['role']})" if m.get("role") else str(m["model"])
        for m in (result.get("panel") or [])
    )
    lines = [
        f"# Council finding\n\n**Question:** {question}\n",
        f"**Panel:** {panel}\n",
        f"**Chair:** {result.get('chair', '')}\n",
    ]
    if result.get("skipped_debate"):
        lines.append("**Debate:** skipped — the panel already agreed after round 1.\n")
    if result.get("stopped_on_budget"):
        lines.append("**Debate:** stopped early — the run hit its token ceiling.\n")
    spend = result.get("spend")
    if spend:
        from .usage import summary_line

        lines.append(f"**Cost:** {summary_line(spend)}\n")
    report = result.get("report") or {}
    for note in report.get("notes") or []:
        lines.append(f"**Note:** {note}\n")
    lines.append("\n---\n\n" + (result.get("consensus") or result.get("error") or ""))

    # Sources last, as markdown links: the finding is the thing to read, and a reader who
    # wants to check a claim wants somewhere to click, not a bare list of hostnames.
    research = result.get("research") or {}
    results = research.get("results") or []
    if results:
        lines.append("\n\n---\n\n## Sources\n")
        queries = research.get("queries") or []
        if queries:
            lines.append("_Searched: " + "; ".join(f"`{q}`" for q in queries) + "_\n")
        for row in results:
            title = (row.get("title") or row.get("url") or "").strip()
            url = (row.get("url") or "").strip()
            lines.append(f"- [{title}]({url})" if url else f"- {title}")
    return "\n".join(lines) + "\n"
