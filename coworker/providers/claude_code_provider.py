"""Claude models through the Claude Code CLI, on the subscription — no API key.

`claude -p` runs one headless turn using the OAuth credentials Claude Code already holds
(`~/.claude/.credentials.json`). That means Claude on the plan you already pay for, with
no `sk-ant-` key to store, rotate, or leak. It is Claude Code being used as Claude Code:
your machine, your login, your allowance.

**Text only, deliberately.** The CLI returns prose, not `tool_use` blocks, so this provider
declares `tools=False` and is for panel members and analysis — not for driving OpenWorker's
own agent loop. Pick a real API provider as the session model; use this one on the council.

**The overhead is the catch, and it is large.** Every `claude -p` call re-sends Claude
Code's own harness — system prompt, tool schemas, skills. Measured 2026-08-16 on this box:

    plain `claude -p "reply ok"`            ~36,000 prompt tokens   ($0.37 at list)
    stripped (this provider's invocation)   ~12,000 prompt tokens   ($0.12 at list)
    the same question via the API             ~93 prompt tokens     (~$0.0001)

So this is not the cheap option — it is the *no-API-key* option. Those tokens come out of
your Claude Code rate limits, the same pool your coding sessions draw on, which is the real
budget to watch. `usd` in the reported usage is what the tokens would have cost at list
rates; on a subscription nothing is charged, so treat it as a size, not a bill.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
from typing import Any, Optional

from .base import AssistantTurn, ModelCapabilities, ProviderClient

DEFAULT_TIMEOUT_S = 300.0


# macOS starts a GUI app with a minimal PATH (/usr/bin:/bin:/usr/sbin:/sbin) — the login
# shell's PATH is never applied. So the sidecar inside OpenWorker.app cannot see a CLI in
# `~/.local/bin`, which is exactly where Claude Code installs itself. Measured on the Mac
# 2026-08-17: `which claude` resolves in a terminal and returns nothing in the app, so the
# council silently convened WITHOUT its Claude member while Settings showed it connected.
# Checking the known install locations is what keeps the desktop panel identical to the one
# a terminal-launched build resolves.
_KNOWN_PATHS = (
    "~/.local/bin/claude",
    "~/.claude/local/claude",
    "/opt/homebrew/bin/claude",
    "/usr/local/bin/claude",
)


def _safe_executable(path: str) -> bool:
    """A real, executable file that only its owner can rewrite.

    PATH entries are chosen by the OS; these locations are not, so a hit here was never
    something the system decided to trust. Two things follow. A directory passes
    `os.access(X_OK)` — searchable, not runnable — so the file type is checked explicitly.
    And a candidate any other account can write to is refused: without that, dropping a file
    in a home directory would be enough to have this app run it.
    """
    try:
        st = os.stat(path)  # follows symlinks on purpose: the target is what executes
    except OSError:
        return False
    if not stat.S_ISREG(st.st_mode) or not os.access(path, os.X_OK):
        return False
    getuid = getattr(os, "getuid", None)
    if getuid is None:
        return True  # Windows has no POSIX ownership; PATH is the only source there anyway
    if st.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        return False
    return st.st_uid in (0, getuid())


def resolve_binary(binary: str = "claude") -> Optional[str]:
    """The CLI's absolute path, or None. PATH first, then the known install locations."""
    found = shutil.which(binary)
    if found:
        # A relative hit would otherwise be resolved against the scratch cwd we run in,
        # which is not the directory the caller measured it from.
        return os.path.abspath(found)
    # An explicit path that does not exist is a mistake to report, not one to guess around.
    if binary != "claude":
        return None
    for candidate in _KNOWN_PATHS:
        path = os.path.expanduser(candidate)
        if _safe_executable(path):
            return path
    return None


def _scratch_dir():
    """An empty directory to run the CLI in, so no project context is discovered."""
    from ..secrets import state_dir

    path = state_dir() / "claude-code-scratch"
    path.mkdir(parents=True, exist_ok=True)
    return path

# Everything Claude Code loads for interactive coding and a council member never uses.
# Dropping them is what takes the per-call overhead from ~36k tokens to ~12k.
_UNUSED_TOOLS = (
    "Bash",
    "Read",
    "Write",
    "Edit",
    "Glob",
    "Grep",
    "WebFetch",
    "WebSearch",
    "Task",
    "TodoWrite",
    "NotebookEdit",
    "Skill",
    "Agent",
)


def _split(messages: list[dict[str, Any]]) -> tuple[str, str]:
    """(system, prompt) from a message list.

    The CLI takes one system prompt and one user turn, so a multi-turn history is flattened
    with role labels rather than dropped — losing earlier turns would silently change the
    question. Council members are single-turn anyway; this is the general case.
    """
    from ..attachments import content_to_text

    systems, body = [], []
    for message in messages:
        text = content_to_text(message.get("content"), image_placeholder="") or ""
        if not text.strip():
            continue
        (systems if message.get("role") == "system" else body).append(
            (message.get("role", "user"), text)
        )
    # Label by how many NON-SYSTEM turns there are, not by the raw message count: a bare
    # [user, assistant] pair is two messages, and concatenating those unlabelled makes the
    # model's own prior reply indistinguishable from the user's question.
    if len(body) > 1:
        turns = [f"{role.upper()}: {text}" for role, text in body]
    else:
        turns = [text for _role, text in body]
    return "\n\n".join(t for _r, t in systems), "\n\n".join(turns)


class ClaudeCodeProvider(ProviderClient):
    """One `claude -p` turn per `complete()`."""

    def __init__(
        self,
        *,
        binary: str = "claude",
        timeout: float = DEFAULT_TIMEOUT_S,
        cwd: Optional[str] = None,
    ) -> None:
        self._binary = binary
        self._timeout = timeout
        # Claude Code discovers CLAUDE.md, skills and settings from its working directory,
        # so the cwd is part of the prompt. Run in a dedicated empty directory: otherwise a
        # council member silently argues from whichever repo the app happens to be open on,
        # while the other members do not — which breaks the panel's core property that
        # everyone reasons from the same brief.
        #
        # `--bare` would also disable global CLAUDE.md and auto-memory, but it skips
        # keychain reads too, which is exactly where the subscription credentials live
        # ("Not logged in · Please run /login"). So it cannot be used here; an empty cwd is
        # the part of the isolation that is compatible with subscription auth.
        self._cwd = cwd or str(_scratch_dir())

    def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]] = None,
        **settings: Any,
    ) -> AssistantTurn:
        if tools:
            raise RuntimeError(
                "The Claude Code provider cannot make tool calls — the CLI returns prose, "
                "not tool_use blocks. Use it for council members and analysis, and pick an "
                "API-keyed provider as the session model."
            )
        binary = resolve_binary(self._binary)
        if binary is None:
            raise RuntimeError(
                f"`{self._binary}` is not on PATH. Install Claude Code and sign in with "
                "`claude` once; this provider reuses that login."
            )

        system, prompt = _split(messages)
        argv = [
            binary,
            "-p",
            prompt,
            "--model",
            model,
            "--output-format",
            "json",
            # A council member is given its whole brief in the prompt; Claude Code's own
            # coding-agent preamble is only noise (and tokens) here.
            "--system-prompt",
            system or "You answer the question you are given, directly.",
            "--exclude-dynamic-system-prompt-sections",
            "--setting-sources",
            "",  # ignore user/project settings — a council answer should be reproducible
            "--disallowedTools",
            *_UNUSED_TOOLS,
        ]
        try:
            proc = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                timeout=self._timeout,
                cwd=self._cwd,
                # The prompt is an argv entry, never shell source: no shell, no quoting to
                # get wrong, and a question containing backticks or $() is just text.
                shell=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"claude -p timed out after {self._timeout:.0f}s"
            ) from exc

        if proc.returncode != 0:
            raise RuntimeError(
                f"claude -p failed ({proc.returncode}): "
                f"{(proc.stderr or proc.stdout or '').strip()[:300]}"
            )
        try:
            payload = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"claude -p returned non-JSON output: {proc.stdout[:200]}"
            ) from exc
        if payload.get("is_error"):
            raise RuntimeError(f"claude -p reported an error: {payload.get('result')}")

        return AssistantTurn(
            text=(payload.get("result") or "").strip() or None,
            finish_reason=payload.get("stop_reason"),
            raw=_Usage(payload),
        )

    def capabilities(self, model: str) -> ModelCapabilities:
        # tools=False is the load-bearing flag: it stops the engine offering this provider
        # a tool loop it cannot serve.
        return ModelCapabilities(
            tools=False, vision=False, pdf=False, parallel_tool_calls=False, streaming=False
        )


class _Usage:
    """Adapts Claude Code's usage JSON to the attribute shape `council.usage` reads.

    Cache-creation tokens are folded into `input_tokens` because on this transport they
    ARE the input: the harness is re-sent on every call, and a report that showed 2 input
    tokens for a 12,000-token prompt would hide the only number that matters here.
    """

    def __init__(self, payload: dict[str, Any]) -> None:
        raw = payload.get("usage") or {}
        self.prompt_tokens = (
            int(raw.get("input_tokens") or 0)
            + int(raw.get("cache_creation_input_tokens") or 0)
            + int(raw.get("cache_read_input_tokens") or 0)
        )
        self.completion_tokens = int(raw.get("output_tokens") or 0)
        # What the same tokens would have cost on the API. On a subscription nothing is
        # charged — kept because it is the only comparable size the CLI reports.
        self.list_cost_usd = payload.get("total_cost_usd")

    @property
    def usage(self) -> "_Usage":  # `usage.extract` looks for `raw.usage`
        return self
