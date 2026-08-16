"""Claude via the Claude Code CLI: argv construction, isolation, and usage reporting."""

from __future__ import annotations

import json
import subprocess

import pytest

from coworker.providers.claude_code_provider import ClaudeCodeProvider, _split
from coworker.providers.registry import build_provider_client, get_descriptor, provider_configured
from coworker.secrets import SecretStore


def _payload(result="ok", **usage):
    base = {"input_tokens": 2, "cache_creation_input_tokens": 12000, "cache_read_input_tokens": 0, "output_tokens": 4}
    return json.dumps(
        {"is_error": False, "result": result, "stop_reason": "end_turn",
         "total_cost_usd": 0.12, "usage": {**base, **usage}}
    )


@pytest.fixture
def run(monkeypatch):
    """Capture the argv the provider builds, and answer with a canned CLI payload."""
    calls: list = []

    def fake_run(argv, **kw):
        calls.append({"argv": argv, **kw})
        return subprocess.CompletedProcess(argv, 0, _payload(), "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr("shutil.which", lambda b: "/usr/bin/" + b)
    return calls


def test_a_turn_returns_text_and_usage(run):
    turn = ClaudeCodeProvider().complete(
        model="claude-opus-5", messages=[{"role": "user", "content": "hi"}]
    )
    assert turn.text == "ok"
    from coworker.council.usage import extract

    # Cache-creation tokens ARE the input on this transport — the harness is re-sent every
    # call. Reporting only `input_tokens: 2` would hide the number that actually matters.
    assert extract(turn.raw) == {"input": 12002, "output": 4, "reasoning": 0}


def test_the_prompt_is_argv_never_shell(run):
    """A question containing backticks or $() must be text, not a command."""
    hostile = "What does `rm -rf /` do; also $(whoami)?"
    ClaudeCodeProvider().complete(model="claude-opus-5", messages=[{"role": "user", "content": hostile}])
    assert run[0]["shell"] is False
    assert hostile in run[0]["argv"]  # one whole argv entry, unquoted and unparsed


def test_it_runs_in_an_empty_directory_not_the_users_project(run, tmp_path, monkeypatch):
    """The cwd is part of the prompt: Claude Code discovers CLAUDE.md, skills and settings
    from it, so a member run inside a repo argues from context the others never saw."""
    monkeypatch.setenv("COWORKER_STATE_DIR", str(tmp_path))
    ClaudeCodeProvider().complete(model="claude-opus-5", messages=[{"role": "user", "content": "hi"}])
    cwd = run[0]["cwd"]
    assert str(tmp_path) in str(cwd)
    assert not any(p.name == "CLAUDE.md" for p in __import__("pathlib").Path(cwd).iterdir())


def test_the_harness_is_stripped(run):
    ClaudeCodeProvider().complete(model="claude-opus-5", messages=[{"role": "user", "content": "hi"}])
    argv = run[0]["argv"]
    assert "--exclude-dynamic-system-prompt-sections" in argv
    assert "--disallowedTools" in argv and "Bash" in argv
    assert "--setting-sources" in argv
    # --bare would strip more, but it also skips keychain reads and so breaks the
    # subscription login entirely ("Not logged in"). Guard against a well-meaning re-add.
    assert "--bare" not in argv


def test_a_system_message_becomes_the_system_prompt(run):
    ClaudeCodeProvider().complete(
        model="claude-opus-5",
        messages=[{"role": "system", "content": "BE TERSE"}, {"role": "user", "content": "hi"}],
    )
    argv = run[0]["argv"]
    assert argv[argv.index("--system-prompt") + 1] == "BE TERSE"


def test_a_bare_two_turn_exchange_is_still_labelled():
    """Two messages with no system turn: labelling by raw message count missed this, so the
    model's own prior reply was indistinguishable from the user's question."""
    _system, prompt = _split(
        [{"role": "user", "content": "first"}, {"role": "assistant", "content": "reply"}]
    )
    assert "USER: first" in prompt and "ASSISTANT: reply" in prompt


def test_a_single_turn_carries_no_role_label():
    """The council case: one turn, so a label would just be noise in the prompt."""
    assert _split([{"role": "system", "content": "S"}, {"role": "user", "content": "Q"}]) == ("S", "Q")


def test_a_longer_history_is_flattened_with_roles_not_dropped():
    system, prompt = _split(
        [
            {"role": "system", "content": "S"},
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "reply"},
            {"role": "user", "content": "second"},
        ]
    )
    assert system == "S"
    assert "USER: first" in prompt and "ASSISTANT: reply" in prompt and "USER: second" in prompt


def test_tool_calls_are_refused_rather_than_silently_dropped(run):
    """The CLI answers in prose. Accepting a tools list would return a turn with no
    tool_calls and look to the engine like the model declined to use them."""
    with pytest.raises(RuntimeError, match="cannot make tool calls"):
        ClaudeCodeProvider().complete(
            model="claude-opus-5",
            messages=[{"role": "user", "content": "hi"}],
            tools=[{"type": "function", "function": {"name": "x"}}],
        )


def test_capabilities_declare_no_tools():
    caps = ClaudeCodeProvider().capabilities("claude-opus-5")
    assert caps.tools is False and caps.streaming is False


def test_a_missing_cli_is_a_clear_error(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda b: None)
    with pytest.raises(RuntimeError, match="not on PATH"):
        ClaudeCodeProvider().complete(model="claude-opus-5", messages=[{"role": "user", "content": "hi"}])


def test_a_cli_level_error_payload_is_raised(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda b: "/usr/bin/claude")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda argv, **kw: subprocess.CompletedProcess(
            argv, 0, json.dumps({"is_error": True, "result": "Not logged in · Please run /login"}), ""
        ),
    )
    with pytest.raises(RuntimeError, match="Not logged in"):
        ClaudeCodeProvider().complete(model="claude-opus-5", messages=[{"role": "user", "content": "hi"}])


def test_malformed_json_from_the_cli_is_a_clear_error(monkeypatch):
    """A CLI upgrade that changes the output shape must fail legibly, not with a
    JSONDecodeError traceback from inside a council member."""
    monkeypatch.setattr("shutil.which", lambda b: "/usr/bin/claude")
    monkeypatch.setattr(
        subprocess, "run",
        lambda argv, **kw: subprocess.CompletedProcess(argv, 0, "Welcome to Claude Code!", ""),
    )
    with pytest.raises(RuntimeError, match="non-JSON"):
        ClaudeCodeProvider().complete(model="claude-opus-5", messages=[{"role": "user", "content": "hi"}])


def test_a_nonzero_exit_reports_stderr(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda b: "/usr/bin/claude")
    monkeypatch.setattr(
        subprocess, "run",
        lambda argv, **kw: subprocess.CompletedProcess(argv, 1, "", "auth token expired"),
    )
    with pytest.raises(RuntimeError, match="auth token expired"):
        ClaudeCodeProvider().complete(model="claude-opus-5", messages=[{"role": "user", "content": "hi"}])


def test_an_empty_but_successful_response_yields_no_text(monkeypatch):
    """The council treats a text-less turn as a failed member and drops it from later
    rounds — so this must return None, not an empty string that reads as an answer."""
    monkeypatch.setattr("shutil.which", lambda b: "/usr/bin/claude")
    monkeypatch.setattr(
        subprocess, "run",
        lambda argv, **kw: subprocess.CompletedProcess(argv, 0, _payload(result="   "), ""),
    )
    turn = ClaudeCodeProvider().complete(model="claude-opus-5", messages=[{"role": "user", "content": "hi"}])
    assert turn.text is None


def test_a_response_with_no_usage_block_does_not_crash(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda b: "/usr/bin/claude")
    monkeypatch.setattr(
        subprocess, "run",
        lambda argv, **kw: subprocess.CompletedProcess(
            argv, 0, json.dumps({"is_error": False, "result": "ok"}), ""
        ),
    )
    turn = ClaudeCodeProvider().complete(model="claude-opus-5", messages=[{"role": "user", "content": "hi"}])
    from coworker.council.usage import extract

    assert turn.text == "ok" and extract(turn.raw) is None


def test_a_timeout_is_reported_not_swallowed(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda b: "/usr/bin/claude")

    def boom(argv, **kw):
        raise subprocess.TimeoutExpired(argv, 1)

    monkeypatch.setattr(subprocess, "run", boom)
    with pytest.raises(RuntimeError, match="timed out"):
        ClaudeCodeProvider().complete(model="claude-opus-5", messages=[{"role": "user", "content": "hi"}])


# -- registry wiring -------------------------------------------------------------------


def test_the_provider_is_keyless_and_needs_no_form():
    d = get_descriptor("claude-code")
    assert d.needs_key is False and d.fields == []
    assert d.recommended_model == "claude-opus-5"


def test_keyless_is_not_the_same_as_available(tmp_path, monkeypatch):
    """Every other keyless provider is configured by definition. This one is only usable if
    the CLI is actually installed, so claiming otherwise puts a dead member on the panel."""
    store = SecretStore(tmp_path / "secrets.json")
    monkeypatch.setattr("shutil.which", lambda b: None)
    assert provider_configured("claude-code", store) is False
    monkeypatch.setattr("shutil.which", lambda b: "/usr/bin/claude")
    assert provider_configured("claude-code", store) is True


def test_the_registry_builds_it(tmp_path):
    client = build_provider_client("claude-code", {}, SecretStore(tmp_path / "s.json"))
    assert isinstance(client, ClaudeCodeProvider)


def test_a_custom_binary_path_is_honoured(tmp_path):
    client = build_provider_client(
        "claude-code", {"binary": "/opt/claude"}, SecretStore(tmp_path / "s.json")
    )
    assert client._binary == "/opt/claude"


def test_subscription_models_price_at_zero_not_unknown():
    """The subscription is not per-token metered, so zero is the truth for the bill — and
    it must not be reported as an unpriced gap."""
    from coworker.council.usage import total

    report = total([[{"model": "claude-code:claude-opus-5", "usage": {"input": 12000, "output": 100}}]])
    assert report["fully_priced"] is True and report["usd_estimate"] == 0.0
