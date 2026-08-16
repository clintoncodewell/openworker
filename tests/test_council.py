"""Council: panel resolution, the debate rounds, and failure isolation."""

from __future__ import annotations

import pytest

from coworker.council import default_panel, make_council_tool, run_council
from coworker.providers.registry import provider_descriptors
from coworker.providers.base import AssistantTurn
from coworker.secrets import SecretStore


class FakeProvider:
    """Records every call; answers with the model name so rounds are distinguishable."""

    def __init__(self, fail: set[str] | None = None, blank: set[str] | None = None):
        self.calls: list[tuple[str, str]] = []  # (model, user prompt)
        self.fail = fail or set()
        self.blank = blank or set()

    def complete(self, *, model, messages, tools=None, **settings):
        self.calls.append((model, messages[-1]["content"]))
        if model in self.fail:
            raise RuntimeError("no key")
        if model in self.blank:
            return AssistantTurn(text="")
        return AssistantTurn(text=f"{model} says POSITION: yes")

    def capabilities(self, model):  # pragma: no cover — unused by the council
        raise NotImplementedError


def run(provider, models, **kw):
    kw.setdefault("research", False)
    return run_council(
        "Should we ship on Friday?",
        provider=provider,
        models=models,
        chair_model="chair-model",
        **kw,
    )


def test_two_rounds_then_chair():
    p = FakeProvider()
    out = run(p, ["a:one", "b:two"])
    # 2 members x 2 rounds + 1 chair
    assert len(p.calls) == 5
    assert [m for m, _ in p.calls[-1:]] == ["chair-model"]
    assert len(out["rounds"]) == 2
    assert out["consensus"].endswith("chair-model says POSITION: yes")
    # Every member carries the lens it was told to argue — identical members are the
    # documented way to waste a debate, so the roles are part of the contract.
    assert out["panel"] == [
        {"model": "a:one", "role": "Advocate"},
        {"model": "b:two", "role": "Skeptic"},
    ]
    assert out["failures"] == []


def test_round_one_members_cannot_see_each_other():
    p = FakeProvider()
    run(p, ["a:one", "b:two"], rounds=1)
    openings = [prompt for model, prompt in p.calls if model != "chair-model"]
    assert all("a:one says" not in o and "b:two says" not in o for o in openings)


def test_debate_round_shows_the_previous_transcript():
    p = FakeProvider()
    run(p, ["a:one", "b:two"])
    second_round = [prompt for model, prompt in p.calls[2:4]]
    assert all("a:one says" in prompt and "b:two says" in prompt for prompt in second_round)


def test_rounds_one_skips_the_debate():
    p = FakeProvider()
    out = run(p, ["a:one", "b:two"], rounds=1)
    assert len(p.calls) == 3  # 2 members + chair
    assert len(out["rounds"]) == 1


def test_a_dead_member_is_reported_and_dropped():
    p = FakeProvider(fail={"b:two"})
    out = run(p, ["a:one", "b:two"])
    assert [(f["model"], f["error"]) for f in out["failures"]] == [
        ("b:two", "RuntimeError: no key")
    ]
    # Round 2 needs two live members, so with one left the debate stops early.
    assert len(out["rounds"]) == 1
    assert out["consensus"].endswith("chair-model says POSITION: yes")


def test_an_empty_answer_counts_as_a_failure():
    p = FakeProvider(blank={"b:two"})
    out = run(p, ["a:one", "b:two", "c:three"])
    assert ("b:two", "empty response") in [
        (f["model"], f["error"]) for f in out["failures"]
    ]
    assert [a["model"] for a in out["rounds"][1]] == ["a:one", "c:three"]


def test_chair_failure_does_not_raise():
    p = FakeProvider(fail={"chair-model"})
    out = run(p, ["a:one", "b:two"])
    assert out["consensus"].startswith("chair failed:")


def test_every_member_failing_reports_an_error_instead_of_a_verdict():
    """With an empty transcript the chair would happily invent a consensus that no member
    ever gave, and it would read exactly like a real one. Refuse instead."""
    p = FakeProvider(fail={"a:one", "b:two"})
    out = run(p, ["a:one", "b:two"])
    assert "error" in out and "consensus" not in out
    assert "chair-model" not in [m for m, _ in p.calls]  # chair was never asked
    assert {f["model"] for f in out["failures"]} == {"a:one", "b:two"}


def test_council_is_serial_not_parallel_safe():
    """The engine runs risk_level="low" tools concurrently. The council must not be one:
    five concurrent panels is up to 125 paid completions in flight."""
    tool = make_council_tool(provider=FakeProvider(), chair_model="c", panel=lambda: [])
    assert tool.__aisuite_tool_metadata__.risk_level != "low"


def test_no_panel_is_an_error_not_a_call():
    p = FakeProvider()
    out = run(p, [])
    assert "error" in out and p.calls == []


def test_duplicate_models_collapse():
    p = FakeProvider()
    out = run(p, ["a:one", "a:one"], rounds=1)
    assert [m["model"] for m in out["panel"]] == ["a:one"]


def test_rounds_are_clamped():
    p = FakeProvider()
    out = run(p, ["a:one", "b:two"], rounds=99)
    assert len(out["rounds"]) == 3  # MAX_ROUNDS


@pytest.fixture
def no_provider_env(monkeypatch):
    """Neutralise every machine-dependent way a provider can read as configured.

    Two of them: each descriptor's `env_key` (a dev box with XAI_API_KEY exported puts an
    extra model on the panel), and the keyless CLI providers, which are configured iff
    their binary is installed — so these tests would pass or fail depending on whether
    Claude Code happens to be on the machine."""
    from coworker.providers import claude_code_provider as ccp

    for d in provider_descriptors():
        if d.env_key:
            monkeypatch.delenv(d.env_key, raising=False)
    monkeypatch.setattr("shutil.which", lambda _binary: None)
    # PATH is not the only place a CLI is looked for any more — a GUI app on macOS gets a
    # minimal PATH, so the resolver also checks the known install locations.
    monkeypatch.setattr(ccp, "_KNOWN_PATHS", ())


def test_default_panel_takes_one_model_per_configured_provider(tmp_path, no_provider_env):
    store = SecretStore(tmp_path / "secrets.json")
    store.put("provider:azure", {"api_key": "k"})
    store.put("provider:gemini", {"api_key": "k"})
    store.put("provider:ollama", {"base_url": "http://localhost:11434"})
    store.put("provider:xai", {})  # present but no key

    # Descriptor order, not insertion order: gemini is declared before azure.
    panel = default_panel(store)
    assert panel == ["gemini:gemini-3.6-flash", "azure:gpt-5.6-sol"]


def test_default_panel_honours_an_env_key(tmp_path, no_provider_env, monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "k")
    panel = default_panel(SecretStore(tmp_path / "secrets.json"))
    assert panel == ["mistral:mistral-large-latest"]


def test_default_panel_includes_a_signed_in_oauth_provider(tmp_path, no_provider_env):
    """ChatGPT is keyless (needs_key=False) but IS configured once signed in. An earlier
    `not d.needs_key: continue` silently dropped it from every panel."""
    store = SecretStore(tmp_path / "secrets.json")
    store.put("provider:chatgpt", {"access_token": "t"})
    assert default_panel(store) == ["chatgpt:gpt-5.4-mini"]


def test_default_panel_skips_ollama_even_though_it_is_keyless(tmp_path, no_provider_env):
    """Keyless means "no key needed", not "running" — a dead localhost would stall the
    whole panel behind its timeout."""
    store = SecretStore(tmp_path / "secrets.json")
    store.put("provider:ollama", {"base_url": "http://localhost:11434"})
    assert default_panel(store) == []


def test_tool_wraps_the_default_panel(tmp_path):
    p = FakeProvider()
    tool = make_council_tool(
        provider=p,
        chair_model="chair-model",
        secrets=SecretStore(tmp_path / "secrets.json"),
        panel=lambda: ["a:one"],
    )
    out = tool(question="ship?", rounds=1, research=False)
    assert [m["model"] for m in out["panel"]] == ["a:one"]
    assert tool.__name__ == "council"
    assert tool.__coworker_schema__["function"]["name"] == "council"


def test_research_failure_is_reported_not_raised(monkeypatch):
    import coworker.web.tool as web_tool

    def boom(secrets=None, **kw):
        raise RuntimeError("network down")

    monkeypatch.setattr(web_tool, "resolve_provider", boom)
    p = FakeProvider()
    out = run_council(
        "ship?",
        provider=p,
        models=["a:one"],
        chair_model="chair-model",
        rounds=1,
        research=True,
    )
    assert out["research"]["ok"] is False
    assert out["consensus"].endswith("chair-model says POSITION: yes")


def test_research_results_reach_every_member(monkeypatch):
    import coworker.web.tool as web_tool

    class Result:
        def to_dict(self):
            return {"title": "T", "url": "http://x", "snippet": "SNIP"}

    class P:
        name = "fake"

        def search(self, q, max_results=5):
            return [Result()]

    monkeypatch.setattr(web_tool, "resolve_provider", lambda secrets=None, **kw: P())
    p = FakeProvider()
    out = run_council(
        "ship?",
        provider=p,
        models=["a:one", "b:two"],
        chair_model="chair-model",
        rounds=1,
        research=True,
    )
    assert out["research"]["provider"] == "fake"
    assert all("SNIP" in prompt for _, prompt in p.calls)


def test_consensus_is_labelled_untrusted():
    """The consensus lands in an agent that HAS shell and write tools, and is built from
    model output and web snippets. It must arrive labelled as data, not instructions."""
    from coworker.council.core import _UNTRUSTED

    out = run(FakeProvider(), ["a:one"], rounds=1)
    assert out["consensus"].startswith(_UNTRUSTED)
    assert "not instructions" in _UNTRUSTED


@pytest.mark.parametrize("question", ["", "   "])
def test_blank_question_is_rejected(question):
    p = FakeProvider()
    out = run_council(
        question, provider=p, models=["a:one"], chair_model="c", research=False
    )
    assert "error" in out and p.calls == []
