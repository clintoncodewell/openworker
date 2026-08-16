"""Spend measurement and the per-run token ceiling."""

from __future__ import annotations

import pytest

from coworker.council import CouncilConfig, run_council
from coworker.council import usage as usage_mod
from coworker.providers.base import AssistantTurn


class _Usage:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


class _Raw:
    def __init__(self, usage=None, usage_metadata=None):
        if usage is not None:
            self.usage = usage
        if usage_metadata is not None:
            self.usage_metadata = usage_metadata


# -- extraction across the three provider shapes ---------------------------------------


def test_openai_shape():
    raw = _Raw(usage=_Usage(prompt_tokens=100, completion_tokens=40))
    assert usage_mod.extract(raw) == {"input": 100, "output": 40, "reasoning": 0}


def test_anthropic_shape():
    raw = _Raw(usage=_Usage(input_tokens=100, output_tokens=40))
    assert usage_mod.extract(raw) == {"input": 100, "output": 40, "reasoning": 0}


def test_gemini_shape_counts_hidden_thinking_as_output():
    """Measured: a Gemini call answering "Ready?" in one word billed 151 thinking tokens
    against 9 of prompt. The visible answer says nothing about the bill."""
    raw = _Raw(
        usage_metadata=_Usage(
            prompt_token_count=9, candidates_token_count=2, thoughts_token_count=151
        )
    )
    assert usage_mod.extract(raw) == {"input": 9, "output": 153, "reasoning": 151}


def test_openai_reasoning_is_a_breakdown_of_completion_not_an_addition():
    """Measured 2026-08-16: `prompt + completion == total` on the OpenAI-compatible shape,
    so `completion_tokens_details.reasoning_tokens` is already inside `completion_tokens`.
    Adding it inflated every cost and could stop a debate early on spend that never
    happened."""
    raw = _Raw(
        usage=_Usage(
            prompt_tokens=100,
            completion_tokens=540,  # already includes the 500 reasoning tokens
            completion_tokens_details=_Usage(reasoning_tokens=500),
        )
    )
    # 540, NOT 1040. `reasoning` is still reported so the hidden spend stays visible.
    assert usage_mod.extract(raw) == {"input": 100, "output": 540, "reasoning": 500}


def test_the_two_families_treat_reasoning_oppositely():
    """The asymmetry is the whole point: applying either rule globally is wrong."""
    openai = usage_mod.extract(
        _Raw(usage=_Usage(prompt_tokens=1, completion_tokens=100,
                          completion_tokens_details=_Usage(reasoning_tokens=90)))
    )
    gemini = usage_mod.extract(
        _Raw(usage_metadata=_Usage(prompt_token_count=1, candidates_token_count=10,
                                   thoughts_token_count=90))
    )
    assert openai["output"] == 100  # reasoning already inside
    assert gemini["output"] == 100  # 10 + 90, reasoning added


@pytest.mark.parametrize("raw", [_Raw(), object(), None])
def test_a_provider_reporting_nothing_is_not_an_error(raw):
    assert usage_mod.extract(raw) is None


# -- pricing ----------------------------------------------------------------------------


def test_the_longest_matching_prefix_wins():
    """gemini-2.5-pro is ~4x gemini-2.5-flash, so a short prefix would understate it."""
    assert usage_mod.price_for("gemini:gemini-2.5-pro") == (1.25, 10.00)
    assert usage_mod.price_for("gemini:gemini-2.5-flash") == (0.30, 2.50)


def test_an_unknown_model_prices_at_zero_rather_than_guessing():
    assert usage_mod.price_for("brandnew:model-x") == (0.0, 0.0)


def test_cost_is_per_million_tokens():
    assert usage_mod.cost("azure:gpt-5.6-sol", {"input": 1_000_000, "output": 0}) == 1.25
    assert usage_mod.cost("azure:gpt-5.6-sol", {"input": 0, "output": 1_000_000}) == 10.0


# -- aggregation ------------------------------------------------------------------------


def _call(model, i, o):
    return {"model": model, "usage": {"input": i, "output": o, "reasoning": 0}}


def test_totals_roll_up_rounds_and_the_chair():
    rounds = [
        [_call("azure:gpt-5.6-sol", 1000, 100), _call("gemini:gemini-3.6-flash", 1000, 100)],
        [_call("azure:gpt-5.6-sol", 3000, 100), _call("gemini:gemini-3.6-flash", 3000, 100)],
    ]
    chair = _call("azure:gpt-5.6-sol", 5000, 400)
    report = usage_mod.total(rounds, chair)
    assert report["calls"] == 5
    assert report["input_tokens"] == 13_000
    assert report["output_tokens"] == 800
    assert report["by_model"]["azure:gpt-5.6-sol"]["calls"] == 3
    assert report["usd_estimate"] > 0


def test_an_unpriced_model_is_flagged_but_the_tokens_are_still_exact():
    report = usage_mod.total([[_call("brandnew:model-x", 1000, 100)]])
    assert report["fully_priced"] is False
    assert report["total_tokens"] == 1100


def test_a_genuinely_free_model_does_not_flag_as_unpriced():
    """A flat-rate plan and an unknown vendor both price at zero; only one is a gap."""
    report = usage_mod.total([[_call("zai-coding:glm-4.6", 1000, 100)]])
    assert report["fully_priced"] is True
    assert report["usd_estimate"] == 0.0


def test_a_failed_member_contributes_no_usage():
    report = usage_mod.total([[{"model": "a:one", "error": "no key"}]])
    assert report["calls"] == 0 and report["total_tokens"] == 0


def test_the_summary_line_names_hidden_reasoning():
    report = usage_mod.total(
        [[{"model": "gemini:gemini-3.6-flash", "usage": {"input": 10, "output": 200, "reasoning": 150}}]]
    )
    line = usage_mod.summary_line(report)
    assert "hidden reasoning" in line and "150" in line


# -- the ceiling -------------------------------------------------------------------------


class _BigProvider:
    """Every call reports a fixed, large token spend."""

    def __init__(self, per_call=100_000):
        self.calls: list[str] = []
        self.per_call = per_call

    def complete(self, *, model, messages, tools=None, **settings):
        self.calls.append(model)
        return AssistantTurn(
            text="STANCE: ship it\nCONFIDENCE: 0.5",
            raw=_Raw(usage=_Usage(prompt_tokens=self.per_call, completion_tokens=0)),
        )

    def capabilities(self, model):  # pragma: no cover
        raise NotImplementedError


def _run(provider, cfg, **kw):
    return run_council(
        "ship?",
        provider=provider,
        models=["a:one", "b:two"],
        chair_model="chair",
        config=cfg,
        research=False,
        save=False,
        **kw,
    )


def test_the_ceiling_stops_further_rounds_but_still_returns_a_finding():
    """A council that stops because of cost must still answer — spending the money and
    then reporting nothing is the worst of both."""
    p = _BigProvider(per_call=200_000)
    cfg = CouncilConfig(max_tokens_per_run=500_000, skip_debate_on_agreement=False)
    out = _run(p, cfg, rounds=3)
    assert out["stopped_on_budget"] is True
    assert len(out["rounds"]) == 1  # round 1 alone already projects over the cap
    assert out["consensus"]  # the chair still ran
    assert "chair" in p.calls


def test_a_generous_ceiling_does_not_fire():
    p = _BigProvider(per_call=1_000)
    cfg = CouncilConfig(max_tokens_per_run=500_000, skip_debate_on_agreement=False)
    out = _run(p, cfg, rounds=3)
    assert out["stopped_on_budget"] is False
    assert len(out["rounds"]) == 3


def test_a_zero_ceiling_means_no_cap():
    p = _BigProvider(per_call=900_000)
    cfg = CouncilConfig(max_tokens_per_run=0, skip_debate_on_agreement=False)
    out = _run(p, cfg, rounds=2)
    assert out["stopped_on_budget"] is False and len(out["rounds"]) == 2


def test_every_run_reports_what_it_spent():
    p = _BigProvider(per_call=1_000)
    out = _run(p, CouncilConfig(), rounds=1)
    assert out["spend"]["calls"] == 3  # 2 members + chair
    assert out["spend"]["input_tokens"] == 3_000
    assert "a:one" in out["spend"]["by_model"]


def test_spend_is_reported_even_when_every_member_failed():
    """The failures still cost money — the whole point of the number is that it appears
    when things go wrong, not only when they go well."""

    class Failing(_BigProvider):
        def complete(self, *, model, messages, tools=None, **settings):
            self.calls.append(model)
            raise RuntimeError("no key")

    out = _run(Failing(), CouncilConfig(), rounds=1)
    assert "error" in out and "spend" in out


@pytest.mark.parametrize("bad", ["lots", None, [], {}])
def test_a_junk_guard_value_falls_back_to_the_documented_default(bad):
    from coworker.council.config import DEFAULT_MAX_TOKENS_PER_RUN

    assert (
        CouncilConfig.from_dict({"max_tokens_per_run": bad}).max_tokens_per_run
        == DEFAULT_MAX_TOKENS_PER_RUN
    )


def test_a_negative_guard_value_does_not_disable_the_guard():
    """0 means "no guard", so clamping -5 to 0 would turn a typo into a silently disabled
    safety limit — the exact opposite of what the typist intended."""
    from coworker.council.config import DEFAULT_MAX_TOKENS_PER_RUN

    assert CouncilConfig.from_dict({"max_tokens_per_run": -5}).max_tokens_per_run == (
        DEFAULT_MAX_TOKENS_PER_RUN
    )
    # An EXPLICIT zero still means unlimited.
    assert CouncilConfig.from_dict({"max_tokens_per_run": 0}).max_tokens_per_run == 0


def test_the_guard_bounds_rounds_but_a_run_can_finish_above_it():
    """This is a debate-round guard, not a hard ceiling, and the test says so out loud —
    round 1 is already spent when it first runs and the chair always runs afterwards. A
    test asserting `spend <= limit` would be asserting a guarantee the design cannot make."""
    p = _BigProvider(per_call=200_000)
    cfg = CouncilConfig(max_tokens_per_run=500_000, skip_debate_on_agreement=False)
    out = _run(p, cfg, rounds=3)

    assert out["stopped_on_budget"] is True
    assert len(out["rounds"]) == 1  # no further rounds were added
    assert out["spend"]["total_tokens"] == 600_000  # …and the run still exceeded the figure
    assert out["consensus"]  # because the chair ran anyway, as documented


def test_a_provider_reporting_no_usage_cannot_trip_the_guard():
    """Unmeasurable spend reads as zero, so the guard never fires on it. Worth pinning:
    the alternative (treat unknown as expensive) would stop councils that cost nothing."""

    class Silent(_BigProvider):
        def complete(self, *, model, messages, tools=None, **settings):
            self.calls.append(model)
            return AssistantTurn(text="STANCE: ship it\nCONFIDENCE: 0.5", raw=None)

    cfg = CouncilConfig(max_tokens_per_run=1, skip_debate_on_agreement=False)
    out = _run(Silent(), cfg, rounds=3)
    assert out["stopped_on_budget"] is False and len(out["rounds"]) == 3


# -- Gemini must never reach the shared, billed Google key -----------------------------


def test_gemini_ignores_the_shared_google_env_keys(tmp_path, monkeypatch):
    """`GEMINI_API_KEY` / `GOOGLE_API_KEY` are the Google SDK's conventional names, so on a
    box running other Google tooling they hold THAT tooling's key — here a billed Cloud
    project. Treating them as OpenWorker's key spends the wrong account, and "remove key"
    in Settings could never turn it off."""
    from coworker.providers.registry import build_provider_client, provider_configured
    from coworker.secrets import SecretStore

    monkeypatch.setenv("GEMINI_API_KEY", "AIza-billed-cloud-project")
    monkeypatch.setenv("GOOGLE_API_KEY", "AIza-also-billed")
    monkeypatch.delenv("OPENWORKER_GEMINI_API_KEY", raising=False)
    store = SecretStore(tmp_path / "secrets.json")

    assert provider_configured("gemini", store) is False
    # Refused at BUILD, not deferred to the first call: the point is that no code path
    # can reach the billed key, and failing early makes that visible in Settings.
    with pytest.raises(RuntimeError, match="No Gemini API key"):
        build_provider_client("gemini", {}, store)


def test_gemini_uses_its_own_env_name_when_set(tmp_path, monkeypatch):
    from coworker.providers.registry import provider_configured
    from coworker.secrets import SecretStore

    monkeypatch.setenv("OPENWORKER_GEMINI_API_KEY", "AIza-chosen-for-this-app")
    assert provider_configured("gemini", SecretStore(tmp_path / "secrets.json")) is True


def test_a_stored_profile_key_still_works(tmp_path, monkeypatch):
    from coworker.providers.registry import build_provider_client
    from coworker.secrets import SecretStore

    monkeypatch.delenv("OPENWORKER_GEMINI_API_KEY", raising=False)
    store = SecretStore(tmp_path / "secrets.json")
    client = build_provider_client("gemini", {"api_key": "AIza-from-settings"}, store)
    assert client._api_key == "AIza-from-settings"
