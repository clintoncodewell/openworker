"""The debate mechanics that the multi-agent-debate literature says decide whether a
panel is worth running at all: role diversity, anti-conformity, adaptive rounds, and the
shared scratchpad."""

from __future__ import annotations

import pytest

from coworker.council import CouncilConfig, Source, run_council
from coworker.council.core import AGREEMENT_CONFIDENCE, _agreed
from coworker.providers.base import AssistantTurn


class ScriptedProvider:
    """Answers from a per-model script, cycling if a round runs past the script's end."""

    def __init__(self, script: dict[str, list[str]] | None = None, default: str = ""):
        self.calls: list[tuple[str, str, str]] = []  # (model, system, user)
        self.script = script or {}
        self.default = default
        self._seen: dict[str, int] = {}

    def complete(self, *, model, messages, tools=None, **settings):
        self.calls.append((model, messages[0]["content"], messages[-1]["content"]))
        lines = self.script.get(model)
        if not lines:
            return AssistantTurn(text=self.default or f"{model}: STANCE: unset\nCONFIDENCE: 0.5")
        i = self._seen.get(model, 0)
        self._seen[model] = i + 1
        return AssistantTurn(text=lines[min(i, len(lines) - 1)])

    def capabilities(self, model):  # pragma: no cover
        raise NotImplementedError

    def systems_for(self, model):
        return [s for m, s, _ in self.calls if m == model]

    def users_for(self, model):
        return [u for m, _, u in self.calls if m == model]


def run(provider, models, cfg=None, **kw):
    kw.setdefault("research", False)
    kw.setdefault("save", False)
    return run_council(
        "Should we ship on Friday?",
        provider=provider,
        models=models,
        chair_model="chair",
        config=cfg or CouncilConfig(),
        **kw,
    )


# -- role diversity -------------------------------------------------------------------


def test_each_member_is_given_a_different_lens():
    """Panels of identically-prompted agents barely beat one model. Different lenses are
    the mechanism, so this is the load-bearing assertion in the whole file."""
    p = ScriptedProvider()
    out = run(p, ["a:one", "b:two", "c:three"], rounds=1)
    roles = [m["role"] for m in out["panel"]]
    assert len(set(roles)) == 3
    assert roles == ["Advocate", "Skeptic", "Pragmatist"]
    # And the lens actually reaches the model, not just the result payload.
    assert "Advocate" in p.systems_for("a:one")[0]
    assert "Skeptic" in p.systems_for("b:two")[0]


def test_the_role_brief_reaches_the_model_not_just_the_name():
    p = ScriptedProvider()
    run(p, ["a:one"], rounds=1)
    assert "strongest honest case FOR" in p.systems_for("a:one")[0]


def test_custom_roles_are_used():
    cfg = CouncilConfig(roles=[{"name": "Accountant", "brief": "follow the money"}])
    p = ScriptedProvider()
    out = run(p, ["a:one"], cfg=cfg, rounds=1)
    assert out["panel"][0]["role"] == "Accountant"
    assert "follow the money" in p.systems_for("a:one")[0]


def test_the_chair_is_told_the_members_argued_assigned_lenses():
    """Without this the chair averages five deliberately one-sided answers and reports the
    mean as consensus."""
    p = ScriptedProvider()
    run(p, ["a:one", "b:two"], rounds=1)
    chair_system = p.systems_for("chair")[0]
    chair_user = p.users_for("chair")[0]
    assert "DELIBERATELY one-sided" in chair_system
    # The roster reaches the chair ANONYMISED. It needs the lenses to weigh the positions;
    # it must not have the model names, because the chair is itself a panel model and would
    # otherwise be marking its own homework. The saved transcript keeps the real names.
    assert "Member A (Advocate)" in chair_user and "Member B (Skeptic)" in chair_user
    # Attribution comes from the labels, and those are what the engine controls. A member
    # that names itself inside its own answer still leaks, which is why this is a bias
    # reducer and not a guarantee.
    assert "--- a:one" not in chair_user and "PANEL: a:one" not in chair_user


# -- anti-conformity ------------------------------------------------------------------


def test_debate_rounds_carry_the_hold_your_position_rule():
    """Sycophancy and disagreement-collapse are the documented failure mode of debate."""
    p = ScriptedProvider()
    run(p, ["a:one", "b:two"], rounds=2)
    debate_system = p.systems_for("a:one")[1]
    assert "Hold your position" in debate_system
    assert "being outnumbered is not evidence" in debate_system


# -- adaptive debate ------------------------------------------------------------------


def _answer(stance: str, confidence: float) -> str:
    return f"STANCE: {stance}\nPOSITION: x\nCONFIDENCE: {confidence}"


def test_agreement_needs_one_stance_and_confidence():
    high = AGREEMENT_CONFIDENCE
    assert _agreed([{"text": _answer("ship it", high)}, {"text": _answer("ship it", high)}])
    # Same stance, but one member is unsure — that is worth arguing about.
    assert not _agreed([{"text": _answer("ship it", high)}, {"text": _answer("ship it", 0.4)}])
    # Confident but split.
    assert not _agreed([{"text": _answer("ship it", high)}, {"text": _answer("wait", high)}])
    # No signal at all → debate. An unnecessary round is a cheaper mistake than a
    # skipped one.
    assert not _agreed([{"text": "no stance line"}, {"text": "nor here"}])
    # A single live member can't agree with anyone.
    assert not _agreed([{"text": _answer("ship it", high)}])


def test_a_unanimous_confident_round_one_skips_the_debate():
    p = ScriptedProvider(default=_answer("ship it", 0.9))
    out = run(p, ["a:one", "b:two"], rounds=2)
    assert out["skipped_debate"] is True
    assert len(out["rounds"]) == 1
    assert len(p.calls) == 3  # 2 members + chair, no second round


def test_a_split_round_one_still_debates():
    p = ScriptedProvider(
        {"a:one": [_answer("ship it", 0.9)], "b:two": [_answer("wait", 0.9)]}
    )
    out = run(p, ["a:one", "b:two"], rounds=2)
    assert out["skipped_debate"] is False
    assert len(out["rounds"]) == 2


def test_the_skip_can_be_turned_off():
    cfg = CouncilConfig(skip_debate_on_agreement=False)
    p = ScriptedProvider(default=_answer("ship it", 0.9))
    out = run(p, ["a:one", "b:two"], cfg=cfg, rounds=2)
    assert out["skipped_debate"] is False and len(out["rounds"]) == 2


def test_stance_and_confidence_are_reported_per_member():
    p = ScriptedProvider(default=_answer("ship it", 0.9))
    out = run(p, ["a:one", "b:two"], rounds=1)
    assert out["rounds"][0][0]["stance"] == "ship it"
    assert out["rounds"][0][0]["confidence"] == 0.9


# -- scratchpad ------------------------------------------------------------------------


def test_notes_from_round_one_reach_every_member_in_round_two():
    p = ScriptedProvider(
        {
            "a:one": [f"{_answer('ship', 0.5)}\nNOTE: the migration is untested", "round2"],
            "b:two": [f"{_answer('wait', 0.5)}\nNOTE: none", "round2"],
        }
    )
    run(p, ["a:one", "b:two"], rounds=2)
    for model in ("a:one", "b:two"):
        second = p.users_for(model)[1]
        assert "PANEL SCRATCHPAD" in second
        assert "the migration is untested" in second


def test_the_chair_sees_the_scratchpad():
    p = ScriptedProvider(default=f"{_answer('ship', 0.5)}\nNOTE: watch the migration")
    run(p, ["a:one", "b:two"], rounds=1)
    assert "watch the migration" in p.users_for("chair")[0]


def test_a_run_writes_its_files(tmp_path, monkeypatch):
    import coworker.council.scratchpad as pad_mod

    monkeypatch.setattr(pad_mod, "runs_dir", lambda: tmp_path)
    p = ScriptedProvider(default=f"{_answer('ship', 0.5)}\nNOTE: watch the migration")
    out = run_council(
        "Should we ship on Friday?",
        provider=p,
        models=["a:one", "b:two"],
        chair_model="chair",
        rounds=1,
        research=False,
    )
    assert set(out["saved"]) == {"scratchpad.md", "transcript.md", "finding.md"}
    written = (tmp_path).rglob("scratchpad.md")
    assert "watch the migration" in next(written).read_text()


# -- sources ---------------------------------------------------------------------------


def test_source_material_reaches_every_member(tmp_path):
    (tmp_path / "brief.md").write_text("THE PROJECT BUDGET IS 40k")
    cfg = CouncilConfig(sources=[Source(kind="folder", target=str(tmp_path), label="Docs")])
    p = ScriptedProvider()
    out = run(p, ["a:one", "b:two"], cfg=cfg, rounds=1)
    for model in ("a:one", "b:two"):
        assert "THE PROJECT BUDGET IS 40k" in p.users_for(model)[0]
    # Labels and errors come back; the text does not — it is already in the prompts, and
    # echoing it would double the tool result the calling agent has to read.
    assert out["sources"] == [{"label": "Docs", "kind": "folder", "truncated": False}]


def test_a_broken_source_does_not_stop_the_council():
    cfg = CouncilConfig(sources=[Source(kind="file", target="/does/not/exist")])
    p = ScriptedProvider()
    out = run(p, ["a:one"], cfg=cfg, rounds=1)
    assert "error" in out["sources"][0]
    assert out["consensus"]  # still produced a finding


def test_the_decision_preset_changes_every_prompt():
    cfg = CouncilConfig(preset="decision")
    p = ScriptedProvider()
    run(p, ["a:one", "b:two"], cfg=cfg, rounds=2)
    assert "REVERSIBILITY" in p.systems_for("a:one")[0]
    assert "PRE-MORTEM" in p.systems_for("a:one")[1]
    assert "RECOMMENDATION" in p.systems_for("chair")[0]


def test_an_edited_prompt_is_what_the_model_actually_receives():
    cfg = CouncilConfig(prompts={"analysis": {"round1": "BE BRIEF. {role_name}"}})
    p = ScriptedProvider()
    run(p, ["a:one"], cfg=cfg, rounds=1)
    assert p.systems_for("a:one")[0] == "BE BRIEF. Advocate"


@pytest.mark.parametrize(
    "template",
    [
        'Reply with JSON like {"answer": 1}',  # braces are ordinary prose in a textarea
        "{unknown}",
        "{0}",
        "{me.__class__.__init__.__globals__}",  # str.format's attribute walk
        "{{escaped}}",
    ],
)
def test_braces_in_an_edited_prompt_are_literal_and_never_evaluated(template):
    """These prompts are typed into a GUI textarea. Under `str.format` the first three
    raise and take the whole council down, and the fourth walks live objects into a prompt
    bound for five external vendors."""
    cfg = CouncilConfig(prompts={"analysis": {"round1": template}})
    p = ScriptedProvider()
    run(p, ["a:one"], cfg=cfg, rounds=1)
    assert p.systems_for("a:one")[0] == template
