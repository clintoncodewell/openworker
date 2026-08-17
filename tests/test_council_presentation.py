"""How a council run is presented: depth, detail, plain-English confidence, an anonymous
chair, and clickable sources in the saved finding.

These are the things that decide whether a reader trusts the output, and every one of them
was wrong in the first real run: 0.76 shown as a measurement, a chair grading its own
argument, and no way to ask for more or less thinking than the default.
"""

from __future__ import annotations

from coworker.council.config import (
    DEPTHS,
    CouncilConfig,
    confidence_label,
)
from coworker.council.core import run_council
from coworker.council.scratchpad import Scratchpad, _finding_markdown
from coworker.providers.base import AssistantTurn


class Panelist:
    def __init__(self, confidence: str = "0.9"):
        self.confidence = confidence
        self.calls: list[tuple[str, str, str]] = []  # model, system, user

    def complete(self, *, model, messages, tools=None, **settings):
        self.calls.append((model, messages[0]["content"], messages[-1]["content"]))
        return AssistantTurn(
            text=f"STANCE: ship it\nPOSITION: yes\nCONFIDENCE: {self.confidence}\nNOTE: watch the queue"
        )

    def capabilities(self, model):  # pragma: no cover
        raise NotImplementedError

    def chair_prompt(self) -> str:
        return next(s for m, s, _ in self.calls if m == "chair")

    def chair_input(self) -> str:
        return next(u for m, _, u in self.calls if m == "chair")


def run(provider, models=("a:one", "b:two"), **kw):
    kw.setdefault("research", False)
    return run_council(
        "ship?", provider=provider, models=list(models), chair_model="chair", **kw
    )


# -- confidence in words ---------------------------------------------------------------


def test_a_number_is_translated_for_the_reader():
    """0.56 reads as a measurement and is really a vibe. The false precision is worse than
    no figure."""
    assert confidence_label(0.95) == "very confident"
    assert confidence_label(0.76) == "confident"
    assert confidence_label(0.56) == "leaning this way"
    assert confidence_label(0.4) == "unsure"
    assert confidence_label(0.1) == "very unsure"


def test_no_number_means_no_label():
    assert confidence_label(None) == ""


def test_every_answer_carries_its_label(monkeypatch):
    out = run(Panelist(confidence="0.55"), rounds=1)
    assert [a["confidence_label"] for a in out["rounds"][0]] == [
        "leaning this way",
        "leaning this way",
    ]


def test_the_chair_is_told_to_write_confidence_in_words():
    p = Panelist()
    run(p, rounds=1)
    assert "Confidence in words, not decimals" in p.chair_prompt()


# -- the chair does not know who argued what -------------------------------------------


def test_the_chair_reads_anonymised_members():
    """The chair is one of the panel models. Benching a strong model to keep the chair
    impartial costs more than the bias does, so hide the names instead."""
    p = Panelist()
    run(p, models=["a:one", "b:two"], rounds=1)
    chair = p.chair_input()
    assert "--- Member A" in chair and "--- Member B" in chair
    assert "--- a:one" not in chair and "--- b:two" not in chair


def test_the_scratchpad_the_chair_reads_is_anonymised_too():
    """Notes carry the model that posted them. Aliasing only the transcript leaks the names
    straight back out through the scratchpad."""
    p = Panelist()
    run(p, models=["a:one", "b:two"], rounds=1)
    chair = p.chair_input()
    assert "watch the queue" in chair  # the note did reach the chair
    assert "[a:one" not in chair and "[b:two" not in chair


def test_members_still_see_real_names():
    """Only the chair is blinded. Members arguing with "Member B" instead of a named model
    would lose the one thing a heterogeneous panel is for."""
    p = Panelist()
    # Both members answer identically here, so adaptive debate would skip round 2 and
    # there would be nothing to assert about.
    run(p, models=["a:one", "b:two"], rounds=2,
        config=CouncilConfig(skip_debate_on_agreement=False, depth="custom", rounds=2))
    debate = [u for m, _, u in p.calls if m == "a:one"][-1]
    assert "b:two" in debate


def test_the_saved_transcript_keeps_the_real_names(tmp_path):
    """The anonymity is a device for the chair. The record is for the reader, and a
    transcript that will not say which model said what is worth much less."""
    p = Panelist()
    out = run_council(
        "ship?",
        provider=p,
        models=["a:one", "b:two"],
        chair_model="chair",
        rounds=1,
        research=False,
        save=True,
    )
    saved = out.get("saved") or {}
    if saved.get("transcript.md"):  # save is best-effort
        from pathlib import Path

        assert "a:one" in Path(saved["transcript.md"]).read_text()


# -- depth -----------------------------------------------------------------------------


def test_depth_decides_rounds_and_panel_size_together():
    """One control the reader can reason about. Exposing rounds and panel size separately
    lets them pick a pair that makes no sense — three rounds of two members."""
    assert CouncilConfig(depth="quick").limits() == (1, 3, False)
    assert CouncilConfig(depth="standard").limits() == (2, 6, True)
    assert CouncilConfig(depth="deep").limits() == (3, DEPTHS["deep"]["max_members"], True)


def test_custom_depth_hands_the_raw_fields_back():
    cfg = CouncilConfig(depth="custom", rounds=3, research=False)
    rounds, _members, research = cfg.limits()
    assert rounds == 3 and research is False


def test_quick_depth_trims_the_panel():
    p = Panelist()
    out = run_council(
        "ship?",
        provider=p,
        models=["a:one", "b:two", "c:three", "d:four"],
        chair_model="chair",
        config=CouncilConfig(depth="quick"),
    )
    assert len(out["panel"]) == 3
    assert len(out["rounds"]) == 1  # quick means no debate round
    assert any("capped at 3" in e["reason"] for e in out["report"]["excluded"])


def test_an_explicit_argument_still_beats_the_depth():
    """The calling agent is allowed to say "one round" for a question that does not need
    more, whatever the stored default says."""
    p = Panelist()
    out = run_council(
        "ship?",
        provider=p,
        models=["a:one", "b:two"],
        chair_model="chair",
        config=CouncilConfig(depth="deep"),
        rounds=1,
        research=False,
    )
    assert len(out["rounds"]) == 1


def test_an_unknown_depth_falls_back_to_standard():
    assert CouncilConfig.from_dict({"depth": "ludicrous"}).depth == "standard"


# -- detail level ----------------------------------------------------------------------


def test_the_detail_level_reaches_the_chair():
    p = Panelist()
    run_council(
        "ship?", provider=p, models=["a:one"], chair_model="chair", rounds=1,
        research=False, config=CouncilConfig(detail="brief"),
    )
    assert "under 200 words" in p.chair_prompt()


def test_every_detail_level_still_leads_with_the_answer():
    """A reader who reads one line should get the verdict, whichever length they picked."""
    for detail in ("brief", "standard", "full"):
        p = Panelist()
        run_council(
            "ship?", provider=p, models=["a:one"], chair_model="chair", rounds=1,
            research=False, config=CouncilConfig(detail=detail),
        )
        assert "ANSWER:" in p.chair_prompt()


def test_a_rewritten_chair_prompt_still_gets_the_length_control():
    """Someone who edits the chair prompt still expects the Settings control to work, and a
    stray placeholder left in their text must not reach the model verbatim."""
    cfg = CouncilConfig(detail="brief", prompts={"analysis": {"chair": "MINE\n{detail}"}})
    text = cfg.prompt("chair")
    assert "under 200 words" in text and "{detail}" not in text


def test_an_unknown_detail_falls_back():
    assert CouncilConfig.from_dict({"detail": "epic"}).detail == "standard"


# -- the saved finding -----------------------------------------------------------------


def test_sources_are_written_as_clickable_links():
    finding = _finding_markdown(
        "ship?",
        {
            "panel": [{"model": "a:one", "role": "Advocate"}],
            "consensus": "yes",
            "research": {
                "queries": ["ship friday risk"],
                "results": [{"title": "Deploy Fridays", "url": "http://x/1"}],
            },
        },
    )
    assert "[Deploy Fridays](http://x/1)" in finding
    assert "`ship friday risk`" in finding


def test_a_run_with_no_sources_has_no_empty_sources_heading():
    finding = _finding_markdown("ship?", {"panel": [], "consensus": "yes", "research": {}})
    assert "## Sources" not in finding


def test_the_finding_carries_the_run_notes():
    finding = _finding_markdown(
        "ship?",
        {"panel": [], "consensus": "yes", "report": {"notes": ["1 of 2 members did not answer"]}},
    )
    assert "1 of 2 members did not answer" in finding


def test_the_scratchpad_renders_without_an_alias():
    pad = Scratchpad("q")
    pad.post("a:one", "Advocate", "a note", 1)
    assert "[a:one · Advocate] a note" in pad.render()
    assert "[Member A · Advocate] a note" in pad.render({"a:one": "Member A"})


# -- what the review caught ------------------------------------------------------------


def test_a_config_saved_before_depth_existed_keeps_its_settings():
    """A stored one-round, no-search council must not silently become a two-round council
    with web search. Adopting the new default would re-price someone's setup for them."""
    cfg = CouncilConfig.from_dict({"rounds": 1, "research": False})
    assert cfg.depth == "custom"
    assert cfg.limits() == (1, 8, False)


def test_an_old_config_that_matched_the_default_adopts_the_named_depth():
    """Nothing to preserve: it already meant "standard", so it should say so and pick up
    the panel cap that comes with it."""
    assert CouncilConfig.from_dict({"rounds": 2, "research": True}).depth == "standard"


def test_an_explicit_depth_is_never_second_guessed():
    assert CouncilConfig.from_dict({"depth": "quick", "rounds": 3}).depth == "quick"


def test_a_member_that_drops_out_mid_debate_is_reported():
    """It answered, so it is not a failure — but it argued with half a voice, and a report
    that stays silent about that is the same silence this report exists to break."""

    class DiesInRound2:
        def __init__(self):
            self.seen = 0

        def complete(self, *, model, messages, tools=None, **settings):
            if model == "b:two":
                self.seen += 1
                if self.seen > 1:
                    raise RuntimeError("rate limited")
            return AssistantTurn(text=f"STANCE: {model}\nPOSITION: yes\nCONFIDENCE: 0.5")

        def capabilities(self, model):  # pragma: no cover
            raise NotImplementedError

    out = run_council(
        "ship?", provider=DiesInRound2(), models=["a:one", "b:two"], chair_model="chair",
        research=False, config=CouncilConfig(depth="custom", rounds=2,
                                             skip_debate_on_agreement=False),
    )
    report = out["report"]
    assert [d["model"] for d in report["dropped"]] == ["b:two"]
    assert any("dropped out of the debate" in n for n in report["notes"])


def test_a_run_where_everything_failed_still_marks_itself_finished(tmp_path, monkeypatch):
    """Otherwise the live panel shows a debate in progress forever, for the run a reader
    most needs told about."""
    monkeypatch.setenv("COWORKER_STATE_DIR", str(tmp_path))
    from coworker.council.scratchpad import read_live

    class Dead:
        def complete(self, **kw):
            raise RuntimeError("no key")

        def capabilities(self, model):  # pragma: no cover
            raise NotImplementedError

    out = run_council("ship?", provider=Dead(), models=["a:one"], chair_model="chair",
                      rounds=1, research=False)
    assert "error" in out
    assert read_live().get("status") == "done"


def test_two_councils_do_not_write_the_same_temp_file(tmp_path):
    """A shared `live.json.tmp` between two sessions interleaves into one file that then
    gets renamed into place — a valid-looking mixture of two runs."""
    a, b = Scratchpad("first question"), Scratchpad("second question")
    assert a.dir.name != b.dir.name
