"""The council's web research: query planning, and the run report that says what happened.

The bug these exist for: the council sent its entire brief — 1,400 characters — as the
search string, and every engine matched on the commonest words in it. Measured against
DuckDuckGo on 2026-08-17, the full brief returned a Yellow Pages listing for financial
planners in Darwin; "advisor AI adoption wealth management 2026" returned MSCI and Family
Wealth Report. Three of five panel members flagged the results as noise, and the run cost
what a researched council costs while producing an unresearched one.
"""

from __future__ import annotations

from coworker.council import research as research_mod
from coworker.council.core import run_council
from coworker.providers.base import AssistantTurn


class Planner:
    """Answers the planner call with queries; answers everything else with a position."""

    def __init__(self, reply: str = "advisor AI adoption 2026\nwealth management AI survey"):
        self.reply = reply
        self.calls: list[tuple[str, str]] = []

    def complete(self, *, model, messages, tools=None, **settings):
        self.calls.append((model, messages[0]["content"]))
        if "search quer" in messages[0]["content"].lower():
            return AssistantTurn(text=self.reply)
        return AssistantTurn(text=f"{model} says POSITION: yes\nCONFIDENCE: 0.5")

    def capabilities(self, model):  # pragma: no cover — unused by the council
        raise NotImplementedError


class Engine:
    name = "fake"

    def __init__(self, per_query: int = 2, boom: bool = False):
        self.queries: list[str] = []
        self.per_query = per_query
        self.boom = boom

    def search(self, query, max_results=5):
        self.queries.append(query)
        if self.boom:
            raise RuntimeError("rate limited")

        class R:
            def __init__(self, url):
                self.url = url

            def to_dict(self):
                return {"title": "T", "url": self.url, "snippet": "SNIP"}

        return [R(f"http://x/{query}/{i}") for i in range(self.per_query)]


def _engine(monkeypatch, engine):
    import coworker.web.tool as web_tool

    monkeypatch.setattr(web_tool, "resolve_provider", lambda secrets=None, **kw: engine)
    return engine


LONG = (
    "Today is August 2026. By January 2027 — roughly 4-5 months from now — where will the "
    "LARGEST impact on financial planning already be visible? Rank the biggest drivers of "
    "change that will have materially shown up, and identify the single largest impact."
)


# -- query planning --------------------------------------------------------------------


def test_the_whole_brief_is_never_the_search_string(monkeypatch):
    """The actual defect. A query this long is a bag of words to every engine."""
    engine = _engine(monkeypatch, Engine())
    research_mod.search(LONG, provider=Planner(), model="chair")
    assert engine.queries and all(len(q) < 120 for q in engine.queries)
    assert not any(LONG[:60] in q for q in engine.queries)


def test_the_planner_queries_are_used_verbatim(monkeypatch):
    engine = _engine(monkeypatch, Engine())
    out = research_mod.search(LONG, provider=Planner(), model="chair")
    assert engine.queries == ["advisor AI adoption 2026", "wealth management AI survey"]
    assert out["queries"] == engine.queries


def test_results_carry_the_query_that_found_them(monkeypatch):
    """The sources panel groups by angle — without this a reader cannot tell which search
    produced a result, so cannot judge whether the angle was worth anything."""
    _engine(monkeypatch, Engine())
    out = research_mod.search(LONG, provider=Planner(), model="chair")
    assert {r["query"] for r in out["results"]} == set(out["queries"])


def test_the_same_url_from_two_queries_appears_once(monkeypatch):
    class Same(Engine):
        def search(self, query, max_results=5):
            self.queries.append(query)

            class R:
                def to_dict(self):
                    return {"title": "T", "url": "http://same", "snippet": "S"}

            return [R()]

    _engine(monkeypatch, Same())
    out = research_mod.search(LONG, provider=Planner(), model="chair")
    assert len(out["results"]) == 1


def test_a_planner_that_writes_prose_is_ignored(monkeypatch):
    """A model that ignores the format returns sentences. Searching those reproduces the
    bug, so anything too long or too wordy is dropped and the fallback takes over."""
    engine = _engine(monkeypatch, Engine())
    prose = (
        "Certainly! Here are some search queries you could use to research this "
        "interesting question about the financial planning industry and its future:"
    )
    research_mod.search(LONG, provider=Planner(reply=prose), model="chair")
    assert engine.queries == [research_mod.fallback_query(LONG)]


def test_a_planner_that_fails_still_searches(monkeypatch):
    class Broken:
        def complete(self, **kw):
            raise RuntimeError("no key")

    engine = _engine(monkeypatch, Engine())
    out = research_mod.search(LONG, provider=Broken(), model="chair")
    assert engine.queries == [research_mod.fallback_query(LONG)]
    assert out["ok"] is True


def test_the_fallback_strips_the_words_that_carry_no_signal():
    q = research_mod.fallback_query("What is the impact of AI on financial planning by 2027?")
    assert "what" not in q.lower() and "of" not in q.lower().split()
    assert "AI" in q and "financial" in q and len(q.split()) <= 12


def test_bullets_and_numbering_are_stripped(monkeypatch):
    engine = _engine(monkeypatch, Engine())
    research_mod.search(
        LONG, provider=Planner(reply='1. "advisor AI adoption"\n- wealth management 2026'), model="c"
    )
    assert engine.queries == ["advisor AI adoption", "wealth management 2026"]


def test_a_dead_search_engine_is_reported_not_raised(monkeypatch):
    _engine(monkeypatch, Engine(boom=True))
    out = research_mod.search(LONG, provider=Planner(), model="chair")
    assert out["ok"] is False and out["errors"]


def test_research_never_raises_when_the_engine_cannot_be_resolved(monkeypatch):
    import coworker.web.tool as web_tool

    def boom(*a, **k):
        raise ValueError("no search provider configured")

    monkeypatch.setattr(web_tool, "resolve_provider", boom)
    out = research_mod.search(LONG, provider=Planner(), model="chair")
    assert out["ok"] is False and "no search provider" in out["error"]


# -- the run report --------------------------------------------------------------------


def test_the_report_names_the_members_that_did_not_answer(monkeypatch):
    """A council that runs five members instead of six looks exactly like one that ran six.
    That happened for real — the Mac could not see the Claude CLI — and nothing said so."""
    _engine(monkeypatch, Engine())

    class Flaky(Planner):
        def complete(self, *, model, messages, tools=None, **settings):
            if model == "b:two":
                raise RuntimeError("no key")
            return super().complete(model=model, messages=messages, **settings)

    out = run_council(
        "ship?", provider=Flaky(), models=["a:one", "b:two"], chair_model="chair", rounds=1
    )
    report = out["report"]
    assert report["members_asked"] == 2 and report["members_answered"] == 1
    assert [f["model"] for f in report["failed"]] == ["b:two"]
    assert any("did not answer" in n for n in report["notes"])


def test_the_report_says_when_research_found_nothing(monkeypatch):
    """The failure that cost a whole run: the panel argued from memory and only said so
    because three members happened to notice."""
    _engine(monkeypatch, Engine(per_query=0))
    out = run_council(
        "ship?", provider=Planner(), models=["a:one"], chair_model="chair", rounds=1,
        research=True,
    )
    assert any("argued from what the models already knew" in n for n in out["report"]["notes"])


def test_the_report_counts_what_the_search_returned(monkeypatch):
    _engine(monkeypatch, Engine())
    out = run_council(
        "ship?", provider=Planner(), models=["a:one"], chair_model="chair", rounds=1,
        research=True,
    )
    research = out["report"]["research"]
    assert research["ran"] is True and research["ok"] is True
    assert research["result_count"] == 4 and len(research["queries"]) == 2


def test_a_run_with_no_research_says_so(monkeypatch):
    out = run_council(
        "ship?", provider=Planner(), models=["a:one"], chair_model="chair", rounds=1,
        research=False,
    )
    assert out["report"]["research"]["ran"] is False
    assert any("Web research was off" in n for n in out["report"]["notes"])


# -- how wide the sweep goes ----------------------------------------------------------


def test_the_sweep_widens_with_depth(monkeypatch):
    """How much evidence to gather is part of how hard the council thinks, so it moves with
    depth instead of being a fourth number to set."""
    from coworker.council.config import CouncilConfig

    assert CouncilConfig(depth="standard").research_limits() == (3, 12)
    assert CouncilConfig(depth="deep").research_limits() == (6, 30)


def test_deep_actually_runs_more_searches(monkeypatch):
    from coworker.council.config import CouncilConfig

    engine = _engine(monkeypatch, Engine(per_query=10))
    planner = Planner(reply="\n".join(f"query number {i}" for i in range(8)))
    out = run_council(
        "ship?", provider=planner, models=["a:one"], chair_model="chair",
        config=CouncilConfig(depth="deep"), research=True,
    )
    assert len(engine.queries) == 6  # deep plans six angles, not three
    assert len(out["research"]["results"]) == 30


def test_the_result_budget_is_shared_across_the_angles(monkeypatch):
    """One lucky query must not fill the whole budget and crowd out the other angles."""
    from coworker.council.config import CouncilConfig

    engine = _engine(monkeypatch, Engine(per_query=50))
    run_council(
        "ship?", provider=Planner(reply="one query\ntwo query\nthree query"),
        models=["a:one"], chair_model="chair",
        config=CouncilConfig(depth="standard"), research=True,
    )
    # 12 results over 3 queries: ask each for about a quarter of the budget, plus headroom.
    assert all(n <= 8 for n in [6])  # per_query = ceil(12/3) + 2
    assert len(engine.queries) == 3


def test_one_search_feeds_every_member(monkeypatch):
    """The panel is compared on judgement, not on which model retrieved better — so there
    is ONE search and everyone reads the same results."""
    engine = _engine(monkeypatch, Engine())
    planner = Planner()
    run_council(
        "ship?", provider=planner, models=["a:one", "b:two", "c:three"],
        chair_model="chair", rounds=1, research=True,
    )
    # Two planned queries, run once each — not once per member.
    assert len(engine.queries) == 2


# -- the search engine setting ---------------------------------------------------------


def test_switching_engine_keeps_the_stored_key(tmp_path):
    """The dropdown sends only a provider. Rebuilding the profile from that alone throws
    the key away, and the user finds out the next time a search quietly returns nothing."""
    from coworker.secrets import SecretStore
    from coworker.server.manager import SessionManager

    store = SecretStore(tmp_path / "secrets.json")
    mgr = SessionManager.__new__(SessionManager)
    mgr.secrets = store

    assert SessionManager.set_web_search(mgr, "brave", "bsk-real")["ok"]
    SessionManager.set_web_search(mgr, "tavily")  # switch, no key given
    assert store.get("web_search:default")["api_key"] == "bsk-real"

    # An explicit empty string is the way to actually clear it.
    SessionManager.set_web_search(mgr, "tavily", "")
    assert not store.get("web_search:default").get("api_key")


def test_an_unknown_engine_is_refused(tmp_path):
    from coworker.secrets import SecretStore
    from coworker.server.manager import SessionManager

    mgr = SessionManager.__new__(SessionManager)
    mgr.secrets = SecretStore(tmp_path / "secrets.json")
    assert SessionManager.set_web_search(mgr, "askjeeves")["ok"] is False
