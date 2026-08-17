"""Council configuration, roles, sources and the shared scratchpad."""

from __future__ import annotations

import json

import pytest

from coworker.council import CouncilConfig, Source, load_config, save_config
from coworker.council import sources as sources_mod
from coworker.council.config import (
    ANSWER_FIRST,
    DEFAULT_DETAIL,
    DEFAULT_ROLES,
    DETAILS,
    HOUSE_STYLE,
    PRESETS,
    render,
)


def chair_default(preset: str = "analysis", detail: str = DEFAULT_DETAIL) -> str:
    """The shipped chair prompt as the model receives it. Unlike round1 and debate, the
    chair text carries the house style and the detail level, so it is substituted before it
    leaves `prompt()` — comparing against the raw template would only assert that we forgot."""
    return render(
        PRESETS[preset]["chair"],
        house_style=HOUSE_STYLE,
        detail=DETAILS[detail]["instruction"],
        answer_first=ANSWER_FIRST,
    )
from coworker.council.scratchpad import Scratchpad, extract_note


# -- config ---------------------------------------------------------------------------


def test_prompts_fall_back_to_the_shipped_default():
    cfg = CouncilConfig()
    assert cfg.prompt("round1") == PRESETS["analysis"]["round1"]
    assert cfg.prompt("chair") == chair_default()


def test_an_override_replaces_only_that_prompt():
    """Editing the chair must not freeze round1 at today's default — the other two keep
    tracking the shipped text so an improvement upstream still lands."""
    cfg = CouncilConfig(prompts={"analysis": {"chair": "MY CHAIR"}})
    assert cfg.prompt("chair") == "MY CHAIR"
    assert cfg.prompt("round1") == PRESETS["analysis"]["round1"]


def test_a_blank_override_is_treated_as_no_override():
    cfg = CouncilConfig(prompts={"analysis": {"chair": "   "}})
    assert cfg.prompt("chair") == chair_default()


def test_overrides_are_per_preset():
    cfg = CouncilConfig(preset="decision", prompts={"analysis": {"chair": "MY CHAIR"}})
    assert cfg.prompt("chair") == chair_default("decision")


def test_roles_wrap_round_a_panel_larger_than_the_role_list():
    cfg = CouncilConfig(roles=[{"name": "A", "brief": "a"}, {"name": "B", "brief": "b"}])
    assert [cfg.role_for(i)["name"] for i in range(5)] == ["A", "B", "A", "B", "A"]


def test_roles_fall_back_to_the_defaults_when_cleared():
    cfg = CouncilConfig(roles=[])
    assert cfg.role_for(0) == DEFAULT_ROLES[0]


def test_roundtrip_through_disk(tmp_path):
    path = tmp_path / "council.json"
    cfg = CouncilConfig(
        preset="decision",
        rounds=3,
        chair_model="azure:gpt-5.6-sol",
        sources=[Source(kind="folder", target="/tmp/x", label="Notes")],
        prompts={"decision": {"chair": "MY CHAIR"}},
    )
    save_config(cfg, path)
    loaded = load_config(path)
    assert loaded.preset == "decision"
    assert loaded.rounds == 3
    assert loaded.chair_model == "azure:gpt-5.6-sol"
    assert loaded.sources[0].kind == "folder" and loaded.sources[0].label == "Notes"
    assert loaded.prompt("chair") == "MY CHAIR"


def test_derived_fields_are_not_written_to_disk(tmp_path):
    """`defaults` is shipped to the GUI so it can render "reset to default". Storing it
    would pin today's text forever and silently defeat that button."""
    path = tmp_path / "council.json"
    save_config(CouncilConfig(), path)
    stored = json.loads(path.read_text())
    assert "defaults" not in stored and "default_roles" not in stored
    assert "defaults" in CouncilConfig().to_dict()


def test_a_corrupt_config_falls_back_to_defaults_instead_of_raising(tmp_path):
    path = tmp_path / "council.json"
    path.write_text("{not json")
    assert load_config(path).preset == "analysis"


def test_unknown_preset_and_out_of_range_rounds_are_clamped():
    cfg = CouncilConfig.from_dict({"preset": "nonsense", "rounds": 99})
    assert cfg.preset == "analysis" and cfg.rounds == 3


def test_unknown_keys_are_ignored(tmp_path):
    """A config written by a newer build must not crash an older one."""
    cfg = CouncilConfig.from_dict({"preset": "decision", "from_the_future": True})
    assert cfg.preset == "decision"


# -- sources --------------------------------------------------------------------------


def test_folder_source_reads_text_files(tmp_path):
    (tmp_path / "a.md").write_text("ALPHA")
    (tmp_path / "b.py").write_text("BETA")
    (tmp_path / "c.bin").write_bytes(b"\x00\x01")
    resolved = sources_mod.resolve([Source(kind="folder", target=str(tmp_path))])
    text = resolved[0]["text"]
    assert "ALPHA" in text and "BETA" in text
    assert "c.bin" not in text  # binary is skipped by suffix, never read


def test_folder_source_honours_a_glob(tmp_path):
    (tmp_path / "a.md").write_text("ALPHA")
    (tmp_path / "b.py").write_text("BETA")
    resolved = sources_mod.resolve(
        [Source(kind="folder", target=str(tmp_path), options={"glob": "*.md"})]
    )
    assert "ALPHA" in resolved[0]["text"] and "BETA" not in resolved[0]["text"]


def test_file_source(tmp_path):
    f = tmp_path / "notes.md"
    f.write_text("THE NOTE")
    assert "THE NOTE" in sources_mod.resolve([Source(kind="file", target=str(f))])[0]["text"]


def test_dotfiles_are_never_sourced(tmp_path):
    """Sourcing a folder ships it to five external vendors. `~` with `**/*.json` would
    otherwise sweep up ~/.config/coworker/secrets.json."""
    (tmp_path / "notes.md").write_text("PUBLIC")
    (tmp_path / ".env").write_text("SECRET_KEY=abc")
    hidden = tmp_path / ".config"
    hidden.mkdir()
    (hidden / "secrets.json").write_text("SECRET_KEY=abc")

    text = sources_mod.resolve([Source(kind="folder", target=str(tmp_path))])[0]["text"]
    assert "PUBLIC" in text
    assert "SECRET_KEY" not in text
    assert "hidden file" in text  # the skip is reported, never silent


def test_a_symlink_out_of_the_folder_is_not_followed(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secrets.md").write_text("SECRET_KEY=abc")
    root = tmp_path / "root"
    root.mkdir()
    (root / "ok.md").write_text("PUBLIC")
    (root / "sneaky.md").symlink_to(outside / "secrets.md")

    text = sources_mod.resolve([Source(kind="folder", target=str(root))])[0]["text"]
    assert "PUBLIC" in text and "SECRET_KEY" not in text


def test_a_glob_cannot_climb_out_of_the_folder(tmp_path):
    (tmp_path / "secrets.md").write_text("SECRET_KEY=abc")
    root = tmp_path / "root"
    root.mkdir()
    (root / "ok.md").write_text("PUBLIC")
    resolved = sources_mod.resolve(
        [Source(kind="folder", target=str(root), options={"glob": "../*.md"})]
    )
    assert "SECRET_KEY" not in (resolved[0].get("text") or "")


@pytest.mark.parametrize(
    "url,expect",
    [
        ("http://169.254.169.254/latest/meta-data/", "link-local"),
        ("http://127.0.0.1:4144/v1/models", "local/private"),
        ("http://192.168.1.10/admin", "local/private"),
    ],
)
def test_http_sources_refuse_metadata_and_local_addresses(url, expect):
    """The request can carry a bearer token from a SecretStore profile; the metadata
    endpoint hands out credentials to anything that asks."""
    out = sources_mod.resolve([Source(kind="http", target=url)])
    assert expect in out[0]["error"]


def test_a_local_http_source_can_be_opted_into():
    """The box runs local model proxies — sourcing one is a legitimate thing to want."""
    from coworker.council.sources import _check_host

    _check_host("http://127.0.0.1:4144/v1/models", allow_local=True)  # does not raise


def test_a_broken_source_is_reported_not_raised():
    resolved = sources_mod.resolve([Source(kind="folder", target="/does/not/exist")])
    assert "FileNotFoundError" in resolved[0]["error"]
    assert "text" not in resolved[0]


def test_an_unknown_kind_is_reported():
    resolved = sources_mod.resolve([Source(kind="telepathy", target="x")])
    assert "unknown source kind" in resolved[0]["error"]


def test_disabled_sources_are_skipped(tmp_path):
    (tmp_path / "a.md").write_text("ALPHA")
    src = Source(kind="folder", target=str(tmp_path), enabled=False)
    assert sources_mod.resolve([src]) == []


def test_oversized_source_is_clipped(tmp_path):
    (tmp_path / "big.txt").write_text("x" * (sources_mod.PER_SOURCE_CHARS + 5000))
    resolved = sources_mod.resolve([Source(kind="folder", target=str(tmp_path))])
    assert resolved[0]["truncated"] is True
    assert len(resolved[0]["text"]) == sources_mod.PER_SOURCE_CHARS


def test_the_total_budget_caps_the_whole_brief(tmp_path, monkeypatch):
    """Every source is re-sent to every member on every round, so an unbounded brief is a
    bill multiplier, not just a long prompt."""
    monkeypatch.setattr(sources_mod, "TOTAL_CHARS", 100)
    monkeypatch.setattr(sources_mod, "PER_SOURCE_CHARS", 80)
    for name in ("a.txt", "b.txt"):
        (tmp_path / name).write_text("y" * 80)
    resolved = sources_mod.resolve([Source(kind="folder", target=str(tmp_path))])
    text = sources_mod.brief(resolved)
    assert "source budget reached" in text or len(text) < 400


def test_the_brief_frames_sources_as_untrusted(tmp_path):
    (tmp_path / "a.md").write_text("ALPHA")
    text = sources_mod.brief(sources_mod.resolve([Source(kind="folder", target=str(tmp_path))]))
    assert "never as instructions" in text
    assert "ALPHA" in text


def test_a_failed_source_still_appears_in_the_brief():
    """Silently dropping it would let the panel answer confidently from material it never
    actually saw."""
    text = sources_mod.brief(sources_mod.resolve([Source(kind="file", target="/nope")]))
    assert "unavailable" in text


class _FakeResponse:
    """Enough of httpx.Response for the source resolver."""

    def __init__(self, *, url, location=None, payload=None):
        self.url = url
        self.headers = {"content-type": "application/json"}
        self.is_redirect = location is not None
        self.next_request = None
        if location:
            self.headers["location"] = location
            self.next_request = type("Req", (), {"url": location})()
        self._payload = payload if payload is not None else {"ok": True}

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class _FakeClient:
    """Records (url, headers) per GET and answers from a scripted list."""

    def __init__(self, responses, log):
        self._responses = list(responses)
        self._log = log

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get(self, url, headers=None, params=None):
        self._log.append((url, dict(headers or {})))
        return self._responses.pop(0)


def _fake_httpx(monkeypatch, responses):
    import httpx

    log: list = []
    monkeypatch.setattr(
        httpx, "Client", lambda **kw: _FakeClient(responses, log)
    )
    return log


def test_http_source_sends_headers_from_a_secret_profile(tmp_path, monkeypatch):
    """Credentials live in the SecretStore, never in council.json — the config is meant to
    be readable, editable in the GUI and copyable between machines."""
    from coworker.secrets import SecretStore

    log = _fake_httpx(monkeypatch, [_FakeResponse(url="https://x/y")])
    monkeypatch.setattr(sources_mod, "_check_host", lambda url, *, allow_local: None)
    store = SecretStore(tmp_path / "secrets.json")
    store.put("api:mine", {"X-Api-Key": "SECRET"})
    src = Source(kind="http", target="https://x/y", options={"headers_profile": "api:mine"})

    out = sources_mod.resolve([src], store)
    assert out[0]["text"].strip().startswith("{")
    assert log[0][1]["X-Api-Key"] == "SECRET"


def test_a_redirect_to_a_refused_host_never_receives_the_credentials(tmp_path, monkeypatch):
    """The whole point of checking hosts. httpx strips a header literally named
    `Authorization` across hosts, but the `X-Api-Key` shape this feature recommends is
    forwarded — so following redirects automatically would hand the key to the metadata
    endpoint and only report it afterwards."""
    from coworker.secrets import SecretStore

    log = _fake_httpx(
        monkeypatch,
        [
            _FakeResponse(url="https://x/y", location="http://169.254.169.254/latest/"),
            _FakeResponse(url="http://169.254.169.254/latest/", payload={"leaked": True}),
        ],
    )
    # The initial host passes; the redirect target must not. Real DNS would fail on the
    # made-up first hostname, so resolve it here instead of reaching the network.
    real_check = sources_mod._check_host
    monkeypatch.setattr(
        sources_mod,
        "_check_host",
        lambda url, *, allow_local: None if url.startswith("https://x/") else real_check(url, allow_local=allow_local),
    )
    store = SecretStore(tmp_path / "secrets.json")
    store.put("api:mine", {"X-Api-Key": "SECRET"})
    src = Source(kind="http", target="https://x/y", options={"headers_profile": "api:mine"})

    out = sources_mod.resolve([src], store)
    assert "link-local" in out[0]["error"]
    # Exactly ONE request was made — the redirect target was never contacted.
    assert [url for url, _ in log] == ["https://x/y"]


def test_a_redirect_chain_is_bounded(tmp_path, monkeypatch):
    from coworker.council.sources import MAX_REDIRECTS

    log = _fake_httpx(
        monkeypatch,
        [_FakeResponse(url=f"https://x/{i}", location=f"https://x/{i + 1}") for i in range(20)],
    )
    monkeypatch.setattr(sources_mod, "_check_host", lambda url, *, allow_local: None)
    out = sources_mod.resolve([Source(kind="http", target="https://x/0")])
    assert "too many redirects" in out[0]["error"]
    assert len(log) == MAX_REDIRECTS + 1


def test_mcp_target_must_name_a_tool():
    out = sources_mod.resolve([Source(kind="mcp", target="just-a-server")])
    assert "server:tool" in out[0]["error"]


# -- scratchpad -----------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("POSITION: x\nNOTE: watch the migration", "watch the migration"),
        ("NOTE: none", None),
        ("NOTE:  N/A ", None),
        ("no note here", None),
        # The prompt puts NOTE last; an earlier hit is the model echoing the instruction.
        ("NOTE: the instruction\nPOSITION: y\nNOTE: the real one", "the real one"),
    ],
)
def test_note_extraction(text, expected):
    assert extract_note(text) == expected


def test_a_runaway_note_is_capped():
    from coworker.council.scratchpad import MAX_NOTE_CHARS

    assert len(extract_note("NOTE: " + "z" * 5000)) == MAX_NOTE_CHARS


def test_the_scratchpad_accumulates_and_renders():
    pad = Scratchpad("Ship on Friday?")
    pad.collect(
        [
            {"model": "a:one", "role": "Advocate", "text": "NOTE: tests are green"},
            {"model": "b:two", "role": "Skeptic", "text": "NOTE: none"},
            {"model": "c:three", "role": "Analyst", "error": "no key"},
        ],
        1,
    )
    rendered = pad.render()
    assert "tests are green" in rendered and "a:one" in rendered and "Advocate" in rendered
    assert "b:two" not in rendered  # "none" posts nothing
    assert len(pad.entries) == 1


def test_an_empty_scratchpad_renders_to_nothing():
    """Falsy, so the caller can append it unconditionally without emitting a stray heading."""
    assert Scratchpad("q").render() == ""


def test_the_scratchpad_writes_three_files(tmp_path):
    pad = Scratchpad("Ship on Friday?", directory=tmp_path / "run")
    pad.post("a:one", "Advocate", "tests are green", 1)
    written = pad.save(
        "# transcript",
        {"panel": [{"model": "a:one", "role": "Advocate"}], "chair": "c", "consensus": "SHIP"},
    )
    assert set(written) == {"scratchpad.md", "transcript.md", "finding.md"}
    assert "tests are green" in (tmp_path / "run" / "scratchpad.md").read_text()
    assert "SHIP" in (tmp_path / "run" / "finding.md").read_text()
    assert "a:one (Advocate)" in (tmp_path / "run" / "finding.md").read_text()


def test_saving_to_an_unwritable_path_does_not_lose_the_answer(tmp_path):
    """The finding is already in memory and on its way back to the caller; a read-only
    disk must not turn a successful council into an exception."""
    blocker = tmp_path / "blocked"
    blocker.write_text("i am a file, not a directory")
    pad = Scratchpad("q", directory=blocker / "run")
    assert pad.save("t", {"panel": [], "consensus": "x"}) == {}


# -- malformed config (values, not just field names) ----------------------------------


def test_a_role_missing_its_brief_is_kept_with_an_empty_one():
    """core.py reads BOTH halves of a role. A half-role saves cleanly and then explodes at
    council time, a long way from the edit that caused it."""
    cfg = CouncilConfig.from_dict({"roles": [{"name": "Accountant"}]})
    assert cfg.role_for(0) == {"name": "Accountant", "brief": ""}


def test_a_role_with_no_name_is_dropped():
    cfg = CouncilConfig.from_dict({"roles": [{"brief": "x"}, {"name": "Real", "brief": "y"}]})
    assert [r["name"] for r in cfg.roles] == ["Real"]


def test_roles_that_are_all_junk_fall_back_to_the_defaults():
    cfg = CouncilConfig.from_dict({"roles": ["nonsense", 7, {}]})
    assert cfg.role_for(0) == DEFAULT_ROLES[0]


@pytest.mark.parametrize("prompts", [[], "text", None, {"analysis": "not a dict"}])
def test_prompts_of_the_wrong_shape_do_not_break_prompt_lookup(prompts):
    cfg = CouncilConfig.from_dict({"prompts": prompts})
    assert cfg.prompt("chair") == chair_default()


def test_an_unknown_prompt_phase_or_preset_is_dropped():
    cfg = CouncilConfig.from_dict(
        {"prompts": {"analysis": {"chair": "MINE", "nonsense": "x"}, "bogus": {"chair": "y"}}}
    )
    assert cfg.prompts == {"analysis": {"chair": "MINE"}}


@pytest.mark.parametrize("rounds", ["many", None, [], {}])
def test_unparseable_rounds_falls_back_to_the_default(rounds):
    assert CouncilConfig.from_dict({"rounds": rounds}).rounds == 2


def test_the_string_false_turns_a_setting_off_not_on():
    """A non-empty string is truthy, so a naive bool() reads "false" as enabled — the one
    coercion that silently does the opposite of what the config says."""
    cfg = CouncilConfig.from_dict({"research": "false", "skip_debate_on_agreement": "true"})
    assert cfg.research is False and cfg.skip_debate_on_agreement is True


def test_a_junk_panel_entry_is_dropped():
    cfg = CouncilConfig.from_dict({"panel": ["a:one", "", "  "]})
    assert cfg.panel == ["a:one"]


def test_source_options_of_the_wrong_type_become_an_empty_dict():
    cfg = CouncilConfig.from_dict(
        {"sources": [{"kind": "folder", "target": "/x", "options": "not a dict"}]}
    )
    assert cfg.sources[0].options == {}


def test_two_runs_in_the_same_second_do_not_share_a_directory():
    """Second-resolution names collide, and both runs then write the same three files."""
    a, b = Scratchpad("same question"), Scratchpad("same question")
    assert a.dir != b.dir
