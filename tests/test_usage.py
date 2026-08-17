"""Provider usage snapshots: inference-header capture and route degradation."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from coworker.providers import AnthropicProvider, OpenAIProvider
from coworker.providers.usage import (
    cached_headers,
    capture_headers,
    clear_cache,
    usage_snapshot,
)


@pytest.fixture(autouse=True)
def _empty_usage_cache():
    clear_cache()
    yield
    clear_cache()


class _RawResponse:
    def __init__(self, parsed, headers):
        self._parsed = parsed
        self.headers = headers

    def parse(self):
        return self._parsed


class _RawResource:
    def __init__(self, parsed, headers):
        self._parsed = parsed
        self._headers = headers
        self.with_raw_response = self

    def create(self, **_kwargs):
        return _RawResponse(self._parsed, self._headers)


def test_openai_inference_headers_are_cached():
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="ok", tool_calls=[]),
                finish_reason="stop",
            )
        ]
    )
    completions = _RawResource(
        response,
        {
            "X-RateLimit-Limit-Requests": "500",
            "x-ratelimit-remaining-requests": "421",
            "x-ratelimit-reset-requests": "12s",
            "x-ratelimit-remaining-tokens": "9876",
            "server": "ignored",
        },
    )
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    OpenAIProvider(client=client).complete(
        model="gpt-5.5", messages=[{"role": "user", "content": "hi"}]
    )

    assert cached_headers("openai") == {
        "x-ratelimit-limit-requests": "500",
        "x-ratelimit-remaining-requests": "421",
        "x-ratelimit-reset-requests": "12s",
        "x-ratelimit-remaining-tokens": "9876",
    }


def test_anthropic_inference_headers_are_cached():
    response = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="ok")], stop_reason="end_turn"
    )
    messages = _RawResource(
        response,
        {
            "anthropic-ratelimit-requests-limit": "100",
            "anthropic-ratelimit-requests-remaining": "88",
            "anthropic-ratelimit-input-tokens-remaining": "12000",
            "request-id": "ignored",
        },
    )
    client = SimpleNamespace(
        messages=messages, beta=SimpleNamespace(messages=messages)
    )

    AnthropicProvider(client=client).complete(
        model="claude-sonnet-4-6",
        messages=[{"role": "user", "content": "hi"}],
    )

    assert cached_headers("anthropic") == {
        "anthropic-ratelimit-requests-limit": "100",
        "anthropic-ratelimit-requests-remaining": "88",
        "anthropic-ratelimit-input-tokens-remaining": "12000",
    }


def test_chatgpt_live_poll_is_supplemented_by_cached_headers(tmp_path, monkeypatch):
    from coworker.providers import usage
    from coworker.secrets import SecretStore

    store = SecretStore(tmp_path / "secrets.json")
    store.put(
        "provider:chatgpt",
        {"type": "oauth", "access_token": "token", "account_id": "acct"},
    )
    capture_headers(
        "chatgpt",
        {
            "x-codex-secondary-used-percent": "25",
            "x-codex-secondary-reset-at": "2000000000",
            "x-codex-credits-balance": "8.5",
        },
    )
    monkeypatch.setattr(
        usage,
        "poll_chatgpt_usage",
        lambda *_args, **_kwargs: {
            "source": "live_poll",
            "plan": "plus",
            "windows": [
                {
                    "id": "primary",
                    "label": "Primary window",
                    "used_percent": 40,
                    "remaining_percent": 60,
                }
            ],
        },
    )

    body = usage_snapshot(
        store,
        chatgpt_auth=SimpleNamespace(
            valid_access_token=lambda: ("token", "acct")
        ),
    )

    chatgpt = body["providers"][0]
    assert [window["id"] for window in chatgpt["windows"]] == [
        "primary",
        "secondary",
    ]
    assert chatgpt["credits"] == {"balance": 8.5, "has_credits": None}
    assert chatgpt["source"] == "live_poll"


def test_usage_route_degrades_failed_poll_and_omits_unconfigured(
    tmp_path, monkeypatch
):
    from coworker.providers import usage
    from coworker.server.app import create_app
    from coworker.server.manager import SessionManager

    for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OPENWORKER_GEMINI_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(usage, "poll_chatgpt_usage", lambda *_args, **_kwargs: None)

    manager = SessionManager(data_dir=tmp_path / "data")
    manager.secrets.put(
        "provider:chatgpt",
        {
            "type": "oauth",
            "access_token": "token",
            "account_id": "acct_test",
            "expires": 9_999_999_999,
        },
    )
    app = create_app(manager)
    route = next(route for route in app.routes if getattr(route, "path", "") == "/v1/usage")
    body = route.endpoint()

    chatgpt = next(provider for provider in body["providers"] if provider["id"] == "chatgpt")
    assert chatgpt["status"] == "unavailable"
    assert chatgpt["message"] == "Temporarily unavailable"
    assert "zai-coding" not in {provider["id"] for provider in body["providers"]}
    assert "ollama" not in {provider["id"] for provider in body["providers"]}
