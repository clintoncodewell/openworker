from __future__ import annotations

import json

import httpx

from coworker.providers import ChatGPTProvider, ToolCall
from coworker.providers.chatgpt_auth import extract_account_id
from coworker.providers.chatgpt_provider import responses_input, responses_tools
from coworker.providers.matrix import models_for_provider
from coworker.providers.usage import cached_headers, clear_cache


class _Auth:
    def valid_access_token(self):
        return "access-token", "acct_123"


def test_responses_input_preserves_tool_loop_and_instructions():
    instructions, items = responses_input(
        [
            {"role": "system", "content": "Be precise."},
            {"role": "user", "content": "Read it"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": '{"path":"a.py"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "contents"},
        ]
    )

    assert instructions == "Be precise."
    assert items[0]["content"] == [{"type": "input_text", "text": "Read it"}]
    assert items[1] == {
        "type": "function_call",
        "call_id": "call_1",
        "name": "read_file",
        "arguments": '{"path":"a.py"}',
    }
    assert items[2] == {
        "type": "function_call_output",
        "call_id": "call_1",
        "output": "contents",
    }


def test_responses_tools_flattens_chat_completions_shape():
    assert responses_tools(
        [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a file",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
    ) == [
        {
            "type": "function",
            "name": "read_file",
            "description": "Read a file",
            "parameters": {"type": "object", "properties": {}},
        }
    ]


def test_stream_parses_text_and_function_calls():
    clear_cache()
    captured = {}
    sse = "\n".join(
        [
            'data: {"type":"response.output_text.delta","delta":"Working "}',
            'data: {"type":"response.output_item.added","output_index":1,"item":{"type":"function_call","call_id":"call_7","name":"write_file","arguments":""}}',
            'data: {"type":"response.function_call_arguments.delta","output_index":1,"delta":"{\\"path\\":\\"x.txt\\"}"}',
            'data: {"type":"response.completed","response":{"output":[]}}',
            "data: [DONE]",
        ]
    )

    def handler(request: httpx.Request):
        captured["headers"] = request.headers
        captured["json"] = json.loads(request.content)
        return httpx.Response(
            200,
            text=sse,
            headers={
                "content-type": "text/event-stream",
                "x-codex-primary-used-percent": "37",
                "x-codex-primary-window-minutes": "300",
                "x-codex-credits-balance": "12.5",
            },
        )

    provider = ChatGPTProvider(
        secrets=None,
        auth=_Auth(),
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    chunks = list(
        provider.stream(
            model="gpt-5.4-mini",
            messages=[{"role": "user", "content": "make it"}],
            tools=[{"type": "function", "function": {"name": "write_file"}}],
        )
    )

    assert [c.text_delta for c in chunks if c.text_delta] == ["Working "]
    assert chunks[-1].turn.tool_calls == [
        ToolCall(id="call_7", name="write_file", arguments={"path": "x.txt"})
    ]
    assert chunks[-1].turn.finish_reason == "tool_calls"
    assert captured["headers"]["chatgpt-account-id"] == "acct_123"
    assert captured["json"]["store"] is False
    assert captured["json"]["tools"][0]["name"] == "write_file"
    assert cached_headers("chatgpt") == {
        "x-codex-primary-used-percent": "37",
        "x-codex-primary-window-minutes": "300",
        "x-codex-credits-balance": "12.5",
    }


def test_extract_account_id_from_nested_claim():
    import base64

    payload = base64.urlsafe_b64encode(
        json.dumps({"https://api.openai.com/auth": {"chatgpt_account_id": "acct_x"}}).encode()
    ).decode().rstrip("=")
    assert extract_account_id(f"x.{payload}.y") == "acct_x"


def test_signed_in_profile_enables_provider_and_picker_model(tmp_path):
    from coworker.server.manager import SessionManager

    manager = SessionManager(data_dir=tmp_path)
    manager.secrets.put(
        "provider:chatgpt",
        {
            "type": "oauth",
            "access_token": "token",
            "refresh_token": "refresh",
            "expires": 9_999_999_999,
            "account_id": "acct_test",
        },
    )
    saved = manager.set_provider("chatgpt", {})

    info = next(p for p in manager.get_providers() if p["name"] == "chatgpt")
    assert saved["ok"] is True
    assert info["configured"] is True
    assert info["auth"] == "oauth"
    assert info["account"] == "acct_test"
    assert info["suggested_models"] == [
        "gpt-5.6-sol",
        "gpt-5.6-terra",
        "gpt-5.6-luna",
        "gpt-5.4-mini",
        "gpt-5.5",
        "gpt-5.4",
        "gpt-5.4-pro",
        "gpt-5-mini",
        "gpt-5.2",
    ]
    assert "chatgpt:gpt-5.4-mini" in manager.get_settings()["models"]

    manager.remove_provider("chatgpt")
    info = next(p for p in manager.get_providers() if p["name"] == "chatgpt")
    assert info["configured"] is False


def test_chatgpt_subscription_catalog_exposes_all_selectable_models():
    assert models_for_provider("chatgpt") == [
        "gpt-5.6-sol",
        "gpt-5.6-terra",
        "gpt-5.6-luna",
        "gpt-5.4-mini",
        "gpt-5.5",
        "gpt-5.4",
        "gpt-5.4-pro",
        "gpt-5-mini",
        "gpt-5.2",
    ]
