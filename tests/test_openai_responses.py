"""OpenAI Responses routing and wire-format translation, with no network calls."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from coworker.council.usage import extract
from coworker.providers import OpenAIProvider, ToolCall, capabilities_for


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
            "strict": True,
        },
    }
]


def _response(*, text="I will inspect it.", status="completed", usage=None):
    output = [
        SimpleNamespace(
            type="message",
            content=[SimpleNamespace(type="output_text", text=text)],
        ),
        SimpleNamespace(
            type="function_call",
            call_id="call_1",
            name="read_file",
            arguments=json.dumps({"path": "a.py"}),
        ),
    ]
    return SimpleNamespace(output=output, status=status, usage=usage)


class _FakeResponses:
    def __init__(self, response, events=None):
        self.response = response
        self.events = events
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return iter(self.events) if kwargs.get("stream") else self.response


class _FakeChat:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise RuntimeError(self.error)
        return self.response


def _client(response, *, chat=None, events=None):
    return SimpleNamespace(
        responses=_FakeResponses(response, events),
        chat=SimpleNamespace(completions=chat or _FakeChat()),
    )


def _chat_response(text="chat worked"):
    message = SimpleNamespace(content=text, tool_calls=None)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="stop")]
    )


def test_responses_tool_turn_returns_text_and_parsed_tool_calls():
    client = _client(_response())
    provider = OpenAIProvider(client=client, supports_responses=True)

    turn = provider.complete(
        model="gpt-5.6-sol",
        messages=[{"role": "user", "content": "Read a.py"}],
        tools=TOOLS,
        reasoning_effort="high",
    )

    assert turn.text == "I will inspect it."
    assert turn.tool_calls == [
        ToolCall(id="call_1", name="read_file", arguments={"path": "a.py"})
    ]
    assert turn.finish_reason == "tool_calls" and turn.raw is client.responses.response
    sent = client.responses.calls[0]
    assert sent["reasoning"] == {"effort": "high"}
    assert sent["tools"][0] == {
        "type": "function",
        "name": "read_file",
        "description": "Read a file",
        "parameters": TOOLS[0]["function"]["parameters"],
        "strict": True,
    }


def test_responses_input_round_trips_assistant_calls_and_tool_results():
    client = _client(_response())
    provider = OpenAIProvider(client=client, supports_responses=True)
    messages = [
        {"role": "system", "content": "System rule"},
        {"role": "developer", "content": "Developer rule"},
        {"role": "user", "content": "Read it"},
        {
            "role": "assistant",
            "content": "Checking.",
            "tool_calls": [
                {
                    "id": "call_old",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": json.dumps({"path": "old.py"}),
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_old", "content": "contents"},
    ]

    provider.complete(model="gpt-5.6-sol", messages=messages, tools=TOOLS)

    assert client.responses.calls[0]["input"] == [
        {"role": "system", "content": "System rule"},
        {"role": "developer", "content": "Developer rule"},
        {"role": "user", "content": "Read it"},
        {"role": "assistant", "content": "Checking."},
        {
            "type": "function_call",
            "call_id": "call_old",
            "name": "read_file",
            "arguments": json.dumps({"path": "old.py"}),
        },
        {"type": "function_call_output", "call_id": "call_old", "output": "contents"},
    ]


def test_chat_compat_model_is_not_routed_to_responses():
    chat = _FakeChat(_chat_response())
    client = _client(_response(), chat=chat)
    provider = OpenAIProvider(
        client=client,
        base_url="http://localhost:11434/v1",
        supports_responses=False,
    )

    turn = provider.complete(model="qwen3-coder", messages=[], tools=TOOLS)

    assert turn.text == "chat worked"
    assert len(chat.calls) == 1 and client.responses.calls == []


def test_chat_effort_400_falls_back_to_responses_once():
    error = (
        "Error code: 400 - Function tools with reasoning_effort are not supported "
        "for gpt-5.7-sol in /v1/chat/completions. Please use /v1/responses instead."
    )
    chat = _FakeChat(error=error)
    client = _client(_response(), chat=chat)
    provider = OpenAIProvider(client=client, supports_responses=False)

    turn = provider.complete(model="gpt-5.7-sol", messages=[], tools=TOOLS)

    assert turn.tool_calls[0].name == "read_file"
    assert len(chat.calls) == 1 and len(client.responses.calls) == 1


def test_responses_raw_preserves_usage_for_council_accounting():
    usage = SimpleNamespace(input_tokens=17, output_tokens=9)
    client = _client(_response(usage=usage))
    turn = OpenAIProvider(client=client, supports_responses=True).complete(
        model="gpt-5.6-sol", messages=[], tools=TOOLS
    )

    assert extract(turn.raw) == {"input": 17, "output": 9, "reasoning": 0}
    assert capabilities_for("azure:gpt-5.6-sol").tools is True


def test_responses_stream_yields_native_deltas_and_final_turn():
    final = _response(text="Hello")
    events = [
        SimpleNamespace(type="response.output_text.delta", delta="Hel"),
        SimpleNamespace(type="response.output_text.delta", delta="lo"),
        SimpleNamespace(type="response.completed", response=final),
    ]
    client = _client(final, events=events)

    chunks = list(
        OpenAIProvider(client=client, supports_responses=True).stream(
            model="gpt-5.6-sol", messages=[], tools=TOOLS
        )
    )

    assert [chunk.text_delta for chunk in chunks if chunk.text_delta] == ["Hel", "lo"]
    assert chunks[-1].turn.text == "Hello"
    assert chunks[-1].turn.tool_calls[0].name == "read_file"
    assert chunks[-1].turn.raw is final


# -- what the review caught ------------------------------------------------------------


def test_two_different_param_fixes_in_a_row_still_send_the_request():
    """Effort first, then max_tokens. The loop was left without the clause that actually
    sends the twice-fixed request, so the caller got an UnboundLocalError instead of either
    the answer or the server's own 400."""

    class _TwoFaults:
        def __init__(self, response):
            self._response = response
            self.calls: list[dict] = []

        def create(self, **kwargs):
            self.calls.append(dict(kwargs))
            if kwargs.get("tools") and kwargs.get("reasoning_effort") != "none":
                raise RuntimeError(
                    "Error code: 400 - Function tools with reasoning_effort are not "
                    "supported for gpt-5.7-x in /v1/chat/completions. To use function "
                    "tools, use /v1/responses or set reasoning_effort to 'none'."
                )
            if "max_tokens" in kwargs:
                raise RuntimeError("Error code: 400 - 'max_tokens' is not supported")
            return self._response

    chat = _TwoFaults(_chat_response())
    client = _client(_response(), chat=chat)
    turn = OpenAIProvider(client=client, supports_responses=False).complete(
        model="gpt-5.7-x", messages=[], tools=TOOLS, max_tokens=64
    )
    assert turn.text == "chat worked"
    assert len(chat.calls) == 3
    assert chat.calls[-1]["reasoning_effort"] == "none"
    assert "max_tokens" not in chat.calls[-1] and chat.calls[-1]["max_completion_tokens"] == 64


def test_a_failed_escalation_reports_the_servers_own_complaint():
    """The escalation is a guess from an error string, and it is attempted even where the
    endpoint may not exist. A 404 from that guess would replace an actionable message
    ("use /v1/responses") with one that explains nothing."""

    class _NoResponses:
        def create(self, **kwargs):
            raise RuntimeError("Error code: 404 - Unrecognized request URL")

    error = (
        "Error code: 400 - Function tools with reasoning_effort are not supported "
        "for gpt-5.7-sol in /v1/chat/completions. Please use /v1/responses instead."
    )
    client = _client(_response(), chat=_FakeChat(error=error))
    client.responses = _NoResponses()

    with pytest.raises(Exception) as caught:
        OpenAIProvider(client=client, supports_responses=False).complete(
            model="gpt-5.7-sol", messages=[], tools=TOOLS
        )
    assert "/v1/responses" in str(caught.value) and "404" not in str(caught.value)


def test_a_failed_generation_raises_instead_of_returning_an_empty_turn():
    """Responses reports trouble in the body of a 200. Passed through, it reaches the agent
    loop as the model choosing to say nothing, with no reason attached."""
    failed = _response(text="")
    failed.status = "failed"
    failed.error = SimpleNamespace(message="content filter")
    with pytest.raises(RuntimeError, match="content filter"):
        OpenAIProvider(client=_client(failed), supports_responses=True).complete(
            model="gpt-5.6-sol", messages=[], tools=TOOLS
        )


def test_a_truncated_response_reports_the_same_word_as_the_chat_path():
    """Responses says "incomplete" where chat/completions says "length". Consumers check
    for one of those, and it is not the new one."""
    cut = _response(text="half a sen")
    cut.output = [cut.output[0]]  # text only: a tool call would set finish_reason first
    cut.status = "incomplete"
    turn = OpenAIProvider(client=_client(cut), supports_responses=True).complete(
        model="gpt-5.6-sol", messages=[], tools=TOOLS
    )
    assert turn.finish_reason == "length"


def test_stock_openai_is_not_rerouted_on_a_hunch():
    """The refusal was measured on Azure. gpt-5.6-sol is also stock OpenAI's default model,
    and changing the main path there on unmeasured evidence is the bigger risk. A provider
    that never declared support keeps the endpoint it has always used."""
    chat = _FakeChat(_chat_response())
    client = _client(_response(), chat=chat)
    OpenAIProvider(client=client).complete(model="gpt-5.6-sol", messages=[], tools=TOOLS)
    assert len(chat.calls) == 1 and client.responses.calls == []
