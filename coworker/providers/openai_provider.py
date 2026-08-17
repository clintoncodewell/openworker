"""OpenAI provider — the v1 model access implementation.

Uses the OpenAI Python SDK, preferring `chat.completions` except for models whose
tool calling is only available through Responses.
"""

from __future__ import annotations

import json
import re
from types import SimpleNamespace
from typing import Any, Optional

from .base import (
    AssistantTurn,
    ModelCapabilities,
    ProviderClient,
    StreamChunk,
    ToolCall,
)
from .capabilities import capabilities_for
from .usage import capture_headers


def resolve_api_key(secrets: Any = None) -> Optional[str]:
    """Resolve the OpenAI API key: env `OPENAI_API_KEY` first, else the SecretStore
    `provider:openai` profile (`{api_key}`). Lets a Tauri-launched sidecar — which does NOT
    inherit the shell env — still find a key the user entered in Settings. The value never
    enters the model context; it only configures the SDK client.
    """
    import os

    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key
    if secrets is not None:
        profile = secrets.get("provider:openai") or {}
        return profile.get("api_key") or None
    return None


# Most GPT-5.6 deployments accept tools on chat/completions only at effort "none".
# Sol is the exception: Azure rejects tools there even at "none", so it takes the
# Responses path below while Terra/Luna keep the narrower compatibility workaround.
_EFFORT_ERROR = "function tools with reasoning_effort are not supported"
_RESPONSES_TOOL_MODELS = frozenset({"gpt-5.6-sol"})
# The server names the endpoint in its own rejection ("Please use /v1/responses instead").
# That is decisive and worth more than our allow-list, so it escalates on the first 400 and
# a future alias recovers without being listed here. A 400 that complains about the effort
# WITHOUT naming an endpoint is the older, narrower problem — Terra and Luna serve tools on
# chat/completions once effort is "none" — and that one still earns its cheap retry.
_RESPONSES_HINT = "/v1/responses"


def _wants_responses(exc: Exception, kwargs: dict[str, Any]) -> bool:
    """Should this rejection move the call to the Responses API?

    Yes when the server named that endpoint, or when it has already refused the effort at
    "none" and so has nothing cheaper left to offer. Escalating any earlier would push a
    model that merely wanted effort="none" onto an endpoint its vendor may not implement.
    """
    msg = str(exc).lower()
    if _EFFORT_ERROR not in msg:
        return False
    # Already refused at "none": nothing cheaper is left, whatever the wording said. This
    # is the backstop that catches any phrasing the two clauses below did not anticipate.
    if kwargs.get("reasoning_effort") == "none":
        return True
    # Otherwise take the server at its word. Sol says "Please use /v1/responses instead"
    # and offers nothing else, so the endpoint is the problem. Other deployments say "use
    # /v1/responses OR set reasoning_effort to 'none'" — that second clause is a cheaper
    # way out on an endpoint their vendor definitely implements, so take it first.
    return _RESPONSES_HINT in msg and "reasoning_effort to" not in msg


def _responses_or_original(
    exc: Exception,
    provider: "OpenAIProvider",
    client: Any,
    model: str,
    messages: list[dict[str, Any]],
    tools: Optional[list[dict[str, Any]]],
    settings: dict[str, Any],
) -> Any:
    """Escalate to the Responses API, but keep the server's own complaint if that fails.

    The escalation is a guess based on an error string, and it is attempted even for
    endpoints not known to implement /v1/responses — a relay in front of Azure will forward
    Azure's "use /v1/responses" verbatim while being perfectly capable of serving it. When
    the guess is wrong the SDK raises a 404, and letting THAT surface would replace an
    actionable message ("use /v1/responses") with one that explains nothing.
    """
    try:
        return _complete_responses(provider, client, model, messages, tools, settings)
    except Exception:
        raise exc from None


def _pin_reasoning_effort(kwargs: dict[str, Any]) -> None:
    if kwargs.get("tools") and str(kwargs.get("model", "")).startswith("gpt-5.6"):
        kwargs.setdefault("reasoning_effort", "none")


def _delta_reasoning(obj: Any) -> Optional[str]:
    """Thinking text off a delta/message: `reasoning_content` (DeepSeek, GLM, Kimi, and
    most compat vendors) or `reasoning` (xAI, OpenRouter). Extra response fields survive
    the OpenAI SDK's models (extra="allow"), so plain getattr sees them."""
    value = getattr(obj, "reasoning_content", None) or getattr(obj, "reasoning", None)
    return value if isinstance(value, str) and value else None


def _strip_foreign_sidecars(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop provider-private message sidecars (underscore-prefixed keys, e.g. `_gemini`
    thought signatures — see providers/base.py): they belong to other providers, and the
    OpenAI wire (and its compat servers) rejects unknown message fields."""
    return [
        (
            {k: v for k, v in m.items() if not k.startswith("_")}
            if any(k.startswith("_") for k in m)
            else m
        )
        for m in messages
    ]


_MAX_TOKENS_ERROR = "'max_tokens' is not supported"


def _param_fix_retry(kwargs: dict[str, Any], exc: Exception) -> dict[str, Any]:
    """Kwargs for the one retry an unsupported-parameter error earns, or re-raise.

    Reasoning-routed OpenAI models reject `max_tokens` outright (they want
    `max_completion_tokens`) — but compat servers (Ollama's /v1) know ONLY
    `max_tokens`, so the swap must happen on rejection, never up front.
    """
    msg = str(exc).lower()
    if _EFFORT_ERROR in msg and kwargs.get("reasoning_effort") != "none":
        return {**kwargs, "reasoning_effort": "none"}
    if _MAX_TOKENS_ERROR in msg and "max_tokens" in kwargs:
        fixed = dict(kwargs)
        fixed["max_completion_tokens"] = fixed.pop("max_tokens")
        return fixed
    raise exc


class OpenAIProvider(ProviderClient):
    def __init__(
        self,
        client: Any = None,
        *,
        default_model: str = "gpt-5.6-sol",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        secrets: Any = None,
        supports_responses: Optional[bool] = None,
        usage_provider_id: Optional[str] = "openai",
    ):
        # The SDK client is built lazily on first use, NOT at construction. This lets an engine
        # be assembled before any key exists — the desktop app lets you enter the key in Settings
        # *after* launch — and the super-agent engine to be built at startup with no key. The key
        # is resolved at call time: explicit `api_key` → env `OPENAI_API_KEY` → SecretStore. Tests
        # inject a `client` directly, bypassing all of this.
        #
        # `base_url` points the same OpenAI SDK at any OpenAI-compatible endpoint — used by the
        # provider router for Ollama (`http://localhost:11434/v1`, with a placeholder key) and,
        # later, other OpenAI-shaped backends. When None, behavior is identical to stock OpenAI.
        self._client = client
        self._api_key = api_key
        self._base_url = base_url
        self._secrets = secrets
        # The stock OpenAI endpoint supports Responses. Custom endpoints opt in at
        # construction because most OpenAI-compatible vendors implement chat only.
        # OPT-IN, and defaulting to off is the point. The refusal that motivates the
        # Responses path was measured on Azure Foundry; nobody has probed whether the same
        # model name behaves the same way on api.openai.com, where it is also the DEFAULT
        # model. A provider that has not been checked keeps the endpoint it has always used
        # and, if it turns out to need the other one, recovers through the server's own 400
        # at the cost of one round trip.
        self._supports_responses = bool(supports_responses)
        # OpenAI-compatible vendors reuse this class but must not overwrite OpenAI's
        # own throughput snapshot. Their v1 usage signals are intentionally out of scope.
        self._usage_provider_id = usage_provider_id
        self.default_model = default_model

    def _create(self, resource: Any, **kwargs: Any) -> Any:
        """Send through the SDK's raw-response facade when present so headers survive."""
        raw_resource = getattr(resource, "with_raw_response", None)
        if raw_resource is not None:
            raw = raw_resource.create(**kwargs)
            capture_headers(self._usage_provider_id, getattr(raw, "headers", None))
            return raw.parse()
        response = resource.create(**kwargs)
        capture_headers(
            self._usage_provider_id,
            getattr(response, "response_headers", None)
            or getattr(getattr(response, "_response", None), "headers", None),
        )
        return response

    def _responses_required(self, model: str, tools: Optional[list[dict[str, Any]]]) -> bool:
        # Keep the proactive list exact: routing a merely OpenAI-shaped vendor to an
        # endpoint it does not implement is worse than waiting for its explicit 400.
        # Deliberately narrow. The refusal was measured on Azure Foundry; nobody has probed
        # whether the same model name behaves the same way on api.openai.com, and it is that
        # provider's DEFAULT model — so guessing there would change the main path on
        # evidence we do not have. Stock OpenAI keeps the endpoint it has always used and,
        # if it turns out to need the other one, recovers through the 400 below at the cost
        # of one round trip.
        return bool(tools and self._supports_responses and model in _RESPONSES_TOOL_MODELS)

    def _ensure_client(self) -> Any:
        if self._client is None:
            # Lazy import so the SDK is only required when actually talking to OpenAI.
            from openai import OpenAI

            key = self._api_key or resolve_api_key(self._secrets)
            if not key:
                raise RuntimeError(
                    "No model API key configured. Set OPENAI_API_KEY in the environment, "
                    "or add your key in Manage → Settings."
                )
            kwargs: dict[str, Any] = {"api_key": key}
            if self._base_url:
                kwargs["base_url"] = self._base_url
            self._client = OpenAI(**kwargs)
        return self._client

    def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]] = None,
        **settings: Any,
    ) -> AssistantTurn:
        client = self._ensure_client()
        if self._responses_required(model, tools):
            return _complete_responses(self, client, model, messages, tools, settings)

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": _strip_foreign_sidecars(messages),
            **settings,
        }
        if tools:
            kwargs["tools"] = tools
        _pin_reasoning_effort(kwargs)

        # One retry is enough: the only chat parameter rewrite left is max_tokens.
        # Two fixes can both be needed, in sequence: effort first, then max_tokens. The
        # `else` is what actually SENDS the twice-fixed request — without it the loop ends
        # with `response` unassigned and the caller gets an UnboundLocalError instead of
        # either the answer or the server's own 400.
        for _ in range(2):
            try:
                response = self._create(client.chat.completions, **kwargs)
                break
            except Exception as exc:
                if tools and _wants_responses(exc, kwargs):
                    return _responses_or_original(
                        exc, self, client, model, messages, tools, settings
                    )
                kwargs = _param_fix_retry(kwargs, exc)
        else:
            response = self._create(client.chat.completions, **kwargs)
        choice = response.choices[0]
        message = choice.message
        text = getattr(message, "content", None)
        tool_calls = _parse_tool_calls(getattr(message, "tool_calls", None))
        text, tool_calls = _maybe_salvage_tool_calls(text, tool_calls, tools=tools)
        return AssistantTurn(
            text=text,
            tool_calls=tool_calls,
            finish_reason=getattr(choice, "finish_reason", None),
            raw=response,
            reasoning=_delta_reasoning(message),
        )

    def capabilities(self, model: str) -> ModelCapabilities:
        return capabilities_for(model)

    def stream(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]] = None,
        **settings: Any,
    ):
        client = self._ensure_client()
        if self._responses_required(model, tools):
            yield from _stream_responses(self, client, model, messages, tools, settings)
            return

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": _strip_foreign_sidecars(messages),
            "stream": True,
            **settings,
        }
        if tools:
            kwargs["tools"] = tools
        _pin_reasoning_effort(kwargs)
        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_accum: dict[int, dict[str, str]] = {}
        finish_reason = None

        # One retry is enough: the only chat parameter rewrite left is max_tokens.
        # See `complete()`: the `else` sends the twice-fixed request, and without it
        # `chunks` is unassigned when both fixes were needed.
        for _ in range(2):
            try:
                chunks = self._create(client.chat.completions, **kwargs)
                break
            except Exception as exc:
                if tools and _wants_responses(exc, kwargs):
                    try:
                        stream = _stream_responses(self, client, model, messages, tools, settings)
                        first = next(stream, None)
                    except Exception:
                        raise exc from None  # see _responses_or_original
                    if first is not None:
                        yield first
                        yield from stream
                    return
                kwargs = _param_fix_retry(kwargs, exc)
        else:
            chunks = self._create(client.chat.completions, **kwargs)
        for chunk in chunks:
            choices = getattr(chunk, "choices", None)
            if not choices:
                continue
            choice = choices[0]
            delta = getattr(choice, "delta", None)
            if delta is not None:
                reasoning = _delta_reasoning(delta)
                if reasoning:
                    reasoning_parts.append(reasoning)
                    yield StreamChunk(reasoning_delta=reasoning)
                content = getattr(delta, "content", None)
                if content:
                    text_parts.append(content)
                    yield StreamChunk(text_delta=content)
                for tc in getattr(delta, "tool_calls", None) or []:
                    acc = tool_accum.setdefault(
                        getattr(tc, "index", 0), {"id": "", "name": "", "args": ""}
                    )
                    if getattr(tc, "id", None):
                        acc["id"] = tc.id
                    fn = getattr(tc, "function", None)
                    if fn is not None:
                        if getattr(fn, "name", None):
                            acc["name"] = fn.name
                        if getattr(fn, "arguments", None):
                            acc["args"] += fn.arguments
            if getattr(choice, "finish_reason", None):
                finish_reason = choice.finish_reason

        tool_calls = []
        for index in sorted(tool_accum):
            acc = tool_accum[index]
            try:
                arguments = json.loads(acc["args"]) if acc["args"] else {}
            except (TypeError, json.JSONDecodeError):
                arguments = {"_raw": acc["args"]}
            tool_calls.append(
                ToolCall(id=acc["id"], name=acc["name"], arguments=arguments)
            )

        text, tool_calls = _maybe_salvage_tool_calls(
            "".join(text_parts) or None, tool_calls, tools=tools
        )
        yield StreamChunk(
            turn=AssistantTurn(
                text=text,
                tool_calls=tool_calls,
                finish_reason=finish_reason,
                reasoning="".join(reasoning_parts) or None,
            )
        )


def _parse_tool_calls(raw_tool_calls: Any) -> list[ToolCall]:
    calls: list[ToolCall] = []
    for tc in raw_tool_calls or []:
        function = getattr(tc, "function", None)
        raw_args = getattr(function, "arguments", None) if function else getattr(tc, "arguments", None)
        try:
            arguments = json.loads(raw_args) if raw_args else {}
        except (TypeError, json.JSONDecodeError):
            # Surface unparseable arguments rather than dropping the call; the engine
            # can return a tool-error so the model corrects itself.
            arguments = {"_raw": raw_args}
        calls.append(
            ToolCall(
                id=getattr(tc, "call_id", None) or getattr(tc, "id", ""),
                name=getattr(function, "name", None) or getattr(tc, "name", ""),
                arguments=arguments,
            )
        )
    return calls


def _get(obj: Any, name: str, default: Any = None) -> Any:
    return obj.get(name, default) if isinstance(obj, dict) else getattr(obj, name, default)


def _responses_content(content: Any, *, assistant: bool = False) -> Any:
    if isinstance(content, str):
        return content
    text_type = "output_text" if assistant else "input_text"
    parts: list[dict[str, Any]] = []
    for part in content or []:
        if not isinstance(part, dict):
            continue
        kind = part.get("type")
        if kind in ("text", "input_text", "output_text") and part.get("text"):
            parts.append({"type": text_type, "text": part["text"]})
        elif kind in ("image_url", "input_image") and not assistant:
            image = part.get("image_url")
            url = image.get("url") if isinstance(image, dict) else image
            if url:
                parts.append({"type": "input_image", "image_url": url})
        elif kind in ("file", "input_file") and not assistant:
            file = part.get("file") or {}
            data = file.get("file_data") or part.get("file_data")
            if data:
                item = {"type": "input_file", "file_data": data}
                filename = file.get("filename") or part.get("filename")
                if filename:
                    item["filename"] = filename
                parts.append(item)
    return parts


def _responses_input(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for message in _strip_foreign_sidecars(messages):
        role = message.get("role")
        content = message.get("content")
        if role == "tool":
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": message.get("tool_call_id") or "",
                    "output": content if isinstance(content, str) else json.dumps(content),
                }
            )
            continue
        converted = _responses_content(content, assistant=role == "assistant")
        if converted:
            items.append({"role": role, "content": converted})
        if role == "assistant":
            for call in message.get("tool_calls") or []:
                function = call.get("function") or {}
                arguments = function.get("arguments") or "{}"
                items.append(
                    {
                        "type": "function_call",
                        "call_id": call.get("id") or "",
                        "name": function.get("name") or "",
                        "arguments": arguments if isinstance(arguments, str) else json.dumps(arguments),
                    }
                )
    return items


def _responses_tools(tools: Optional[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for tool in tools or []:
        function = tool.get("function") or {}
        item: dict[str, Any] = {
            "type": "function",
            "name": function.get("name") or "",
            "parameters": function.get("parameters") or {"type": "object", "properties": {}},
        }
        for key in ("description", "strict"):
            if key in function:
                item[key] = function[key]
        converted.append(item)
    return converted


def _responses_kwargs(
    model: str,
    messages: list[dict[str, Any]],
    tools: Optional[list[dict[str, Any]]],
    settings: dict[str, Any],
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"model": model, "input": _responses_input(messages)}
    converted_tools = _responses_tools(tools)
    if converted_tools:
        kwargs["tools"] = converted_tools
    for key in ("temperature", "top_p", "parallel_tool_calls", "tool_choice"):
        if key in settings:
            kwargs[key] = settings[key]
    max_tokens = settings.get("max_output_tokens")
    if max_tokens is None:
        max_tokens = settings.get("max_completion_tokens", settings.get("max_tokens"))
    if max_tokens is not None:
        kwargs["max_output_tokens"] = max_tokens
    if "reasoning_effort" in settings:
        kwargs["reasoning"] = {"effort": settings["reasoning_effort"]}
    return kwargs


def _responses_text(response: Any) -> Optional[str]:
    direct = _get(response, "output_text")
    if isinstance(direct, str) and direct:
        return direct
    parts: list[str] = []
    for item in _get(response, "output", []) or []:
        if _get(item, "type") != "message":
            continue
        for content in _get(item, "content", []) or []:
            if _get(content, "type") in ("output_text", "text") and _get(content, "text"):
                parts.append(str(_get(content, "text")))
    return "".join(parts) or None


def _responses_reasoning(response: Any) -> Optional[str]:
    parts: list[str] = []
    for item in _get(response, "output", []) or []:
        if _get(item, "type") != "reasoning":
            continue
        for part in (_get(item, "summary", []) or []) + (_get(item, "content", []) or []):
            if _get(part, "text"):
                parts.append(str(_get(part, "text")))
    return "".join(parts) or None


def _responses_tool_calls(response: Any) -> list[ToolCall]:
    return _parse_tool_calls(
        item for item in (_get(response, "output", []) or []) if _get(item, "type") == "function_call"
    )


# Responses reports trouble in the BODY of a 200, where chat/completions would have raised.
# Passing that through as an ordinary turn hands the agent loop an empty assistant message
# and no reason for it, so a genuine failure reads as the model choosing to say nothing.
_RESPONSES_TRUNCATED = "incomplete"


def _responses_finish_reason(response: Any, tool_calls: list[ToolCall]) -> Optional[str]:
    if tool_calls:
        return "tool_calls"
    status = _get(response, "status")
    if status == "completed":
        return "stop"
    # Truncation is "incomplete" here and "length" on chat/completions. Report the name
    # every existing consumer already checks for, rather than a second word for one thing.
    if status == _RESPONSES_TRUNCATED:
        return "length"
    return status


def _raise_if_failed(response: Any) -> None:
    """A failed Responses call arrives as HTTP 200 with `status: "failed"`. Raise, so it
    reaches the caller the same way a chat/completions failure would."""
    if _get(response, "status") != "failed":
        return
    error = _get(response, "error")
    detail = _get(error, "message") if error is not None else None
    raise RuntimeError(f"the Responses API reported a failed generation: {detail or 'no detail given'}")


def _complete_responses(
    provider: OpenAIProvider,
    client: Any,
    model: str,
    messages: list[dict[str, Any]],
    tools: Optional[list[dict[str, Any]]],
    settings: dict[str, Any],
) -> AssistantTurn:
    response = provider._create(
        client.responses, **_responses_kwargs(model, messages, tools, settings)
    )
    _raise_if_failed(response)
    tool_calls = _responses_tool_calls(response)
    return AssistantTurn(
        text=_responses_text(response),
        tool_calls=tool_calls,
        finish_reason=_responses_finish_reason(response, tool_calls),
        raw=response,
        reasoning=_responses_reasoning(response),
    )


def _stream_responses(
    provider: OpenAIProvider,
    client: Any,
    model: str,
    messages: list[dict[str, Any]],
    tools: Optional[list[dict[str, Any]]],
    settings: dict[str, Any],
):
    kwargs = _responses_kwargs(model, messages, tools, settings)
    events = provider._create(client.responses, **kwargs, stream=True)
    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    calls: dict[int, dict[str, str]] = {}
    final_response: Any = None
    for event in events:
        kind = _get(event, "type")
        if kind == "response.output_text.delta" and _get(event, "delta"):
            delta = str(_get(event, "delta"))
            text_parts.append(delta)
            yield StreamChunk(text_delta=delta)
        elif kind in ("response.reasoning_summary_text.delta", "response.reasoning_text.delta") and _get(event, "delta"):
            delta = str(_get(event, "delta"))
            reasoning_parts.append(delta)
            yield StreamChunk(reasoning_delta=delta)
        elif kind in ("response.output_item.added", "response.output_item.done"):
            item = _get(event, "item")
            if _get(item, "type") == "function_call":
                index = int(_get(event, "output_index", 0) or 0)
                calls[index] = {
                    "id": str(_get(item, "call_id") or _get(item, "id") or ""),
                    "name": str(_get(item, "name") or ""),
                    "args": str(_get(item, "arguments") or calls.get(index, {}).get("args", "")),
                }
        elif kind == "response.function_call_arguments.delta":
            index = int(_get(event, "output_index", 0) or 0)
            calls.setdefault(index, {"id": "", "name": "", "args": ""})["args"] += str(
                _get(event, "delta") or ""
            )
        elif kind == "response.completed":
            final_response = _get(event, "response")

    if final_response is not None:
        final_text = _responses_text(final_response)
        if not text_parts and final_text:
            text_parts.append(final_text)
            yield StreamChunk(text_delta=final_text)
        if not reasoning_parts:
            final_reasoning = _responses_reasoning(final_response)
            if final_reasoning:
                reasoning_parts.append(final_reasoning)
        final_calls = _responses_tool_calls(final_response)
    else:
        final_calls = []
    tool_calls = final_calls or _parse_tool_calls(
        SimpleNamespace(
            call_id=call["id"], name=call["name"], arguments=call["args"]
        )
        for _, call in sorted(calls.items())
    )
    yield StreamChunk(
        turn=AssistantTurn(
            text="".join(text_parts) or None,
            tool_calls=tool_calls,
            finish_reason=_responses_finish_reason(final_response, tool_calls),
            raw=final_response,
            reasoning="".join(reasoning_parts) or None,
        )
    )


# Some OpenAI-compatible backends — notably Ollama for several local models (qwen, etc.) —
# fail to populate the structured `tool_calls` field and instead emit the call as TEXT, in
# wildly varied shapes: a `<tool_call>{…}</tool_call>` block, a bare `{"name","arguments"}` object
# (often mixed in with prose), or a `toolname {args}` / `toolname [args]` shorthand. Our agent
# loop needs structured calls, so we recover them — using the requested tool SCHEMAS to recognize
# tool-name forms and to filter out anything whose name isn't a real tool (no false positives).
# Gated on: tools were requested AND no structured calls came back. Never fires for OpenAI.
_TOOLCALL_OPEN = re.compile(r"<tool_call>\s*", re.IGNORECASE)

# Qwen/Hermes native tool-call template — NOT JSON. The model writes the call as nested XML:
#   <function=write_file><parameter=path>hello.txt</parameter><parameter=content>hi</parameter></function>
# (usually wrapped in <tool_call>…</tool_call>). qwen3-coder emits exactly this, so we parse the
# function/parameter tags directly. Values are taken verbatim (stripped); only no-whitespace JSON
# tokens (numbers, bools, objects/arrays) are coerced, so free-text content stays a string.
_FUNCTION_BLOCK = re.compile(
    r"<function\s*=\s*(?P<name>[^>\s]+)\s*>(?P<body>.*?)</function\s*>",
    re.IGNORECASE | re.DOTALL,
)
_PARAM_BLOCK = re.compile(
    r"<parameter\s*=\s*(?P<key>[^>\s]+)\s*>(?P<val>.*?)</parameter\s*>",
    re.IGNORECASE | re.DOTALL,
)


def _coerce_param(raw: str) -> Any:
    """Keep free-text verbatim (the common case: file content), but recover real JSON values when
    the whole token is unambiguous JSON (no embedded whitespace) — e.g. `3`, `true`, `{"a":1}`.
    """
    s = raw.strip()
    if s and not any(c.isspace() for c in s):
        v = _loads(s)
        if isinstance(v, (dict, list, int, float, bool)):
            return v
    return s


def _maybe_salvage_tool_calls(
    text: Optional[str],
    tool_calls: list[ToolCall],
    *,
    tools: Optional[list[dict[str, Any]]],
) -> tuple[Optional[str], list[ToolCall]]:
    """If the model returned tool calls as text, convert them. Returns (text, tool_calls):
    on success the salvaged calls replace `tool_calls` and `text` is cleared."""
    if tool_calls or not tools or not text:
        return text, tool_calls
    salvaged = _salvage_tool_calls_from_text(text, tools)
    if salvaged:
        return None, salvaged
    return text, tool_calls


def _tool_index(
    tools: Optional[list[dict[str, Any]]],
) -> tuple[Optional[set[str]], dict[str, Optional[str]]]:
    """(known tool names, {name: sole-parameter-name}) from OpenAI tool schemas. The sole-param
    map lets us map a bare `toolname [args]` to `{param: args}` when a tool has one parameter.
    """
    if not tools:
        return None, {}
    names: set[str] = set()
    single: dict[str, Optional[str]] = {}
    for t in tools:
        fn = (t or {}).get("function") or {}
        name = fn.get("name")
        if not isinstance(name, str) or not name:
            continue
        names.add(name)
        params = fn.get("parameters") or {}
        props = params.get("properties") or {}
        if len(props) == 1:
            single[name] = next(iter(props))
        else:
            required = params.get("required") or []
            single[name] = required[0] if len(required) == 1 else None
    return names, single


def _loads(s: str) -> Any:
    try:
        return json.loads(s)
    except (TypeError, json.JSONDecodeError):
        return None


def _extract_balanced(text: str, start: int) -> Optional[str]:
    """Return the balanced `{…}`/`[…]` substring beginning at `text[start]` (string-aware), or
    None if it doesn't close — so nested braces/brackets are handled correctly."""
    open_ch = text[start]
    close_ch = "]" if open_ch == "[" else "}"
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _iter_top_objects(text: str):
    """Yield balanced `{…}` substrings at brace-depth 0 (array brackets ignored), so embedded
    JSON objects are found even amid surrounding prose."""
    i = 0
    while i < len(text):
        if text[i] == "{":
            sub = _extract_balanced(text, i)
            if sub:
                yield sub
                i += len(sub)
                continue
        i += 1


def _call_from_dict(d: Any, names: Optional[set[str]]) -> Optional[ToolCall]:
    """Build a ToolCall from a `{"name","arguments"}` dict, or None if it isn't one / the name
    isn't a known tool."""
    if not isinstance(d, dict):
        return None
    name = d.get("name")
    if not isinstance(name, str) or not name:
        return None
    if names is not None and name not in names:
        return None
    args = d.get("arguments", d.get("parameters"))
    if args is None:
        args = {}
    if isinstance(args, str):
        args = _loads(args)
        if not isinstance(args, dict):
            args = {"_raw": d.get("arguments")}
    if not isinstance(args, dict):
        args = {"_raw": args}
    return ToolCall(id="", name=name, arguments=args)


def _renumber(calls: list[ToolCall]) -> list[ToolCall]:
    return [
        ToolCall(id=f"call_salvaged_{i}", name=c.name, arguments=c.arguments)
        for i, c in enumerate(calls)
    ]


def _salvage_tool_calls_from_text(
    content: str, tools: Optional[list[dict[str, Any]]] = None
) -> list[ToolCall]:
    """Best-effort recovery of tool calls embedded in assistant text. Tries, in order:
    1. `<tool_call>…</tool_call>` blocks (anywhere, balanced); 2. embedded `{"name","arguments"}`
    objects (even mixed with prose); 3. `toolname {args}` / `toolname [args]` for known tools.
    Returns [] (treat as plain text) when nothing tool-shaped is found."""
    text = (content or "").strip()
    if not text:
        return []
    names, single = _tool_index(tools)

    # 1) <tool_call> … </tool_call> blocks.
    calls: list[ToolCall] = []
    for m in _TOOLCALL_OPEN.finditer(text):
        j = m.end()
        if j < len(text) and text[j] in "{[":
            sub = _extract_balanced(text, j)
            parsed = _loads(sub) if sub else None
            for d in parsed if isinstance(parsed, list) else [parsed]:
                c = _call_from_dict(d, names)
                if c:
                    calls.append(c)
    if calls:
        return _renumber(calls)

    # 1b) Qwen/Hermes XML calls: <function=NAME><parameter=KEY>VAL</parameter>…</function>.
    for fm in _FUNCTION_BLOCK.finditer(text):
        name = fm.group("name").strip()
        if names is not None and name not in names:
            continue
        args = {
            pm.group("key").strip(): _coerce_param(pm.group("val"))
            for pm in _PARAM_BLOCK.finditer(fm.group("body"))
        }
        calls.append(ToolCall(id="", name=name, arguments=args))
    if calls:
        return _renumber(calls)

    # 2) Embedded {"name": …, "arguments": …} objects, even surrounded by prose.
    for sub in _iter_top_objects(text):
        d = _loads(sub)
        if isinstance(d, dict) and "name" in d:
            c = _call_from_dict(d, names)
            if c:
                calls.append(c)
    if calls:
        return _renumber(calls)

    # 3) `toolname {args}` / `toolname [args]` shorthand — only for tools we actually offered.
    if names:
        for name in names:
            for m in re.finditer(re.escape(name) + r"\s*[:=]?\s*", text):
                j = m.end()
                if j >= len(text) or text[j] not in "{[":
                    continue
                sub = _extract_balanced(text, j)
                parsed = _loads(sub) if sub else None
                if parsed is None:
                    continue
                if isinstance(parsed, dict):
                    args = parsed
                else:
                    param = single.get(name)
                    if not param:
                        continue
                    args = {param: parsed}
                calls.append(ToolCall(id="", name=name, arguments=args))
                break  # one salvaged call per tool name
    return _renumber(calls)
