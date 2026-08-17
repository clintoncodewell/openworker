"""ChatGPT subscription adapter for OpenWork's provider seam.

The transport is the same undocumented ChatGPT Responses endpoint used by Muesli.  All
wire-format conversion stays here so the engine continues to speak canonical messages,
tools, AssistantTurn, and StreamChunk.
"""

from __future__ import annotations

import json
from typing import Any, Optional

import httpx

from .base import AssistantTurn, ModelCapabilities, ProviderClient, StreamChunk, ToolCall
from .capabilities import capabilities_for
from .chatgpt_auth import ChatGPTAuthManager
from .usage import capture_headers

WHAM_URL = "https://chatgpt.com/backend-api/wham/responses"


def _content_parts(content: Any, *, assistant: bool = False) -> list[dict[str, Any]]:
    text_type = "output_text" if assistant else "input_text"
    if isinstance(content, str):
        return [{"type": text_type, "text": content}] if content else []
    parts: list[dict[str, Any]] = []
    for part in content or []:
        if not isinstance(part, dict):
            continue
        kind = part.get("type")
        if kind in ("text", "input_text", "output_text"):
            text = part.get("text")
            if text:
                parts.append({"type": text_type, "text": text})
        elif kind in ("image_url", "input_image") and not assistant:
            image = part.get("image_url")
            url = image.get("url") if isinstance(image, dict) else image
            url = url or part.get("image_url")
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


def responses_input(messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    instructions: list[str] = []
    items: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if role in ("system", "developer"):
            if isinstance(content, str) and content:
                instructions.append(content)
            continue
        if role == "tool":
            output = content if isinstance(content, str) else json.dumps(content)
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": message.get("tool_call_id") or "",
                    "output": output,
                }
            )
            continue
        if role == "assistant":
            parts = _content_parts(content, assistant=True)
            if parts:
                items.append({"role": "assistant", "content": parts})
            for call in message.get("tool_calls") or []:
                fn = call.get("function") or {}
                arguments = fn.get("arguments") or "{}"
                if not isinstance(arguments, str):
                    arguments = json.dumps(arguments)
                items.append(
                    {
                        "type": "function_call",
                        "call_id": call.get("id") or "",
                        "name": fn.get("name") or "",
                        "arguments": arguments,
                    }
                )
            continue
        parts = _content_parts(content)
        if parts:
            items.append({"role": "user", "content": parts})
    return "\n\n".join(instructions), items


def responses_tools(tools: Optional[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for tool in tools or []:
        fn = tool.get("function") or {}
        if tool.get("type") != "function" or not fn.get("name"):
            continue
        item: dict[str, Any] = {
            "type": "function",
            "name": fn["name"],
            "parameters": fn.get("parameters") or {"type": "object", "properties": {}},
        }
        if fn.get("description"):
            item["description"] = fn["description"]
        if "strict" in fn:
            item["strict"] = fn["strict"]
        out.append(item)
    return out


def _error_message(response: Any) -> str:
    try:
        data = response.json()
        error = data.get("error") or {}
        return str(error.get("message") or data.get("message") or data.get("detail") or data)
    except (ValueError, TypeError, AttributeError):
        return str(getattr(response, "text", "") or "unknown error")


def _completed_text(response: dict[str, Any]) -> str:
    parts: list[str] = []
    for item in response.get("output") or []:
        if item.get("type") != "message":
            continue
        for content in item.get("content") or []:
            if content.get("type") in ("output_text", "text") and content.get("text"):
                parts.append(str(content["text"]))
    return "".join(parts)


class ChatGPTProvider(ProviderClient):
    def __init__(self, *, secrets: Any, auth: Any = None, http_client: Any = None) -> None:
        self.auth = auth or ChatGPTAuthManager(secrets)
        self._http = http_client or httpx.Client(timeout=120)

    def capabilities(self, model: str) -> ModelCapabilities:
        return capabilities_for(f"chatgpt:{model}")

    def complete(self, **kwargs: Any) -> AssistantTurn:
        final: Optional[AssistantTurn] = None
        for chunk in self.stream(**kwargs):
            if chunk.turn is not None:
                final = chunk.turn
        if final is None:
            raise RuntimeError("ChatGPT returned no completed response")
        return final

    def stream(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]] = None,
        **settings: Any,
    ):
        token, account_id = self.auth.valid_access_token()
        instructions, input_items = responses_input(messages)
        body: dict[str, Any] = {
            "model": model,
            "store": False,
            "stream": True,
            "input": input_items,
        }
        if instructions:
            body["instructions"] = instructions
        converted_tools = responses_tools(tools)
        if converted_tools:
            body["tools"] = converted_tools
            body["parallel_tool_calls"] = True
        for key in ("temperature", "top_p", "max_output_tokens"):
            if key in settings:
                body[key] = settings[key]
        if "max_tokens" in settings or "max_completion_tokens" in settings:
            body["max_output_tokens"] = settings.get("max_output_tokens") or settings.get(
                "max_completion_tokens"
            ) or settings.get("max_tokens")

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
        if account_id:
            headers["ChatGPT-Account-Id"] = account_id

        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        calls: dict[int, dict[str, str]] = {}
        final_response: dict[str, Any] = {}
        with self._http.stream("POST", WHAM_URL, headers=headers, json=body) as response:
            capture_headers("chatgpt", response.headers)
            if response.status_code != 200:
                response.read()
                raise RuntimeError(
                    f"ChatGPT failed with status {response.status_code}: "
                    f"{_error_message(response)}"
                )
            for line in response.iter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[6:].strip()
                if not payload or payload in ("[DONE]", "ping", "heartbeat", "keep-alive"):
                    continue
                try:
                    event = json.loads(payload)
                except json.JSONDecodeError as exc:
                    raise RuntimeError("Malformed ChatGPT stream payload") from exc
                kind = event.get("type")
                if kind == "response.output_text.delta" and event.get("delta"):
                    delta = str(event["delta"])
                    text_parts.append(delta)
                    yield StreamChunk(text_delta=delta)
                elif kind in (
                    "response.reasoning_summary_text.delta",
                    "response.reasoning_text.delta",
                ) and event.get("delta"):
                    delta = str(event["delta"])
                    reasoning_parts.append(delta)
                    yield StreamChunk(reasoning_delta=delta)
                elif kind == "response.output_item.added":
                    item = event.get("item") or {}
                    if item.get("type") == "function_call":
                        index = int(event.get("output_index") or 0)
                        calls[index] = {
                            "id": str(item.get("call_id") or item.get("id") or ""),
                            "name": str(item.get("name") or ""),
                            "args": str(item.get("arguments") or ""),
                        }
                elif kind == "response.function_call_arguments.delta":
                    index = int(event.get("output_index") or 0)
                    calls.setdefault(index, {"id": "", "name": "", "args": ""})[
                        "args"
                    ] += str(event.get("delta") or "")
                elif kind == "response.output_item.done":
                    item = event.get("item") or {}
                    if item.get("type") == "function_call":
                        index = int(event.get("output_index") or 0)
                        acc = calls.setdefault(index, {"id": "", "name": "", "args": ""})
                        acc["id"] = str(item.get("call_id") or item.get("id") or acc["id"])
                        acc["name"] = str(item.get("name") or acc["name"])
                        if item.get("arguments"):
                            acc["args"] = str(item["arguments"])
                elif kind == "response.completed":
                    final_response = event.get("response") or {}

        for index, item in enumerate(final_response.get("output") or []):
            if item.get("type") == "function_call" and index not in calls:
                calls[index] = {
                    "id": str(item.get("call_id") or item.get("id") or ""),
                    "name": str(item.get("name") or ""),
                    "args": str(item.get("arguments") or ""),
                }
        if not text_parts:
            completed_text = _completed_text(final_response)
            if completed_text:
                text_parts.append(completed_text)
                yield StreamChunk(text_delta=completed_text)
        tool_calls: list[ToolCall] = []
        for index in sorted(calls):
            call = calls[index]
            try:
                arguments = json.loads(call["args"] or "{}")
            except json.JSONDecodeError:
                arguments = {"_raw": call["args"]}
            tool_calls.append(ToolCall(call["id"], call["name"], arguments))
        yield StreamChunk(
            turn=AssistantTurn(
                text="".join(text_parts) or None,
                tool_calls=tool_calls,
                finish_reason="tool_calls" if tool_calls else "stop",
                raw=final_response or None,
                reasoning="".join(reasoning_parts) or None,
            )
        )
