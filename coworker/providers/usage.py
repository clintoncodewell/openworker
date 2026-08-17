"""Best-effort, process-local provider usage snapshots.

Only the providers with a useful v1 signal live here.  The two quota endpoints are
unofficial and intentionally polled on demand; inference response headers are cached for
the process lifetime.  Nothing in this module may make a model request fail.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Mapping, Optional

import httpx

CHATGPT_USAGE_URL = "https://chatgpt.com/backend-api/wham/usage"
ZAI_USAGE_URL = "https://api.z.ai/api/monitor/usage/quota/limit"

_OPENAI_HEADERS = (
    "x-ratelimit-limit-requests",
    "x-ratelimit-remaining-requests",
    "x-ratelimit-reset-requests",
    "x-ratelimit-limit-tokens",
    "x-ratelimit-remaining-tokens",
    "x-ratelimit-reset-tokens",
)
_ANTHROPIC_PREFIXES = (
    "anthropic-ratelimit-requests-",
    "anthropic-ratelimit-tokens-",
    "anthropic-ratelimit-input-tokens-",
    "anthropic-ratelimit-output-tokens-",
)
_CHATGPT_HEADERS = (
    "x-codex-primary-used-percent",
    "x-codex-primary-window-minutes",
    "x-codex-primary-reset-at",
    "x-codex-secondary-used-percent",
    "x-codex-secondary-window-minutes",
    "x-codex-secondary-reset-at",
    "x-codex-credits-balance",
    "x-codex-has-credits",
)

_CACHE: dict[str, dict[str, str]] = {}
_LOCK = threading.Lock()


def capture_headers(provider: Optional[str], headers: Any) -> None:
    """Keep only usage-related response headers, case-insensitively and best-effort."""
    if not provider or not headers:
        return
    try:
        source = {str(k).lower(): str(v) for k, v in dict(headers).items()}
    except (TypeError, ValueError, AttributeError):
        return
    if provider == "openai":
        captured = {key: source[key] for key in _OPENAI_HEADERS if key in source}
    elif provider == "anthropic":
        captured = {
            key: value
            for key, value in source.items()
            if any(key.startswith(prefix) for prefix in _ANTHROPIC_PREFIXES)
        }
    elif provider == "chatgpt":
        captured = {key: source[key] for key in _CHATGPT_HEADERS if key in source}
    else:
        captured = {}
    if captured:
        with _LOCK:
            _CACHE[provider] = captured


def cached_headers(provider: str) -> dict[str, str]:
    with _LOCK:
        return dict(_CACHE.get(provider) or {})


def clear_cache() -> None:
    """Test helper; normal operation keeps snapshots until process exit."""
    with _LOCK:
        _CACHE.clear()


def _number(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "1", "yes"):
            return True
        if lowered in ("false", "0", "no"):
            return False
    return None


def _plan(value: Any) -> Optional[str]:
    """Plan labels come from undocumented endpoints and are rendered as a React child.
    Anything that is not already a plain scalar is dropped rather than risking an
    object reaching the tree and unmounting the whole Usage tab."""
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    return None


def _window(
    window_id: str,
    label: str,
    *,
    used_percent: Any = None,
    remaining: Any = None,
    reset_at: Any = None,
    window_seconds: Any = None,
) -> dict[str, Any]:
    used = _number(used_percent)
    out: dict[str, Any] = {"id": window_id, "label": label}
    if used is not None:
        out["used_percent"] = max(0.0, min(100.0, used))
        out["remaining_percent"] = max(0.0, min(100.0, 100.0 - used))
    remaining_number = _number(remaining)
    out["remaining"] = remaining_number if remaining_number is not None else remaining
    if out["remaining"] is None:
        out.pop("remaining")
    if reset_at not in (None, ""):
        out["reset_at"] = reset_at
    seconds = _number(window_seconds)
    if seconds is not None:
        out["limit_window_seconds"] = seconds
    return out


def _chatgpt_cached_snapshot() -> Optional[dict[str, Any]]:
    headers = cached_headers("chatgpt")
    windows = []
    for window_id, label in (("primary", "Primary window"), ("secondary", "Weekly window")):
        prefix = f"x-codex-{window_id}-"
        if not any(key.startswith(prefix) for key in headers):
            continue
        minutes = _number(headers.get(prefix + "window-minutes"))
        windows.append(
            _window(
                window_id,
                label,
                used_percent=headers.get(prefix + "used-percent"),
                reset_at=headers.get(prefix + "reset-at"),
                window_seconds=minutes * 60 if minutes is not None else None,
            )
        )
    if not windows and not any(key.startswith("x-codex-credits-") for key in headers):
        return None
    result: dict[str, Any] = {"windows": windows, "source": "inference_headers"}
    balance = _number(headers.get("x-codex-credits-balance"))
    has_credits = _bool(headers.get("x-codex-has-credits"))
    if balance is not None or has_credits is not None:
        result["credits"] = {"balance": balance, "has_credits": has_credits}
    return result


def _merge_chatgpt_snapshots(
    live: Optional[dict[str, Any]], cached: Optional[dict[str, Any]]
) -> Optional[dict[str, Any]]:
    """Prefer the live poll while filling fields its unofficial shape omitted."""
    if not live:
        return cached
    if not cached:
        return live
    merged = dict(cached)
    merged.update({key: value for key, value in live.items() if key != "windows"})
    if isinstance(cached.get("credits"), Mapping) and isinstance(
        live.get("credits"), Mapping
    ):
        merged["credits"] = {**cached["credits"], **live["credits"]}
    windows = {
        window.get("id"): window
        for window in cached.get("windows", [])
        if isinstance(window, Mapping) and window.get("id")
    }
    for window in live.get("windows", []):
        if isinstance(window, Mapping) and window.get("id"):
            windows[window["id"]] = window
    merged["windows"] = [
        windows[window_id]
        for window_id in ("primary", "secondary")
        if window_id in windows
    ]
    merged["windows"].extend(
        window
        for window_id, window in windows.items()
        if window_id not in ("primary", "secondary")
    )
    return merged


def poll_chatgpt_usage(
    token: str,
    account_id: str = "",
    *,
    http_client: Any = None,
) -> Optional[dict[str, Any]]:
    """Poll ChatGPT's unofficial subscription-usage endpoint; None means unavailable."""
    headers = {"Authorization": f"Bearer {token}"}
    if account_id:
        headers["ChatGPT-Account-Id"] = account_id
    try:
        if http_client is None:
            response = httpx.get(CHATGPT_USAGE_URL, headers=headers, timeout=5.0)
        else:
            response = http_client.get(CHATGPT_USAGE_URL, headers=headers, timeout=5.0)
        if response.status_code != 200:
            return None
        payload = response.json()
        if not isinstance(payload, Mapping):
            return None
        rate_limit = payload.get("rate_limit")
        rate_limit = rate_limit if isinstance(rate_limit, Mapping) else {}
        windows = []
        for key, label in (("primary_window", "Primary window"), ("secondary_window", "Weekly window")):
            raw = rate_limit.get(key)
            if not isinstance(raw, Mapping):
                continue
            windows.append(
                _window(
                    key.removesuffix("_window"),
                    label,
                    used_percent=raw.get("used_percent"),
                    reset_at=raw.get("reset_at"),
                    window_seconds=raw.get("limit_window_seconds"),
                )
            )
        if not windows and not isinstance(payload.get("credits"), Mapping):
            return None
        result: dict[str, Any] = {
            "windows": windows,
            "source": "live_poll",
            "plan": _plan(payload.get("plan_type")),
        }
        # Whitelist rather than copy: this endpoint is undocumented, so an added field
        # (an account identifier, say) would otherwise leave the backend untouched, and a
        # field that turns into an object would reach React as a child and unmount the tab.
        credits = payload.get("credits")
        if isinstance(credits, Mapping):
            result["credits"] = {
                "balance": _number(credits.get("balance")),
                "has_credits": _bool(credits.get("has_credits")),
            }
        return result
    except Exception:
        return None


def poll_zai_usage(api_key: str, *, http_client: Any = None) -> Optional[dict[str, Any]]:
    """Poll Z AI's unofficial coding-plan endpoint; tolerate wrapper/field changes."""
    try:
        kwargs = {
            "headers": {"Authorization": f"Bearer {api_key}"},
            "timeout": 5.0,
        }
        response = (
            httpx.get(ZAI_USAGE_URL, **kwargs)
            if http_client is None
            else http_client.get(ZAI_USAGE_URL, **kwargs)
        )
        if response.status_code != 200:
            return None
        payload = response.json()
        if not isinstance(payload, Mapping):
            return None
        body = payload.get("data") if isinstance(payload.get("data"), Mapping) else payload
        raw_limits = body.get("limits") if isinstance(body, Mapping) else None
        if not isinstance(raw_limits, list):
            return None
        windows = []
        for index, raw in enumerate(raw_limits[:2]):
            if not isinstance(raw, Mapping):
                continue
            window_id = "primary" if index == 0 else "secondary"
            label = "Primary window" if index == 0 else "Weekly window"
            windows.append(
                _window(
                    window_id,
                    label,
                    used_percent=raw.get("percentage"),
                    remaining=raw.get("remaining"),
                    reset_at=raw.get("nextResetTime"),
                )
            )
        if not windows:
            return None
        return {
            "windows": windows,
            "source": "live_poll",
            "plan": _plan(
                body.get("level") or body.get("planLevel") or body.get("plan")
            ),
        }
    except Exception:
        return None


def _headroom(provider: str) -> dict[str, Any]:
    headers = cached_headers(provider)
    metrics = []
    if provider == "openai":
        for metric in ("requests", "tokens"):
            prefix = "x-ratelimit-"
            metrics.append(
                {
                    "id": metric,
                    "label": metric.title(),
                    "limit": headers.get(f"{prefix}limit-{metric}"),
                    "remaining": headers.get(f"{prefix}remaining-{metric}"),
                    "reset": headers.get(f"{prefix}reset-{metric}"),
                }
            )
    else:
        for metric, label in (
            ("requests", "Requests"),
            ("tokens", "Tokens"),
            ("input-tokens", "Input tokens"),
            ("output-tokens", "Output tokens"),
        ):
            prefix = f"anthropic-ratelimit-{metric}-"
            if any(key.startswith(prefix) for key in headers):
                metrics.append(
                    {
                        "id": metric,
                        "label": label,
                        "limit": headers.get(prefix + "limit"),
                        "remaining": headers.get(prefix + "remaining"),
                        "reset": headers.get(prefix + "reset"),
                    }
                )
    metrics = [{k: v for k, v in metric.items() if v is not None} for metric in metrics]
    metrics = [metric for metric in metrics if len(metric) > 2]
    return {
        "status": "ok" if metrics else "unavailable",
        "kind": "rate_limit_headroom",
        "metrics": metrics,
        "message": None if metrics else "No recent rate-limit headers captured yet.",
    }


def usage_snapshot(
    secrets: Any, *, chatgpt_auth: Any = None
) -> dict[str, list[dict[str, Any]]]:
    """Return configured, supported providers only; one failed poll never affects another."""
    from .registry import get_descriptor, provider_configured

    def _profile(pid: str) -> Mapping[str, Any]:
        # A stored profile is user-writable JSON on disk; a corrupt non-Mapping one must
        # not raise past a provider's own guard and 500 the whole snapshot.
        stored = secrets.get(f"provider:{pid}")
        return stored if isinstance(stored, Mapping) else {}

    def _chatgpt_live() -> Optional[dict[str, Any]]:
        try:
            auth = chatgpt_auth
            if auth is None:
                from .chatgpt_auth import ChatGPTAuthManager

                auth = ChatGPTAuthManager(secrets)
            token, account_id = auth.valid_access_token()
            return poll_chatgpt_usage(token, account_id)
        except Exception:
            return None

    configured = [
        p
        for p in ("chatgpt", "zai-coding", "openai", "anthropic", "claude-code", "gemini")
        if provider_configured(p, secrets)
    ]

    # Both live polls wait up to 5s on an external host and neither depends on the other,
    # so run them together: one unreachable vendor then costs a single timeout, not two
    # in series while the user watches a spinner.
    polls: dict[str, Any] = {}
    with ThreadPoolExecutor(max_workers=2) as pool:
        if "chatgpt" in configured:
            polls["chatgpt"] = pool.submit(_chatgpt_live)
        if "zai-coding" in configured:
            polls["zai-coding"] = pool.submit(
                poll_zai_usage, str(_profile("zai-coding").get("api_key") or "")
            )

    def _poll_result(pid: str) -> Optional[dict[str, Any]]:
        future = polls.get(pid)
        if future is None:
            return None
        try:
            return future.result()
        except Exception:
            return None

    providers: list[dict[str, Any]] = []
    for provider_id in configured:
        descriptor = get_descriptor(provider_id)
        title = descriptor.title if descriptor else provider_id
        base = {"id": provider_id, "title": title}
        if provider_id == "chatgpt":
            data = _merge_chatgpt_snapshots(
                _poll_result("chatgpt"), _chatgpt_cached_snapshot()
            )
            providers.append(
                {
                    **base,
                    "status": "ok" if data else "unavailable",
                    "kind": "quota_window",
                    "label": "Subscription window (approximate, unofficial endpoint)",
                    "windows": (data or {}).get("windows", []),
                    **({k: v for k, v in data.items() if k != "windows"} if data else {}),
                    "message": None if data else "Temporarily unavailable",
                }
            )
        elif provider_id == "zai-coding":
            data = _poll_result("zai-coding")
            providers.append(
                {
                    **base,
                    "status": "ok" if data else "unavailable",
                    "kind": "quota_window",
                    "label": "Subscription window (approximate, unofficial endpoint)",
                    "windows": (data or {}).get("windows", []),
                    **({k: v for k, v in data.items() if k != "windows"} if data else {}),
                    "message": None if data else "Temporarily unavailable",
                }
            )
        elif provider_id in ("openai", "anthropic"):
            providers.append(
                {
                    **base,
                    **_headroom(provider_id),
                    "label": "Current throughput headroom — not a billing figure",
                }
            )
        elif provider_id == "claude-code":
            providers.append(
                {
                    **base,
                    "status": "ok",
                    "kind": "status_only",
                    # provider_configured() proves the CLI exists, nothing more — claiming
                    # "signed in" here would assert an auth state we never checked.
                    "message": "Installed — uses your existing Claude Code sign-in",
                }
            )
        elif provider_id == "gemini":
            providers.append(
                {
                    **base,
                    "status": "ok",
                    "kind": "status_only",
                    "message": "Configured",
                    "link": "https://aistudio.google.com/usage",
                }
            )
    return {"providers": providers}
