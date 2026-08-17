"""What a council actually cost — measured, not estimated.

Every provider reports token usage on its raw response, in its own shape. This normalizes
the three shapes the app talks to, sums them per run, and turns them into a dollar
estimate.

Two different confidence levels, deliberately kept apart:

* **Tokens are measured.** They come from the vendor's own response. Trust them.
* **Dollars are an estimate.** They come from `PRICES` below, which is a hand-maintained
  table that goes stale every time a vendor changes a price. Treat the number as an order
  of magnitude, not an invoice.

Reasoning tokens are the trap. A Gemini call answering "Ready?" with one word billed 151
thinking tokens against 9 of prompt — the visible answer length tells you almost nothing
about the bill. They are counted as output here, which is how they are charged.
"""

from __future__ import annotations

from typing import Any, Optional

# USD per 1M tokens, (input, output). Checked 2026-08-16 — vendor prices drift, so treat a
# dollar figure as an order of magnitude and re-check before making a decision on it.
# Keys are matched as a PREFIX of the routed model id, longest first, so "azure:gpt-5.6"
# covers every tier without a row each.
# Sentinel for "this provider does not meter tokens", as distinct from "no price row" —
# both are (0.0, 0.0), and only the first should report as fully priced.
UNMETERED = (0.0, 0.0)

PRICES: dict[str, tuple[float, float]] = {
    "azure:gpt-5.6": (1.25, 10.00),
    "azure:gpt-5.5": (1.25, 10.00),
    "gpt-5.6": (1.25, 10.00),
    "gpt-5.5": (1.25, 10.00),
    "chatgpt:": UNMETERED,  # subscription allowance, not metered per token
    "anthropic:claude-fable": (10.00, 50.00),
    "anthropic:claude-opus": (5.00, 25.00),
    "anthropic:claude-sonnet": (3.00, 15.00),
    "anthropic:claude-haiku": (1.00, 5.00),
    # Claude Code runs on the subscription allowance, not a per-token meter. Zero here is
    # the truth for the bill; the cost that matters is rate-limit headroom, not dollars.
    "claude-code:": UNMETERED,
    "gemini:gemini-3": (0.30, 2.50),
    "gemini:gemini-2.5-pro": (1.25, 10.00),
    "gemini:gemini-2.5": (0.30, 2.50),
    "zai-coding:": UNMETERED,  # flat-rate coding plan, not per token
    "zai:": (0.60, 2.20),
    "azure-oss:kimi": (0.60, 2.50),
    "azure-oss:deepseek": (0.27, 1.10),
    "deepseek:": (0.27, 1.10),
    "kimi:": (0.60, 2.50),
    "xai:": (3.00, 15.00),
    "minimax:": (0.30, 1.20),
    "qwen:": (1.60, 6.40),
    "mistral:": (2.00, 6.00),
    "together:": (0.60, 2.20),
    "fireworks:": (0.60, 2.20),
    "ollama:": UNMETERED,  # local
}


# A provider is "not metered" when its zero price is a FACT (flat-rate plan, subscription,
# local compute) rather than a missing table row — derived from PRICES itself, so adding a
# provider is one edit. A second hand-maintained list would drift, and the failure is
# silent: a new vendor would read as free instead of as an unpriced gap.
NOT_METERED = frozenset(p for p, v in PRICES.items() if v == UNMETERED)


def price_for(model: str) -> tuple[float, float]:
    """(input, output) USD per 1M tokens. Longest matching prefix wins; unknown → 0.

    Zero for an unknown model rather than a guess: a made-up price presented as a number
    is worse than an obvious gap, and the token counts beside it are still exact.
    """
    for prefix in sorted(PRICES, key=len, reverse=True):
        if model.startswith(prefix):
            return PRICES[prefix]
    return (0.0, 0.0)


def _first_int(obj: Any, *names: str) -> int:
    for name in names:
        value = getattr(obj, name, None)
        if isinstance(value, (int, float)):
            return int(value)
    return 0


def extract(raw: Any) -> Optional[dict[str, int]]:
    """Token usage from a provider's raw response, or None if it reported none.

    Three shapes: OpenAI-compatible (`prompt_tokens`/`completion_tokens`), Anthropic
    (`input_tokens`/`output_tokens`), Gemini (`prompt_token_count`/`candidates_token_count`).

    **Reasoning tokens are counted differently by the two families, and getting this wrong
    silently misreports every cost.** Measured 2026-08-16:

    * OpenAI-compatible — `completion_tokens_details.reasoning_tokens` is a BREAKDOWN of
      `completion_tokens`, which already includes it (`prompt + completion == total`).
      Adding it inflates output, cost, and budget consumption.
    * Gemini — `thoughts_token_count` is SEPARATE from `candidates_token_count` and must be
      added (prompt 9 + candidates 2 + thoughts 151 == total 162).

    So `reasoning` is reported either way for visibility, but only added to `output` for
    the family that excludes it.
    """
    usage = getattr(raw, "usage", None) or getattr(raw, "usage_metadata", None)
    if usage is None:
        return None
    prompt = _first_int(usage, "prompt_tokens", "input_tokens", "prompt_token_count")
    output = _first_int(usage, "completion_tokens", "output_tokens", "candidates_token_count")

    gemini_thoughts = _first_int(usage, "thoughts_token_count")  # additive
    # The Responses API nests the same breakdown under a different name. Reading only the
    # chat/completions one reports zero thinking for every Responses turn — the tokens are
    # still counted and still billed (they sit inside `output_tokens`), but the split that
    # tells you WHY a turn was expensive silently disappears.
    details = getattr(usage, "completion_tokens_details", None) or getattr(
        usage, "output_tokens_details", None
    )
    openai_reasoning = _first_int(details, "reasoning_tokens") if details is not None else 0

    output += gemini_thoughts  # NOT openai_reasoning — already inside `output`
    reasoning = gemini_thoughts + openai_reasoning
    if not (prompt or output or reasoning):
        return None
    return {"input": prompt, "output": output, "reasoning": reasoning}


def cost(model: str, usage: dict[str, int]) -> float:
    """Estimated USD for one call."""
    price_in, price_out = price_for(model)
    return (
        usage.get("input", 0) * price_in + usage.get("output", 0) * price_out
    ) / 1_000_000


def total(rounds: list[list[dict[str, Any]]], chair: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """Roll every member turn (and the chair) into one report.

    `by_model` is the useful part in practice: it is how you find out that one member is
    responsible for most of the bill, which is the actionable fact.
    """
    calls = [a for r in rounds for a in r] + ([chair] if chair else [])
    by_model: dict[str, dict[str, Any]] = {}
    tokens_in = tokens_out = reasoning = 0
    usd = 0.0
    priced = True
    for call in calls:
        usage = call.get("usage")
        if not usage:
            continue
        model = call.get("model", "?")
        call_cost = cost(model, usage)
        if price_for(model) == UNMETERED and not model.startswith(tuple(NOT_METERED)):
            priced = False  # an unknown model, not a genuinely free one
        entry = by_model.setdefault(
            model, {"calls": 0, "input": 0, "output": 0, "usd": 0.0}
        )
        entry["calls"] += 1
        entry["input"] += usage.get("input", 0)
        entry["output"] += usage.get("output", 0)
        entry["usd"] = round(entry["usd"] + call_cost, 4)
        tokens_in += usage.get("input", 0)
        tokens_out += usage.get("output", 0)
        reasoning += usage.get("reasoning", 0)
        usd += call_cost
    return {
        "calls": len([c for c in calls if c.get("usage")]),
        "input_tokens": tokens_in,
        "output_tokens": tokens_out,
        "reasoning_tokens": reasoning,
        "total_tokens": tokens_in + tokens_out,
        "usd_estimate": round(usd, 4),
        # False when some model on the panel has no price row — the tokens are still
        # exact, but the dollar figure is missing part of the run.
        "fully_priced": priced,
        "by_model": by_model,
    }


def summary_line(report: dict[str, Any]) -> str:
    """One line for a human: what this run cost."""
    tokens = report.get("total_tokens", 0)
    usd = report.get("usd_estimate", 0.0)
    note = "" if report.get("fully_priced", True) else " (some models unpriced)"
    thinking = report.get("reasoning_tokens", 0)
    thinking_note = f", {thinking:,} of it hidden reasoning" if thinking else ""
    return (
        f"{report.get('calls', 0)} model calls · {tokens:,} tokens{thinking_note} · "
        f"about ${usd:.3f}{note}"
    )
