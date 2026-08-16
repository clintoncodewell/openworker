"""The curated model matrix — the only models we actively suggest, label, and vouch for.

Keyed by the FULL routed id, exactly as the ProviderRouter receives it — including reseller
"ugly names" like ``together:zai-org/GLM-5.2`` (bare ids route to the OpenAI default). Each
entry carries the UI display label and the model's capabilities, making this the single
source of truth the capability probe and the GUI's pickers read from.

Deliberately SMALL (owner call, 2026-07-04): current-generation, agent-capable (tool-calling)
models only. That "tool-calling" clause is load-bearing and enforced by a test: the matrix
feeds the composer's SESSION-model picker, so a text-only transport (the `claude-code`
subscription provider) is deliberately absent here even though it is a first-class council
member. Its models are reached through its descriptor's `recommended_model`, or by naming
them in the council panel. It is not user-editable — users can still add any custom model string, which
falls back to the conservative heuristics in ``capabilities.py`` at their own risk of
degraded results. Ids verified against vendor/reseller catalogs on 2026-07-04; refresh the
reseller rows when catalogs rotate (they rename on every model generation).

Resellers: Together + Fireworks for now. TODO: add Groq and OpenRouter entries here AND their
descriptors in ``registry.py`` once the current provider surface is tested — deliberately
deferred to bound how much needs verifying at once.
"""

from __future__ import annotations

from dataclasses import dataclass

from .base import ModelCapabilities

_AGENTIC = ModelCapabilities(
    tools=True, vision=False, parallel_tool_calls=True, streaming=True
)
# The native three (OpenAI, Anthropic, Gemini) all take PDFs directly; every
# OpenAI-compatible vendor and reseller in the matrix does not (their chat APIs have
# no inline file part — checked 2026-07-17), so those fall back via pdf_support.py.
_AGENTIC_VISION = ModelCapabilities(
    tools=True, vision=True, pdf=True, parallel_tool_calls=True, streaming=True
)
_CHATGPT_SUBSCRIPTION = ModelCapabilities(
    tools=True, vision=True, pdf=False, parallel_tool_calls=True, streaming=True
)
# Azure AI Foundry serves the same models over an OpenAI-compatible surface, but its
# chat API has no inline `file` part, so PDFs go through the pdf_support.py fallback.
_AGENTIC_VISION_NO_PDF = ModelCapabilities(
    tools=True, vision=True, pdf=False, parallel_tool_calls=True, streaming=True
)


@dataclass(frozen=True)
class ModelEntry:
    label: str  # UI display name, e.g. "GLM-5.2 · via Together"
    caps: ModelCapabilities = _AGENTIC


MATRIX: dict[str, ModelEntry] = {
    # -- first-party ------------------------------------------------------------
    # GPT-5.6 (2026-07-09): number = generation, Sol/Terra/Luna = capability tiers.
    # Bare "gpt-5.6" aliases to Sol server-side; we list the explicit tier ids only.
    # Rolling out — accounts without access get a friendly error (providers/errors.py).
    "gpt-5.6-sol": ModelEntry("GPT-5.6 Sol · OpenAI", _AGENTIC_VISION),
    "gpt-5.6-terra": ModelEntry("GPT-5.6 Terra · OpenAI", _AGENTIC_VISION),
    "gpt-5.6-luna": ModelEntry("GPT-5.6 Luna · OpenAI", _AGENTIC_VISION),
    "gpt-5.5": ModelEntry("GPT-5.5 · OpenAI", _AGENTIC_VISION),
    # ChatGPT account transport (distinct from API-billed OpenAI models).
    # Subscription-backed catalogue. Availability is still determined by the signed-in
    # account because this ChatGPT transport is undocumented and entitlement-gated.
    "chatgpt:gpt-5.6-sol": ModelEntry(
        "GPT-5.6 Sol · ChatGPT subscription", _CHATGPT_SUBSCRIPTION
    ),
    "chatgpt:gpt-5.6-terra": ModelEntry(
        "GPT-5.6 Terra · ChatGPT subscription", _CHATGPT_SUBSCRIPTION
    ),
    "chatgpt:gpt-5.6-luna": ModelEntry(
        "GPT-5.6 Luna · ChatGPT subscription", _CHATGPT_SUBSCRIPTION
    ),
    "chatgpt:gpt-5.4-mini": ModelEntry(
        "GPT-5.4 Mini · ChatGPT subscription", _CHATGPT_SUBSCRIPTION
    ),
    "chatgpt:gpt-5.5": ModelEntry(
        "GPT-5.5 · ChatGPT subscription", _CHATGPT_SUBSCRIPTION
    ),
    "chatgpt:gpt-5.4": ModelEntry(
        "GPT-5.4 · ChatGPT subscription", _CHATGPT_SUBSCRIPTION
    ),
    "chatgpt:gpt-5.4-pro": ModelEntry(
        "GPT-5.4 Pro · ChatGPT subscription", _CHATGPT_SUBSCRIPTION
    ),
    "chatgpt:gpt-5-mini": ModelEntry(
        "GPT-5 Mini · ChatGPT subscription", _CHATGPT_SUBSCRIPTION
    ),
    "chatgpt:gpt-5.2": ModelEntry(
        "GPT-5.2 · ChatGPT subscription", _CHATGPT_SUBSCRIPTION
    ),
    # Fable 5 (2026-06-09) is GA; its Mythos 5 sibling is approved-orgs-only, so it
    # stays out of a picker meant for the public.
    "anthropic:claude-opus-5": ModelEntry("Claude Opus 5 · Anthropic", _AGENTIC_VISION),
    "anthropic:claude-sonnet-5": ModelEntry(
        "Claude Sonnet 5 · Anthropic", _AGENTIC_VISION
    ),
    "anthropic:claude-fable-5": ModelEntry(
        "Claude Fable 5 · Anthropic", _AGENTIC_VISION
    ),
    "anthropic:claude-opus-4-8": ModelEntry(
        "Claude Opus 4.8 · Anthropic", _AGENTIC_VISION
    ),
    "anthropic:claude-sonnet-4-6": ModelEntry(
        "Claude Sonnet 4.6 · Anthropic", _AGENTIC_VISION
    ),
    "anthropic:claude-haiku-4-5": ModelEntry(
        "Claude Haiku 4.5 · Anthropic", _AGENTIC_VISION
    ),
    # Gemini 3 (thought signatures required in tool loops — carried via the `_gemini`
    # message sidecar, see gemini_provider.py; ids from the vendor catalog 2026-07-22).
    "gemini:gemini-3.1-pro-preview": ModelEntry(
        "Gemini 3.1 Pro · Google", _AGENTIC_VISION
    ),
    "gemini:gemini-3.6-flash": ModelEntry("Gemini 3.6 Flash · Google", _AGENTIC_VISION),
    "gemini:gemini-2.5-pro": ModelEntry("Gemini 2.5 Pro · Google", _AGENTIC_VISION),
    "gemini:gemini-2.5-flash": ModelEntry("Gemini 2.5 Flash · Google", _AGENTIC_VISION),
    # -- Azure AI Foundry (your own deployments, OpenAI-compatible) ---------------
    "azure:gpt-5.6-sol": ModelEntry(
        "GPT-5.6 Sol · Azure Foundry", _AGENTIC_VISION_NO_PDF
    ),
    "azure:gpt-5.6-terra": ModelEntry(
        "GPT-5.6 Terra · Azure Foundry", _AGENTIC_VISION_NO_PDF
    ),
    "azure:gpt-5.6-luna": ModelEntry(
        "GPT-5.6 Luna · Azure Foundry", _AGENTIC_VISION_NO_PDF
    ),
    "azure:gpt-5.5": ModelEntry("GPT-5.5 · Azure Foundry", _AGENTIC_VISION_NO_PDF),
    "azure-oss:kimi-k3": ModelEntry("Kimi K3 · Azure Foundry"),
    "azure-oss:deepseek-v4-flash": ModelEntry("DeepSeek V4 Flash · Azure Foundry"),
    # -- direct OpenAI-compatible vendors ----------------------------------------
    "zai:glm-5.3": ModelEntry("GLM-5.3 · Z AI"),
    "zai:glm-5.2": ModelEntry("GLM-5.2 · Z AI"),
    "zai-coding:glm-5.3": ModelEntry("GLM-5.3 · Z AI (coding plan)"),
    "zai-coding:glm-4.6": ModelEntry("GLM-4.6 · Z AI (coding plan)"),
    "zai-coding:glm-4.5": ModelEntry("GLM-4.5 · Z AI (coding plan)"),
    "deepseek:deepseek-v4-flash": ModelEntry("DeepSeek V4 Flash · DeepSeek"),
    "deepseek:deepseek-v4-pro": ModelEntry("DeepSeek V4 Pro · DeepSeek"),
    "kimi:kimi-k2.6": ModelEntry("Kimi K2.6 · Moonshot"),
    "minimax:MiniMax-M2.5": ModelEntry("MiniMax M2.5 · MiniMax"),
    "qwen:qwen3-max": ModelEntry("Qwen3 Max · Alibaba"),
    "xai:grok-4.6": ModelEntry("Grok 4.6 · xAI"),
    "xai:grok-4.5": ModelEntry("Grok 4.5 · xAI"),
    "xai:grok-4.3": ModelEntry("Grok 4.3 · xAI"),
    "mistral:mistral-large-latest": ModelEntry("Mistral Large · Mistral"),
    # -- resellers (their model namespaces, verbatim) -----------------------------
    "together:thinkingmachines/Inkling": ModelEntry("Inkling · via Together"),
    "together:zai-org/GLM-5.3": ModelEntry("GLM-5.3 · via Together"),
    "together:zai-org/GLM-5.2": ModelEntry("GLM-5.2 · via Together"),
    # Kimi K3 (2026-07-16) is not on Together yet — weights land ~07-27; revisit then.
    "together:moonshotai/Kimi-K2.7-Code": ModelEntry("Kimi K2.7 Code · via Together"),
    "together:moonshotai/Kimi-K2.6": ModelEntry("Kimi K2.6 · via Together"),
    "together:deepseek-ai/DeepSeek-V4-Pro": ModelEntry(
        "DeepSeek V4 Pro · via Together"
    ),
    "together:meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8": ModelEntry(
        "Llama 4 Maverick · via Together"
    ),
    "fireworks:accounts/fireworks/models/glm-5p2": ModelEntry(
        "GLM-5.2 · via Fireworks"
    ),
    "fireworks:accounts/fireworks/models/kimi-k2p6": ModelEntry(
        "Kimi K2.6 · via Fireworks"
    ),
    "fireworks:accounts/fireworks/models/deepseek-v4-pro": ModelEntry(
        "DeepSeek V4 Pro · via Fireworks"
    ),
    "fireworks:accounts/fireworks/models/llama4-maverick-instruct-basic": ModelEntry(
        "Llama 4 Maverick · via Fireworks"
    ),
}


def entry_for(model: str) -> ModelEntry | None:
    return MATRIX.get(model)


def model_labels() -> dict[str, str]:
    """Full-id → display-label map, shipped to the GUI so every picker shows human names."""
    return {mid: e.label for mid, e in MATRIX.items()}


def models_for_provider(provider: str) -> list[str]:
    """BARE model ids (prefix stripped) the matrix curates for a provider — feeds the
    Settings pane's suggestions and the composer picker so both stay in lockstep with the
    matrix. OpenAI entries are stored without a prefix (bare ids route to the OpenAI
    default), so its list is every un-prefixed id."""
    if provider == "openai":
        return [mid for mid in MATRIX if ":" not in mid]
    prefix = provider + ":"
    return [mid[len(prefix) :] for mid in MATRIX if mid.startswith(prefix)]
