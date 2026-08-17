"""Model-provider registry — descriptors + a factory, mirroring the connector
(`connectors/descriptors.py`) and web-search (`web/providers.py`) patterns.

A `ProviderDescriptor` declares a provider's UI config `fields` (rendered dynamically by the
GUI, same `to_dict()` shape connectors use) and a `build(profile, secrets)` factory that returns
a `ProviderClient`. The `ProviderRouter` selects a descriptor by the `provider:` prefix of a
model string and builds (and caches) its client from the matching SecretStore profile.

Today: `openai` (the default, with an optional custom endpoint that covers Azure OpenAI's
`/openai/v1` and any OpenAI-compliant gateway), `anthropic` (native Messages API via
`AnthropicProvider`), `gemini` (native Google GenAI API via `GeminiProvider`), and `ollama`
(local, OpenAI-compatible `/v1`). Bedrock/Vertex auth for Claude is future work.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .anthropic_provider import AnthropicProvider
from .claude_code_provider import ClaudeCodeProvider, resolve_binary
from .base import ProviderClient
from .chatgpt_provider import ChatGPTProvider
from .gemini_provider import GeminiProvider
from .openai_provider import OpenAIProvider

DEFAULT_OLLAMA_URL = "http://localhost:11434"


@dataclass(frozen=True)
class ProviderField:
    """One config input for a provider, rendered by the GUI (mirrors connectors' `Field`)."""

    key: str
    label: str
    secret: bool = False
    required: bool = True
    help: str = ""
    placeholder: str = ""
    # Pre-filled (still editable) form value — e.g. an OpenAI-compatible vendor's official
    # endpoint, so the user only has to paste a key. Distinct from `placeholder` (grey hint).
    default: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "secret": self.secret,
            "required": self.required,
            "help": self.help,
            "placeholder": self.placeholder,
            "default": self.default,
        }


@dataclass(frozen=True)
class ProviderDescriptor:
    """A model provider: its UI fields + a factory that builds its `ProviderClient`."""

    name: str
    title: str
    needs_key: bool
    fields: list[ProviderField]
    build: Callable[[dict[str, Any], Any], ProviderClient] = field(repr=False)
    recommended_model: Optional[str] = (
        None  # pre-filled in the UI; auto-added on configure
    )
    env_key: Optional[str] = (
        None  # env var that can supply the API key (e.g. ANTHROPIC_API_KEY)
    )
    # One-line note under the provider title (e.g. "Connects through X's OpenAI-compatible API").
    blurb: str = ""
    # Browser account sign-in instead of form fields. Empty means the normal key/keyless flow.
    auth: str = ""
    # Liveness test for a KEYLESS provider, where "configured" is otherwise vacuously true.
    # Takes the provider's stored profile, so a configured custom binary is what gets tested.
    # Must be cheap enough to run on every Settings render — a `which`, not a network call.
    available: Optional[Callable[[dict[str, Any]], bool]] = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "title": self.title,
            "needs_key": self.needs_key,
            "fields": [f.to_dict() for f in self.fields],
            "recommended_model": self.recommended_model,
            "blurb": self.blurb,
            "auth": self.auth or None,
        }


def _normalize_ollama_url(url: Optional[str]) -> str:
    """Accept `http://host:11434` or `.../v1` and return an OpenAI-compatible base URL.

    Ollama serves its OpenAI-compatible API under `/v1`; the native API lives at the root, so we
    always target `<root>/v1`.
    """
    base = (url or DEFAULT_OLLAMA_URL).strip().rstrip("/")
    if not base:
        base = DEFAULT_OLLAMA_URL
    if not base.endswith("/v1"):
        base = base + "/v1"
    return base


def _claude_binary(profile: Optional[dict[str, Any]]) -> str:
    """The Claude Code binary this profile asks for — used to BUILD it and to test whether
    it exists, so those two can never disagree about which binary they mean."""
    return ((profile or {}).get("binary") or "claude").strip() or "claude"


def _build_openai(profile: dict[str, Any], secrets: Any) -> ProviderClient:
    # Key resolution stays in OpenAIProvider/resolve_api_key (explicit → env → SecretStore),
    # so we just hand it the SecretStore. An optional custom endpoint (Azure OpenAI /openai/v1,
    # OpenRouter, vLLM, …) comes from the stored profile.
    base_url = ((profile or {}).get("base_url") or "").strip() or None
    return OpenAIProvider(secrets=secrets, base_url=base_url)


def _build_chatgpt(profile: dict[str, Any], secrets: Any) -> ProviderClient:
    return ChatGPTProvider(secrets=secrets)


def _build_anthropic(profile: dict[str, Any], secrets: Any) -> ProviderClient:
    # Key resolution stays in AnthropicProvider/resolve_api_key (explicit → env → SecretStore),
    # deferred to first call so the provider can be built before a key exists.
    # thinking_budget: hidden profile override — absent/invalid → the default (ON),
    # explicit 0 → off (see DEFAULT_THINKING_BUDGET).
    from .anthropic_provider import DEFAULT_THINKING_BUDGET

    api_key = ((profile or {}).get("api_key") or "").strip() or None
    try:
        thinking_budget = int(str((profile or {}).get("thinking_budget") or "").strip())
    except ValueError:
        thinking_budget = DEFAULT_THINKING_BUDGET
    return AnthropicProvider(
        api_key=api_key, secrets=secrets, thinking_budget=thinking_budget
    )


def _build_zai_anthropic(profile: dict[str, Any], secrets: Any) -> ProviderClient:
    # Z AI's coding-plan key is billed ONLY on its Anthropic-compatible endpoint; the OpenAI-style
    # /paas API (the `zai` provider) returns 429 code 1113 "insufficient balance" for it. So reach
    # GLM through AnthropicProvider pointed at /api/anthropic, keyed from THIS provider's own
    # profile (never the shared Anthropic env/SecretStore key — a different vendor's endpoint).
    api_key = ((profile or {}).get("api_key") or "").strip() or None
    base_url = ((profile or {}).get("base_url") or "").strip() or "https://api.z.ai/api/anthropic"
    if not api_key:
        raise RuntimeError("No Z AI API key configured — add it in Settings ▸ Models.")
    return AnthropicProvider(api_key=api_key, base_url=base_url, secrets=secrets)


def _build_gemini(profile: dict[str, Any], secrets: Any) -> ProviderClient:
    # An EXPLICIT key only — never GeminiProvider's own resolution, which reads the shared
    # `GEMINI_API_KEY`/`GOOGLE_API_KEY` env vars. Passing `secrets=None` closes that path:
    # on a box running other Google tooling those names hold a different account's key, and
    # falling back to it spends money the user did not choose to spend.
    api_key = ((profile or {}).get("api_key") or "").strip() or os.environ.get(
        "OPENWORKER_GEMINI_API_KEY", ""
    ).strip()
    if not api_key:
        raise RuntimeError(
            "No Gemini API key configured — add one in Settings ▸ Models. "
            "Get it from aistudio.google.com; a key made in a Cloud project bills that project."
        )
    return GeminiProvider(api_key=api_key)


def _build_ollama(profile: dict[str, Any], secrets: Any) -> ProviderClient:
    # Ollama's OpenAI-compatible endpoint ignores the key but the SDK requires a non-empty
    # string, so we pass a placeholder. `base_url` comes from the stored profile (or the default).
    base_url = _normalize_ollama_url((profile or {}).get("base_url"))
    return OpenAIProvider(api_key="ollama", base_url=base_url)


def _openai_compat(vendor: str, default_base_url: str, env_key: Optional[str] = None):
    """Builder factory for vendors reached through their OpenAI-compatible API (Z AI, DeepSeek,
    Kimi, MiniMax, Qwen, xAI, Mistral). The key is resolved from the vendor's OWN profile (or its
    env var) — deliberately NOT from the OpenAI env/SecretStore fallback, so a configured OpenAI
    key is never silently sent to a different vendor's endpoint. Missing key ⇒ fail fast with a
    vendor-named error (these are only built on demand, when one of their models is selected).
    """

    def build(profile: dict[str, Any], secrets: Any) -> ProviderClient:
        base_url = ((profile or {}).get("base_url") or "").strip() or default_base_url
        api_key = ((profile or {}).get("api_key") or "").strip() or (
            os.environ.get(env_key, "").strip() if env_key else ""
        )
        if not base_url:
            # Only reachable for a per-tenant provider (Azure Foundry), which ships no
            # default endpoint. Without this the SDK falls back to api.openai.com and the
            # user's Foundry key goes to OpenAI — a wrong-vendor key leak, reported as a
            # baffling 401.
            raise RuntimeError(
                f"No {vendor} endpoint configured — add it in Settings ▸ Models."
            )
        if not api_key:
            raise RuntimeError(
                f"No {vendor} API key configured — add it in Settings ▸ Models."
            )
        return OpenAIProvider(
            api_key=api_key,
            base_url=base_url,
            supports_responses=vendor == "Azure Foundry GPT",
        )

    return build


def _compat(
    name: str,
    title: str,
    *,
    base_url: str,
    recommended_model: str,
    env_key: str,
    endpoint_help: str = "",
    vendor: str = "",
    blurb: str = "",
) -> ProviderDescriptor:
    """Descriptor for an OpenAI-compatible vendor: key + a prefilled, editable endpoint.

    `vendor` overrides the name used in field labels. It is normally derived from the
    title, but that collapses two providers from the same vendor into one label — the two
    Azure Foundry entries both rendered "Azure AI Foundry API key" and were
    indistinguishable in Settings.
    """
    vendor = vendor or title.split(" (")[0]
    # An endpoint the user MUST supply (their own tenant) is a required field with no
    # prefill; a vendor's public endpoint is optional and prefilled.
    self_hosted = not base_url
    return ProviderDescriptor(
        name=name,
        title=title,
        needs_key=True,
        fields=[
            ProviderField(
                "api_key",
                f"{vendor} API key",
                secret=True,
            ),
            ProviderField(
                "base_url",
                "Endpoint",
                required=self_hosted,
                default=base_url,
                placeholder=base_url,
                help=endpoint_help
                or f"Prefilled with {vendor}'s official endpoint; edit only for a regional or proxy variant.",
            ),
        ],
        build=_openai_compat(vendor, base_url, env_key),
        recommended_model=recommended_model,
        env_key=env_key,
        blurb=blurb
        or f"Uses {vendor}'s OpenAI-compatible API — the endpoint is prefilled, just add your key.",
    )


DESCRIPTORS: list[ProviderDescriptor] = [
    ProviderDescriptor(
        name="openai",
        title="OpenAI",
        needs_key=True,
        fields=[
            ProviderField(
                "api_key",
                "OpenAI API key",
                secret=True,
                placeholder="sk-…",
            ),
            ProviderField(
                "base_url",
                "Custom endpoint (optional)",
                secret=False,
                required=False,
                placeholder="https://…/openai/v1",
                help="For Azure OpenAI, OpenRouter, vLLM, or any OpenAI-compliant server. Leave blank for api.openai.com.",
            ),
        ],
        build=_build_openai,
        recommended_model="gpt-5.6-sol",
        env_key="OPENAI_API_KEY",
    ),
    ProviderDescriptor(
        name="chatgpt",
        title="ChatGPT subscription",
        needs_key=False,
        fields=[],
        build=_build_chatgpt,
        recommended_model="gpt-5.4-mini",
        blurb="Use your ChatGPT account and subscription allowance — no API key or separate API billing.",
        auth="oauth",
    ),
    ProviderDescriptor(
        name="claude-code",
        title="Claude (via Claude Code subscription)",
        needs_key=False,
        fields=[],
        build=lambda profile, secrets: ClaudeCodeProvider(binary=_claude_binary(profile)),
        available=lambda profile: resolve_binary(_claude_binary(profile)) is not None,
        recommended_model="claude-opus-5",
        blurb="Runs Claude on your Claude Code subscription — no API key. Text only, so it "
        "works as a council member but cannot be the session model.",
    ),
    ProviderDescriptor(
        name="anthropic",
        title="Claude (Anthropic)",
        needs_key=True,
        fields=[
            ProviderField(
                "api_key",
                "Anthropic API key",
                secret=True,
                placeholder="sk-ant-…",
            ),
            # No thinking_budget field (owner call 2026-07-23): extended thinking is
            # on by default; the profile key stays a hidden override (0 = off).
        ],
        build=_build_anthropic,
        recommended_model="claude-fable-5",
        env_key="ANTHROPIC_API_KEY",
    ),
    ProviderDescriptor(
        name="gemini",
        title="Gemini (Google)",
        needs_key=True,
        fields=[
            ProviderField(
                "api_key",
                "Gemini API key",
                secret=True,
                placeholder="AIza…",
                help="From aistudio.google.com. A key created in a Google Cloud project bills that project.",
            ),
        ],
        build=_build_gemini,
        recommended_model="gemini-3.6-flash",
        # Deliberately NOT `GEMINI_API_KEY` (owner call, 2026-08-16). That name is the
        # Google SDK convention, so on a machine that already runs other Google tooling it
        # is set to whatever key that tooling uses — here, a billed Cloud project. Treating
        # it as OpenWorker's key silently spends the wrong account and cannot be turned off
        # from Settings, because "remove key" would not remove the env var. A dedicated
        # name means the env path only ever holds a key chosen FOR this app.
        env_key="OPENWORKER_GEMINI_API_KEY",
    ),
    # OpenAI-compatible vendors, listed as first-class providers so users don't need to know the
    # "point the OpenAI slot at a different endpoint" trick (owner call, 2026-07-04). Each keeps
    # its own key profile; the endpoint is prefilled and editable (regional variants in `help`).
    _compat(
        "zai",
        "Z AI (GLM)",
        base_url="https://api.z.ai/api/paas/v4",
        recommended_model="glm-5.3",
        env_key="ZAI_API_KEY",
        endpoint_help="Prefilled with Z AI's international endpoint. China mainland: https://open.bigmodel.cn/api/paas/v4",
    ),
    # Z AI coding-plan keys are billed on Z AI's ANTHROPIC-compatible endpoint, not the OpenAI-style
    # /paas API the `zai` provider above uses (that returns 1113 "insufficient balance" for them).
    # This provider reuses the native Anthropic engine pointed at /api/anthropic. Separate from
    # `anthropic` so real Claude and GLM can be configured side by side.
    ProviderDescriptor(
        name="zai-coding",
        title="GLM (Z AI · coding plan)",
        needs_key=True,
        fields=[
            ProviderField("api_key", "Z AI API key", secret=True),
            ProviderField(
                "base_url",
                "Endpoint",
                required=False,
                default="https://api.z.ai/api/anthropic",
                placeholder="https://api.z.ai/api/anthropic",
                help="Z AI's Anthropic-compatible endpoint, where coding-plan keys are billed. China mainland: https://open.bigmodel.cn/api/anthropic",
            ),
        ],
        build=_build_zai_anthropic,
        recommended_model="glm-5.3",
        blurb="Spends your Z AI coding-plan allowance via Z AI's Anthropic-compatible API (not the pay-as-you-go /paas endpoint).",
    ),
    # Azure AI Foundry. Unlike every other entry here there is no vendor-wide endpoint to
    # prefill: a Foundry resource is per-tenant, with its own hostname, key and deployment
    # names, so the endpoint is REQUIRED and the user supplies it. Two descriptors because
    # one Foundry resource means one endpoint + one key, and GPT and open-weight models
    # usually live in different resources (different regions and quotas).
    _compat(
        "azure",
        "Azure AI Foundry (GPT)",
        base_url="",
        recommended_model="gpt-5.6-sol",
        env_key="AZURE_OPENAI_API_KEY",
        vendor="Azure Foundry GPT",
        endpoint_help="Your Foundry resource's OpenAI-compatible surface, e.g. https://<resource>.openai.azure.com/openai/v1",
        blurb="Your own Azure AI Foundry GPT deployments. Paste the resource endpoint and key from the Foundry portal.",
    ),
    _compat(
        "azure-oss",
        "Azure AI Foundry (open models)",
        base_url="",
        recommended_model="kimi-k3",
        env_key="AZURE_OSS_API_KEY",
        vendor="Azure Foundry open-model",
        endpoint_help="The Foundry resource hosting your open-weight deployments, e.g. https://<resource>.cognitiveservices.azure.com/openai/v1",
        blurb="Open-weight models (Kimi, DeepSeek, …) on your own Azure AI Foundry resource. Model ids are YOUR deployment names.",
    ),
    _compat(
        "deepseek",
        "DeepSeek",
        base_url="https://api.deepseek.com",
        recommended_model="deepseek-v4-flash",
        env_key="DEEPSEEK_API_KEY",
    ),
    _compat(
        "kimi",
        "Kimi (Moonshot AI)",
        base_url="https://api.moonshot.ai/v1",
        recommended_model="kimi-k2.6",
        env_key="MOONSHOT_API_KEY",
        endpoint_help="Prefilled with Moonshot's international endpoint. China mainland: https://api.moonshot.cn/v1",
    ),
    _compat(
        "minimax",
        "MiniMax",
        base_url="https://api.minimax.io/v1",
        recommended_model="MiniMax-M2.5",
        env_key="MINIMAX_API_KEY",
    ),
    _compat(
        "qwen",
        "Qwen (Alibaba)",
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        recommended_model="qwen3-max",
        env_key="DASHSCOPE_API_KEY",
        endpoint_help="Prefilled with Alibaba Model Studio's international endpoint. China (Beijing): https://dashscope.aliyuncs.com/compatible-mode/v1",
    ),
    _compat(
        "xai",
        "xAI (Grok)",
        base_url="https://api.x.ai/v1",
        recommended_model="grok-4.6",
        env_key="XAI_API_KEY",
    ),
    _compat(
        "mistral",
        "Mistral",
        base_url="https://api.mistral.ai/v1",
        recommended_model="mistral-large-latest",
        env_key="MISTRAL_API_KEY",
    ),
    # Resellers: many labs' models behind one key, using THEIR model namespaces (the curated
    # ids + display labels live in providers/matrix.py). TODO: add Groq and OpenRouter here
    # (+ their matrix rows) once the current provider surface is tested — deliberately
    # deferred to bound how much needs verifying at once (owner call, 2026-07-04).
    _compat(
        "together",
        "Together AI",
        base_url="https://api.together.xyz/v1",
        recommended_model="zai-org/GLM-5.3",
        env_key="TOGETHER_API_KEY",
    ),
    _compat(
        "fireworks",
        "Fireworks AI",
        base_url="https://api.fireworks.ai/inference/v1",
        recommended_model="accounts/fireworks/models/glm-5p2",
        env_key="FIREWORKS_API_KEY",
    ),
    ProviderDescriptor(
        name="ollama",
        title="Ollama (local models)",
        needs_key=False,
        fields=[
            ProviderField(
                "base_url",
                "Ollama server URL",
                secret=False,
                required=False,
                placeholder=DEFAULT_OLLAMA_URL,
                help="Where `ollama serve` is listening. The OpenAI-compatible /v1 path is added automatically.",
            ),
        ],
        build=_build_ollama,
        # Reliable native tool-calling + strong coding quality (verified). Pull with
        # `ollama pull qwen3-coder:30b`.
        recommended_model="qwen3-coder:30b",
    ),
]

_BY_NAME = {d.name: d for d in DESCRIPTORS}


def provider_descriptors() -> list[ProviderDescriptor]:
    return list(DESCRIPTORS)


def provider_names() -> list[str]:
    return [d.name for d in DESCRIPTORS]


def get_descriptor(name: str) -> Optional[ProviderDescriptor]:
    return _BY_NAME.get(name)


def provider_available(d: ProviderDescriptor, profile: Optional[dict[str, Any]]) -> bool:
    """Run a keyless provider's liveness check against ITS OWN stored profile.

    The profile matters: a user who points `claude-code` at a custom binary must be judged
    on that binary. Checking the default instead reproduces, in the custom-binary path, the
    exact Settings-says-yes / council-says-no split this check exists to close.

    A check that raises reads as unavailable rather than propagating — this runs on every
    Settings render, and one broken descriptor must not take out the whole provider list.
    """
    if d.available is None:
        return True
    try:
        return bool(d.available(profile or {}))
    except Exception:
        return False


def provider_configured(name: str, secrets: Any) -> bool:
    """Is this provider usable — has it a key (stored or from its env var), or is it
    keyless / signed in?

    The single definition of "configured". `SessionManager._provider_configured` and the
    council's panel resolution both defer to it; when they each had their own copy they
    disagreed, and the council silently skipped the OAuth ChatGPT provider. Reads through
    the SecretStore so `${VAR}` refs and the local `.env` resolve the same way here as
    they do at call time.
    """
    d = _BY_NAME.get(name)
    if d is None:
        return False
    profile = (secrets.get(f"provider:{name}") if secrets is not None else None) or {}
    if d.auth == "oauth":
        return bool(profile.get("access_token"))
    if not d.needs_key:
        # Keyless says nothing about usable. A provider that shells out to a CLI is only
        # configured if that CLI is actually installed, so ask before claiming it works.
        return provider_available(d, profile)
    return bool(profile.get("api_key")) or bool(
        d.env_key and os.environ.get(d.env_key)
    )


def build_provider_client(
    name: str, profile: dict[str, Any], secrets: Any
) -> ProviderClient:
    """Build a `ProviderClient` for `name` from its stored profile. Unknown → OpenAI default."""
    descriptor = _BY_NAME.get(name) or _BY_NAME["openai"]
    return descriptor.build(profile or {}, secrets)


def detect_provider(api_key: str) -> Optional[str]:
    """Best-effort provider guess from an API key's shape, for the onboarding auto-detect.
    Returns a known provider name or None. Mirrors the GUI's client-side detection so both agree.
    """
    key = (api_key or "").strip()
    if not key:
        return None
    if key.startswith("sk-ant-"):
        return "anthropic"
    if key.startswith("AIza"):
        return "gemini"
    if key.startswith(("sk-", "sk_")):
        return "openai"
    return None


def verify_provider_key(
    name: str,
    *,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout: float = 10.0,
) -> dict[str, Any]:
    """Validate a provider's credentials with one cheap, read-only call (list models) — the same
    pattern connectors use to validate tokens. Transient: callers pass the key directly so a user
    can Test before saving. Never raises; returns {ok, error?}.
    """
    import httpx

    d = _BY_NAME.get(name) or _BY_NAME["openai"]
    key = (api_key or "").strip()
    try:
        if name == "anthropic":
            resp = httpx.get(
                "https://api.anthropic.com/v1/models",
                headers={"x-api-key": key, "anthropic-version": "2023-06-01"},
                timeout=timeout,
            )
        elif name == "gemini":
            resp = httpx.get(
                "https://generativelanguage.googleapis.com/v1beta/models",
                params={"key": key},
                timeout=timeout,
            )
        elif name == "ollama":
            base = _normalize_ollama_url(base_url)
            resp = httpx.get(base.rstrip("/") + "/models", timeout=timeout)
        else:  # openai + any OpenAI-compatible endpoint (Azure, OpenRouter, vendors, vLLM…)
            default_base = next(
                (f.default for f in d.fields if f.key == "base_url" and f.default), ""
            )
            base = (
                (base_url or "").strip().rstrip("/")
                or default_base.rstrip("/")
                or "https://api.openai.com/v1"
            )
            resp = httpx.get(
                base + "/models",
                headers={"Authorization": f"Bearer {key}"},
                timeout=timeout,
            )
    except Exception as exc:  # DNS/connection/timeout — never let it bubble to a 500
        return {
            "ok": False,
            "error": f"Couldn't reach {d.title} ({exc.__class__.__name__}).",
        }

    if resp.status_code < 300:
        return {"ok": True}
    if resp.status_code in (401, 403):
        if name == "ollama":
            return {"ok": False, "error": "Server rejected the request."}
        return {"ok": False, "error": "Invalid API key."}
    if resp.status_code == 404 and name == "ollama":
        return {
            "ok": False,
            "error": "Reached the server, but no OpenAI-compatible /v1 API there.",
        }
    return {"ok": False, "error": f"{d.title} returned HTTP {resp.status_code}."}
