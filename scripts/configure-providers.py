#!/usr/bin/env python3
"""Point OpenWorker at every model this machine can already reach.

The credentials for those models are scattered across the box in the files each CLI
made for itself (`~/.codex/azure.key`, `~/.config/clm/*`, `~/.config/glm/key`). This
collects them into OpenWorker's own SecretStore so the app's model picker offers the
same models the terminal does.

  configure-providers.py                 collect from this box, apply here, verify
  configure-providers.py --out FILE      collect and write a bundle (0600) instead
  configure-providers.py --apply FILE    apply a bundle written on another machine
  configure-providers.py --verify-only   just report what is configured and live

`--out`/`--apply` exist because the Mac has none of these key files; the bundle is the
transport. It is plaintext key material — ship it over ssh and delete it after.

Two providers are reached through always-on proxies on the VM rather than directly:

* **Grok** — its xAI OAuth bearer expires hourly and the LiteLLM proxy refreshes it, so
  no key is stored at all.
* **Azure Foundry GPT** — a VNet/firewall rule on that resource accepts only the VM's
  IP, so every other machine gets 403 with a perfectly valid key.

`--proxy-host` is where those two proxies listen: leave it at loopback on the VM, and
pass the VM's tailnet IP when building a bundle for another machine.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from coworker.providers.registry import verify_provider_key  # noqa: E402
from coworker.secrets import SecretStore  # noqa: E402

VM_TAILNET_IP = "100.65.245.83"


def _from_file(*paths: str) -> str:
    for p in paths:
        f = Path(p).expanduser()
        if f.is_file():
            value = f.read_text(encoding="utf-8").strip()
            if value:
                return value
    return ""


def _from_env_file(path: str, var: str) -> str:
    """One VAR=value line out of a systemd EnvironmentFile."""
    f = Path(path).expanduser()
    if not f.is_file():
        return ""
    for line in f.read_text(encoding="utf-8").splitlines():
        if line.startswith(f"{var}="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


def _reachable(base_url: str, timeout: float = 3.0) -> bool:
    """Does something answer HTTP here? Any status counts — the question is whether the
    proxy is up, not whether it likes an unauthenticated request."""
    import httpx

    try:
        httpx.get(base_url.rstrip("/") + "/models", timeout=timeout)
        return True
    except Exception:
        return False


def collect(proxy_host: str) -> dict[str, dict[str, str]]:
    """Provider profiles this machine has credentials for. Missing key ⇒ omitted."""
    profiles: dict[str, dict[str, str]] = {}

    azure = os.environ.get("AZURE_OPENAI_API_KEY", "").strip() or _from_file(
        "~/.codex/azure.key"
    )
    if azure:
        # Direct on the VM (the only IP the firewall allows); via the VM's proxy elsewhere.
        base = (
            "https://foundry-codex-dev.openai.azure.com/openai/v1"
            if proxy_host in ("127.0.0.1", "localhost")
            else f"http://{proxy_host}:8802/openai/v1"
        )
        profiles["azure"] = {"api_key": azure, "base_url": base}

    oss = _from_file("~/.config/clm/azure-foundry-key")
    if oss:
        profiles["azure-oss"] = {
            "api_key": oss,
            "base_url": "https://aw-kimi-k3-eval.cognitiveservices.azure.com/openai/v1",
        }

    zai = _from_file("~/.config/glm/key") or _from_env_file(
        "~/.openclaw/gateway.systemd.env", "ZAI_API_KEY"
    )
    if zai:
        # The coding-plan key is billed only on Z AI's Anthropic-compatible endpoint.
        profiles["zai-coding"] = {
            "api_key": zai,
            "base_url": "https://api.z.ai/api/anthropic",
        }

    # Gemini is deliberately NOT harvested from GEMINI_API_KEY or the OpenClaw gateway
    # env (owner call, 2026-08-16). That key is shared across the whole stack and belongs
    # to the `aw-gemini-api-central` GCP project — billed work. OpenWorker is to use the
    # free tier under the personal Google account instead, so it reads one dedicated file
    # that only ever holds that key. No file, no Gemini: silently falling back to the
    # billed key is exactly the mistake this exists to prevent.
    gemini = _from_file("~/.config/coworker/gemini-aistudio-key")
    if gemini:
        profiles["gemini"] = {"api_key": gemini}

    # Only when the proxy actually answers. Adding it unconditionally would overwrite a
    # real xAI key and endpoint with a placeholder pointing at a proxy that isn't running,
    # silently breaking xAI for anyone who configured it properly.
    grok = f"http://{proxy_host}:4144/v1"
    if _reachable(grok):
        # The proxy ignores the key but the OpenAI SDK insists on a non-empty string.
        profiles["xai"] = {"api_key": "local-grok-proxy", "base_url": grok}

    anthropic = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if anthropic:
        profiles["anthropic"] = {"api_key": anthropic}

    return profiles


def apply(profiles: dict[str, dict[str, str]]) -> None:
    store = SecretStore()
    for name, fields in profiles.items():
        existing = store.get(f"provider:{name}") or {}
        store.put(f"provider:{name}", {**existing, **fields})
        print(f"configured provider:{name}")
    print(f"→ {store.path}")


def verify(profiles: dict[str, dict[str, str]]) -> int:
    failures = 0
    for name, fields in sorted(profiles.items()):
        result = verify_provider_key(
            name, api_key=fields.get("api_key"), base_url=fields.get("base_url")
        )
        ok = result.get("ok")
        failures += 0 if ok else 1
        print(f"{'OK  ' if ok else 'FAIL'} {name:<12} {result.get('error', '')}")
    return failures


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", metavar="FILE", help="write a bundle instead of applying")
    ap.add_argument("--apply", metavar="FILE", help="apply a bundle from another machine")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument(
        "--proxy-host",
        default="127.0.0.1",
        help=f"host running the Grok + Foundry proxies (use {VM_TAILNET_IP} off the VM)",
    )
    args = ap.parse_args()

    if args.apply:
        profiles = json.loads(Path(args.apply).read_text(encoding="utf-8"))
    else:
        profiles = collect(args.proxy_host)

    if not profiles:
        print("no credentials found on this machine", file=sys.stderr)
        return 1

    if args.out:
        out = Path(args.out)
        # Create at 0600 rather than writing then chmod-ing: every key in this file would
        # otherwise be world-readable for the length of the write, and O_EXCL refuses to
        # follow a symlink someone planted at the path.
        payload = json.dumps(profiles, indent=2).encode("utf-8")
        try:
            fd = os.open(out, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            print(
                f"{out} already exists — refusing to overwrite it. Delete it first "
                "(it holds API keys, so check what it is before you do).",
                file=sys.stderr,
            )
            return 1
        try:
            os.write(fd, payload)
        finally:
            os.close(fd)
        print(f"wrote {len(profiles)} provider profiles to {out} (0600) — delete after use")
        return 0

    if not args.verify_only:
        apply(profiles)
    return 1 if verify(profiles) else 0


if __name__ == "__main__":
    raise SystemExit(main())
