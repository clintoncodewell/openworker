"""Local ChatGPT OAuth (PKCE) for the subscription-backed model provider.

This is deliberately separate from OpenAI API-key auth.  It mirrors the flow used by
Muesli/Codex-style clients and stores tokens in the existing SecretStore profile
``provider:chatgpt``.
"""

from __future__ import annotations

import base64
import hashlib
import html
import json
import secrets as random_secrets
import threading
import time
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Callable, Optional

import httpx

CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
AUTHORIZE_URL = "https://auth.openai.com/oauth/authorize"
TOKEN_URL = "https://auth.openai.com/oauth/token"
REDIRECT_URI = "http://localhost:1455/auth/callback"
SCOPES = "openid profile email offline_access"
PROFILE = "provider:chatgpt"


class ChatGPTAuthError(RuntimeError):
    pass


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def extract_account_id(token: str) -> str:
    """Read the non-secret account id claim without validating the bearer JWT."""
    try:
        payload = token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        claims = json.loads(base64.urlsafe_b64decode(payload))
    except (IndexError, ValueError, TypeError, json.JSONDecodeError):
        return ""
    direct = claims.get("chatgpt_account_id")
    if isinstance(direct, str):
        return direct
    auth = claims.get("https://api.openai.com/auth")
    if isinstance(auth, dict) and isinstance(auth.get("chatgpt_account_id"), str):
        return auth["chatgpt_account_id"]
    organizations = claims.get("organizations")
    if isinstance(organizations, list) and organizations:
        first = organizations[0]
        if isinstance(first, dict) and isinstance(first.get("id"), str):
            return first["id"]
    return ""


def authorization_url(challenge: str, state: str) -> str:
    query = urllib.parse.urlencode(
        {
            "client_id": CLIENT_ID,
            "redirect_uri": REDIRECT_URI,
            "response_type": "code",
            "scope": SCOPES,
            "state": state,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "id_token_add_organizations": "true",
            "codex_cli_simplified_flow": "true",
            "originator": "opencode",
        }
    )
    return f"{AUTHORIZE_URL}?{query}"


class ChatGPTAuthManager:
    """Owns OAuth lifecycle; callers only need ``valid_access_token`` and status."""

    def __init__(
        self,
        secret_store: Any,
        *,
        http_client: Any = None,
        browser_open: Callable[[str], Any] = webbrowser.open,
    ) -> None:
        self.secrets = secret_store
        self._http = http_client or httpx.Client(timeout=30)
        self._browser_open = browser_open
        self._lock = threading.Lock()
        self._signing_in = False
        self._last_error: Optional[str] = None

    def status(self) -> dict[str, Any]:
        profile = self.secrets.get(PROFILE) or {}
        return {
            "connected": bool(profile.get("access_token")),
            "authorizing": self._signing_in,
            "account": profile.get("account_id") or "",
            "last_error": self._last_error,
        }

    def start_sign_in(self, on_success: Optional[Callable[[], None]] = None) -> dict[str, Any]:
        with self._lock:
            if self._signing_in:
                return {"ok": True, "started": False}
            self._signing_in = True
            self._last_error = None
        thread = threading.Thread(
            target=self._sign_in_worker, args=(on_success,), daemon=True
        )
        thread.start()
        return {"ok": True, "started": True}

    def sign_out(self) -> None:
        self.secrets.delete(PROFILE)
        self._last_error = None

    def valid_access_token(self) -> tuple[str, str]:
        with self._lock:
            profile = dict(self.secrets.get(PROFILE) or {})
            token = str(profile.get("access_token") or "")
            if not token:
                raise ChatGPTAuthError("Not signed in to ChatGPT")
            expires = float(profile.get("expires") or 0)
            if not expires or expires > time.time() + 30:
                return token, str(profile.get("account_id") or "")
            refresh_token = str(profile.get("refresh_token") or "")
            if not refresh_token:
                raise ChatGPTAuthError("ChatGPT sign-in expired; sign in again")
            response = self._http.post(
                TOKEN_URL,
                data={
                    "grant_type": "refresh_token",
                    "client_id": CLIENT_ID,
                    "refresh_token": refresh_token,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            tokens = self._token_payload(response, "Token refresh")
            if not tokens.get("refresh_token"):
                tokens["refresh_token"] = refresh_token
            self.secrets.put(PROFILE, tokens)
            return tokens["access_token"], tokens["account_id"]

    def _sign_in_worker(self, on_success: Optional[Callable[[], None]]) -> None:
        try:
            verifier = _b64url(random_secrets.token_bytes(32))
            challenge = _b64url(hashlib.sha256(verifier.encode("ascii")).digest())
            state = _b64url(random_secrets.token_bytes(32))
            code = self._wait_for_callback(authorization_url(challenge, state), state)
            response = self._http.post(
                TOKEN_URL,
                data={
                    "grant_type": "authorization_code",
                    "client_id": CLIENT_ID,
                    "code": code,
                    "redirect_uri": REDIRECT_URI,
                    "code_verifier": verifier,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            self.secrets.put(PROFILE, self._token_payload(response, "Token exchange"))
            if on_success:
                on_success()
        except Exception as exc:
            self._last_error = str(exc)
        finally:
            with self._lock:
                self._signing_in = False

    def _wait_for_callback(self, url: str, expected_state: str) -> str:
        result: dict[str, str] = {}
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802 - stdlib handler contract
                parsed = urllib.parse.urlparse(self.path)
                params = urllib.parse.parse_qs(parsed.query)
                code = (params.get("code") or [""])[0]
                state = (params.get("state") or [""])[0]
                error = (params.get("error_description") or params.get("error") or [""])[0]
                if error:
                    result["error"] = error
                elif not code:
                    result["error"] = "OAuth callback did not include a code"
                elif not random_secrets.compare_digest(state, expected_state):
                    result["error"] = "OAuth state mismatch; please try again"
                else:
                    result["code"] = code
                ok = "code" in result
                self.send_response(200 if ok else 400)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                title = "Signed in to OpenWork" if ok else "OpenWork sign-in failed"
                detail = (
                    "You can close this window."
                    if ok
                    else result.get("error", "Please try again.")
                )
                safe_title = html.escape(title)
                safe_detail = html.escape(detail)
                self.wfile.write(
                    (
                        f"<!doctype html><title>{safe_title}</title>"
                        "<body style='font-family:system-ui;display:grid;place-items:center;"
                        "height:100vh;background:#171717;color:white'>"
                        f"<main><h2>{safe_title}</h2><p>{safe_detail}</p></main></body>"
                    ).encode()
                )

            def log_message(self, format: str, *args: Any) -> None:
                return

        try:
            server = HTTPServer(("127.0.0.1", 1455), Handler)
        except OSError as exc:
            raise ChatGPTAuthError("Callback port 1455 is already in use") from exc
        with server:
            server.timeout = 300
            outer._browser_open(url)
            server.handle_request()
        if "code" in result:
            return result["code"]
        raise ChatGPTAuthError(result.get("error") or "ChatGPT sign-in timed out")

    @staticmethod
    def _token_payload(response: Any, action: str) -> dict[str, Any]:
        if response.status_code != 200:
            body = getattr(response, "text", "")
            raise ChatGPTAuthError(f"{action} failed: {body[:500] or response.status_code}")
        try:
            data = response.json()
        except (ValueError, TypeError) as exc:
            raise ChatGPTAuthError(f"{action} returned invalid JSON") from exc
        access_token = str(data.get("access_token") or "")
        if not access_token:
            raise ChatGPTAuthError(f"{action} did not return an access token")
        return {
            "type": "oauth",
            "access_token": access_token,
            "refresh_token": str(data.get("refresh_token") or ""),
            "expires": time.time() + float(data.get("expires_in") or 3600),
            "account_id": extract_account_id(access_token),
        }
