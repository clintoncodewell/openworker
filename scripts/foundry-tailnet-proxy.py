#!/usr/bin/env python3
"""Reach a VNet-restricted Azure Foundry endpoint from another machine on the tailnet.

`foundry-codex-dev.openai.azure.com` accepts requests only from the VM's IP (an Azure
Virtual Network/Firewall rule), so OpenWorker on the Mac gets 403 with a perfectly valid
key. This forwards HTTP from the VM's tailnet address to that endpoint over TLS,
rewriting `Host` — Azure's front door rejects any other hostname.

The security boundary is unchanged in the way that matters: the listener binds ONE
tailnet address, never 0.0.0.0, so only Clinton's own devices reach it. The alternative
is adding each client's public IP to the Foundry firewall, which is worse — home IPs
rotate, and a public IP is a far broader grant than a tailnet identity.

It holds no credentials. The client's `Authorization` header passes through untouched,
so an unauthenticated caller gets 401 from Azure exactly as it would directly.

Deliberately protocol-agnostic: bodies are relayed as opaque bytes in both directions.
`~/.codex/tool-strip-proxy.py` looks similar but is NOT reusable here — it enforces a
Responses-API terminal event and so kills a healthy chat-completions stream.

  foundry-tailnet-proxy.py [--bind IP] [--port N] [--upstream HOST]
"""

from __future__ import annotations

import argparse
import http.client
import http.server
import ipaddress
import socketserver
import ssl
import sys

CONNECT_TIMEOUT_S = 30
# Time-to-first-byte, NOT the connect timeout. A reasoning model thinks for minutes before
# it emits a header; leaving the 30s connect timeout in force through `getresponse()` turns
# a healthy slow answer into a 502 (measured: a Kimi K3 completion took 29s).
HEADER_TIMEOUT_S = 600
# Above any sane inter-chunk gap: this fires only when upstream has gone silent, which
# otherwise strands a worker thread and an Azure connection forever.
IDLE_TIMEOUT_S = 600
READ_SIZE = 64 * 1024
MAX_BODY_BYTES = 32 * 1024 * 1024
HOP_BY_HOP = {
    "connection",
    "content-length",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}

UPSTREAM_HOST = "foundry-codex-dev.openai.azure.com"


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # one line per request, not urllib's default
        sys.stderr.write(f"{self.address_string()} {fmt % args}\n")

    def do_GET(self):
        self._proxy("GET")

    def do_POST(self):
        self._proxy("POST")

    def do_HEAD(self):
        self._proxy("HEAD")

    def _headers_for_upstream(self, body: bytes) -> dict[str, str]:
        headers = {
            k: v
            for k, v in self.headers.items()
            if k.lower() not in HOP_BY_HOP and k.lower() != "host"
        }
        headers["Host"] = UPSTREAM_HOST  # Azure 400s "Invalid Hostname" otherwise
        headers["Accept-Encoding"] = "identity"  # relay bytes, never re-encode
        headers["Content-Length"] = str(len(body))
        return headers

    def _fail(self, status: int, message: str) -> None:
        payload = f'{{"error": {{"message": "{message}"}}}}'.encode()
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):
            pass
        self.close_connection = True

    def _proxy(self, method: str) -> None:
        # Framing is validated before anything else, and inside the request handler so a
        # bad header answers 400 instead of raising past the error path and cutting the
        # connection with no status.
        if "chunked" in self.headers.get("Transfer-Encoding", "").lower():
            # `rfile` holds raw chunk framing that this proxy does not decode. Forwarding
            # it as a body would send the chunk headers to Azure as content AND leave the
            # remaining bytes in the socket to be parsed as the next request. Refuse.
            return self._fail(411, "chunked request bodies are not supported; send Content-Length")
        raw_length = self.headers.get("Content-Length", "0") or "0"
        try:
            length = int(raw_length)
            if length < 0:
                raise ValueError
        except ValueError:
            return self._fail(400, "invalid Content-Length")
        if length > MAX_BODY_BYTES:
            # Checked BEFORE the read: a thread per connection times an unbounded
            # Content-Length is a one-device memory exhaustion, self-inflicted or not.
            # Azure itself rejects bodies near 1MB, so nothing legitimate is lost.
            return self._fail(413, "request body too large")
        body = self.rfile.read(length) if length else b""
        upstream = None
        started = False
        try:
            upstream = http.client.HTTPSConnection(
                UPSTREAM_HOST,
                443,
                timeout=CONNECT_TIMEOUT_S,
                context=ssl.create_default_context(),
            )
            # Connect under the short timeout (a dead host should fail fast), then raise it
            # before the request — `getresponse()` otherwise inherits the connect timeout.
            upstream.connect()
            if upstream.sock is not None:
                upstream.sock.settimeout(HEADER_TIMEOUT_S)
            upstream.request(
                method, self.path, body=body, headers=self._headers_for_upstream(body)
            )
            response = upstream.getresponse()
            if upstream.sock is not None:
                upstream.sock.settimeout(IDLE_TIMEOUT_S)

            self.send_response(response.status, response.reason)
            for key, value in response.getheaders():
                if key.lower() not in HOP_BY_HOP:
                    self.send_header(key, value)
            has_body = method != "HEAD" and response.status not in (204, 304)
            if has_body:
                self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            started = True

            if has_body:
                # `read1` returns whatever has arrived rather than waiting for a full
                # buffer, which is what keeps a streamed response actually streaming.
                while chunk := response.read1(READ_SIZE):
                    self.wfile.write(b"%X\r\n%s\r\n" % (len(chunk), chunk))
                    self.wfile.flush()
                try:
                    self.wfile.write(b"0\r\n\r\n")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    # A client that closes the instant the last event lands leaves the
                    # terminator nowhere to go. The response itself is complete.
                    self.close_connection = True
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True
        except Exception as exc:
            sys.stderr.write(f"upstream_failure {method} {self.path}: {exc!r}\n")
            if not started:
                self._fail(502, f"proxy could not reach {UPSTREAM_HOST}")
            else:
                self.close_connection = True  # headers are already out; just cut it
        finally:
            if upstream is not None:
                upstream.close()


class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main() -> int:
    global UPSTREAM_HOST

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bind", default="100.65.245.83", help="tailnet address to listen on")
    ap.add_argument("--port", type=int, default=8802)
    ap.add_argument("--upstream", default=UPSTREAM_HOST)
    args = ap.parse_args()

    UPSTREAM_HOST = args.upstream
    # The whole security argument for this proxy is that it is tailnet-only, so check that
    # rather than trusting the operator to pass the right thing. Refusing only 0.0.0.0
    # still lets a LAN address through, which would put a bearer token on plain HTTP over
    # the local network. Tailscale hands out 100.64.0.0/10 (CGNAT); loopback is fine too.
    try:
        addr = ipaddress.ip_address(args.bind)
    except ValueError:
        ap.error(f"--bind must be an IP address, not a hostname: {args.bind}")
    if not (addr.is_loopback or addr in ipaddress.ip_network("100.64.0.0/10")):
        ap.error(
            f"refusing to bind {args.bind}: not a Tailscale (100.64.0.0/10) or loopback "
            "address. This proxy forwards Authorization headers over plain HTTP."
        )

    sys.stderr.write(f"listening {args.bind}:{args.port} -> https://{UPSTREAM_HOST}\n")
    Server((args.bind, args.port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
