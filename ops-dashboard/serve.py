#!/usr/bin/env python3
"""Standalone /ops dashboard: a static HTML/CSS/JS viewer for the real
production backend's /api/admin/ops/* endpoints, launched on demand (not
a persistent deployment -- no hosting, no login screen, nothing left
running once you close it).

This script does two things:
1. Serves the static files in this directory (index.html, app.js, etc.).
2. Proxies any /api/* request to the real production backend
   ($FPAI_PROD_URL, from the repo root's .env), attaching the admin
   x-app-token header itself -- so the token never reaches browser-visible
   JS at all, and the browser only ever talks to localhost, sidestepping
   CORS entirely (no Railway config change needed).

Usage: python3 ops-dashboard/serve.py [port]  (default port 8765)
Then open http://localhost:<port> in a browser.

Stdlib only -- no new dependencies for a tool this small (ponytail rung 2/3).
"""

from __future__ import annotations

import http.server
import sys
import urllib.request
import urllib.error
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = Path(__file__).resolve().parent
DEFAULT_PORT = 8765


def _load_env(env_path: Path) -> dict[str, str]:
    """Minimal KEY=VALUE .env parser -- only needs to extract the two
    vars this script actually uses, not full .env semantics (quoting,
    multiline, etc. -- the real .env's FPAI_PROD_URL/FPAI_PROD_TOKEN lines
    are plain, unquoted KEY=VALUE, same as every other var in that file)."""
    values: dict[str, str] = {}
    if not env_path.exists():
        return values
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip()
    return values


def main() -> None:
    env = _load_env(REPO_ROOT / ".env")
    prod_url = env.get("FPAI_PROD_URL")
    prod_token = env.get("FPAI_PROD_TOKEN")
    if not prod_url or not prod_token:
        print("FPAI_PROD_URL/FPAI_PROD_TOKEN not found in the repo root's .env -- cannot proxy to production.")
        sys.exit(1)

    port = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PORT

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

        def do_GET(self):
            if self.path.startswith("/api/"):
                self._proxy()
            else:
                super().do_GET()

        def end_headers(self):
            # This tool gets edited and re-viewed in the same browser tab
            # across a single session -- a stale cached app.js silently
            # running old chart logic (looking like a real bug, not a
            # cache issue) is worse than never caching a few KB of static
            # files that are never a performance concern at this size.
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
            super().end_headers()

        def _proxy(self) -> None:
            target = f"{prod_url}{self.path}"
            req = urllib.request.Request(target, headers={"x-app-token": prod_token})
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    body = resp.read()
                    self.send_response(resp.status)
                    self.send_header("Content-Type", resp.headers.get("Content-Type", "application/json"))
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
            except urllib.error.HTTPError as exc:
                body = exc.read()
                self.send_response(exc.code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except urllib.error.URLError as exc:
                body = f'{{"error": "proxy request failed: {exc.reason}"}}'.encode("utf-8")
                self.send_response(502)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        def log_message(self, format: str, *args) -> None:
            # Quieter than the default (which logs every request to stderr
            # with a full HTTP-version string) -- just method + path + status.
            print(f"{self.command} {self.path} -> {args[1] if len(args) > 1 else '?'}")

    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Ops dashboard: http://localhost:{port}  (proxying to {prod_url})")
    print("Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
