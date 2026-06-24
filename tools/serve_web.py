#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Serve Sophia AGI thesis web UI + /api/ask agent endpoint.

Usage:
  python tools/build_web_data.py
  python tools/serve_web.py
  Open http://127.0.0.1:8765
"""

from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PORT = 8765


class SophiaHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args) -> None:
        print(f"[web] {args[0]}")

    def _send_json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path, content_type: str) -> None:
        if not path.exists() or not path.is_file():
            self.send_error(404)
            return
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        route = parsed.path.lstrip("/") or "index.html"
        if route == "api/health":
            self._send_json(200, {"ok": True, "version": (ROOT / "VERSION").read_text().strip()})
            return
        path = WEB / route
        if route.endswith(".css"):
            self._send_file(path, "text/css; charset=utf-8")
        elif route.endswith(".js"):
            self._send_file(path, "application/javascript; charset=utf-8")
        elif route.endswith(".json"):
            self._send_file(path, "application/json; charset=utf-8")
        elif route == "index.html" or not path.suffix:
            self._send_file(WEB / "index.html", "text/html; charset=utf-8")
        else:
            self._send_file(path, "application/octet-stream")

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/ask":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", 0))
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except json.JSONDecodeError:
            self._send_json(400, {"error": "invalid JSON"})
            return

        mode = payload.get("mode", "advisor")
        question = (payload.get("question") or "").strip()
        if mode not in ("advisor", "repo", "life") or not question:
            self._send_json(400, {"error": "mode and question required"})
            return

        try:
            from agent.gate import check_response
            from agent.llm import complete
            from agent.memory import log_decision
            from agent.prompts import MODE_PROMPTS
            from agent.retrieval import format_context, retrieve
            from tools.sophia_agent import build_user_prompt

            chunks = retrieve(question, top_k=6)
            answer = complete(MODE_PROMPTS[mode], build_user_prompt(mode, question))
            gate = check_response(
                answer,
                mode=mode,
                question=question,
                sources=[c.path for c in chunks],
            )
            log_decision(
                mode=mode,
                question=question,
                answer=answer,
                sources=[c.path for c in chunks],
                gate=gate,
            )
            self._send_json(200, {"mode": mode, "answer": answer, "gate": gate, "sources": [c.path for c in chunks]})
        except Exception as exc:
            self._send_json(500, {"error": str(exc)})


def main() -> int:
    if not WEB.exists():
        print(f"Missing {WEB}")
        return 1
    print(f"Sophia AGI web → http://127.0.0.1:{PORT}")
    print("Agent API: POST /api/ask  {mode, question}")
    server = ThreadingHTTPServer(("127.0.0.1", PORT), SophiaHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())