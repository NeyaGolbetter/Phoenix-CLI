"""A tiny OpenAI-compatible chat-completions server used by the test suite.

Pure stdlib (http.server + threads) so tests never need external services.
Scenario selection happens through the *model name* sent in the request:

    ok               normal markdown-ish SSE stream
    usage            stream that also emits a `usage` chunk
    echo             replies with the user's message (no markdown)
    nonstream        ignores `stream: true` and sends one complete message
    fail-auth        401  (also triggered by `Authorization: Bearer bad-key`)
    fail-model       404 model-not-found
    fail-model-400   400 with a model-not-found style message
    fail-rate        429 rate limit
    fail-server      500
    fail-html        200 with an HTML content-type (wrong endpoint)
    slow             sleeps 1s before the first token
"""

from __future__ import annotations

import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import List, Tuple

MARKDOWN_REPLY = (
    "Hello from the **mock provider**!\n\n"
    "Here is a code block:\n\n"
    "```python\n"
    "def greet(name: str) -> str:\n"
    "    return f\"Hello, {name}!\"\n"
    "\n"
    "print(greet('Phoenix'))\n"
    "```\n\n"
    "That's all, [phoenix](https://example.com) out."
)


class MockOpenAIHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "MockOpenAI/1.0"

    # -- plumbing -----------------------------------------------------------

    def log_message(self, *args):  # silence request logging
        pass

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, status: int, html: str) -> None:
        body = html.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # -- routes -------------------------------------------------------------

    def do_POST(self):  # noqa: N802 (http.server API)
        length = int(self.headers.get("Content-Length") or 0)
        try:
            request = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            return self._send_json(400, {"error": {"message": "bad json"}})

        model = str(request.get("model", ""))
        auth = self.headers.get("Authorization", "")

        if self.path.rstrip("/").endswith("/chat/completions"):
            return self._chat_completions(request, model, auth)

        # Any other endpoint: behave like a server that is not an OpenAI API.
        return self._send_html(200, "<html><body><h1>It works!</h1></body></html>")

    def _chat_completions(self, request: dict, model: str, auth: str) -> None:
        if model == "fail-auth" or auth == "Bearer bad-key":
            return self._send_json(401, {"error": {"message": "Invalid API key"}})
        if model == "fail-model":
            return self._send_json(
                404, {"error": {"message": f"model {model!r} not found"}}
            )
        if model == "fail-model-400":
            return self._send_json(
                400, {"error": {"message": "The model 'nope' does not exist"}}
            )
        if model == "fail-rate":
            return self._send_json(429, {"error": {"message": "Too many requests"}})
        if model == "fail-server":
            return self._send_json(500, {"error": {"message": "internal boom"}})
        if model == "fail-html":
            return self._send_html(200, "<html><body><h1>Welcome!</h1></body></html>")

        if model == "nonstream":
            reply = MARKDOWN_REPLY
            return self._send_json(
                200,
                {
                    "id": "chatcmpl-nonstream",
                    "object": "chat.completion",
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": reply},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 11,
                        "completion_tokens": 40,
                        "total_tokens": 51,
                    },
                },
            )

        if model == "echo":
            messages = request.get("messages") or []
            reply = messages[-1].get("content", "") if messages else ""
        else:
            reply = MARKDOWN_REPLY

        self._sse_stream(model, reply, with_usage=(model == "usage"),
                         slow=(model == "slow"))

    # -- SSE streaming -------------------------------------------------------

    def _sse_stream(self, model: str, content: str, *, with_usage: bool,
                    slow: bool) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        pieces: List[str] = re.findall(r"\S+\s*", content)
        if slow:
            time.sleep(0.7)
        for index, piece in enumerate(pieces):
            chunk = {
                "id": "chatcmpl-mock",
                "object": "chat.completion.chunk",
                "model": model,
                "choices": [
                    {"index": 0, "delta": {"content": piece}, "finish_reason": None}
                ],
            }
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode("utf-8"))
            self.wfile.flush()
            time.sleep(0.001)

        if with_usage:
            usage = {
                "id": "chatcmpl-mock",
                "object": "chat.completion.chunk",
                "model": model,
                "choices": [],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": len(pieces),
                    "total_tokens": 10 + len(pieces),
                },
            }
            self.wfile.write(f"data: {json.dumps(usage)}\n\n".encode("utf-8"))
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


def start_server() -> Tuple[ThreadingHTTPServer, int]:
    """Start the mock server on an ephemeral port. Returns (server, port)."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), MockOpenAIHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, server.server_address[1]
