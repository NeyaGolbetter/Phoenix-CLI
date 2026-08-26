"""Tests for phoenix_cli.mcp."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from phoenix_cli.mcp import (
    MCPClient,
    MCPError,
    MCPManager,
    MCPTool,
    load_mcp_config,
    save_mcp_config,
)


# ---------------------------------------------------------------------------
# MCPTool
# ---------------------------------------------------------------------------


def test_tool_to_openai():
    tool = MCPTool(
        name="create_part",
        description="Create a Roblox part",
        input_schema={
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "size": {"type": "array"},
            },
        },
        server_name="roblox",
    )
    result = tool.to_openai()
    assert result["type"] == "function"
    assert result["function"]["name"] == "mcp__roblox__create_part"
    assert result["function"]["description"] == "Create a Roblox part"
    assert "properties" in result["function"]["parameters"]


def test_tool_qualified_name():
    tool = MCPTool(name="echo", description="", input_schema={}, server_name="myserver")
    assert tool.qualified_name == "mcp__myserver__echo"

    tool2 = MCPTool(name="echo", description="", input_schema={}, server_name="")
    assert tool2.qualified_name == "echo"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_load_mcp_config_empty(tmp_path, monkeypatch):
    """Returns empty list when no config file exists."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    assert load_mcp_config() == []


def test_save_and_load_mcp_config(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    servers = [
        {"name": "roblox", "command": ["npx", "-y", "roblox-mcp"]},
        {"name": "remote", "url": "https://mcp.example.com"},
    ]
    path = save_mcp_config(servers)
    assert path.is_file()
    loaded = load_mcp_config()
    assert len(loaded) == 2
    assert loaded[0]["name"] == "roblox"
    assert loaded[1]["url"] == "https://mcp.example.com"


def test_load_mcp_config_corrupt(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    (tmp_path / ".phoenix_mcp.json").write_text("{ invalid json")
    assert load_mcp_config() == []


# ---------------------------------------------------------------------------
# MCPManager
# ---------------------------------------------------------------------------


def test_manager_no_servers():
    """Manager with no servers has no tools."""
    manager = MCPManager()
    assert manager.get_openai_tools() == []
    assert manager.get_tool_names() == []
    assert manager.connected_servers == []


# ---------------------------------------------------------------------------
# Stdio transport round-trip (real subprocess)
# ---------------------------------------------------------------------------

FAKE_STDIO_SERVER = r"""
import json
import sys

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        msg = json.loads(line)
    except json.JSONDecodeError:
        continue
    method = msg.get("method")
    req_id = msg.get("id")
    if method == "initialize":
        print(json.dumps({"jsonrpc": "2.0", "id": req_id, "result": {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "fake", "version": "1.0"},
        }}), flush=True)
    elif method == "tools/list":
        print(json.dumps({"jsonrpc": "2.0", "id": req_id, "result": {"tools": [{
            "name": "create_part",
            "description": "Create a part",
            "inputSchema": {"type": "object", "properties": {}},
        }]}}), flush=True)
    elif method == "tools/call":
        args = (msg.get("params") or {}).get("arguments") or {}
        print(json.dumps({"jsonrpc": "2.0", "id": req_id, "result": {
            "content": [{"type": "text", "text": "made " + str(args.get("name"))}],
            "isError": False,
        }}), flush=True)
"""


def test_stdio_transport_round_trip():
    async def _run():
        client = MCPClient.from_config({
            "name": "roblox",
            "command": [sys.executable, "-c", FAKE_STDIO_SERVER],
        })
        try:
            await client.connect()
            tools = await client.list_tools()
            assert [t.name for t in tools] == ["create_part"]
            result = await client.call_tool(
                "mcp__roblox__create_part", {"name": "Floor"}
            )
            assert result == "made Floor"
        finally:
            await client.close()

    asyncio.run(_run())


def test_stdio_transport_fails_fast_when_server_exits():
    """A server that dies immediately must fail fast with its stderr."""

    async def _run():
        dying = (
            "import sys, time\n"
            "sys.stderr.write('npm error 404: package does not exist\\n')\n"
            "sys.stderr.flush()\n"
            "sys.exit(1)\n"
        )
        client = MCPClient.from_config({
            "name": "broken",
            "command": [sys.executable, "-c", dying],
        })
        t0 = time.monotonic()
        with pytest.raises(MCPError) as exc_info:
            await client.connect()
        elapsed = time.monotonic() - t0
        assert "package does not exist" in str(exc_info.value)
        assert "exit code 1" in str(exc_info.value)
        assert elapsed < 10, "early exit must fail fast, took %.1fs" % elapsed
        await client.close()

    asyncio.run(_run())


def test_stdio_transport_command_not_found():
    async def _run():
        client = MCPClient.from_config({
            "name": "ghost",
            "command": ["definitely-not-a-real-command-xyz"],
        })
        with pytest.raises(MCPError) as exc_info:
            await client.connect()
        assert "command not found" in str(exc_info.value)
        await client.close()

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# SSE transport — session-bound server
# ---------------------------------------------------------------------------

class _SessionSSEHandler(BaseHTTPRequestHandler):
    """Minimal MCP-over-SSE server.

    Every GET /sse creates a NEW session id, and POSTs are only accepted
    for the *current* session's stream. A client that closes and reopens
    the SSE stream (the old buggy behavior) can never receive replies.
    """

    streams = {}          # sid -> handler (writer queue)
    lock = threading.Lock()
    next_sid = 0

    def log_message(self, *args):
        pass

    def _sid(self):
        return urllib.parse.parse_qs(
            urllib.parse.urlparse(self.path).query
        ).get("session_id", [None])[0]

    def do_GET(self):
        if not self.path.rstrip("/").endswith("/sse"):
            return self._json(404, {"error": "not found"})
        with _SessionSSEHandler.lock:
            _SessionSSEHandler.next_sid += 1
            sid = f"s{_SessionSSEHandler.next_sid}"
            _SessionSSEHandler.streams[sid] = self
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        try:
            self.wfile.write(
                f"event: endpoint\ndata: /messages?session_id={sid}\n\n".encode()
            )
            self.wfile.flush()
            # Keep the stream open until the client disconnects.
            while True:
                try:
                    self.wfile.write(b": keep-alive\n\n")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    break
                time.sleep(0.05)
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            with _SessionSSEHandler.lock:
                _SessionSSEHandler.streams.pop(sid, None)

    def do_POST(self):
        qs = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        sid = qs.get("session_id", [None])[0]
        length = int(self.headers.get("Content-Length") or 0)
        try:
            msg = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            return self._json(400, {"error": "bad json"})
        with _SessionSSEHandler.lock:
            handler = _SessionSSEHandler.streams.get(sid)
        if handler is None:
            # Old sessions are invalid — this catches the reopen bug.
            return self._json(404, {"error": "unknown session"})
        req_id = msg.get("id")
        method = msg.get("method", "")
        if method == "initialize":
            result = {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "sse-fake", "version": "1.0"},
            }
        elif method == "tools/list":
            result = {"tools": [{
                "name": "sse_tool",
                "description": "A tool over SSE",
                "inputSchema": {"type": "object", "properties": {}},
            }]}
        else:
            result = {}
        payload = json.dumps(
            {"jsonrpc": "2.0", "id": req_id, "result": result}
        )
        try:
            handler.wfile.write(f"event: message\ndata: {payload}\n\n".encode())
            handler.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            return self._json(404, {"error": "session gone"})
        self._json(202, {"ok": True})

    def _json(self, status, payload):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def test_sse_transport_keeps_session_stream_open():
    """Replies must arrive on the SAME /sse stream we opened.

    Regression test: the old client closed its first SSE stream and opened
    a second one, so session-bound servers never delivered responses.
    """

    async def _run():
        _SessionSSEHandler.streams = {}
        _SessionSSEHandler.lock = threading.Lock()

        server = ThreadingHTTPServer(("127.0.0.1", 0), _SessionSSEHandler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        port = server.server_address[1]
        try:
            client = MCPClient.from_config({
                "name": "remote",
                "url": f"http://127.0.0.1:{port}",
            })
            await client.connect()
            tools = await client.list_tools()
            assert [t.name for t in tools] == ["sse_tool"]
            await client.close()

            # The stream is closed on disconnect — no leak.
            deadline = time.monotonic() + 5
            while _SessionSSEHandler.streams and time.monotonic() < deadline:
                time.sleep(0.05)
            assert len(_SessionSSEHandler.streams) == 0
        finally:
            server.shutdown()
            server.server_close()

    asyncio.run(_run())
