"""MCP (Model Context Protocol) client for Phoenix CLI.

MCP lets AI models use external tools — e.g. a Roblox MCP server can
expose tools like ``create_part``, ``edit_script``, ``get_hierarchy`` etc.
so the LLM can drive the Roblox engine during a chat session.

This module implements a lightweight MCP client that supports:

* **stdio transport** — spawns the server as a subprocess and talks
  JSON-RPC over stdin/stdout. This is the most common transport for
  local MCP servers (e.g. ``npx @anthropic/mcp-server-...``).
* **sse transport** — connects to a remote MCP server over HTTP+SSE
  (Server-Sent Events). The server pushes a message endpoint via SSE;
  the client POSTs JSON-RPC to that endpoint.

Usage (high-level):
    async with MCPClient.from_config(server_cfg) as client:
        tools = await client.list_tools()       # OpenAI-style tool defs
        result = await client.call_tool(name, arguments)

The tool definitions returned by ``list_tools`` are already shaped for
the OpenAI chat-completions ``tools`` parameter, so they can be passed
straight through to ``PhoenixClient.chat_stream``.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import httpx


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class MCPError(Exception):
    """Something went wrong while talking to an MCP server."""


class MCPConnectionError(MCPError):
    """Could not connect to the MCP server."""


class MCPToolError(MCPError):
    """The MCP server rejected a tool call."""


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------


@dataclass
class MCPTool:
    """One tool advertised by an MCP server."""

    name: str
    description: str
    input_schema: Dict[str, Any]
    server_name: str = ""  # which MCP server provides it

    def to_openai(self) -> Dict[str, Any]:
        """Return the OpenAI ``tools[]`` entry for this tool."""
        return {
            "type": "function",
            "function": {
                "name": self.qualified_name,
                "description": self.description or f"Tool: {self.name}",
                "parameters": self.input_schema or {"type": "object", "properties": {}},
            },
        }

    @property
    def qualified_name(self) -> str:
        """Tool name prefixed with the server name to avoid collisions."""
        if self.server_name:
            return f"mcp__{self.server_name}__{self.name}"
        return self.name


# ---------------------------------------------------------------------------
# JSON-RPC helpers
# ---------------------------------------------------------------------------

def _rpc_request(method: str, params: Any = None, req_id: int = 1) -> Dict[str, Any]:
    msg: Dict[str, Any] = {
        "jsonrpc": "2.0",
        "id": req_id,
        "method": method,
    }
    if params is not None:
        msg["params"] = params
    return msg


def _rpc_notification(method: str, params: Any = None) -> Dict[str, Any]:
    msg: Dict[str, Any] = {
        "jsonrpc": "2.0",
        "method": method,
    }
    if params is not None:
        msg["params"] = params
    return msg


# ---------------------------------------------------------------------------
# Stdio transport
# ---------------------------------------------------------------------------


class _StdioTransport:
    """Talk JSON-RPC with a subprocess over stdin/stdout."""

    def __init__(self, command: Sequence[str], env: Optional[Dict[str, str]] = None,
                 cwd: Optional[str] = None) -> None:
        self._command = list(command)
        self._env = env
        self._cwd = cwd
        self._proc: Optional[asyncio.subprocess.Process] = None
        self._pending: Dict[int, asyncio.Future] = {}
        self._reader_task: Optional[asyncio.Task] = None
        self._next_id: int = 1

    async def start(self) -> None:
        env = dict(os.environ)
        if self._env:
            env.update(self._env)
        try:
            self._proc = await asyncio.create_subprocess_exec(
                *self._command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=self._cwd,
            )
        except FileNotFoundError as exc:
            raise MCPConnectionError(
                f"MCP server command not found: {self._command[0]!r}.\n"
                "Make sure the command is installed and on your PATH."
            ) from exc
        except OSError as exc:
            raise MCPConnectionError(
                f"Could not start MCP server: {exc}"
            ) from exc
        self._reader_task = asyncio.create_task(self._read_loop())

    async def stop(self) -> None:
        if self._reader_task:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except (asyncio.CancelledError, Exception):
                pass
        if self._proc and self._proc.returncode is None:
            self._proc.terminate()
            try:
                await asyncio.wait_for(self._proc.wait(), timeout=5)
            except asyncio.TimeoutError:
                self._proc.kill()

    async def send_request(self, method: str, params: Any = None) -> Any:
        if not self._proc or not self._proc.stdin or not self._proc.stdout:
            raise MCPConnectionError("Transport not started")
        req_id = self._next_id
        self._next_id += 1
        msg = _rpc_request(method, params, req_id)
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        self._pending[req_id] = fut
        data = json.dumps(msg) + "\n"
        self._proc.stdin.write(data.encode())
        await self._proc.stdin.drain()
        return await asyncio.wait_for(fut, timeout=60)

    async def send_notification(self, method: str, params: Any = None) -> None:
        if not self._proc or not self._proc.stdin:
            raise MCPConnectionError("Transport not started")
        msg = _rpc_notification(method, params)
        data = json.dumps(msg) + "\n"
        self._proc.stdin.write(data.encode())
        await self._proc.stdin.drain()

    async def _read_loop(self) -> None:
        assert self._proc and self._proc.stdout
        try:
            while True:
                line = await self._proc.stdout.readline()
                if not line:
                    break
                line = line.decode().strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue
                msg_id = msg.get("id")
                if msg_id is not None and msg_id in self._pending:
                    fut = self._pending.pop(msg_id)
                    if "error" in msg:
                        err = msg["error"]
                        fut.set_exception(
                            MCPToolError(
                                f"MCP error {err.get('code', '?')}: "
                                f"{err.get('message', 'unknown')}"
                            )
                        )
                    else:
                        fut.set_result(msg.get("result"))
        except asyncio.CancelledError:
            pass
        except Exception:
            pass


# ---------------------------------------------------------------------------
# SSE transport
# ---------------------------------------------------------------------------


class _SSETransport:
    """Talk JSON-RPC with a remote MCP server over HTTP + SSE."""

    def __init__(self, url: str, headers: Optional[Dict[str, str]] = None,
                 timeout: float = 60.0) -> None:
        self._url = url.rstrip("/")
        self._headers = dict(headers or {})
        self._timeout = timeout
        self._message_endpoint: Optional[str] = None
        self._client: Optional[httpx.AsyncClient] = None
        self._pending: Dict[int, asyncio.Future] = {}
        self._reader_task: Optional[asyncio.Task] = None
        self._next_id: int = 1

    async def start(self) -> None:
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(self._timeout, connect=15.0),
            headers=self._headers,
            follow_redirects=True,
        )
        # Connect to the SSE endpoint to discover the message endpoint.
        try:
            async with self._client.stream("GET", self._url + "/sse") as resp:
                if resp.status_code != 200:
                    await resp.aread()
                    raise MCPConnectionError(
                        f"MCP SSE endpoint returned HTTP {resp.status_code}: {self._url}/sse"
                    )
                # Read lines until we get the endpoint event.
                event_type = ""
                async for raw_line in resp.aiter_lines():
                    line = raw_line.strip()
                    if line.startswith("event:"):
                        event_type = line[6:].strip()
                    elif line.startswith("data:"):
                        data = line[5:].strip()
                        if event_type == "endpoint":
                            # The endpoint may be relative or absolute.
                            if data.startswith("http://") or data.startswith("https://"):
                                self._message_endpoint = data
                            elif data.startswith("/"):
                                base = self._url.split("/sse")[0]
                                self._message_endpoint = base + data
                            else:
                                base = self._url.rsplit("/", 1)[0]
                                self._message_endpoint = base + "/" + data
                            break
        except httpx.ConnectError as exc:
            raise MCPConnectionError(
                f"Could not connect to MCP server at {self._url}.\n"
                "Is the server running and is the URL correct?"
            ) from exc
        except httpx.TimeoutException as exc:
            raise MCPConnectionError(
                f"Timed out connecting to MCP server at {self._url}"
            ) from exc

        if not self._message_endpoint:
            raise MCPConnectionError(
                f"MCP server at {self._url} did not provide a message endpoint."
            )

        # Start a background task to read SSE events for responses.
        self._reader_task = asyncio.create_task(self._read_sse_loop())

    async def stop(self) -> None:
        if self._reader_task:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except (asyncio.CancelledError, Exception):
                pass
        if self._client:
            await self._client.aclose()
            self._client = None

    async def send_request(self, method: str, params: Any = None) -> Any:
        if not self._client or not self._message_endpoint:
            raise MCPConnectionError("SSE transport not started")
        req_id = self._next_id
        self._next_id += 1
        msg = _rpc_request(method, params, req_id)
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        self._pending[req_id] = fut
        try:
            resp = await self._client.post(
                self._message_endpoint,
                json=msg,
                headers={"Content-Type": "application/json"},
            )
            if resp.status_code >= 400:
                raise MCPToolError(
                    f"MCP server returned HTTP {resp.status_code}: {resp.text[:200]}"
                )
        except httpx.ConnectError as exc:
            self._pending.pop(req_id, None)
            raise MCPConnectionError(f"Lost connection to MCP server: {exc}") from exc
        return await asyncio.wait_for(fut, timeout=60)

    async def send_notification(self, method: str, params: Any = None) -> None:
        if not self._client or not self._message_endpoint:
            raise MCPConnectionError("SSE transport not started")
        msg = _rpc_notification(method, params)
        await self._client.post(
            self._message_endpoint,
            json=msg,
            headers={"Content-Type": "application/json"},
        )

    async def _read_sse_loop(self) -> None:
        if not self._client:
            return
        try:
            async with self._client.stream("GET", self._url + "/sse") as resp:
                if resp.status_code != 200:
                    return
                event_type = ""
                async for raw_line in resp.aiter_lines():
                    line = raw_line.strip()
                    if line.startswith("event:"):
                        event_type = line[6:].strip()
                    elif line.startswith("data:"):
                        data = line[5:].strip()
                        if event_type == "message" and data:
                            try:
                                msg = json.loads(data)
                            except json.JSONDecodeError:
                                continue
                            msg_id = msg.get("id")
                            if msg_id is not None and msg_id in self._pending:
                                fut = self._pending.pop(msg_id)
                                if "error" in msg:
                                    err = msg["error"]
                                    fut.set_exception(
                                        MCPToolError(
                                            f"MCP error {err.get('code', '?')}: "
                                            f"{err.get('message', 'unknown')}"
                                        )
                                    )
                                else:
                                    fut.set_result(msg.get("result"))
        except asyncio.CancelledError:
            pass
        except Exception:
            pass


# ---------------------------------------------------------------------------
# High-level MCP client
# ---------------------------------------------------------------------------


class MCPClient:
    """High-level client for one MCP server.

    Usage::

        async with MCPClient.from_config(server_cfg) as mcp:
            tools = await mcp.list_tools()
            result = await mcp.call_tool("some_tool", {"arg": "val"})
    """

    def __init__(self, transport: Any, name: str = "mcp") -> None:
        self._transport = transport
        self.name = name
        self._tools: List[MCPTool] = []
        self._initialized = False

    @classmethod
    def from_config(cls, server_cfg: Dict[str, Any]) -> "MCPClient":
        """Build an ``MCPClient`` from a config dict.

        The dict should have either:
        - ``command`` (list of strings) for stdio transport
        - ``url`` (string) for SSE transport

        Optional keys: ``env``, ``cwd``, ``headers``, ``name``.
        """
        name = server_cfg.get("name", "mcp")
        if "command" in server_cfg:
            transport = _StdioTransport(
                command=server_cfg["command"],
                env=server_cfg.get("env"),
                cwd=server_cfg.get("cwd"),
            )
        elif "url" in server_cfg:
            transport = _SSETransport(
                url=server_cfg["url"],
                headers=server_cfg.get("headers"),
                timeout=server_cfg.get("timeout", 60.0),
            )
        else:
            raise MCPError(
                f"MCP server {name!r} has neither 'command' nor 'url' configured."
            )
        return cls(transport=transport, name=name)

    async def connect(self) -> None:
        from . import __version__
        await self._transport.start()
        # MCP handshake.
        try:
            await self._transport.send_request("initialize", {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "phoenix-cli", "version": __version__},
            })
            await self._transport.send_notification(
                "notifications/initialized"
            )
            self._initialized = True
        except Exception as exc:
            await self._transport.stop()
            raise MCPConnectionError(
                f"MCP handshake failed with {self.name!r}: {exc}"
            ) from exc

    async def close(self) -> None:
        await self._transport.stop()

    async def __aenter__(self) -> "MCPClient":
        await self.connect()
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        await self.close()

    async def list_tools(self) -> List[MCPTool]:
        """Fetch tools from the server and cache them."""
        if not self._initialized:
            raise MCPError("Not connected; call connect() first")
        result = await self._transport.send_request("tools/list")
        tools: List[MCPTool] = []
        for item in (result or {}).get("tools") or []:
            tools.append(MCPTool(
                name=item.get("name", ""),
                description=item.get("description", ""),
                input_schema=item.get("inputSchema") or {},
                server_name=self.name,
            ))
        self._tools = tools
        return tools

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Call a tool by name and return the text content of the result.

        ``tool_name`` should be the *qualified* name (``mcp__server__tool``)
        as advertised in ``to_openai()``.
        """
        if not self._initialized:
            raise MCPError("Not connected; call connect() first")
        # Strip the server prefix if present.
        bare_name = tool_name
        prefix = f"mcp__{self.name}__"
        if bare_name.startswith(prefix):
            bare_name = bare_name[len(prefix):]
        result = await self._transport.send_request("tools/call", {
            "name": bare_name,
            "arguments": arguments,
        })
        # Extract text content from the result.
        contents = (result or {}).get("content") or []
        parts: List[str] = []
        for c in contents:
            if isinstance(c, dict):
                if c.get("type") == "text":
                    parts.append(c.get("text", ""))
                elif c.get("text"):
                    parts.append(c["text"])
        is_error = (result or {}).get("isError", False)
        text = "\n".join(parts) if parts else "(no output)"
        if is_error:
            raise MCPToolError(f"MCP tool {tool_name!r} returned an error: {text}")
        return text


# ---------------------------------------------------------------------------
# Multi-server manager
# ---------------------------------------------------------------------------


class MCPManager:
    """Manage multiple MCP server connections.

    Connects to all configured servers, aggregates their tools, and
    dispatches tool calls to the right server.
    """

    def __init__(self) -> None:
        self._clients: Dict[str, MCPClient] = {}
        self._tools: Dict[str, MCPTool] = {}  # qualified name -> tool

    async def connect_servers(self, servers: List[Dict[str, Any]]) -> List[str]:
        """Connect to all configured MCP servers.

        Returns a list of warnings for servers that failed to connect
        (non-fatal — we continue with the ones that work).
        """
        warnings: List[str] = []
        for cfg in servers:
            name = cfg.get("name", "mcp")
            try:
                client = MCPClient.from_config(cfg)
                await client.connect()
                self._clients[name] = client
            except MCPError as exc:
                warnings.append(f"  ✖ {name}: {exc}")
                continue
        # Collect tools from all connected servers.
        for name, client in self._clients.items():
            try:
                tools = await client.list_tools()
                for tool in tools:
                    self._tools[tool.qualified_name] = tool
            except MCPError as exc:
                warnings.append(f"  ✖ {name}: could not list tools: {exc}")
        return warnings

    async def close(self) -> None:
        for client in self._clients.values():
            try:
                await client.close()
            except Exception:
                pass
        self._clients.clear()
        self._tools.clear()

    async def __aenter__(self) -> "MCPManager":
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        await self.close()

    def get_openai_tools(self) -> List[Dict[str, Any]]:
        """Return OpenAI-style tool definitions for all connected MCP tools."""
        return [tool.to_openai() for tool in self._tools.values()]

    def get_tool_names(self) -> List[str]:
        """Return the qualified names of all available tools."""
        return sorted(self._tools.keys())

    @property
    def connected_servers(self) -> List[str]:
        return list(self._clients.keys())

    async def call_tool(self, qualified_name: str, arguments: Dict[str, Any]) -> str:
        """Route a tool call to the right server."""
        # Find the server.
        for name, client in self._clients.items():
            prefix = f"mcp__{name}__"
            if qualified_name.startswith(prefix):
                return await client.call_tool(qualified_name, arguments)
        raise MCPToolError(
            f"No MCP server provides tool {qualified_name!r}. "
            f"Available: {list(self._tools.keys())}"
        )


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def load_mcp_config() -> List[Dict[str, Any]]:
    """Load MCP server configs from ``~/.phoenix_mcp.json``.

    The file is a JSON object with a ``servers`` key containing a list::

        {
          "servers": [
            {
              "name": "roblox",
              "command": ["npx", "-y", "@anthropic/mcp-server-roblox"],
              "env": {}
            },
            {
              "name": "remote-tools",
              "url": "https://my-mcp-server.example.com"
            }
          ]
        }
    """
    path = Path.home() / ".phoenix_mcp.json"
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    if isinstance(data, dict):
        servers = data.get("servers") or []
        if isinstance(servers, list):
            return [s for s in servers if isinstance(s, dict)]
    return []


def save_mcp_config(servers: List[Dict[str, Any]]) -> Path:
    """Write MCP server configs to ``~/.phoenix_mcp.json``."""
    path = Path.home() / ".phoenix_mcp.json"
    payload = {"servers": servers}
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp_path, path)
    return path
