"""Command-line interface for Phoenix CLI.

Commands
--------
phoenix "prompt"        one-shot prompt (also: phoenix ask "prompt")
phoenix chat            interactive chat with 30+ slash commands
phoenix setup           configure BASE_URL / API_KEY / MODEL_NAME
phoenix models          list / pick the provider's available models
phoenix status          show the current configuration (+ optional probe)
phoenix mcp             manage MCP server connections

Design notes for Termux:
* all output goes through rich, which re-measures the terminal on every
  render, so the layout survives window resizes;
* the interactive prompt uses prompt_toolkit (pure Python) when available
  and falls back to plain ``input()`` otherwise;
* Ctrl+C cancels the current request instead of killing the app.
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import click
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

from . import __version__
from .client import (
    APIKeyError,
    ConfigurationError,
    Conversation,
    ModelNotFoundError,
    NetworkError,
    PhoenixClient,
    PhoenixError,
    ProviderError,
    RateLimitError,
)
from .config import (
    ENV_API_KEY,
    ENV_BASE_URL,
    ENV_MODEL,
    check_configured,
    config_path,
    load_config,
    mask_secret,
    normalize_base_url,
    save_config,
)
from .mcp import (
    MCPClient,
    MCPError,
    MCPManager,
    MCPTool,
    load_mcp_config,
    save_mcp_config,
)

# Brand color (phoenix orange).
ACCENT = "#ff6b00"

# A short prompt shown by prompt_toolkit / the input() fallback.
PROMPT_LABEL = "phoenix ❯ "

BANNER = r"""[bold #ff6b00]
   ██████╗ ██╗  ██╗ ██████╗ ███████╗███╗   ██╗██╗██╗  ██╗
   ██╔══██╗██║  ██║██╔═══██╗██╔════╝████╗  ██║██║╚██╗██╔╝
   ██████╔╝███████║██║   ██║█████╗  ██╔██╗ ██║██║ ╚███╔╝
   ██╔═══╝ ██╔══██║██║   ██║██╔══╝  ██║╚██╗██║██║ ██╔██╗
   ██║     ██║  ██║╚██████╝███████╗██║ ╚████║██║██╔╝ ██╗
   ═╝     ╚═╝  ╚═╝ ╚═════╝ ╚══════╝═╝  ╚═══╝╚═╝╚═╝  ╚═╝
[/bold #ff6b00]"""

TAGLINE = "AI that rises with you"

# Code themes supported by rich.
CODE_THEMES = ("monokai", "github_dark", "dracula", "vs_dark", "fruity", "native")

# Human-readable hints appended to specific errors.
ERROR_HINTS: Dict[type, str] = {
    ConfigurationError: (
        "Run `phoenix setup` to configure BASE_URL, API_KEY and MODEL_NAME."
    ),
    APIKeyError: "Run `phoenix setup` and enter a valid API key.",
    ModelNotFoundError: (
        "Run `phoenix models --select` to pick from available models.\n"
        "Tip: `phoenix models` lists all models the provider offers."
    ),
    RateLimitError: "Free tiers often allow only a few requests per minute.",
    NetworkError: (
        "Termux tips:\n"
        "  * local servers must listen on 0.0.0.0 (or 127.0.0.1 inside Termux);\n"
        "  * remote APIs need the phone online and DNS working;\n"
        "  * check the URL with `phoenix status --probe`."
    ),
}

console = Console(highlight=False)

# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def print_banner() -> None:
    """Print the ASCII logo (no figlet/toilet needed, unlike on desktop)."""
    console.print(BANNER)
    console.print(f"[bold]{TAGLINE}[/bold]", justify="center")
    console.print(f"[dim]v{__version__} • OpenAI-compatible APIs + MCP[/dim]", justify="center")
    console.print()


def print_error(exc: Exception) -> None:
    """Print an exception with a clean layout and an actionable hint."""
    console.print()
    console.print(Text(f"✖ {type(exc).__name__}", style="bold red"))
    console.print(Text(str(exc)))
    hint = ERROR_HINTS.get(type(exc))
    if hint is None:
        for klass, text in ERROR_HINTS.items():
            if isinstance(exc, klass):
                hint = text
                break
    if hint:
        console.print(Text(hint, style="dim"))


def _params(
    cfg: Dict[str, str],
    *,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    """Bundle config + per-invocation overrides into one dict."""
    return {
        "base_url": cfg["base_url"],
        "api_key": cfg.get("api_key", ""),
        "model_name": model or cfg["model_name"],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }


def make_client(params: Dict[str, Any]) -> PhoenixClient:
    """Build a client from a params dict (see ``_params``)."""
    return PhoenixClient(
        base_url=params["base_url"],
        api_key=params.get("api_key", ""),
        model_name=params["model_name"],
        extra_headers=params.get("extra_headers"),
        timeout=params.get("timeout", 300.0),
    )


# ---------------------------------------------------------------------------
# MCP helpers
# ---------------------------------------------------------------------------


async def _connect_mcp(cfg: Dict[str, str]) -> Optional[MCPManager]:
    """Connect to all configured MCP servers. Returns None if MCP is off."""
    mcp_enabled = cfg.get("mcp_enabled")
    if not mcp_enabled:
        return None
    servers = load_mcp_config()
    if not servers:
        return None
    manager = MCPManager()
    warnings = await manager.connect_servers(servers)
    if warnings:
        for w in warnings:
            console.print(Text(w, style="yellow"))
    if not manager.connected_servers:
        await manager.close()
        return None
    tool_names = manager.get_tool_names()
    if tool_names:
        console.print(
            Text(
                f"🔧 MCP: {len(tool_names)} tool(s) from "
                f"{len(manager.connected_servers)} server(s)",
                style="dim",
            )
        )
    return manager


def _ask_user_yn(prompt_text: str, default: bool = True) -> bool:
    """Ask the user a yes/no question. Returns True for yes."""
    suffix = " [Y/n] " if default else " [y/N] "
    try:
        answer = input(prompt_text + suffix).strip().lower()
    except (EOFError, KeyboardInterrupt):
        return default
    if answer == "":
        return default
    return answer in ("y", "yes", "1", "true")


def _copy_to_clipboard(text: str) -> bool:
    """Try to copy text to the system clipboard. Returns True on success."""
    # Termux
    if shutil.which("termux-clipboard-set"):
        try:
            subprocess.run(["termux-clipboard-set"], input=text, text=True, timeout=5)
            return True
        except Exception:
            return False
    # macOS
    if shutil.which("pbcopy"):
        try:
            subprocess.run(["pbcopy"], input=text, text=True, timeout=5)
            return True
        except Exception:
            return False
    # Linux
    if shutil.which("xclip"):
        try:
            subprocess.run(["xclip", "-selection", "clipboard"], input=text,
                           text=True, timeout=5)
            return True
        except Exception:
            return False
    if shutil.which("xsel"):
        try:
            subprocess.run(["xsel", "--clipboard", "--input"], input=text,
                           text=True, timeout=5)
            return True
        except Exception:
            return False
    if shutil.which("wl-copy"):
        try:
            subprocess.run(["wl-copy"], input=text, text=True, timeout=5)
            return True
        except Exception:
            return False
    return False


# ---------------------------------------------------------------------------
# Conversation tool-call storage
# ---------------------------------------------------------------------------


def _add_assistant_with_tool_calls(
    conversation: Conversation,
    content: str,
    tool_calls: List[Dict[str, Any]],
) -> None:
    """Add an assistant message with tool_calls to the conversation."""
    msg: Dict[str, Any] = {
        "role": "assistant",
        "content": content or None,
        "tool_calls": [
            {
                "id": tc.get("id", ""),
                "type": tc.get("type", "function"),
                "function": tc.get("function", {}),
            }
            for tc in tool_calls
        ],
    }
    # Add placeholder Message.
    from .client import Message
    conversation.messages.append(Message(role="assistant", content=content or ""))
    idx = len(conversation.messages) - 1
    if not hasattr(conversation, '_tool_msgs'):
        conversation._tool_msgs: Dict[int, Dict[str, Any]] = {}
    conversation._tool_msgs[idx] = msg


def _add_tool_result(
    conversation: Conversation,
    tool_call_id: str,
    tool_name: str,
    result: str,
) -> None:
    """Add a tool result message to the conversation."""
    from .client import Message
    conversation.messages.append(Message(role="tool", content=result))
    idx = len(conversation.messages) - 1
    if not hasattr(conversation, '_tool_call_ids'):
        conversation._tool_call_ids: Dict[int, Dict[str, str]] = {}
    conversation._tool_call_ids[idx] = {
        "tool_call_id": tool_call_id,
        "name": tool_name,
    }


def _build_history_with_tools(conversation: Conversation) -> List[Dict[str, Any]]:
    """Build the full OpenAI message list including tool_call messages."""
    out: List[Dict[str, Any]] = []
    if conversation.system:
        out.append({"role": "system", "content": conversation.system})

    tool_msgs: Dict[int, Dict[str, Any]] = getattr(conversation, '_tool_msgs', {})
    tool_ids: Dict[int, Dict[str, str]] = getattr(conversation, '_tool_call_ids', {})

    for i, msg in enumerate(conversation.messages):
        if i in tool_msgs:
            out.append(tool_msgs[i])
        elif i in tool_ids:
            out.append({
                "role": "tool",
                "content": msg.content,
                "tool_call_id": tool_ids[i]["tool_call_id"],
                "name": tool_ids[i]["name"],
            })
        else:
            out.append(msg.to_dict())
    return out


_original_history = Conversation.history


def _patched_history(self: Conversation) -> List[Dict[str, Any]]:
    if hasattr(self, '_tool_msgs') and self._tool_msgs:
        return _build_history_with_tools(self)
    return _original_history(self)


Conversation.history = _patched_history  # type: ignore


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------

MAX_TOOL_ROUNDS = 10


async def _stream_turn(
    params: Dict[str, Any],
    conversation: Conversation,
    *,
    use_live: bool,
    mcp_manager: Optional[MCPManager] = None,
    auto_approve: bool = True,
    theme: str = "monokai",
    verbose: bool = False,
) -> tuple[bool, str]:
    """Run one chat-completions request and print the streamed reply.

    Handles the full tool-use loop when ``mcp_manager`` is provided.
    When ``auto_approve`` is False, prompts the user before each tool call.
    """
    buffer: List[str] = []
    usage: Dict[str, Any] = {}
    ok = False
    last_update = 0.0
    last_len = 0

    tools: Optional[List[Dict[str, Any]]] = None
    if mcp_manager:
        openai_tools = mcp_manager.get_openai_tools()
        if openai_tools:
            tools = openai_tools

    def renderable():
        text = "".join(buffer)
        if not text:
            return Spinner("dots", text=" waiting for the model...", style="dim")
        return Markdown(text, code_theme=theme, justify="left")

    try:
        for _round in range(MAX_TOOL_ROUNDS + 1):
            buffer.clear()
            usage = {}
            last_update = 0.0
            last_len = 0
            tool_calls_received: List[Dict[str, Any]] = []

            if use_live:
                with Live(
                    renderable(),
                    console=console,
                    refresh_per_second=10,
                    transient=False,
                ) as live:
                    try:
                        async with make_client(params) as client:
                            async for chunk in client.chat_stream(
                                conversation.history(),
                                temperature=params.get("temperature"),
                                max_tokens=params.get("max_tokens"),
                                tools=tools,
                            ):
                                content = chunk.get("content")
                                if content:
                                    buffer.append(content)
                                    if verbose:
                                        console.print(
                                            Text(f"[raw] {content!r}", style="dim")
                                        )
                                    now = time.monotonic()
                                    if (
                                        "\n" in content
                                        or now - last_update >= 0.1
                                        or len(buffer) - last_len >= 128
                                    ):
                                        live.update(renderable())
                                        last_update = now
                                        last_len = len(buffer)
                                elif "tool_calls" in chunk:
                                    tool_calls_received = chunk["tool_calls"]
                                    if verbose:
                                        console.print(
                                            Text(
                                                f"[tool_calls] "
                                                f"{json.dumps(tool_calls_received)}",
                                                style="dim",
                                            )
                                        )
                                elif "usage" in chunk:
                                    usage = chunk["usage"]
                    except PhoenixError:
                        if not buffer:
                            live.update(Text(""))
                        raise
                    live.update(renderable())
            else:
                async with make_client(params) as client:
                    async for chunk in client.chat_stream(
                        conversation.history(),
                        temperature=params.get("temperature"),
                        max_tokens=params.get("max_tokens"),
                        tools=tools,
                    ):
                        content = chunk.get("content")
                        if content:
                            buffer.append(content)
                        elif "tool_calls" in chunk:
                            tool_calls_received = chunk["tool_calls"]
                        elif "usage" in chunk:
                            usage = chunk["usage"]
                if buffer:
                    console.print(Markdown("".join(buffer), code_theme=theme))

            # Handle tool calls.
            if tool_calls_received and mcp_manager:
                _add_assistant_with_tool_calls(
                    conversation, "".join(buffer), tool_calls_received
                )

                for tc in tool_calls_received:
                    fn = tc.get("function", {})
                    fn_name = fn.get("name", "")
                    try:
                        fn_args = json.loads(fn.get("arguments", "{}"))
                    except json.JSONDecodeError:
                        fn_args = {}

                    # Permission prompt.
                    if not auto_approve:
                        console.print()
                        console.print(
                            Panel(
                                f"[bold]Tool:[/bold] {fn_name}\n"
                                f"[bold]Args:[/bold]\n"
                                f"```json\n{json.dumps(fn_args, indent=2)}\n```",
                                border_style="yellow",
                                title="🔧 Tool call",
                            )
                        )
                        if not _ask_user_yn("Execute this tool?", default=True):
                            console.print(Text("  ✖ skipped", style="dim yellow"))
                            _add_tool_result(
                                conversation,
                                tc.get("id", ""),
                                fn_name,
                                "(skipped by user)",
                            )
                            continue

                    console.print(
                        Text(f"  🔧 Calling {fn_name}...", style="dim cyan")
                    )
                    try:
                        result = await mcp_manager.call_tool(fn_name, fn_args)
                    except MCPError as exc:
                        result = f"Error: {exc}"
                        console.print(
                            Text(f"  ✖ {fn_name}: {exc}", style="red")
                        )

                    _add_tool_result(
                        conversation, tc.get("id", ""), fn_name, result
                    )

                continue  # loop again with tool results
            else:
                break

    except PhoenixError as exc:
        if buffer:
            console.print(Text("… (reply truncated)", style="dim italic"))
        print_error(exc)
        return False, "".join(buffer)

    full = "".join(buffer)
    if full:
        conversation.add("assistant", full)
        ok = True
    if usage and console.is_terminal:
        tokens = usage.get("total_tokens")
        if tokens:
            console.print(Text(f"⚡ {tokens} tokens used", style="dim italic"))
    return ok, full


# ---------------------------------------------------------------------------
# Interactive chat
# ---------------------------------------------------------------------------


HELP_TEXT = """\
[bold #ff6b00]Phoenix CLI — all commands[/bold #ff6b00]

[bold]Navigation[/bold]
  /help              show this help
  /exit, /quit, /q   leave the chat
  /clear             forget conversation history

[bold]Models & system[/bold]
  /model             open the interactive model picker
  /model NAME        switch to model NAME directly
  /system            show current system prompt
  /system TEXT       set/replace the system prompt (empty clears it)
  /models            list provider models inline

[bold]Generation[/bold]
  /temp              show current temperature
  /temp N            set temperature to N
  /max-tokens        show current max-tokens cap
  /max-tokens N      cap the reply length to N tokens

[bold]Conversation[/bold]
  /history           show message count and trim limit
  /save, /save FILE  save the conversation as markdown
  /undo              remove the last user+assistant exchange
  /retry             resend the last user message
  /search WORD       search conversation history
  /compact           summarize the conversation to save context
  /export, /export FILE.json  export conversation as JSON
  /import FILE.json  import a previously exported conversation
  /pin TEXT          pin a note into the system prompt
  /pinned            list pinned notes
  /unpin N           remove pinned note N

[bold]MCP & tools[/bold]
  /tools             list available MCP tools
  /mcp               show MCP server status
  /auto              show tool auto-approve state
  /auto on           auto-approve tool calls (default)
  /auto off          ask before each tool call

[bold]Session[/bold]
  /status            show current session config
  /ping              measure latency to the provider
  /config            show and edit BASE_URL / API_KEY / MODEL_NAME
  /theme NAME        switch code theme (monokai, github_dark, dracula, vs_dark)
  /verbose           toggle verbose mode (show raw tokens)
  /context           show token count and context window usage
  /copy              copy last AI reply to clipboard
  /reset             full reset — clear history, restore defaults

[bold]Keys[/bold]
  Ctrl+C   cancel the reply currently streaming
  Ctrl+D   quit (same as /exit)
"""


def _handle_slash_command(
    command: str,
    conversation: Conversation,
    state: Dict[str, Any],
    cfg: Dict[str, str],
    mcp_manager: Optional[MCPManager] = None,
) -> Optional[str]:
    """Interpret a ``/command`` typed in chat.

    Returns ``"exit"`` when the user wants to leave, else ``None``.
    """
    parts = command.split(None, 1)
    verb = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""

    # --- Navigation ---------------------------------------------------------
    if verb in ("/exit", "/quit", "/q"):
        return "exit"

    if verb == "/help":
        console.print(Panel(Markdown(HELP_TEXT), border_style=ACCENT, title="Phoenix CLI"))

    elif verb == "/clear":
        conversation.clear()
        # Clear tool-call metadata too.
        for attr in ("_tool_msgs", "_tool_call_ids"):
            if hasattr(conversation, attr):
                getattr(conversation, attr).clear()
        console.print("[dim]✔ conversation history cleared[/dim]")

    # --- Models & system ----------------------------------------------------
    elif verb == "/model":
        if arg:
            # Direct switch.
            state["model"] = arg
            console.print(
                Text(f"✔ model switched to [bold]{arg}[/bold]", style="dim green")
            )
        else:
            # Interactive picker.
            console.print("[dim]Fetching available models...[/dim]")
            params = _params(cfg, model=state["model"])
            params["timeout"] = 15.0
            try:
                ids = asyncio.run(_list_models(params))
                if not ids:
                    console.print("[dim]Provider returned no models.[/dim]")
                else:
                    console.print()
                    table = Table(border_style="bright_black", show_header=True,
                                  box=None, header_style="bold")
                    table.add_column("#", style="dim", width=5)
                    table.add_column("Model", style="bold")
                    for i, mid in enumerate(ids, 1):
                        marker = "  ✓" if mid == state["model"] else ""
                        table.add_row(str(i), mid + marker)
                    console.print(table)
                    console.print()
                    try:
                        choice = click.prompt(
                            "Enter number (0 to skip)",
                            type=int, default=0, show_default=True,
                        )
                    except (KeyboardInterrupt, click.Abort):
                        console.print()
                        return None
                    if 1 <= choice <= len(ids):
                        state["model"] = ids[choice - 1]
                        console.print(
                            Text(f"✔ switched to: {state['model']}",
                                 style="bold green")
                        )
                    elif choice != 0:
                        console.print(
                            Text(f"Invalid number. Enter 1-{len(ids)}.", style="red")
                        )
            except PhoenixError as exc:
                print_error(exc)
            except Exception as exc:
                console.print(Text(f" could not fetch models: {exc}", style="red"))

    elif verb == "/system":
        if len(parts) > 1:
            # Set (or clear if the text is empty / "clear").
            if arg.lower() == "clear":
                conversation.system = None
                console.print(Text("✔ system prompt cleared", style="dim green"))
            else:
                conversation.system = arg
                console.print(
                    Text(f"✔ system prompt set: {arg}", style="dim green")
                )
        else:
            cur = conversation.system or "(none)"
            console.print(f"[dim]current system prompt:[/dim] {cur}")

    elif verb == "/models":
        console.print("[dim]Fetching models...[/dim]")
        params = _params(cfg, model=state["model"])
        params["timeout"] = 15.0
        try:
            ids = asyncio.run(_list_models(params))
            if not ids:
                console.print("[dim]Provider returned no models.[/dim]")
                return None
            table = Table(border_style="bright_black", show_header=True,
                          box=None, header_style="bold")
            table.add_column("#", style="dim", width=5)
            table.add_column("Model", style="bold")
            for i, mid in enumerate(ids, 1):
                marker = "  ✓" if mid == state["model"] else ""
                table.add_row(str(i), mid + marker)
            console.print(table)
            console.print()
            console.print("[dim]Tip: /model NUMBER to switch[/dim]")
        except PhoenixError as exc:
            print_error(exc)

    # --- Generation ---------------------------------------------------------
    elif verb in ("/temp", "/temperature"):
        if not arg:
            console.print(
                f"[dim]temperature = "
                f"{state['temperature'] if state['temperature'] is not None else '(default)'}[/dim]"
            )
        else:
            try:
                state["temperature"] = float(arg)
                console.print(
                    Text(f"✔ temperature = {state['temperature']}", style="dim green")
                )
            except ValueError:
                console.print("[red]usage: /temp 0.8[/red]")

    elif verb in ("/max-tokens", "/maxtokens"):
        if not arg:
            val = state.get("max_tokens")
            console.print(
                f"[dim]max-tokens = {val if val is not None else '(no cap)'}[/dim]"
            )
        else:
            try:
                state["max_tokens"] = int(arg)
                console.print(
                    Text(f"✔ max-tokens = {state['max_tokens']}", style="dim green")
                )
            except ValueError:
                console.print("[red]usage: /max-tokens 1024[/red]")

    # --- Conversation -------------------------------------------------------
    elif verb == "/history":
        count = len(conversation.messages)
        pinned = len(getattr(conversation, '_pinned_notes', []))
        console.print(
            f"[dim]{count} messages in memory "
            f"(limit {conversation.max_messages}, oldest are dropped)"
        )
        if pinned:
            console.print(f"[dim]{pinned} pinned note(s) in system prompt[/dim]")

    elif verb == "/save":
        target = Path(arg).expanduser() if arg else Path("phoenix_chat.md")
        _save_conversation(conversation, target, state["model"])
        console.print(Text(f"✔ conversation saved to {target}", style="dim green"))

    elif verb == "/undo":
        if not conversation.messages:
            console.print("[dim]nothing to undo[/dim]")
        else:
            # Remove the last assistant message (and the user message before it).
            removed = 0
            while conversation.messages and conversation.messages[-1].role in (
                "assistant", "tool"
            ):
                idx = len(conversation.messages) - 1
                for attr in ("_tool_msgs", "_tool_call_ids"):
                    if hasattr(conversation, attr):
                        getattr(conversation, attr).pop(idx, None)
                conversation.messages.pop()
                removed += 1
            if conversation.messages and conversation.messages[-1].role == "user":
                conversation.messages.pop()
                removed += 1
            console.print(
                Text(f"✔ removed {removed} message(s)", style="dim green")
            )

    elif verb == "/retry":
        if not conversation.messages:
            console.print("[dim]nothing to retry[/dim]")
        else:
            # Remove the last assistant message(s).
            while conversation.messages and conversation.messages[-1].role in (
                "assistant", "tool"
            ):
                idx = len(conversation.messages) - 1
                for attr in ("_tool_msgs", "_tool_call_ids"):
                    if hasattr(conversation, attr):
                        getattr(conversation, attr).pop(idx, None)
                conversation.messages.pop()
            if conversation.messages and conversation.messages[-1].role == "user":
                console.print(
                    Text(
                        f"✔ resending: {conversation.messages[-1].content[:80]}",
                        style="dim green",
                    )
                )
                # Will be re-sent by the caller — signal via state flag.
                state["_retry"] = True
            else:
                console.print("[dim]no user message to retry[/dim]")

    elif verb == "/search":
        if not arg:
            console.print("[red]usage: /search WORD[/red]")
        else:
            needle = arg.lower()
            hits: List[tuple] = []
            for i, msg in enumerate(conversation.messages):
                if needle in msg.content.lower():
                    hits.append((i, msg.role, msg.content))
            if not hits:
                console.print(f"[dim]no matches for '{arg}'[/dim]")
            else:
                console.print(
                    Text(f"{len(hits)} match(es) for '{arg}':", style="bold")
                )
                console.print()
                for idx, role, text in hits:
                    snippet = text.replace("\n", " ")[:80]
                    console.print(
                        f"[dim]#{idx:3d}  {role:12s}  {snippet}[/dim]"
                    )

    elif verb == "/compact":
        console.print(
            "[dim]Asking the model to summarize the conversation...[/dim]"
        )
        # Build a compact prompt.
        original_system = conversation.system
        conversation.system = (
            "Summarize the entire conversation above in a single paragraph "
            "preserving all important facts, decisions, and context. "
            "Only output the summary — nothing else."
        )
        params = _params(cfg, model=state["model"])
        params["temperature"] = 0.0
        use_live = console.is_terminal
        try:
            ok, summary = asyncio.run(
                _stream_turn(
                    params, conversation, use_live=use_live,
                    mcp_manager=None,  # no tools during compact
                    auto_approve=True,
                    theme=state.get("theme", "monokai"),
                    verbose=state.get("verbose", False),
                )
            )
        except KeyboardInterrupt:
            console.print(Text(" interrupted", style="dim"))
            ok, summary = False, ""

        conversation.system = original_system
        if ok and summary:
            # Replace history with a single user message containing the summary.
            conversation.clear()
            for attr in ("_tool_msgs", "_tool_call_ids"):
                if hasattr(conversation, attr):
                    getattr(conversation, attr).clear()
            conversation.add("user", f"[compact summary of prior chat]\n{summary}")
            console.print(
                Text(f"✔ conversation compacted ({len(summary)} chars)",
                     style="bold green")
            )
        else:
            console.print("[red]compact failed — history unchanged[/red]")

    elif verb == "/export":
        target = Path(arg).expanduser() if arg else Path("phoenix_chat.json")
        data = {
            "model": state["model"],
            "saved": datetime.now().isoformat(timespec="seconds"),
            "system": conversation.system,
            "messages": [m.to_dict() for m in conversation.messages],
        }
        target.write_text(
            json.dumps(data, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        console.print(
            Text(f"✔ exported {len(conversation.messages)} message(s) to {target}",
                 style="dim green")
        )

    elif verb == "/import":
        if not arg:
            console.print("[red]usage: /import FILE.json[/red]")
        else:
            target = Path(arg).expanduser()
            if not target.is_file():
                console.print(Text(f"✖ {target} not found", style="red"))
            else:
                try:
                    data = json.loads(target.read_text(encoding="utf-8"))
                except json.JSONDecodeError as exc:
                    console.print(Text(f"✖ invalid JSON: {exc}", style="red"))
                else:
                    conversation.clear()
                    for attr in ("_tool_msgs", "_tool_call_ids"):
                        if hasattr(conversation, attr):
                            getattr(conversation, attr).clear()
                    if data.get("system"):
                        conversation.system = data["system"]
                    for msg in data.get("messages", []):
                        role = msg.get("role", "user")
                        content = msg.get("content", "")
                        if role in ("user", "assistant"):
                            conversation.add(role, content)
                    n = len(conversation.messages)
                    console.print(
                        Text(f"✔ imported {n} message(s) from {target}",
                             style="bold green")
                    )

    # --- MCP & tools --------------------------------------------------------
    elif verb == "/tools":
        if mcp_manager and mcp_manager.get_tool_names():
            console.print("[bold]Available MCP tools:[/bold]")
            for name in mcp_manager.get_tool_names():
                console.print(f"  🔧 {name}")
        else:
            console.print("[dim]No MCP tools connected.[/dim]")
            console.print("[dim]Use `phoenix mcp add` to add an MCP server.[/dim]")

    elif verb == "/mcp":
        if mcp_manager:
            servers = mcp_manager.connected_servers
            tools = mcp_manager.get_tool_names()
            console.print(
                f"[bold]MCP status:[/bold] {len(servers)} server(s), "
                f"{len(tools)} tool(s)"
            )
            for s in servers:
                console.print(f"  ● {s}")
            auto = state.get("auto_approve", True)
            console.print(
                f"[dim]auto-approve: {'ON' if auto else 'OFF'}[/dim]"
            )
        else:
            console.print("[dim]MCP not connected.[/dim]")
            console.print(
                "[dim]Enable in `phoenix setup` or add a server with "
                "`phoenix mcp add`.[/dim]"
            )

    elif verb == "/auto":
        if not arg:
            cur = state.get("auto_approve", True)
            console.print(
                f"[dim]tool auto-approve = {'ON' if cur else 'OFF'}[/dim]"
            )
        elif arg.lower() in ("on", "1", "true", "yes"):
            state["auto_approve"] = True
            console.print(
                Text("✔ auto-approve ON — tools run without prompts",
                     style="bold green")
            )
        elif arg.lower() in ("off", "0", "false", "no"):
            state["auto_approve"] = False
            console.print(
                Text("✔ auto-approve OFF — will ask before each tool call",
                     style="bold yellow")
            )
        else:
            console.print("[red]usage: /auto on | /auto off[/red]")

    # --- Session ------------------------------------------------------------
    elif verb == "/status":
        console.print("[bold]Session status[/bold]")
        console.print()
        table = Table(border_style="bright_black", show_header=False, box=None)
        table.add_column(style="bold")
        table.add_column()
        table.add_row("model", state["model"])
        table.add_row("url", cfg["base_url"])
        table.add_row("api_key", mask_secret(cfg.get("api_key", "")))
        table.add_row("temperature",
                       str(state.get("temperature") or "(default)"))
        table.add_row("max-tokens",
                       str(state.get("max_tokens") or "(no cap)"))
        table.add_row("auto-approve",
                       "ON" if state.get("auto_approve", True) else "OFF")
        table.add_row("theme", state.get("theme", "monokai"))
        table.add_row("verbose",
                       "ON" if state.get("verbose", False) else "OFF")
        table.add_row("mcp",
                       "ON" if cfg.get("mcp_enabled") else "off")
        table.add_row("messages", str(len(conversation.messages)))
        console.print(table)

    elif verb in ("/ping", "/probe"):
        console.print("[dim]sending ping...[/dim]")
        params = _params(cfg, model=state["model"])
        params["timeout"] = 15.0
        t0 = time.perf_counter()
        try:
            ok = asyncio.run(_probe(params))
        except PhoenixError as exc:
            print_error(exc)
            ok = False
        if ok:
            latency = (time.perf_counter() - t0) * 1000
            console.print(
                Text(f"✓ provider reachable — {latency:.0f} ms",
                     style="bold green")
            )

    elif verb == "/config":
        console.print("[bold]Current config[/bold]")
        console.print()
        table = Table(border_style="bright_black", show_header=False, box=None)
        table.add_column(style="bold")
        table.add_column()
        table.add_row("BASE_URL", cfg["base_url"])
        table.add_row("API_KEY", mask_secret(cfg.get("api_key", "")))
        table.add_row("MODEL_NAME", cfg["model_name"])
        table.add_row("MCP", "enabled" if cfg.get("mcp_enabled") else "disabled")
        console.print(table)
        console.print()
        console.print("[dim]To edit, run `phoenix setup` from a new session.[/dim]")

    elif verb == "/theme":
        if not arg:
            console.print(
                "[dim]available themes: " + ", ".join(CODE_THEMES)
            )
            console.print(f"[dim]current: {state.get('theme', 'monokai')}[/dim]")
        elif arg in CODE_THEMES:
            state["theme"] = arg
            console.print(
                Text(f"✔ theme switched to {arg}", style="bold green")
            )
        else:
            console.print(
                Text(f"unknown theme {arg!r} — choose from: "
                     + ", ".join(CODE_THEMES), style="red")
            )

    elif verb == "/verbose":
        if not arg:
            cur = state.get("verbose", False)
            console.print(
                f"[dim]verbose = {'ON' if cur else 'OFF'}[/dim]"
            )
        elif arg.lower() in ("on", "1", "true", "yes"):
            state["verbose"] = True
            console.print(Text("✔ verbose ON", style="bold green"))
        elif arg.lower() in ("off", "0", "false", "no"):
            state["verbose"] = False
            console.print(Text("✔ verbose OFF", style="bold green"))
        else:
            console.print("[red]usage: /verbose on | /verbose off[/red]")

    elif verb == "/context":
        # Rough token count estimate (chars / 4).
        total_chars = sum(len(m.content) for m in conversation.messages)
        estimated = total_chars // 4
        console.print(f"[dim]~{estimated} tokens in context "
                      f"({total_chars} chars, {len(conversation.messages)} msgs)[/dim]")

    elif verb == "/copy":
        # Find the last assistant message.
        last_reply = ""
        for msg in reversed(conversation.messages):
            if msg.role == "assistant":
                last_reply = msg.content
                break
        if not last_reply:
            console.print("[dim]no assistant reply to copy[/dim]")
        else:
            if _copy_to_clipboard(last_reply):
                console.print(
                    Text(f"✔ copied {len(last_reply)} chars to clipboard",
                         style="bold green")
                )
            else:
                # Fallback: print plain for manual copy.
                console.print()
                console.print("[dim]— copy below —[/dim]")
                console.print(last_reply)
                console.print("[dim]— end —[/dim]")

    elif verb == "/reset":
        conversation.clear()
        for attr in ("_tool_msgs", "_tool_call_ids", "_pinned_notes"):
            if hasattr(conversation, attr):
                getattr(conversation, attr).clear()
        # Reload defaults.
        fresh = load_config()
        state["model"] = fresh["model_name"]
        state["temperature"] = None
        state["max_tokens"] = None
        state["auto_approve"] = True
        state["theme"] = "monokai"
        state["verbose"] = False
        console.print(Text("✔ full reset — defaults restored", style="bold green"))

    elif verb == "/pin":
        if not arg:
            console.print("[red]usage: /pin TEXT[/red]")
        else:
            if not hasattr(conversation, '_pinned_notes'):
                conversation._pinned_notes: List[str] = []
            conversation._pinned_notes.append(arg)
            n = len(conversation._pinned_notes)
            console.print(
                Text(f"✔ pinned note #{n}: {arg}", style="bold green")
            )
            # Update system prompt to include pinned notes.
            _update_system_with_pins(conversation)

    elif verb == "/pinned":
        notes = getattr(conversation, '_pinned_notes', [])
        if not notes:
            console.print("[dim]no pinned notes[/dim]")
        else:
            console.print("[bold]Pinned notes:[/bold]")
            for i, note in enumerate(notes, 1):
                console.print(f"  {i}. {note}")

    elif verb == "/unpin":
        notes = getattr(conversation, '_pinned_notes', [])
        if not arg:
            console.print("[red]usage: /unpin N[/red]")
        else:
            try:
                idx = int(arg) - 1
            except ValueError:
                console.print("[red]usage: /unpin N (number)[/red]")
            else:
                if 0 <= idx < len(notes):
                    removed = notes.pop(idx)
                    console.print(
                        Text(f"✔ removed pinned note #{idx+1}: {removed}",
                             style="bold green")
                    )
                    _update_system_with_pins(conversation)
                else:
                    console.print(
                        Text(f"invalid number — have {len(notes)} note(s)",
                             style="red")
                    )

    # --- Unknown ------------------------------------------------------------
    else:
        console.print(
            Text(f"unknown command {verb!r} — try /help", style="red")
        )

    return None


def _update_system_with_pins(conversation: Conversation) -> None:
    """Merge pinned notes into the conversation's system prompt."""
    notes = getattr(conversation, '_pinned_notes', [])
    if notes:
        pinned_block = "\n\n[Pinned notes — always follow these]\n" + "\n".join(
            f"• {n}" for n in notes
        )
        # If the current system prompt was previously set without pins,
        # re-append.  We don't try to detect this — just always set it.
        base = conversation.system or ""
        if "[Pinned notes" in base:
            base = base.split("[Pinned notes")[0].rstrip()
        conversation.system = base + pinned_block
    else:
        # Remove any pinned block.
        base = conversation.system or ""
        if "[Pinned notes" in base:
            conversation.system = base.split("[Pinned notes")[0].rstrip() or None


def _save_conversation(conversation: Conversation, path: Path, model: str) -> None:
    """Write the conversation history to a markdown file."""
    lines = [
        "# Phoenix CLI conversation",
        f"- model: {model}",
        f"- saved: {datetime.now().isoformat(timespec='seconds')}",
        "",
    ]
    for msg in conversation.messages:
        lines.append(f"## {msg.role.title()}")
        lines.append("")
        lines.append(msg.content)
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _read_user_input(session: Any, interactive: bool) -> Optional[str]:
    """Read one line from the user (prompt_toolkit when available)."""
    if interactive and session is not None:
        from prompt_toolkit.styles import Style

        style = Style.from_dict({"prompt": f"{ACCENT} bold"})
        return session.prompt(
            [("class:prompt", "phoenix"), ("", " ❯ ")], style=style
        )
    return input(PROMPT_LABEL)


def _make_session(interactive: bool) -> Optional[Any]:
    """Create a prompt_toolkit session, or ``None`` if unavailable."""
    if not interactive:
        return None
    try:
        from prompt_toolkit import PromptSession

        return PromptSession()
    except Exception:
        return None


def run_chat(cfg: Dict[str, str], system: Optional[str], model: Optional[str],
             temperature: Optional[float], max_tokens: Optional[int]) -> int:
    """The interactive chat loop. Returns the process exit code."""
    interactive = sys.stdin.isatty() and console.is_terminal
    if not interactive:
        print_error(
            ConfigurationError(
                "`phoenix chat` needs an interactive terminal.\n"
                "For one-shot use (scripts, pipes), run: phoenix \"your prompt\""
            )
        )
        return 1

    conversation = Conversation(system=system)
    state: Dict[str, Any] = {
        "model": model or cfg["model_name"],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "auto_approve": True,
        "theme": "monokai",
        "verbose": False,
        "_pinned_notes": [],
    }
    conversation._pinned_notes = []
    conversation._tool_msgs = {}
    conversation._tool_call_ids = {}

    print_banner()
    console.print(f"[bold]Model:[/bold] [bold #ff6b00]{state['model']}[/bold #ff6b00]")
    console.print(f"[bold]API:[/bold]   {cfg['base_url']}")

    mcp_manager = None
    try:
        mcp_manager = asyncio.run(_connect_mcp(cfg))
    except Exception:
        pass

    if mcp_manager:
        tool_count = len(mcp_manager.get_tool_names())
        console.print(
            f"[bold]MCP:[/bold]   {tool_count} tool(s) from "
            f"{len(mcp_manager.connected_servers)} server(s)"
        )
        auto = state.get("auto_approve", True)
        console.print(
            f"[bold]Auto-approve:[/bold] {'ON' if auto else 'OFF'}"
        )

    console.print("[dim]Type /help for 30+ commands • Ctrl+C cancels a reply[/dim]")
    console.print()

    session = _make_session(interactive)
    while True:
        try:
            user_input = _read_user_input(session, interactive)
        except (KeyboardInterrupt, EOFError):
            console.print()
            console.print(
                "[dim]Ctrl+D or /exit to quit; Ctrl+C once more to force.[/dim]"
            )
            continue
        if user_input is None:
            console.print("\n[dim]Bye![/dim]")
            break

        text = user_input.strip()
        if not text:
            continue
        if text.startswith("/"):
            if _handle_slash_command(
                text, conversation, state, cfg, mcp_manager,
            ) == "exit":
                console.print("[dim]Bye![/dim]")
                break
            if state.pop("_retry", False):
                # /retry: re-send last user message.
                pass
            else:
                continue

        conversation.add("user", text)
        params = _params(
            cfg,
            model=state["model"],
            temperature=state["temperature"],
            max_tokens=state["max_tokens"],
        )
        try:
            asyncio.run(
                _stream_turn(
                    params,
                    conversation,
                    use_live=True,
                    mcp_manager=mcp_manager,
                    auto_approve=state.get("auto_approve", True),
                    theme=state.get("theme", "monokai"),
                    verbose=state.get("verbose", False),
                )
            )
        except KeyboardInterrupt:
            console.print(Text(" interrupted — reply cancelled", style="dim"))
            if conversation.messages and conversation.messages[-1].role == "user":
                conversation.messages.pop()
        console.print()

    if mcp_manager:
        asyncio.run(mcp_manager.close())
    return 0


# ---------------------------------------------------------------------------
# Single-prompt mode
# ---------------------------------------------------------------------------


def run_single_prompt(
    cfg: Dict[str, str],
    prompt: str,
    *,
    system: Optional[str],
    model: Optional[str],
    temperature: Optional[float],
    max_tokens: Optional[int],
    no_stream: bool,
) -> int:
    """Run a one-shot prompt. Returns the process exit code."""
    conversation = Conversation(system=system)
    conversation.add("user", prompt)
    params = _params(cfg, model=model, temperature=temperature, max_tokens=max_tokens)
    use_live = console.is_terminal and not no_stream

    mcp_manager = None
    try:
        mcp_manager = asyncio.run(_connect_mcp(cfg))
    except Exception:
        pass

    try:
        ok, _ = asyncio.run(
            _stream_turn(
                params, conversation, use_live=use_live,
                mcp_manager=mcp_manager, auto_approve=True,
            )
        )
    except KeyboardInterrupt:
        console.print(Text("✖ interrupted — reply cancelled", style="dim"))
        return 130
    finally:
        if mcp_manager:
            asyncio.run(mcp_manager.close())
    return 0 if ok else 1


async def _probe(params: Dict[str, Any]) -> bool:
    """Send a tiny request to verify connectivity. Returns True on success."""
    conversation = Conversation()
    conversation.add("user", "ping")
    t0 = time.perf_counter()
    try:
        async with make_client(params) as client:
            async for _chunk in client.chat_stream(
                conversation.history(), max_tokens=2
            ):
                pass
    except PhoenixError as exc:
        print_error(exc)
        return False
    latency = (time.perf_counter() - t0) * 1000
    console.print(
        Text(f"✓ provider reachable — {latency:.0f} ms", style="bold green")
    )
    return True


async def _list_models(params: Dict[str, Any]) -> List[str]:
    """Fetch the provider's model list."""
    async with make_client(params) as client:
        return await client.list_models()


# ---------------------------------------------------------------------------
# Interactive model selector (used by setup / --select)
# ---------------------------------------------------------------------------


def _select_model_interactive(model_ids: List[str], current: str = "") -> Optional[str]:
    """Show a numbered list and let the user pick a model."""
    console.print()
    console.print("[bold]Select a model (enter its number):[/bold]")
    console.print()

    table = Table(border_style="bright_black", show_header=True, box=None,
                  header_style="bold")
    table.add_column("#", style="dim", width=5)
    table.add_column("Model", style="bold")

    for i, model_id in enumerate(model_ids, 1):
        marker = "  ✓" if model_id == current else ""
        table.add_row(str(i), model_id + marker)

    console.print(table)
    console.print()

    while True:
        try:
            choice = click.prompt(
                "Enter number (0 to skip)",
                type=int, default=0, show_default=True,
            )
        except (KeyboardInterrupt, click.Abort):
            return None

        if choice == 0:
            return None
        if 1 <= choice <= len(model_ids):
            selected = model_ids[choice - 1]
            console.print(
                Text(f"✓ selected: {selected}", style="bold green")
            )
            return selected
        console.print(
            Text(f"Invalid number. Enter 1-{len(model_ids)} or 0 to skip.",
                 style="red")
        )


# ---------------------------------------------------------------------------
# Click command group
# ---------------------------------------------------------------------------


class PhoenixGroup(click.Group):
    """Group that lets the first argument be a prompt instead of a command."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("invoke_without_command", True)
        kwargs.setdefault("no_args_is_help", False)
        super().__init__(*args, **kwargs)

    def resolve_command(
        self, ctx: click.Context, args: List[str]
    ) -> tuple[Optional[str], Optional[click.Command], List[str]]:
        try:
            return super().resolve_command(ctx, args)
        except click.UsageError:
            ask_cmd = self.commands.get("ask")
            if ask_cmd is None:  # pragma: no cover - defensive
                raise
            return "ask", ask_cmd, list(args)


@click.group(cls=PhoenixGroup)
@click.version_option(version=__version__, prog_name="phoenix")
@click.option("--system", "-s", default=None, help="System prompt for this request/chat.")
@click.option("--model", "-m", default=None, help="Model override for this request/chat.")
@click.option("--temperature", "-t", type=float, default=None, help="Sampling temperature.")
@click.option("--max-tokens", type=click.IntRange(min=1), default=None, help="Cap the reply length.")
@click.option("--no-stream", is_flag=True, help="Print the reply only when complete (scripts/pipes).")
@click.pass_context
def cli(
    ctx: click.Context,
    system: Optional[str],
    model: Optional[str],
    temperature: Optional[float],
    max_tokens: Optional[int],
    no_stream: bool,
) -> None:
    """Phoenix CLI — AI that rises with you.

    Works with any OpenAI-compatible API and supports MCP for tool use.
    Type `phoenix --help` for the full command list.
    """
    if ctx.invoked_subcommand is None:
        print_banner()
        console.print(
            Panel.fit(
                "  [bold]Quick start[/bold]\n"
                "  [bold #ff6b00]phoenix setup[/bold #ff6b00]                    "
                "configure provider + model\n"
                "  [bold #ff6b00]phoenix \"explain quicksort\"[/bold #ff6b00]    "
                "one-shot prompt\n"
                "  [bold #ff6b00]phoenix chat[/bold #ff6b00]                    "
                "interactive chat with 30+ commands\n"
                "  [bold #ff6b00]phoenix models --select[/bold #ff6b00]         "
                "pick a model from a numbered list\n"
                "  [bold #ff6b00]phoenix mcp add[/bold #ff6b00]                 "
                "add an MCP server (e.g. Roblox)\n"
                "  [bold #ff6b00]phoenix status[/bold #ff6b00]                  "
                "show current configuration\n"
                "\n"
                "  [dim]Type /help inside chat to see every command.[/dim]",
                border_style=ACCENT,
                title="Phoenix CLI",
            )
        )


@cli.command()
def setup() -> None:
    """Configure BASE_URL, API_KEY and MODEL_NAME (~/.phoenix_config.json).

    After entering the URL and API key, Phoenix auto-fetches models and
    lets you pick one interactively.
    """
    print_banner()
    current = load_config()

    console.print(
        "[bold]Phoenix setup[/bold] — the provider can be any OpenAI-compatible "
        "API. Press Enter to keep the current value."
    )
    console.print()

    def url_hint() -> None:
        console.print(
            "[dim]examples:\n"
            "  Ollama (local)     http://localhost:11434\n"
            "  LM Studio (local)  http://localhost:1234\n"
            "  vLLM (local)       http://localhost:8000\n"
            "  OpenRouter         https://openrouter.ai/api\n"
            "  Together AI        https://api.together.xyz\n"
            "  Groq               https://api.groq.com/openai\n"
            "  Any custom server    myserver.example.com:8080[/dim]"
        )

    url_hint()
    base_url = ""
    for _attempt in range(3):
        raw = click.prompt(
            "BASE_URL", default=current["base_url"] or "", show_default=False
        )
        base_url = normalize_base_url(raw)
        if not base_url or " " in base_url:
            console.print("[red]That does not look like a valid URL.[/red]")
            url_hint()
            continue
        break
    if not base_url:
        console.print("[red]Giving up on BASE_URL — run `phoenix setup` again.[/red]")
        raise SystemExit(1)

    api_key = click.prompt(
        "API_KEY (Enter for none — local servers usually need none)",
        default=current["api_key"] or "",
        hide_input=True,
        show_default=False,
    ).strip()

    console.print()
    console.print("[dim]Connecting to the provider to fetch models...[/dim]")
    params = {
        "base_url": base_url,
        "api_key": api_key,
        "model_name": "temp",
        "timeout": 15.0,
    }
    model_name = ""
    try:
        ids = asyncio.run(_list_models(params))
        if ids:
            console.print(
                Text(f"✓ found {len(ids)} model(s)", style="bold green")
            )
            selected = _select_model_interactive(
                ids, current=current["model_name"]
            )
            if selected:
                model_name = selected
    except PhoenixError as exc:
        console.print(
            Text(f"⚠ could not fetch models: {type(exc).__name__}", style="yellow")
        )
        console.print("[dim]You can enter the model name manually.[/dim]")
    except Exception:
        console.print(
            Text("⚠ could not fetch models (connection error)", style="yellow")
        )
        console.print("[dim]You can enter the model name manually.[/dim]")

    if not model_name:
        console.print()
        for _attempt in range(3):
            model_name = click.prompt(
                "MODEL_NAME", default=current["model_name"] or "", show_default=False
            ).strip()
            if model_name:
                break
            console.print("[red]MODEL_NAME cannot be empty.[/red]")
        if not model_name:
            console.print("[red]Giving up on MODEL_NAME — run `phoenix setup` again.[/red]")
            raise SystemExit(1)

    console.print()
    mcp_enabled = click.confirm(
        "Enable MCP (Model Context Protocol) tools? (for Roblox MCP etc.)",
        default=bool(current.get("mcp_enabled")),
    )

    path = save_config(
        base_url=base_url, api_key=api_key, model_name=model_name,
        mcp_enabled=mcp_enabled,
    )

    console.print()
    console.print(Panel(
        f"  [bold]BASE_URL[/bold]     {base_url}\n"
        f"  [bold]API_KEY[/bold]      {mask_secret(api_key)}\n"
        f"  [bold]MODEL_NAME[/bold]   {model_name}\n"
        f"  [bold]MCP[/bold]          {'enabled' if mcp_enabled else 'disabled'}",
        border_style="green",
        title="Saved",
    ))
    console.print(f"[dim]Config written to {path}[/dim]")

    if mcp_enabled:
        servers = load_mcp_config()
        if servers:
            console.print(
                f"[dim]{len(servers)} MCP server(s) configured — "
                f"use `phoenix mcp list` to see them.[/dim]"
            )
        else:
            console.print(
                "[dim]No MCP servers configured yet. Run "
                "[bold #ff6b00]phoenix mcp add[/bold #ff6b00] to add one.[/dim]"
            )

    console.print()
    console.print(
        "[bold]Next:[/bold] try [bold #ff6b00]phoenix \"hello!\"[/bold #ff6b00] "
        "or start a conversation with [bold #ff6b00]phoenix chat[/bold #ff6b00]"
    )


@cli.command()
@click.option("--probe", is_flag=True, help="Send a test request and measure latency.")
def status(probe: bool) -> None:
    """Show the current configuration and where it comes from."""
    cfg = load_config()
    path = config_path()

    console.print("[bold]Phoenix status[/bold]")
    console.print()
    table = Table(border_style="bright_black", show_header=False, box=None)
    table.add_column(style="bold")
    table.add_column()

    def source(env_name: str) -> str:
        return (
            f"environment ({env_name})"
            if os.environ.get(env_name)
            else f"file ({path})"
        )

    table.add_row("BASE_URL", f"{cfg['base_url'] or '(not set)'}  [dim]← {source(ENV_BASE_URL)}[/dim]")
    table.add_row("API_KEY", f"{mask_secret(cfg['api_key'])}  [dim]← {source(ENV_API_KEY)}[/dim]")
    table.add_row("MODEL_NAME", f"{cfg['model_name'] or '(not set)'}  [dim]← {source(ENV_MODEL)}[/dim]")
    table.add_row("MCP", f"{'enabled' if cfg.get('mcp_enabled') else 'disabled'}")
    mcp_servers = load_mcp_config()
    if mcp_servers:
        table.add_row("MCP servers", f"{len(mcp_servers)} configured")
    table.add_row("Config file", str(path))
    console.print(table)
    console.print()

    problem = check_configured(cfg)
    if problem:
        console.print(Text(f"⚠ {problem.splitlines()[0]}", style="yellow"))
        raise SystemExit(1)

    console.print(Text("✓ configuration complete", style="green"))
    if probe:
        console.print()
        params = _params(cfg)
        params["timeout"] = 15.0
        if not asyncio.run(_probe(params)):
            raise SystemExit(1)


@cli.command()
@click.option("--raw", is_flag=True, help="Print one model ID per line (for scripts).")
@click.option("--select", "-s", "do_select", is_flag=True,
              help="Interactively select a model and save it to config.")
def models(raw: bool, do_select: bool) -> None:
    """List the models available from the configured provider.

    Use --select to pick a model interactively and save it as your default.
    """
    cfg = load_config()
    problem = check_configured(cfg)
    if problem:
        print_error(ConfigurationError(problem))
        raise SystemExit(1)

    params = _params(cfg)
    params["timeout"] = 15.0
    try:
        ids = asyncio.run(_list_models(params))
    except PhoenixError as exc:
        print_error(exc)
        raise SystemExit(1)

    if not ids:
        console.print("[dim]The provider returned an empty model list.[/dim]")
        raise SystemExit(1)

    if raw and not do_select:
        for model_id in ids:
            console.print(model_id)
        return

    current = cfg["model_name"]

    if do_select:
        selected = _select_model_interactive(ids, current=current)
        if selected:
            save_config(
                base_url=cfg["base_url"],
                api_key=cfg.get("api_key", ""),
                model_name=selected,
                mcp_enabled=bool(cfg.get("mcp_enabled")),
            )
            console.print(
                Text(f"\n✓ MODEL_NAME saved as '{selected}'", style="bold green")
            )
        return

    console.print(
        Text(
            f"{len(ids)} model(s) available from {cfg['base_url']}",
            style="bold",
        )
    )
    console.print()
    for model_id in ids:
        if model_id == current:
            console.print(Text(f"✓ {model_id}", style="bold #ff6b00"))
        else:
            console.print(Text(f"  {model_id}"))
    console.print()
    console.print(
        "[dim]Tip: run `phoenix models --select` to pick a model, or "
        "`phoenix -m NAME \"prompt\"` for a one-off.[/dim]"
    )


# ---------------------------------------------------------------------------
# MCP commands
# ---------------------------------------------------------------------------


@cli.group("mcp")
def mcp_group() -> None:
    """Manage MCP (Model Context Protocol) server connections.

    MCP servers expose tools that the AI can use — e.g. a Roblox MCP
    server lets the model create parts, edit scripts, etc.

    Servers are configured in ~/.phoenix_mcp.json.
    """


@mcp_group.command("list")
def mcp_list() -> None:
    """List configured MCP servers."""
    servers = load_mcp_config()
    if not servers:
        console.print("[dim]No MCP servers configured.[/dim]")
        console.print(
            "[dim]Add one with: [bold #ff6b00]phoenix mcp add[/bold #ff6b00][/dim]"
        )
        return

    cfg = load_config()
    mcp_enabled = bool(cfg.get("mcp_enabled"))

    console.print("[bold]Configured MCP servers:[/bold]")
    console.print()
    table = Table(border_style="bright_black", show_header=True, box=None,
                  header_style="bold")
    table.add_column("#", style="dim", width=4)
    table.add_column("Name", style="bold")
    table.add_column("Type")
    table.add_column("Details")

    for i, srv in enumerate(servers, 1):
        name = srv.get("name", f"server-{i}")
        if "command" in srv:
            srv_type = "stdio"
            details = " ".join(srv["command"])
        elif "url" in srv:
            srv_type = "sse"
            details = srv["url"]
        else:
            srv_type = "?"
            details = "(no command or url)"
        table.add_row(str(i), name, srv_type, details)

    console.print(table)
    console.print()
    if mcp_enabled:
        console.print(Text("✓ MCP is enabled in your config", style="green"))
    else:
        console.print(
            Text("⚠ MCP is disabled — run `phoenix setup` to enable it", style="yellow")
        )


@mcp_group.command("add")
def mcp_add() -> None:
    """Add an MCP server to ~/.phoenix_mcp.json.

    Supports two types:
    - stdio: a local command (e.g. npx, python, node)
    - sse: a remote server URL
    """
    servers = load_mcp_config()

    console.print("[bold]Add MCP server[/bold]")
    console.print()
    console.print("[dim]Types:\n"
                  "  stdio  — local command (e.g. npx, python, node)\n"
                  "  sse    — remote server URL[/dim]")
    console.print()

    transport = click.prompt(
        "Transport type",
        type=click.Choice(["stdio", "sse"]),
        default="stdio",
    )

    server: Dict[str, Any] = {}
    name = click.prompt("Server name (e.g. roblox)").strip()
    if not name:
        console.print("[red]Name cannot be empty.[/red]")
        raise SystemExit(1)
    server["name"] = name

    if transport == "stdio":
        console.print(
            "[dim]Examples:\n"
            "  npx -y @anthropic/mcp-server-roblox\n"
            "  python /path/to/server.py\n"
            "  node /path/to/server.js[/dim]"
        )
        cmd_str = click.prompt("Command (space-separated)").strip()
        if not cmd_str:
            console.print("[red]Command cannot be empty.[/red]")
            raise SystemExit(1)
        server["command"] = cmd_str.split()

        env_str = click.prompt(
            "Environment variables (KEY=VAL,KEY2=VAL2 or empty)",
            default="",
        ).strip()
        if env_str:
            env = {}
            for pair in env_str.split(","):
                pair = pair.strip()
                if "=" in pair:
                    k, v = pair.split("=", 1)
                    env[k.strip()] = v.strip()
            if env:
                server["env"] = env
    else:
        console.print(
            "[dim]Examples:\n"
            "  https://my-mcp-server.example.com\n"
            "  http://localhost:3000[/dim]"
        )
        url = click.prompt("Server URL").strip()
        if not url:
            console.print("[red]URL cannot be empty.[/red]")
            raise SystemExit(1)
        server["url"] = url.rstrip("/")

        mcp_key = click.prompt(
            "API key for MCP server (Enter for none)",
            default="",
            hide_input=True,
        ).strip()
        if mcp_key:
            server["headers"] = {"Authorization": f"Bearer {mcp_key}"}

    servers.append(server)
    path = save_mcp_config(servers)

    console.print()
    console.print(
        Text(f"✓ MCP server '{name}' added", style="bold green")
    )
    console.print(f"[dim]Saved to {path}[/dim]")

    cfg = load_config()
    if not cfg.get("mcp_enabled"):
        console.print()
        enable = click.confirm("Enable MCP in your config? (needed to use tools)")
        if enable:
            save_config(
                base_url=cfg["base_url"],
                api_key=cfg.get("api_key", ""),
                model_name=cfg["model_name"],
                mcp_enabled=True,
            )
            console.print(Text("✓ MCP enabled", style="bold green"))

    console.print()
    console.print(
        "[dim]Test with: [bold #ff6b00]phoenix mcp test[/bold #ff6b00][/dim]"
    )


@mcp_group.command("remove")
@click.argument("name", required=False)
def mcp_remove(name: Optional[str]) -> None:
    """Remove an MCP server by name."""
    servers = load_mcp_config()
    if not servers:
        console.print("[dim]No MCP servers configured.[/dim]")
        return

    if not name:
        console.print("[bold]Select a server to remove:[/bold]")
        for i, srv in enumerate(servers, 1):
            console.print(f"  {i}. {srv.get('name', f'server-{i}')}")
        console.print()
        try:
            idx = click.prompt("Enter number (0 to cancel)", type=int, default=0)
        except (KeyboardInterrupt, click.Abort):
            return
        if idx == 0:
            return
        if 1 <= idx <= len(servers):
            name = servers[idx - 1].get("name", "")
        else:
            console.print("[red]Invalid number.[/red]")
            return

    original_count = len(servers)
    servers = [s for s in servers if s.get("name") != name]
    if len(servers) == original_count:
        console.print(Text(f"Server '{name}' not found.", style="red"))
        return

    save_mcp_config(servers)
    console.print(Text(f"✓ MCP server '{name}' removed", style="bold green"))


@mcp_group.command("test")
@click.argument("name", required=False)
def mcp_test(name: Optional[str]) -> None:
    """Test connection to an MCP server.

    If no name is given, tests all configured servers.
    """
    servers = load_mcp_config()
    if not servers:
        console.print("[dim]No MCP servers configured.[/dim]")
        return

    if name:
        servers = [s for s in servers if s.get("name") == name]
        if not servers:
            console.print(Text(f"Server '{name}' not found.", style="red"))
            return

    async def _test_server(srv: Dict[str, Any]) -> None:
        srv_name = srv.get("name", "unnamed")
        console.print(f"Testing [bold]{srv_name}[/bold]...", end=" ")
        try:
            async with MCPClient.from_config(srv) as client:
                tools = await client.list_tools()
                console.print(
                    Text(f"✓ connected — {len(tools)} tool(s)", style="bold green")
                )
                if tools:
                    for tool in tools[:10]:
                        desc = (tool.description or "")[:60]
                        console.print(f"    🔧 {tool.name} — {desc}")
                    if len(tools) > 10:
                        console.print(f"    ... and {len(tools) - 10} more")
        except MCPError as exc:
            console.print(Text(f"✖ {exc}", style="red"))

    async def _run_all():
        for srv in servers:
            await _test_server(srv)
            console.print()

    asyncio.run(_run_all())


# ---------------------------------------------------------------------------
# The two interaction modes
# ---------------------------------------------------------------------------


@cli.command(context_settings={"ignore_unknown_options": True})
@click.argument("prompt", nargs=-1, required=False)
@click.option("--system", "-s", default=None, help="System prompt for this request.")
@click.option("--model", "-m", default=None, help="Model override for this request.")
@click.option("--temperature", "-t", type=float, default=None, help="Sampling temperature.")
@click.option("--max-tokens", type=click.IntRange(min=1), default=None, help="Cap the reply length.")
@click.option("--no-stream", is_flag=True, help="Print the reply only when complete (for scripts/pipes).")
@click.pass_context
def ask(
    ctx: click.Context,
    prompt: tuple[str, ...],
    system: Optional[str],
    model: Optional[str],
    temperature: Optional[float],
    max_tokens: Optional[int],
    no_stream: bool,
) -> None:
    """Ask a single question and print the reply (also the default command)."""
    parent = ctx.parent.params if ctx.parent is not None else {}
    system = system or parent.get("system")
    model = model or parent.get("model")
    if temperature is None:
        temperature = parent.get("temperature")
    if max_tokens is None:
        max_tokens = parent.get("max_tokens")
    no_stream = no_stream or bool(parent.get("no_stream"))

    text = " ".join(prompt).strip()
    if not text:
        raise click.UsageError(
            'Provide a prompt, e.g.  phoenix "write a python script"'
        )

    cfg = load_config()
    problem = check_configured(cfg)
    if problem:
        print_error(ConfigurationError(problem))
        raise SystemExit(1)

    raise SystemExit(
        run_single_prompt(
            cfg,
            text,
            system=system,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            no_stream=no_stream,
        )
    )


@cli.command()
@click.option("--system", "-s", default=None, help="System prompt for the conversation.")
@click.option("--model", "-m", default=None, help="Model override for this chat.")
@click.option("--temperature", "-t", type=float, default=None, help="Sampling temperature.")
@click.option("--max-tokens", type=click.IntRange(min=1), default=None, help="Cap each reply length.")
@click.pass_context
def chat(
    ctx: click.Context,
    system: Optional[str],
    model: Optional[str],
    temperature: Optional[float],
    max_tokens: Optional[int],
) -> None:
    """Start an interactive chat session with 30+ slash commands.

    Type /help once inside for the full reference.
    """
    parent = ctx.parent.params if ctx.parent is not None else {}
    system = system or parent.get("system")
    model = model or parent.get("model")
    if temperature is None:
        temperature = parent.get("temperature")
    if max_tokens is None:
        max_tokens = parent.get("max_tokens")

    cfg = load_config()
    problem = check_configured(cfg)
    if problem:
        print_error(ConfigurationError(problem))
        raise SystemExit(1)
    raise SystemExit(
        run_chat(cfg, system=system, model=model, temperature=temperature, max_tokens=max_tokens)
    )


if __name__ == "__main__":
    cli()
