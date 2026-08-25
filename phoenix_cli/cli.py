"""Command-line interface for Phoenix CLI.

Commands
--------
phoenix "prompt"        one-shot prompt (also: phoenix ask "prompt")
phoenix chat            interactive chat with in-memory history
phoenix setup           configure BASE_URL / API_KEY / MODEL_NAME
phoenix models          list the provider's available models
phoenix status          show the current configuration (+ optional probe)

Design notes for Termux:
* all output goes through rich, which re-measures the terminal on every
  render, so the layout survives window resizes;
* the interactive prompt uses prompt_toolkit (pure Python) when available
  and falls back to plain ``input()`` otherwise;
* Ctrl+C cancels the current request instead of killing the app.
"""

from __future__ import annotations

import asyncio
import os
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

# Brand color (phoenix orange).
ACCENT = "#ff6b00"

# A short prompt shown by prompt_toolkit / the input() fallback.
PROMPT_LABEL = "phoenix ❯ "

BANNER = r"""[bold #ff6b00]
   ██████╗ ██╗  ██╗ ██████╗ ███████╗███╗   ██╗██╗██╗  ██╗
   ██╔══██╗██║  ██║██╔═══██╗██╔════╝████╗  ██║██║╚██╗██╔╝
   ██████╔╝███████║██║   ██║█████╗  ██╔██╗ ██║██║ ╚███╔╝
   ██╔═══╝ ██╔══██║██║   ██║██╔══╝  ██║╚██╗██║██║ ██╔██╗
   ██║     ██║  ██║╚██████╔╝███████╗██║ ╚████║██║██╔╝ ██╗
   ╚═╝     ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝╚═╝╚═╝  ╚═╝
[/bold #ff6b00]"""

TAGLINE = "A provider-agnostic AI assistant for the terminal"

# Human-readable hints appended to specific errors.
ERROR_HINTS: Dict[type, str] = {
    ConfigurationError: (
        "Run `phoenix setup` to configure BASE_URL, API_KEY and MODEL_NAME."
    ),
    APIKeyError: "Run `phoenix setup` and enter a valid API key.",
    ModelNotFoundError: (
        "Run `phoenix setup` and double-check the exact model name.\n"
        "Tip: ask the provider for its model list, e.g. `ollama list`."
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
    console.print(f"[dim]v{__version__} • OpenAI-compatible APIs[/dim]", justify="center")
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
# Streaming
# ---------------------------------------------------------------------------


async def _stream_turn(
    params: Dict[str, Any],
    conversation: Conversation,
    *,
    use_live: bool,
) -> tuple[bool, str]:
    """Run one chat-completions request and print the streamed reply.

    ``use_live=True`` re-renders the growing markdown buffer in place with
    rich's ``Live`` (nice on a real terminal). ``use_live=False`` buffers
    quietly and prints the final markdown once -- the clean choice when
    stdout is piped.

    Returns ``(ok, response_text)``. Errors are printed here; only Ctrl+C
    propagates (as ``KeyboardInterrupt``) so callers can cancel cleanly.
    """
    buffer: List[str] = []
    usage: Dict[str, Any] = {}
    ok = False
    last_update = 0.0
    last_len = 0

    def renderable():
        text = "".join(buffer)
        if not text:
            return Spinner("dots", text=" waiting for the model...", style="dim")
        return Markdown(text, code_theme="monokai", justify="left")

    try:
        if use_live:
            # Refresh at most ~10 times/sec; updates also happen at line
            # boundaries, so big code dumps stay snappy on a phone.
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
                        ):
                            content = chunk.get("content")
                            if content:
                                buffer.append(content)
                                now = time.monotonic()
                                if (
                                    "\n" in content
                                    or now - last_update >= 0.1
                                    or len(buffer) - last_len >= 128
                                ):
                                    live.update(renderable())
                                    last_update = now
                                    last_len = len(buffer)
                            elif "usage" in chunk:
                                usage = chunk["usage"]
                except PhoenixError:
                    # Erase the "waiting..." spinner before the error is
                    # printed below (partial replies stay visible).
                    if not buffer:
                        live.update(Text(""))
                    raise
                live.update(renderable())  # final frame (also flushes spinner)
        else:
            async with make_client(params) as client:
                async for chunk in client.chat_stream(
                    conversation.history(),
                    temperature=params.get("temperature"),
                    max_tokens=params.get("max_tokens"),
                ):
                    content = chunk.get("content")
                    if content:
                        buffer.append(content)
                    elif "usage" in chunk:
                        usage = chunk["usage"]
            if buffer:
                console.print(Markdown("".join(buffer), code_theme="monokai"))
    except PhoenixError as exc:
        # If we already streamed part of the reply, keep it visible and mark
        # the truncation point.
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
[bold]Commands[/bold]
  [bold #ff6b00]/help[/]              show this help
  [bold #ff6b00]/exit[/], [bold #ff6b00]/quit[/]     leave the chat
  [bold #ff6b00]/clear[/]             forget the conversation history
  [bold #ff6b00]/model NAME[/]        switch to a different model (current session)
  [bold #ff6b00]/system TEXT[/]       set/replace the system prompt (empty clears it)
  [bold #ff6b00]/temp 0.8[/]          set sampling temperature
  [bold #ff6b00]/max-tokens 1024[/]   cap the reply length
  [bold #ff6b00]/history[/]           show how many messages are in memory
  [bold #ff6b00]/save FILE[/]         save the conversation as markdown

[bold]Keys[/bold]
  Ctrl+C   cancel the reply currently streaming
  Ctrl+D   quit (same as /exit)
"""


def _handle_slash_command(
    command: str,
    conversation: Conversation,
    state: Dict[str, Any],
) -> Optional[str]:
    """Interpret a ``/command`` typed in chat.

    Returns ``"exit"`` when the user wants to leave, else ``None``.
    """
    parts = command.split(None, 1)
    verb = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""

    if verb in ("/exit", "/quit", "/q"):
        return "exit"

    if verb == "/help":
        console.print(Panel(Markdown(HELP_TEXT), border_style=ACCENT, title="Phoenix CLI"))
    elif verb == "/clear":
        conversation.clear()
        console.print("[dim]✔ conversation history cleared[/dim]")
    elif verb == "/model":
        if not arg:
            console.print(f"[dim]current model: {state['model']}[/dim]")
        else:
            state["model"] = arg
            console.print(f"[dim]✔ model switched to [bold]{arg}[/bold][/dim]")
    elif verb == "/system":
        conversation.system = arg or None
        console.print(
            "[dim]✔ system prompt "
            + ("cleared[/dim]" if not arg else f"set to: {arg}[/dim]")
        )
    elif verb in ("/temp", "/temperature"):
        try:
            state["temperature"] = float(arg)
            console.print(f"[dim]✔ temperature = {state['temperature']}[/dim]")
        except ValueError:
            console.print("[red]usage: /temp 0.8[/red]")
    elif verb in ("/max-tokens", "/maxtokens"):
        try:
            state["max_tokens"] = int(arg)
            console.print(f"[dim]✔ max-tokens = {state['max_tokens']}[/dim]")
        except ValueError:
            console.print("[red]usage: /max-tokens 1024[/red]")
    elif verb == "/history":
        count = len(conversation.messages)
        console.print(
            f"[dim]{count} messages in memory "
            f"(limit {conversation.max_messages}, oldest are dropped)[/dim]"
        )
    elif verb == "/save":
        target = Path(arg).expanduser() if arg else Path("phoenix_chat.md")
        _save_conversation(conversation, target, state["model"])
        console.print(f"[dim]✔ conversation saved to {target}[/dim]")
    else:
        console.print(
            f"[red]unknown command {verb!r} — try /help[/red]"
        )
    return None


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
    }

    print_banner()
    console.print(f"[bold]Model:[/bold] [bold #ff6b00]{state['model']}[/bold #ff6b00]")
    console.print(f"[bold]API:[/bold]   {cfg['base_url']}")
    console.print("[dim]Type /help for commands • Ctrl+C cancels a reply[/dim]")
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
        if user_input is None:  # EOF (Ctrl+D)
            console.print("\n[dim]Bye![/dim]")
            break

        text = user_input.strip()
        if not text:
            continue
        if text.startswith("/"):
            if _handle_slash_command(text, conversation, state) == "exit":
                console.print("[dim]Bye![/dim]")
                break
            continue

        conversation.add("user", text)
        params = _params(
            cfg,
            model=state["model"],
            temperature=state["temperature"],
            max_tokens=state["max_tokens"],
        )
        try:
            asyncio.run(_stream_turn(params, conversation, use_live=True))
        except KeyboardInterrupt:
            # Ctrl+C mid-stream: the request is cancelled (the `Live`
            # context cleans itself up while unwinding). Drop the
            # un-answered prompt so the history stays consistent.
            console.print(Text("✖ interrupted — reply cancelled", style="dim"))
            if conversation.messages and conversation.messages[-1].role == "user":
                conversation.messages.pop()
        console.print()
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
    try:
        ok, _ = asyncio.run(_stream_turn(params, conversation, use_live=use_live))
    except KeyboardInterrupt:
        console.print(Text("✖ interrupted — reply cancelled", style="dim"))
        return 130
    return 0 if ok else 1


async def _probe(params: Dict[str, Any]) -> bool:
    """Send a tiny request to verify connectivity; print timing info."""
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
    """Fetch the provider's model list (errors are printed by the caller)."""
    async with make_client(params) as client:
        return await client.list_models()


# ---------------------------------------------------------------------------
# Click command group
# ---------------------------------------------------------------------------


class PhoenixGroup(click.Group):
    """Group that lets the first argument be a prompt instead of a command.

    ``phoenix "write a script"`` must behave like ``phoenix ask "write a
    script"``. Click normally raises "No such command" for an unknown first
    token, so we route it to the default ``ask`` command instead.
    """

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
            # The first token is not a subcommand: it is the beginning of a
            # prompt. Hand everything to the default `ask` command.
            ask_cmd = self.commands.get("ask")
            if ask_cmd is None:  # pragma: no cover - defensive
                raise
            return "ask", ask_cmd, list(args)


@click.group(cls=PhoenixGroup)
@click.version_option(version=__version__, prog_name="phoenix")
# These options are shared by `ask` and `chat` and may appear before the
# subcommand/prompt (e.g. `phoenix -m llama3 "hello"`). Each subcommand also
# declares its own copy so `phoenix chat -m llama3` works; subcommands merge
# the group-level value when their own option was not used.
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
    """Phoenix CLI — a provider-agnostic AI assistant for the terminal.

    Works with any OpenAI-compatible API: Ollama, LM Studio, vLLM,
    OpenRouter, Together AI, Groq, DeepSeek, and more.
    """
    if ctx.invoked_subcommand is None:
        print_banner()
        console.print(
            Panel.fit(
                "  [bold]Quick start[/bold]\n"
                "  [bold #ff6b00]phoenix setup[/bold #ff6b00]                     "
                "configure provider + model\n"
                "  [bold #ff6b00]phoenix \"explain quicksort\"[/bold #ff6b00]     "
                "one-shot prompt\n"
                "  [bold #ff6b00]phoenix chat[/bold #ff6b00]                     "
                "interactive conversation\n"
                "  [bold #ff6b00]phoenix models[/bold #ff6b00]                   "
                "list the provider's models\n"
                "  [bold #ff6b00]phoenix status[/bold #ff6b00]                   "
                "show current configuration\n"
                "\n"
                "  [dim]Run `phoenix --help` for every option.[/dim]",
                border_style=ACCENT,
                title="Phoenix CLI",
            )
        )


@cli.command()
def setup() -> None:
    """Configure BASE_URL, API_KEY and MODEL_NAME (~/.phoenix_config.json)."""
    print_banner()
    current = load_config()

    console.print(
        "[bold]Phoenix setup[/bold] — the provider can be any OpenAI-compatible "
        "API. Press Enter to keep the current value."
    )
    console.print()

    # -- BASE_URL -----------------------------------------------------------
    def url_hint() -> None:
        console.print(
            "[dim]examples:\n"
            "  Ollama (local)     http://localhost:11434\n"
            "  LM Studio (local)  http://localhost:1234\n"
            "  vLLM (local)       http://localhost:8000\n"
            "  OpenRouter         https://openrouter.ai/api\n"
            "  Together AI        https://api.together.xyz\n"
            "  Groq               https://api.groq.com/openai[/dim]"
        )

    url_hint()
    base_url = ""
    for _attempt in range(3):
        raw = click.prompt(
            "BASE_URL", default=current["base_url"] or "", show_default=False
        )
        base_url = normalize_base_url(raw)
        if "://" not in base_url or " " in base_url:
            console.print("[red]That does not look like a valid URL.[/red]")
            url_hint()
            continue
        break
    if "://" not in base_url:
        console.print("[red]Giving up on BASE_URL — run `phoenix setup` again.[/red]")
        raise SystemExit(1)

    # -- API_KEY ------------------------------------------------------------
    api_key = click.prompt(
        "API_KEY (Enter for none — local servers usually need none)",
        default=current["api_key"] or "",
        hide_input=True,
        show_default=False,
    ).strip()

    # -- MODEL_NAME ---------------------------------------------------------
    model_name = ""
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

    path = save_config(base_url=base_url, api_key=api_key, model_name=model_name)

    console.print()
    console.print(Panel(
        f"  [bold]BASE_URL[/bold]   {base_url}\n"
        f"  [bold]API_KEY[/bold]    {mask_secret(api_key)}\n"
        f"  [bold]MODEL_NAME[/bold] {model_name}",
        border_style="green",
        title="Saved",
    ))
    console.print(f"[dim]Config written to {path}[/dim]")
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
        params["timeout"] = 15.0  # keep the probe snappy
        if not asyncio.run(_probe(params)):
            raise SystemExit(1)


@cli.command()
@click.option("--raw", is_flag=True, help="Print one model ID per line (for scripts).")
def models(raw: bool) -> None:
    """List the models available from the configured provider."""
    cfg = load_config()
    problem = check_configured(cfg)
    if problem:
        print_error(ConfigurationError(problem))
        raise SystemExit(1)

    params = _params(cfg)
    params["timeout"] = 15.0  # listing should never hang for long
    try:
        ids = asyncio.run(_list_models(params))
    except PhoenixError as exc:
        print_error(exc)
        raise SystemExit(1)

    if not ids:
        console.print("[dim]The provider returned an empty model list.[/dim]")
        raise SystemExit(1)

    if raw:
        for model_id in ids:
            console.print(model_id)
        return

    current = cfg["model_name"]
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
        "[dim]Tip: `phoenix setup` to change MODEL_NAME, or "
        "`phoenix -m NAME \"prompt\"` for a one-off.[/dim]"
    )


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
    # Merge values given at the group level (`phoenix -m llama3 "hi"`).
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
    """Start an interactive chat session (history kept in memory)."""
    # Merge values given at the group level (`phoenix -m llama3 chat`).
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
