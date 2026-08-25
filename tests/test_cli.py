"""End-to-end tests for the Phoenix CLI command surface."""

from __future__ import annotations

import json
import os
import pty
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest
from click.testing import CliRunner

from phoenix_cli.cli import cli
from phoenix_cli.config import save_config

REPO_ROOT = Path(__file__).resolve().parent.parent


def make_env(tmp_path, base_url=None, model=None, api_key=None):
    """A clean subprocess environment with isolated PHOENIX_* variables."""
    env = dict(os.environ)
    for var in ("PHOENIX_BASE_URL", "PHOENIX_API_KEY", "PHOENIX_MODEL"):
        env.pop(var, None)
    env["PHOENIX_CONFIG"] = str(tmp_path / "no_config_here.json")
    env["PYTHONPATH"] = str(REPO_ROOT)
    if base_url:
        env["PHOENIX_BASE_URL"] = base_url
    if model:
        env["PHOENIX_MODEL"] = model
    if api_key:
        env["PHOENIX_API_KEY"] = api_key
    return env


def run_phoenix(argv, env, input_text=None):
    return subprocess.run(
        [sys.executable, "-m", "phoenix_cli", *argv],
        input=input_text,
        capture_output=True,
        text=True,
        env=env,
        cwd=REPO_ROOT,
        timeout=60,
    )


# ---------------------------------------------------------------------------
# Pure click-level behavior (no network)
# ---------------------------------------------------------------------------


def test_version_flag():
    result = CliRunner().invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert "phoenix" in result.output.lower()


def test_help_lists_commands():
    result = CliRunner().invoke(cli, ["--help"])
    assert result.exit_code == 0
    for name in ("setup", "chat", "status", "ask"):
        assert name in result.output


def test_no_args_prints_banner_and_quick_start():
    result = CliRunner().invoke(cli, [])
    assert result.exit_code == 0
    assert "Quick start" in result.output
    assert "phoenix setup" in result.output


def test_unknown_first_token_routes_to_prompt(clean_env):
    # `phoenix hello` must behave like `phoenix ask hello`: it reaches the
    # ask command, which then complains about missing configuration.
    result = CliRunner().invoke(cli, ["hello", "world"])
    assert result.exit_code == 1
    assert "phoenix setup" in result.output


def test_setup_writes_config(clean_env):
    runner = CliRunner()
    # URL, API_KEY (empty), model (auto-fetch fails so manual entry), MCP (no)
    result = runner.invoke(
        cli, ["setup"], input="http://localhost:11434\n\nllama3.2\nn\n"
    )
    assert result.exit_code == 0, result.output
    cfg = json.loads(clean_env.read_text(encoding="utf-8"))
    assert cfg["base_url"].endswith("/v1")
    assert cfg["api_key"] == ""
    assert cfg["model_name"] == "llama3.2"


def test_setup_rejects_garbage_url(clean_env):
    runner = CliRunner()
    # bad URL, good URL, API_KEY (empty), model (manual), MCP (no)
    result = runner.invoke(cli, ["setup"], input="not a url\nhttp://localhost:11434\n\nllama3\nn\n")
    assert result.exit_code == 0
    assert "valid URL" in result.output  # it warned about the bad one


def test_status_shows_config(clean_env):
    save_config("http://localhost:11434", "sk-secret-key", "llama3")
    result = CliRunner().invoke(cli, ["status"])
    assert result.exit_code == 0
    assert "llama3" in result.output
    assert "sk-****" in result.output
    assert "sk-secret-key" not in result.output  # key is masked


def test_status_probe_reaches_provider(mock_api, clean_env):
    save_config(mock_api, "", "ok")
    result = CliRunner().invoke(cli, ["status", "--probe"])
    assert result.exit_code == 0, result.output
    assert "reachable" in result.output


def test_status_probe_reports_auth_error(mock_api, clean_env):
    save_config(mock_api, "bad-key", "ok")
    result = CliRunner().invoke(cli, ["status", "--probe"])
    assert result.exit_code == 1
    assert "APIKeyError" in result.output


def test_models_command_lists_and_marks_current(mock_api, clean_env):
    save_config(mock_api, "", "qwen2.5-coder")
    result = CliRunner().invoke(cli, ["models"])
    assert result.exit_code == 0, result.output
    assert "✓ qwen2.5-coder" in result.output  # current model is marked
    assert "ok" in result.output
    assert "echo" in result.output


def test_models_raw_prints_bare_ids(mock_api, clean_env):
    save_config(mock_api, "", "ok")
    result = CliRunner().invoke(cli, ["models", "--raw"])
    assert result.exit_code == 0, result.output
    lines = result.output.strip().splitlines()
    assert "ok" in lines
    assert "✓" not in result.output  # no decoration in raw mode


def test_models_command_end_to_end(tmp_path, mock_api):
    env = make_env(tmp_path, base_url=mock_api, model="ok")
    result = run_phoenix(["models", "--raw"], env)
    assert result.returncode == 0, result.stderr
    assert "qwen2.5-coder" in result.stdout.splitlines()


# ---------------------------------------------------------------------------
# Full end-to-end via subprocess (real stdio, real HTTP)
# ---------------------------------------------------------------------------


def test_single_prompt_end_to_end(tmp_path, mock_api):
    env = make_env(tmp_path, base_url=mock_api, model="ok")
    result = run_phoenix(["Explain me this"], env)
    assert result.returncode == 0, result.stderr
    assert "mock provider" in result.stdout
    # rich renders markdown: the ``` fence itself is consumed, the code
    # content remains.
    assert "def greet" in result.stdout
    assert "\x1b[" not in result.stdout  # no ANSI codes when piped


def test_single_prompt_with_options_before_prompt(tmp_path, mock_api):
    env = make_env(tmp_path, base_url=mock_api, model="ok")
    result = run_phoenix(["--model", "echo", "just echo this back"], env)
    assert result.returncode == 0, result.stderr
    assert "just echo this back" in result.stdout


def test_single_prompt_auth_error(tmp_path, mock_api):
    env = make_env(tmp_path, base_url=mock_api, model="ok", api_key="bad-key")
    result = run_phoenix(["hi"], env)
    assert result.returncode == 1
    assert "APIKeyError" in result.stdout
    assert "setup" in result.stdout


def test_single_prompt_unconfigured(tmp_path):
    env = make_env(tmp_path)
    result = run_phoenix(["hi"], env)
    assert result.returncode == 1
    assert "phoenix setup" in result.stdout


def test_chat_requires_a_terminal(tmp_path, mock_api):
    env = make_env(tmp_path, base_url=mock_api, model="ok")
    result = run_phoenix(["chat"], env, input_text="hi\n/exit\n")
    assert result.returncode == 1
    assert "interactive terminal" in result.stdout


@pytest.mark.skipif(shutil.which("script") is None, reason="needs util-linux `script`")
def test_interactive_chat_over_pty(tmp_path, mock_api):
    """Drive `phoenix chat` through a pseudo-terminal like a real user."""
    env = make_env(tmp_path, base_url=mock_api, model="echo")
    env["TERM"] = "xterm-256color"
    result = subprocess.run(
        ["script", "-qec", f"{sys.executable} -m phoenix_cli chat", "/dev/null"],
        input="hello from the pty\n/exit\n",
        capture_output=True,
        text=True,
        env=env,
        cwd=REPO_ROOT,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "hello from the pty" in result.stdout  # the echoed reply


def test_no_stream_flag_prints_complete_reply(tmp_path, mock_api):
    env = make_env(tmp_path, base_url=mock_api, model="usage")
    result = run_phoenix(["--no-stream", "hi"], env)
    assert result.returncode == 0, result.stderr
    assert "mock provider" in result.stdout


# ---------------------------------------------------------------------------
# Model selection (interactive --select)
# ---------------------------------------------------------------------------


def test_models_select_interactive(mock_api, clean_env):
    """`phoenix models --select` lets you pick a model from a numbered list."""
    save_config(mock_api, "", "ok")
    runner = CliRunner()
    # The mock server has models: echo, ok, qwen2.5-coder (sorted).
    # Select the first one (number 1).
    result = runner.invoke(cli, ["models", "--select"], input="1\n")
    assert result.exit_code == 0, result.output
    assert "selected" in result.output.lower()
    assert "saved" in result.output.lower()


def test_models_select_cancel(mock_api, clean_env):
    """Entering 0 skips the selection."""
    save_config(mock_api, "", "ok")
    runner = CliRunner()
    result = runner.invoke(cli, ["models", "--select"], input="0\n")
    assert result.exit_code == 0, result.output


# ---------------------------------------------------------------------------
# MCP commands
# ---------------------------------------------------------------------------


def test_mcp_list_empty(clean_env, monkeypatch, tmp_path):
    """`phoenix mcp list` shows a helpful message when no servers configured."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    save_config("http://localhost:11434", "", "llama3", mcp_enabled=True)
    runner = CliRunner()
    result = runner.invoke(cli, ["mcp", "list"])
    assert result.exit_code == 0, result.output
    assert "No MCP servers" in result.output


def test_mcp_list_with_servers(clean_env, monkeypatch, tmp_path):
    """`phoenix mcp list` shows configured servers."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    save_config("http://localhost:11434", "", "llama3", mcp_enabled=True)
    # Write an MCP config.
    from phoenix_cli.mcp import save_mcp_config
    save_mcp_config([
        {"name": "roblox", "command": ["npx", "-y", "roblox-mcp"]},
        {"name": "remote", "url": "https://mcp.example.com"},
    ])
    runner = CliRunner()
    result = runner.invoke(cli, ["mcp", "list"])
    assert result.exit_code == 0, result.output
    assert "roblox" in result.output
    assert "remote" in result.output
    assert "stdio" in result.output
    assert "sse" in result.output


def test_mcp_remove(clean_env, monkeypatch, tmp_path):
    """`phoenix mcp remove` removes a server by name."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    save_config("http://localhost:11434", "", "llama3", mcp_enabled=True)
    from phoenix_cli.mcp import save_mcp_config, load_mcp_config
    save_mcp_config([
        {"name": "roblox", "command": ["npx", "-y", "roblox-mcp"]},
        {"name": "other", "command": ["node", "server.js"]},
    ])
    runner = CliRunner()
    result = runner.invoke(cli, ["mcp", "remove", "roblox"])
    assert result.exit_code == 0, result.output
    assert "removed" in result.output.lower()
    remaining = load_mcp_config()
    assert len(remaining) == 1
    assert remaining[0]["name"] == "other"


def test_status_shows_mcp(clean_env, monkeypatch, tmp_path):
    """`phoenix status` shows MCP status."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    save_config("http://localhost:11434", "", "llama3", mcp_enabled=True)
    runner = CliRunner()
    result = runner.invoke(cli, ["status"])
    assert result.exit_code == 0, result.output
    assert "MCP" in result.output
    assert "enabled" in result.output


# ---------------------------------------------------------------------------
# API key secure-input regression tests
# ---------------------------------------------------------------------------


def test_hidden_input_raw_reads_key_in_pty():
    """_hidden_input_raw must read a typed API key via PTY and not hang."""
    import os, pty, select, signal, time as _time

    pid, fd = pty.fork()
    if pid == 0:
        import os as _os
        _os.environ["PHOENIX_CONFIG"] = "/tmp/test_hidden_input_config.json"
        for v in ("PHOENIX_BASE_URL", "PHOENIX_API_KEY", "PHOENIX_MODEL"):
            _os.environ.pop(v, None)
        sys.path.insert(0, str(REPO_ROOT))
        from phoenix_cli.cli import _hidden_input_raw
        result = _hidden_input_raw("prompt> ")
        print(f"__GOT__={result!r}", flush=True)
        os._exit(0)

    all_out = b""


# ---------------------------------------------------------------------------
# API key secure-input regression tests
# ---------------------------------------------------------------------------


def _drain_pty(fd, timeout=2):
    """Read all available data from a PTY fd with a generous timeout."""
    import select as _sel, time as _t
    deadline = _t.time() + timeout
    out = b""
    got_data_recently = True
    while _t.time() < deadline:
        r, _, _ = _sel.select([fd], [], [], 0.2)
        if fd in r:
            try:
                d = os.read(fd, 4096)
                if d:
                    out += d
                    got_data_recently = True
                    deadline = _t.time() + 1.0  # extend timeout on data
                else:
                    if got_data_recently:
                        got_data_recently = False
                        _t.sleep(0.3)  # wait briefly for more
                    else:
                        break
            except OSError:
                break
        else:
            if got_data_recently:
                got_data_recently = False
    return out


def _wait_for_prompt_and_send(fd, marker, response, wait_before_send=0.5, initial_timeout=10):
    """Wait for marker then send response; return all bytes read."""
    import select as _sel, time as _t
    all_out = b""
    deadline = _t.time() + initial_timeout
    sent = False
    while _t.time() < deadline:
        r, _, _ = _sel.select([fd], [], [], 0.3)
        if fd in r:
            try:
                data = os.read(fd, 4096)
                if not data:
                    break
                all_out += data
            except OSError:
                break
        if not sent and marker in all_out:
            _t.sleep(wait_before_send)
            os.write(fd, response)
            sent = True
            break
    return all_out, sent


@pytest.mark.skipif(not hasattr(pty, "fork") or not hasattr(os, "openpty"),
                    reason="PTY tests require a PTY-capable POSIX system")
def test_hidden_input_raw_reads_key_in_pty():
    """_hidden_input_raw must read a typed API key via PTY and not hang."""
    import signal, time as _time

    pid, fd = pty.fork()
    if pid == 0:
        os.environ["PHOENIX_CONFIG"] = "/tmp/test_hidden_input_config.json"
        for v in ("PHOENIX_BASE_URL", "PHOENIX_API_KEY", "PHOENIX_MODEL"):
            os.environ.pop(v, None)
        sys.path.insert(0, str(REPO_ROOT))
        # Re-flush before exec-like behavior since we forked
        sys.stdout.flush()
        from phoenix_cli.cli import _hidden_input_raw
        result = _hidden_input_raw("prompt> ")
        # Use os write directly to print result to avoid Rich interference
        os.write(1, f"__GOT__={result!r}\n".encode())
        os._exit(0)

    _, sent = _wait_for_prompt_and_send(fd, b"prompt>", b"sk-regression-test-42\r")
    assert sent, "Prompt marker never appeared"
    all_out = _drain_pty(fd, timeout=3)

    try:
        _, s = os.waitpid(pid, os.WNOHANG)
        if s == 0:
            os.kill(pid, signal.SIGTERM)
            _time.sleep(0.3)
            os.waitpid(pid, os.WNOHANG)
    except (ChildProcessError, ProcessLookupError):
        pass

    out = all_out.decode("utf-8", errors="replace")
    assert "sk-regression-test-42" in out, f"Key not received; tail: {out[-400:]}"


@pytest.mark.skipif(not hasattr(pty, "fork") or not hasattr(os, "openpty"),
                    reason="PTY tests require a PTY-capable POSIX system")
def test_hidden_input_raw_handles_backspace_in_pty():
    """Backspace must erase the previous character (typed 'abcd' + 3 BS + 'x' → 'ax')."""
    import signal, time as _time

    pid, fd = pty.fork()
    if pid == 0:
        os.environ["PHOENIX_CONFIG"] = "/tmp/test_bs_config.json"
        for v in ("PHOENIX_BASE_URL", "PHOENIX_API_KEY", "PHOENIX_MODEL"):
            os.environ.pop(v, None)
        sys.path.insert(0, str(REPO_ROOT))
        sys.stdout.flush()
        from phoenix_cli.cli import _hidden_input_raw
        result = _hidden_input_raw("pw> ")
        os.write(1, f"__BS__={result!r}\n".encode())
        os._exit(0)

    _, sent = _wait_for_prompt_and_send(
        fd, b"pw>", b"abcd" + b"\x7f\x7f\x7f" + b"x\r"
    )
    assert sent, "Prompt marker never appeared"
    all_out = _drain_pty(fd, timeout=3)

    try:
        _, s = os.waitpid(pid, os.WNOHANG)
        if s == 0:
            os.kill(pid, signal.SIGTERM)
            _time.sleep(0.3)
            os.waitpid(pid, os.WNOHANG)
    except (ChildProcessError, ProcessLookupError):
        pass

    out = all_out.decode("utf-8", errors="replace")
    assert "__BS__=" in out, f"Result marker missing; tail: {out[-400:]}"
    assert "'ax'" in out, f"Backspace result wrong; tail: {out[-400:]}"
