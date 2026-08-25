"""End-to-end tests for the Phoenix CLI command surface."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
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
    result = runner.invoke(
        cli, ["setup"], input="http://localhost:11434\n\nllama3.2\n"
    )
    assert result.exit_code == 0, result.output
    cfg = json.loads(clean_env.read_text(encoding="utf-8"))
    assert cfg["base_url"].endswith("/v1")
    assert cfg["api_key"] == ""
    assert cfg["model_name"] == "llama3.2"


def test_setup_rejects_garbage_url(clean_env):
    runner = CliRunner()
    result = runner.invoke(cli, ["setup"], input="not a url\nhttp://localhost:11434\n\nllama3\n")
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
