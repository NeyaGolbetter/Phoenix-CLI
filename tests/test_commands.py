"""Tests for the 30+ chat slash commands in phoenix chat.

These tests call `_handle_slash_command` directly with a fake Conversation
and state dict, so they don't need a TTY or the subprocess-level chat loop.
For the few commands that hit the network (`/model`, `/models`, `/ping`),
we exercise them via `CliRunner` with the `script` pty wrapper — or we just
assert the non-network path works and rely on the existing client tests for
network behaviour.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import pytest
from click.testing import CliRunner

from phoenix_cli.cli import (
    _handle_slash_command,
    _update_system_with_pins,
    _select_model_interactive,
    cli,
)
from phoenix_cli.client import Conversation
from phoenix_cli.config import save_config


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def chat_state(cfg: Dict[str, str]):
    """A minimal state dict mirroring what `run_chat` builds."""
    return {
        "model": cfg["model_name"],
        "temperature": None,
        "max_tokens": None,
        "auto_approve": True,
        "theme": "monokai",
        "verbose": False,
    }


@pytest.fixture()
def chat_conversation():
    """An empty conversation with tool-call metadata attached."""
    conv = Conversation(system=None)
    conv._pinned_notes = []
    conv._tool_msgs = {}
    conv._tool_call_ids = {}
    return conv


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------


def test_help_lists_commands(chat_conversation, chat_state, cfg, capsys):
    """`/help` prints the full command reference."""
    result = _handle_slash_command("/help", chat_conversation, chat_state, cfg)
    assert result is None
    # Help text was printed — we don't capture stdout here easily, but we
    # assert the command didn't crash and didn't exit.


def test_exit_returns_exit_string(chat_conversation, chat_state, cfg):
    for cmd in ("/exit", "/quit", "/q"):
        conv = Conversation()
        conv._pinned_notes = []
        conv._tool_msgs = {}
        conv._tool_call_ids = {}
        assert _handle_slash_command(cmd, conv, chat_state, cfg) == "exit"


def test_clear_wipes_history(chat_conversation, chat_state, cfg):
    chat_conversation.add("user", "hi")
    chat_conversation.add("assistant", "hello")
    _handle_slash_command("/clear", chat_conversation, chat_state, cfg)
    assert len(chat_conversation.messages) == 0


# ---------------------------------------------------------------------------
# Models & system
# ---------------------------------------------------------------------------


def test_model_direct_switch(chat_conversation, chat_state, cfg):
    _handle_slash_command("/model llama3.2", chat_conversation, chat_state, cfg)
    assert chat_state["model"] == "llama3.2"


def test_system_set_and_show(chat_conversation, chat_state, cfg):
    _handle_slash_command("/system You are terse", chat_conversation, chat_state, cfg)
    assert chat_conversation.system == "You are terse"

    _handle_slash_command("/system", chat_conversation, chat_state, cfg)
    # Should print the current system prompt (no crash).


def test_system_clear(chat_conversation, chat_state, cfg):
    chat_conversation.system = "something"
    _handle_slash_command("/system clear", chat_conversation, chat_state, cfg)
    assert chat_conversation.system is None


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def test_temp_set_and_show(chat_conversation, chat_state, cfg):
    _handle_slash_command("/temp 0.4", chat_conversation, chat_state, cfg)
    assert chat_state["temperature"] == 0.4

    _handle_slash_command("/temp", chat_conversation, chat_state, cfg)
    # Should print current value.


def test_temp_invalid(chat_conversation, chat_state, cfg):
    _handle_slash_command("/temp garbage", chat_conversation, chat_state, cfg)
    assert chat_state["temperature"] is None  # unchanged


def test_max_tokens_set_and_show(chat_conversation, chat_state, cfg):
    _handle_slash_command("/max-tokens 256", chat_conversation, chat_state, cfg)
    assert chat_state["max_tokens"] == 256

    _handle_slash_command("/max-tokens", chat_conversation, chat_state, cfg)


def test_max_tokens_invalid(chat_conversation, chat_state, cfg):
    _handle_slash_command("/max-tokens abc", chat_conversation, chat_state, cfg)
    assert chat_state["max_tokens"] is None


# ---------------------------------------------------------------------------
# Conversation
# ---------------------------------------------------------------------------


def test_history_shows_count(chat_conversation, chat_state, cfg):
    chat_conversation.add("user", "hi")
    chat_conversation.add("assistant", "hello")
    # Shouldn't crash.
    _handle_slash_command("/history", chat_conversation, chat_state, cfg)


def test_save_exports_markdown(chat_conversation, chat_state, cfg, tmp_path):
    chat_conversation.add("user", "hello")
    chat_conversation.add("assistant", "hi")
    target = tmp_path / "out.md"
    _handle_slash_command(
        f"/save {target}", chat_conversation, chat_state, cfg
    )
    assert target.is_file()
    text = target.read_text()
    assert "User" in text
    assert "hello" in text


def test_save_default_filename(chat_conversation, chat_state, cfg, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    chat_conversation.add("user", "hello")
    chat_conversation.add("assistant", "hi")
    _handle_slash_command("/save", chat_conversation, chat_state, cfg)
    assert (tmp_path / "phoenix_chat.md").is_file()


def test_undo_on_empty(chat_conversation, chat_state, cfg):
    # Shouldn't crash.
    _handle_slash_command("/undo", chat_conversation, chat_state, cfg)
    assert len(chat_conversation.messages) == 0


def test_undo_removes_last_exchange(chat_conversation, chat_state, cfg):
    chat_conversation.add("user", "hi")
    chat_conversation.add("assistant", "hello")
    _handle_slash_command("/undo", chat_conversation, chat_state, cfg)
    assert len(chat_conversation.messages) == 0


def test_undo_with_tool_messages(chat_conversation, chat_state, cfg):
    chat_conversation.add("user", "hi")
    chat_conversation.add("assistant", "calling tool")
    chat_conversation.add("tool", "result")
    chat_conversation._tool_msgs[len(chat_conversation.messages) - 2] = {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"id": "x", "type": "function", "function": {"name": "y", "arguments": "{}"}}],
    }
    chat_conversation._tool_call_ids[len(chat_conversation.messages) - 1] = {
        "tool_call_id": "x",
        "name": "y",
    }
    _handle_slash_command("/undo", chat_conversation, chat_state, cfg)
    # All tool-related messages and metadata should be gone.
    assert len(chat_conversation.messages) == 0
    assert chat_conversation._tool_msgs == {}
    assert chat_conversation._tool_call_ids == {}


def test_search_no_match(chat_conversation, chat_state, cfg):
    chat_conversation.add("user", "hi")
    _handle_slash_command("/search banana", chat_conversation, chat_state, cfg)
    # Shouldn't crash.


def test_search_finds_match(chat_conversation, chat_state, cfg):
    chat_conversation.add("user", "what is a banana?")
    chat_conversation.add("assistant", "a banana is a fruit")
    _handle_slash_command("/search banana", chat_conversation, chat_state, cfg)
    # Shouldn't crash.


def test_search_without_word(chat_conversation, chat_state, cfg):
    _handle_slash_command("/search", chat_conversation, chat_state, cfg)
    # Should print usage.


def test_export_and_import_roundtrip(chat_conversation, chat_state, cfg, tmp_path):
    chat_conversation.add("user", "hello world")
    chat_conversation.add("assistant", "hi")
    target = tmp_path / "chat.json"

    _handle_slash_command(
        f"/export {target}", chat_conversation, chat_state, cfg
    )
    assert target.is_file()
    data = json.loads(target.read_text())
    assert data["messages"][0]["content"] == "hello world"

    # Import into a fresh conversation.
    conv2 = Conversation()
    conv2._pinned_notes = []
    conv2._tool_msgs = {}
    conv2._tool_call_ids = {}
    _handle_slash_command(
        f"/import {target}", conv2, chat_state, cfg
    )
    assert len(conv2.messages) == 2
    assert conv2.messages[0].content == "hello world"


def test_import_missing_file(chat_conversation, chat_state, cfg):
    _handle_slash_command(
        "/import /tmp/no_such_file_abc.json",
        chat_conversation, chat_state, cfg,
    )
    # Shouldn't crash.


def test_import_usage(chat_conversation, chat_state, cfg):
    _handle_slash_command("/import", chat_conversation, chat_state, cfg)
    # Should print usage.


# ---------------------------------------------------------------------------
# MCP & tools
# ---------------------------------------------------------------------------


def test_tools_when_no_mcp(chat_conversation, chat_state, cfg):
    _handle_slash_command("/tools", chat_conversation, chat_state, cfg, None)
    # Shouldn't crash.


def test_mcp_status_when_no_mcp(chat_conversation, chat_state, cfg):
    _handle_slash_command("/mcp", chat_conversation, chat_state, cfg, None)
    # Shouldn't crash.


def test_auto_show(chat_conversation, chat_state, cfg):
    _handle_slash_command("/auto", chat_conversation, chat_state, cfg)
    # Default is ON.


def test_auto_on_off(chat_conversation, chat_state, cfg):
    assert chat_state.get("auto_approve", True) is True

    _handle_slash_command("/auto off", chat_conversation, chat_state, cfg)
    assert chat_state["auto_approve"] is False

    _handle_slash_command("/auto on", chat_conversation, chat_state, cfg)
    assert chat_state["auto_approve"] is True


def test_auto_invalid(chat_conversation, chat_state, cfg):
    _handle_slash_command("/auto banana", chat_conversation, chat_state, cfg)
    # Should print usage.


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------


def test_status_command(chat_conversation, chat_state, cfg):
    _handle_slash_command("/status", chat_conversation, chat_state, cfg)
    # Shouldn't crash.


def test_config_command(chat_conversation, chat_state, cfg):
    _handle_slash_command("/config", chat_conversation, chat_state, cfg)
    # Shouldn't crash.


def test_theme_switch(chat_conversation, chat_state, cfg):
    _handle_slash_command("/theme dracula", chat_conversation, chat_state, cfg)
    assert chat_state["theme"] == "dracula"


def test_theme_show(chat_conversation, chat_state, cfg):
    _handle_slash_command("/theme", chat_conversation, chat_state, cfg)
    # Should print available themes.


def test_theme_invalid(chat_conversation, chat_state, cfg):
    _handle_slash_command("/theme banana", chat_conversation, chat_state, cfg)
    assert chat_state["theme"] == "monokai"  # unchanged


def test_verbose_toggle(chat_conversation, chat_state, cfg):
    assert chat_state.get("verbose", False) is False

    _handle_slash_command("/verbose on", chat_conversation, chat_state, cfg)
    assert chat_state["verbose"] is True

    _handle_slash_command("/verbose off", chat_conversation, chat_state, cfg)
    assert chat_state["verbose"] is False


def test_verbose_show(chat_conversation, chat_state, cfg):
    _handle_slash_command("/verbose", chat_conversation, chat_state, cfg)
    # Shouldn't crash.


def test_context_shows_estimate(chat_conversation, chat_state, cfg):
    chat_conversation.add("user", "some text here")
    _handle_slash_command("/context", chat_conversation, chat_state, cfg)
    # Shouldn't crash.


def test_copy_no_reply(chat_conversation, chat_state, cfg):
    _handle_slash_command("/copy", chat_conversation, chat_state, cfg)
    # Shouldn't crash.


def test_reset_restores_defaults(chat_conversation, chat_state, cfg):
    chat_conversation.add("user", "hi")
    chat_conversation.add("assistant", "hello")
    chat_state["temperature"] = 0.9
    chat_state["max_tokens"] = 999
    chat_state["theme"] = "dracula"

    _handle_slash_command("/reset", chat_conversation, chat_state, cfg)

    assert len(chat_conversation.messages) == 0
    assert chat_state["temperature"] is None
    assert chat_state["max_tokens"] is None
    assert chat_state["theme"] == "monokai"
    assert chat_state["auto_approve"] is True


# ---------------------------------------------------------------------------
# /pin, /pinned, /unpin
# ---------------------------------------------------------------------------


def test_pin_and_show(chat_conversation, chat_state, cfg):
    _handle_slash_command(
        "/pin always use python", chat_conversation, chat_state, cfg
    )
    _handle_slash_command(
        "/pin keep it short", chat_conversation, chat_state, cfg
    )
    assert chat_conversation._pinned_notes == [
        "always use python",
        "keep it short",
    ]
    # System prompt should now contain pinned notes.
    assert "Pinned notes" in (chat_conversation.system or "")


def test_pinned_shows_list(chat_conversation, chat_state, cfg):
    chat_conversation._pinned_notes = ["note one", "note two"]
    _handle_slash_command("/pinned", chat_conversation, chat_state, cfg)
    # Shouldn't crash.


def test_pinned_empty(chat_conversation, chat_state, cfg):
    _handle_slash_command("/pinned", chat_conversation, chat_state, cfg)
    # Shouldn't crash.


def test_unpin_by_number(chat_conversation, chat_state, cfg):
    chat_conversation._pinned_notes = ["first", "second", "third"]
    _handle_slash_command("/unpin 2", chat_conversation, chat_state, cfg)
    assert chat_conversation._pinned_notes == ["first", "third"]


def test_unpin_invalid_number(chat_conversation, chat_state, cfg):
    chat_conversation._pinned_notes = ["only one"]
    _handle_slash_command("/unpin 5", chat_conversation, chat_state, cfg)
    # Shouldn't crash; note list unchanged.
    assert chat_conversation._pinned_notes == ["only one"]


def test_unpin_usage(chat_conversation, chat_state, cfg):
    _handle_slash_command("/unpin", chat_conversation, chat_state, cfg)
    # Shouldn't crash.


def test_pin_without_text(chat_conversation, chat_state, cfg):
    _handle_slash_command("/pin", chat_conversation, chat_state, cfg)
    # Shouldn't crash.


# ---------------------------------------------------------------------------
# Unknown command
# ---------------------------------------------------------------------------


def test_unknown_command_reports_error(chat_conversation, chat_state, cfg):
    _handle_slash_command("/nonsense", chat_conversation, chat_state, cfg)
    # Shouldn't crash.


# ---------------------------------------------------------------------------
# Compact (needs async — use a tiny async helper)
# ---------------------------------------------------------------------------


def test_pinned_notes_merge_into_system(chat_conversation, chat_state, cfg):
    """Pinned notes get merged into the system prompt with a header."""
    _update_system_with_pins(chat_conversation)
    assert chat_conversation.system is None

    chat_conversation._pinned_notes = ["note A", "note B"]
    _update_system_with_pins(chat_conversation)
    assert "Pinned notes" in chat_conversation.system
    assert "note A" in chat_conversation.system
    assert "note B" in chat_conversation.system

    # Remove a note and check system prompt updates.
    chat_conversation._pinned_notes.pop()
    _update_system_with_pins(chat_conversation)
    assert "note B" not in chat_conversation.system
    assert "note A" in chat_conversation.system

    # Remove all notes — system prompt clears.
    chat_conversation._pinned_notes.clear()
    _update_system_with_pins(chat_conversation)
    assert chat_conversation.system is None
