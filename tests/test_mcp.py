"""Tests for phoenix_cli.mcp."""

from __future__ import annotations

import json
import os
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
