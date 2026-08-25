"""Shared pytest fixtures."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Make `import mock_server` work regardless of how pytest is invoked.
sys.path.insert(0, str(Path(__file__).parent))

from mock_server import start_server  # noqa: E402


@pytest.fixture(scope="session")
def mock_api():
    """An OpenAI-compatible chat-completions endpoint (ephemeral port)."""
    server, port = start_server()
    yield f"http://127.0.0.1:{port}/v1"
    server.shutdown()
    server.server_close()


@pytest.fixture()
def clean_env(monkeypatch, tmp_path):
    """Isolate the environment: no config file, no PHOENIX_* variables."""
    for var in ("PHOENIX_BASE_URL", "PHOENIX_API_KEY", "PHOENIX_MODEL"):
        monkeypatch.delenv(var, raising=False)
    config_file = tmp_path / ".phoenix_config.json"
    monkeypatch.setenv("PHOENIX_CONFIG", str(config_file))
    return config_file
