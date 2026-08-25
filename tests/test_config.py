"""Tests for phoenix_cli.config."""

from __future__ import annotations

import json

from phoenix_cli.config import (
    check_configured,
    config_path,
    load_config,
    mask_secret,
    normalize_base_url,
    save_config,
)


def test_normalize_appends_v1_when_no_path():
    assert normalize_base_url("http://localhost:11434") == "http://localhost:11434/v1"
    assert normalize_base_url(" http://localhost:11434/ ") == "http://localhost:11434/v1"


def test_normalize_keeps_explicit_paths():
    assert normalize_base_url("http://localhost:11434/v1") == "http://localhost:11434/v1"
    assert (
        normalize_base_url("https://api.groq.com/openai/v1")
        == "https://api.groq.com/openai/v1"
    )
    assert (
        normalize_base_url("https://openrouter.ai/api/v1/")
        == "https://openrouter.ai/api/v1"
    )
    assert normalize_base_url("http://localhost:1234/custom") == "http://localhost:1234/custom"


def test_save_and_load_roundtrip(clean_env):
    path = save_config("http://localhost:11434", "secret-key", "llama3")
    assert path == config_path()
    assert path.is_file()
    cfg = load_config()
    assert cfg["base_url"] == "http://localhost:11434/v1"
    assert cfg["api_key"] == "secret-key"
    assert cfg["model_name"] == "llama3"


def test_env_vars_take_precedence(clean_env, monkeypatch):
    save_config("http://file-url", "file-key", "file-model")
    monkeypatch.setenv("PHOENIX_BASE_URL", "http://env-url:9999")
    monkeypatch.setenv("PHOENIX_API_KEY", "env-key")
    monkeypatch.setenv("PHOENIX_MODEL", "env-model")
    cfg = load_config()
    assert cfg["base_url"] == "http://env-url:9999/v1"
    assert cfg["api_key"] == "env-key"
    assert cfg["model_name"] == "env-model"


def test_corrupt_config_file_fails_soft(clean_env):
    clean_env.write_text("{ this is not json", encoding="utf-8")
    cfg = load_config()  # must not raise
    assert cfg["base_url"] == ""


def test_missing_config_file_fails_soft(clean_env):
    cfg = load_config()
    assert cfg["model_name"] == ""


def test_mask_secret():
    assert mask_secret("") == "(none)"
    assert mask_secret("abc") == "****"
    assert mask_secret("sk-abcdefgh1234") == "sk-****34"


def test_check_configured(clean_env):
    save_config("http://localhost:11434", "", "llama3")
    assert check_configured(load_config()) is None

    clean_env.unlink()
    assert "setup" in check_configured(load_config())

    clean_env.write_text(
        json.dumps({"base_url": "http://x", "api_key": "", "model_name": ""}),
        encoding="utf-8",
    )
    assert "MODEL_NAME" in check_configured(load_config())


def test_config_path_env_override(clean_env):
    assert config_path() == clean_env
