"""Configuration management for Phoenix CLI.

The config lives in ``~/.phoenix_config.json`` and holds the three values that
make Phoenix CLI provider-agnostic:

* ``base_url``  - the OpenAI-compatible endpoint (Ollama, LM Studio, vLLM,
                  OpenRouter, Together, Groq, DeepSeek, ...)
* ``api_key``   - the provider's API key (can be anything for local servers)
* ``model_name``- the model identifier the provider understands

Everything here is plain stdlib so configuration never depends on network
access or optional packages.
"""

from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path
from typing import Any, Dict

# Name of the config file in the user's home directory.
CONFIG_FILENAME = ".phoenix_config.json"

# Environment variables that can be used INSTEAD of the config file. They are
# checked first, so an exported value always wins. This also makes it easy to
# keep secrets out of files entirely.
ENV_BASE_URL = "PHOENIX_BASE_URL"
ENV_API_KEY = "PHOENIX_API_KEY"
ENV_MODEL = "PHOENIX_MODEL"

# Optional: point Phoenix CLI at a different config file.
ENV_CONFIG_PATH = "PHOENIX_CONFIG"


def config_path() -> Path:
    """Return the full path to the config file (~/.phoenix_config.json).

    Honors the ``PHOENIX_CONFIG`` environment variable, which is handy for
    tests and for keeping several provider profiles around.
    """
    override = os.environ.get(ENV_CONFIG_PATH)
    if override:
        return Path(override).expanduser()
    return Path.home() / CONFIG_FILENAME


def normalize_base_url(raw: str) -> str:
    """Clean up a user-supplied base URL.

    Strips whitespace and trailing slashes. If the URL has *no path at all*
    (e.g. ``http://localhost:11434``), the OpenAI-compatibility suffix ``/v1``
    is appended. Explicit paths are always respected as given, so
    ``https://api.openrouter.ai/api/v1`` or ``https://api.groq.com/openai/v1``
    are never rewritten.
    """
    url = (raw or "").strip().rstrip("/")
    if not url:
        return url
    rest = url.split("://", 1)[-1]
    path = rest.split("/", 1)[1] if "/" in rest else ""
    if not path:
        url += "/v1"
    return url


def load_config() -> Dict[str, str]:
    """Read the configuration.

    Resolution order (highest priority first):

    1. Environment variables (``PHOENIX_BASE_URL``, ``PHOENIX_API_KEY``,
       ``PHOENIX_MODEL``).
    2. ``~/.phoenix_config.json``.

    Returns a dict that always contains the keys ``base_url``, ``api_key`` and
    ``model_name`` (possibly empty strings when unset). Never raises.
    """
    cfg: Dict[str, str] = {
        "base_url": os.environ.get(ENV_BASE_URL, ""),
        "api_key": os.environ.get(ENV_API_KEY, ""),
        "model_name": os.environ.get(ENV_MODEL, ""),
    }
    if os.environ.get(ENV_BASE_URL):
        cfg["base_url"] = normalize_base_url(cfg["base_url"])

    path = config_path()
    if not path.is_file():
        return cfg

    # Fail soft: a corrupt config should never crash the whole app. The user
    # can always run `phoenix setup` to overwrite it.
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return cfg

    if isinstance(raw, dict):
        for key in ("base_url", "api_key", "model_name"):
            value = raw.get(key)
            if isinstance(value, str) and value.strip() and not cfg[key]:
                if key == "base_url":
                    cfg[key] = normalize_base_url(value)
                else:
                    cfg[key] = value.strip()

    return cfg


def save_config(
    base_url: str,
    api_key: str,
    model_name: str,
    extra_headers: Dict[str, str] | None = None,
) -> Path:
    """Write the configuration to ``~/.phoenix_config.json``.

    ``extra_headers`` is stored for backwards/forwards compatibility with
    future Phoenix CLI releases but is not used by the 1.0 client.
    """
    path = config_path()
    payload: Dict[str, Any] = {
        "base_url": base_url,
        "api_key": api_key,
        "model_name": model_name,
    }
    if extra_headers:
        payload["extra_headers"] = extra_headers

    # Write atomically (temp file + rename) so an interrupted write never
    # leaves a half-written config behind.
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    # Restrict permissions on POSIX systems (Termux included): the API key is
    # a secret, so keep the file readable only by the owner when possible.
    try:
        if hasattr(os, "chmod"):
            os.chmod(tmp_path, stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        pass
    os.replace(tmp_path, path)
    return path


def check_configured(cfg: Dict[str, str]) -> str | None:
    """Return a human-readable error message if the config is unusable.

    Returns ``None`` when everything needed is present.
    """
    if not cfg.get("base_url"):
        return (
            "No BASE_URL configured.\n"
            "Run `phoenix setup` to configure Phoenix CLI, or export "
            f"{ENV_BASE_URL}."
        )
    if not cfg.get("model_name"):
        return (
            "No MODEL_NAME configured.\n"
            "Run `phoenix setup` to configure Phoenix CLI, or export "
            f"{ENV_MODEL}."
        )
    return None


def mask_secret(secret: str) -> str:
    """Return a masked version of an API key for safe display."""
    if not secret:
        return "(none)"
    if len(secret) <= 4:
        return "****"
    return secret[:3] + "****" + secret[-2:]


def main() -> int:  # pragma: no cover - kept for symmetry with cli.py
    """Thin guard so the module can run standalone (`python config.py`)."""
    import argparse

    parser = argparse.ArgumentParser(description="Phoenix CLI config module")
    parser.parse_args()
    print(f"Config file location: {config_path()}")
    print(f"Configured: {check_configured(load_config()) is None}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
