"""Tests for phoenix_cli.client against a live mock server."""

from __future__ import annotations

import asyncio

import httpx
import pytest

from phoenix_cli.client import (
    APIKeyError,
    ConfigurationError,
    Conversation,
    ModelNotFoundError,
    NetworkError,
    PhoenixClient,
    ProviderError,
    RateLimitError,
)

MOCK_REPLY_FRAGMENT = "Hello from the **mock provider**!"


def _collect(params, **kwargs):
    """Run one streamed request and return (chunks, final_chunk)."""

    async def run():
        client = PhoenixClient(**params)
        chunks = []
        final = None
        try:
            async with client:
                async for item in client.chat_stream(
                    [{"role": "user", "content": "hi"}], **kwargs
                ):
                    if "content" in item:
                        chunks.append(item["content"])
                    else:
                        final = item
        finally:
            await client.close()
        return chunks, final

    return asyncio.run(run())


def make_client(mock_api, model="ok", api_key=""):
    return {
        "base_url": mock_api,
        "api_key": api_key,
        "model_name": model,
    }


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


def test_stream_collects_full_reply(mock_api):
    chunks, final = _collect(make_client(mock_api))
    full = "".join(chunks)
    assert "Hello from the **mock provider**!" in full
    assert "```python" in full
    assert full.rstrip().endswith("out.")
    assert final is not None
    assert final["finish_reason"] is None  # mock does not send finish_reason


def test_usage_chunk_is_reported(mock_api):
    _, final = _collect(make_client(mock_api, model="usage"))
    assert final["usage"]["total_tokens"] > 0


def test_non_streaming_server_is_tolerated(mock_api):
    chunks, _ = _collect(make_client(mock_api, model="nonstream"))
    assert "mock provider" in "".join(chunks)


def test_custom_headers_and_key_are_sent(mock_api):
    chunks, _ = _collect(make_client(mock_api, model="echo", api_key="good-key"))
    assert "".join(chunks) == "hi"


# ---------------------------------------------------------------------------
# Error mapping
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model,error",
    [
        ("fail-auth", APIKeyError),
        ("fail-model", ModelNotFoundError),
        ("fail-model-400", ModelNotFoundError),
        ("fail-rate", RateLimitError),
        ("fail-server", ProviderError),
        ("fail-html", ProviderError),
    ],
)
def test_error_mapping(mock_api, model, error):
    with pytest.raises(error):
        _collect(make_client(mock_api, model=model))


def test_bad_key_raises_api_error(mock_api):
    with pytest.raises(APIKeyError) as excinfo:
        _collect(make_client(mock_api, api_key="bad-key"))
    assert "API key" in str(excinfo.value)


def test_html_response_mentions_endpoint(mock_api):
    with pytest.raises(ProviderError) as excinfo:
        _collect(make_client(mock_api, model="fail-html"))
    assert "HTML" in str(excinfo.value)


def test_connection_refused_raises_network_error():
    # Grab a port and close it so nothing is listening.
    import socket

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()

    with pytest.raises(NetworkError):
        _collect(make_client(f"http://127.0.0.1:{port}/v1"))


def test_missing_base_url_rejected():
    with pytest.raises(ConfigurationError):
        _collect({"base_url": "", "api_key": "", "model_name": "x"})


def test_url_with_spaces_rejected():
    with pytest.raises(ConfigurationError):
        _collect({"base_url": "http://bad url", "api_key": "", "model_name": "x"})


def test_url_without_scheme_is_accepted():
    """URLs without an explicit scheme should be normalized (http:// added)."""
    # We just verify it doesn't raise a ConfigurationError for missing scheme.
    # The actual connection will fail, but that's a NetworkError, not a config error.
    import socket
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()

    with pytest.raises(NetworkError):
        _collect({"base_url": f"127.0.0.1:{port}", "api_key": "", "model_name": "x"})


# ---------------------------------------------------------------------------
# Model listing
# ---------------------------------------------------------------------------


def test_list_models_returns_sorted_ids(mock_api):
    async def run():
        client = PhoenixClient(base_url=mock_api, api_key="", model_name="ok")
        try:
            async with client:
                return await client.list_models()
        finally:
            await client.close()

    ids = asyncio.run(run())
    assert "echo" in ids
    assert "qwen2.5-coder" in ids
    assert ids == sorted(set(ids))


def test_list_models_bad_key_raises_api_error(mock_api):
    async def run():
        client = PhoenixClient(base_url=mock_api, api_key="bad-key", model_name="ok")
        try:
            async with client:
                return await client.list_models()
        finally:
            await client.close()

    with pytest.raises(APIKeyError):
        asyncio.run(run())


# ---------------------------------------------------------------------------
# Conversation trimming
# ---------------------------------------------------------------------------


def test_conversation_trims_oldest_but_keeps_last():
    conv = Conversation(system="be brief", max_messages=10)
    conv.add("user", "first")
    for i in range(12):
        conv.add("assistant", f"reply {i}")
    conv.add("user", "final prompt")
    history = conv.history()
    assert history[0] == {"role": "system", "content": "be brief"}
    assert len(history) == 1 + 10  # system + 10 messages
    assert history[-1] == {"role": "user", "content": "final prompt"}


def test_conversation_clear_keeps_system():
    conv = Conversation(system="be brief")
    conv.add("user", "x")
    conv.add("assistant", "y")
    conv.clear()
    assert conv.history() == [{"role": "system", "content": "be brief"}]
