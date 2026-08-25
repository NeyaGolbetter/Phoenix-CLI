"""HTTP client for Phoenix CLI.

Talks the *OpenAI-compatible* Chat Completions protocol, which is supported
by Ollama, LM Studio, vLLM, OpenRouter, Together AI, Groq, DeepSeek and many
others. That is the only contract Phoenix CLI relies on, so any provider that
speaks it can be plugged in via the config file -- nothing is hardcoded.

Key design decisions:

* ``httpx.AsyncClient`` with the **streaming** chat-completions endpoint, so
  tokens are printed as the model produces them (great over slow mobile
  connections in Termux).
* Strict, fail-fast URL handling: a mistyped BASE_URL raises a clear
  ``ProviderError`` immediately instead of hanging for a minute.
* A small internal taxonomy (``ProviderError`` + subclasses) so the UI layer
  can print precise, actionable messages for the cases that actually happen:
  wrong URL, invalid API key, unknown model, rate limits, and network
  failures.
* A pure-Python tokenizer fallback used *only* to colorize pieces of code
  after the stream finishes. ``tiktoken`` is a nice-to-have that needs a Rust
  toolchain to build -- not Termux-friendly -- so it is optional.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class PhoenixError(Exception):
    """Base class for every error Phoenix CLI raises on purpose."""


class ProviderError(PhoenixError):
    """The remote provider rejected the request or something broke on the wire."""


class ConfigurationError(PhoenixError):
    """Local configuration is missing or invalid."""


class APIKeyError(ProviderError):
    """The provider refused our API key (HTTP 401/403)."""


class ModelNotFoundError(ProviderError):
    """The provider does not know the configured model (HTTP 404, model errors)."""


class RateLimitError(ProviderError):
    """The provider rate-limited us (HTTP 429) or asked us to back off."""


class NetworkError(ProviderError):
    """We could not reach the provider at all (DNS, refused, timeout, TLS)."""


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------


@dataclass
class Message:
    """A single chat message, OpenAI-style."""

    role: str
    content: str

    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class Conversation:
    """An in-memory conversation with automatic history trimming.

    The oldest messages are dropped once ``max_messages`` is exceeded, always
    keeping the system prompt and the final user message. This keeps memory
    flat during long chats -- which matters on a phone.
    """

    system: Optional[str] = None
    messages: List[Message] = field(default_factory=list)
    max_messages: int = 60

    def add(self, role: str, content: str) -> None:
        self.messages.append(Message(role=role, content=content))
        self._trim()

    def history(self) -> List[Dict[str, str]]:
        """Return the full OpenAI payload: [system, ...messages]."""
        out: List[Dict[str, str]] = []
        if self.system:
            out.append({"role": "system", "content": self.system})
        out.extend(m.to_dict() for m in self.messages)
        return out

    def clear(self) -> None:
        """Drop everything except the system prompt."""
        self.messages.clear()

    def _trim(self) -> None:
        """Keep the conversation inside ``max_messages`` entries."""
        if self.max_messages <= 0:
            return
        overflow = len(self.messages) - self.max_messages
        while overflow > 0:
            # Never evict the *last* message: it is the prompt we are about
            # to send.
            del self.messages[0]
            overflow -= 1


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

DEFAULT_TIMEOUT_SECONDS = 300.0


class PhoenixClient:
    """Async client for an OpenAI-compatible chat-completions endpoint.

    ``extra_headers`` exists so power users can pass provider-specific
    headers (e.g. ``X-Title`` for OpenRouter) without touching the core code.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model_name: str,
        extra_headers: Optional[Dict[str, str]] = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        if not base_url:
            raise ConfigurationError("BASE_URL is empty")
        if not model_name:
            raise ConfigurationError("MODEL_NAME is empty")

        # Reject clearly-invalid URLs before the first network round-trip.
        if "://" not in base_url or " " in base_url:
            raise ConfigurationError(
                f"BASE_URL must include a scheme, e.g. http://localhost:11434/v1 "
                f"(got: {base_url!r}). Did you mean to run `phoenix setup`?"
            )

        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model_name = model_name
        self.extra_headers = dict(extra_headers or {})
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    # -- lifecycle ----------------------------------------------------------

    async def open(self) -> None:
        """Create the underlying httpx client (call before chatting)."""
        headers = {"Content-Type": "application/json", **self.extra_headers}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout, connect=15.0),
            headers=headers,
            follow_redirects=True,
        )

    async def close(self) -> None:
        """Close the underlying httpx client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> "PhoenixClient":
        await self.open()
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        await self.close()

    # -- helpers ------------------------------------------------------------

    def _url(self, path: str) -> str:
        return f"{self.base_url}/{path.lstrip('/')}"

    def _payload(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: Optional[float],
        max_tokens: Optional[int],
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "stream": True,
        }
        # Only include optional knobs when the user actually set them: a few
        # providers reject `temperature: null`, and omitting is the safest.
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        return payload

    # -- the actual API call --------------------------------------------------

    async def chat_stream(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Yield parsed SSE deltas from the chat-completions stream.

        Yields dictionaries of the form ``{"content": str}`` (a content
        delta) or ``{"usage": {...}, "finish_reason": ...}`` (a final usage
        chunk, when the provider sends one). Raises one of the error classes
        above on failure.
        """
        if self._client is None:
            raise ProviderError("Client is not open; call `open()` first")

        payload = self._payload(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        try:
            async with self._client.stream(
                "POST", self._url("chat/completions"), json=payload
            ) as response:
                # --- HTTP-level errors: read the body for a useful message ---
                if response.status_code != 200:
                    await self._raise_for_response(response)
                await self._check_media_type(response)

                usage: Dict[str, Any] = {}
                finish_reason: Optional[str] = None
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    if line.startswith(":"):
                        continue  # SSE comment/keep-alive
                    if line.startswith("data:"):
                        data = line[len("data:"):].strip()
                        if data == "[DONE]":
                            break
                    else:
                        # Not an SSE frame: a few providers answer with a
                        # plain JSON body (full completion or error). Try to
                        # parse the line as-is instead of dropping it.
                        data = line.strip()
                        if not data:
                            continue
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        # Some servers interleave non-JSON noise; skip it
                        # rather than killing a long generation.
                        continue

                    if "usage" in chunk and chunk["usage"]:
                        usage = chunk["usage"]
                    if chunk.get("choices"):
                        choice = chunk["choices"][0]
                        finish_reason = choice.get("finish_reason") or finish_reason
                        delta = choice.get("delta") or {}
                        text = delta.get("content")
                        if not text:
                            # Fallback: some servers ignore `stream: true` and
                            # send one complete message instead of deltas.
                            text = (choice.get("message") or {}).get("content")
                        if text:
                            yield {"content": text}

                # vLLM sends `finish_reason` on the *last* chunk; emit a
                # final bookkeeping item so callers can show token usage.
                yield {"finish_reason": finish_reason, "usage": usage}
        except httpx.TimeoutException as exc:
            raise NetworkError(
                f"Request timed out after {self.timeout:.0f}s. The model may be "
                "slow, or the provider is overloaded."
            ) from exc
        except httpx.ConnectError as exc:
            raise NetworkError(
                f"Could not connect to {self.base_url}.\n"
                "  * Is the server running (Ollama / LM Studio / vLLM / ...)?\n"
                "  * Is the BASE_URL correct? Check it with `phoenix status`."
            ) from exc
        except httpx.TransportError as exc:
            raise NetworkError(f"Network error while talking to {self.base_url}: {exc}") from exc

    # -- model listing -------------------------------------------------------

    async def list_models(self) -> List[str]:
        """Return the model IDs advertised by the provider (GET ``/models``).

        OpenRouter, Groq, Together AI and most self-hosted stacks implement
        this endpoint; Ollama exposes it through its ``/v1`` compatibility
        layer. Raises ``ProviderError`` with a hint when the provider does
        not support model listing.
        """
        if self._client is None:
            raise ProviderError("Client is not open; call `open()` first")
        try:
            response = await self._client.get(self._url("models"))
        except httpx.TimeoutException as exc:
            raise NetworkError(
                f"Request timed out after {self.timeout:.0f}s while listing models."
            ) from exc
        except httpx.ConnectError as exc:
            raise NetworkError(
                f"Could not connect to {self.base_url}.\n"
                "  * Is the server running (Ollama / LM Studio / vLLM / ...)?\n"
                "  * Is the BASE_URL correct? Check it with `phoenix status`."
            ) from exc
        except httpx.TransportError as exc:
            raise NetworkError(
                f"Network error while talking to {self.base_url}: {exc}"
            ) from exc

        if response.status_code != 200:
            await self._raise_models_error(response)

        try:
            data = response.json()
        except ValueError as exc:
            raise ProviderError(
                "The models endpoint answered with non-JSON data. Is BASE_URL "
                "pointing at the provider's OpenAI-compatible endpoint?"
            ) from exc

        ids: List[str] = []
        if isinstance(data, dict):
            for item in data.get("data") or []:
                if isinstance(item, dict) and item.get("id"):
                    ids.append(str(item["id"]))
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, str):
                    ids.append(item)
                elif isinstance(item, dict) and item.get("id"):
                    ids.append(str(item["id"]))
        return sorted(set(ids))

    async def _raise_models_error(self, response: httpx.Response) -> None:
        """Convert a non-200 ``/models`` response into a precise error."""
        status = response.status_code
        detail = self._extract_error_message(response.text[:500])
        if status in (401, 403):
            raise APIKeyError(
                f"Authentication failed (HTTP {status}). Your API key was rejected.\n"
                "Run `phoenix setup` to update it.\n"
                f"Provider said: {detail}"
            )
        if status == 429:
            raise RateLimitError(
                "Rate limited by the provider (HTTP 429). Wait a moment and retry."
            )
        if status >= 500:
            raise ProviderError(
                f"The provider had a server error (HTTP {status}): {detail}"
            )
        if status == 404:
            raise ProviderError(
                "This provider does not expose a model list (GET .../models "
                "answered HTTP 404).\n"
                "Tips:\n"
                "  * local Ollama: run `ollama list` in Termux instead\n"
                "  * cloud providers: check their docs / model dashboard\n"
                "  * make sure BASE_URL points at the `/v1` endpoint"
            )
        raise ProviderError(f"Unexpected response (HTTP {status}): {detail}")

    async def _check_media_type(self, response: httpx.Response) -> None:
        """Fail fast with a helpful message when the URL is not an API."""
        ctype = response.headers.get("content-type", "")
        if ctype and "text/html" in ctype:
            await response.aread()  # streaming response: read before .text
            body = response.text[:200]
            raise ProviderError(
                "The server answered with an HTML page, not the OpenAI API.\n"
                f"URL used: {self._url('chat/completions')}\n"
                "Common fixes:\n"
                "  * add `/v1` to BASE_URL (Ollama needs it: http://localhost:11434/v1)\n"
                "  * point BASE_URL at the provider's OpenAI-compatible endpoint\n"
                f"Sample response: {body!r}"
            )

    async def _raise_for_response(self, response: httpx.Response) -> None:
        """Convert a non-200 response into a precise, actionable error."""
        status = response.status_code
        text = ""
        try:
            await response.aread()  # streaming response: read before .text
            text = response.text[:500]
        except Exception:  # pragma: no cover - defensive
            pass
        detail = self._extract_error_message(text)

        if status in (401, 403):
            raise APIKeyError(
                "Authentication failed (HTTP "
                f"{status}). Your API key was rejected.\n"
                "Run `phoenix setup` to update it.\n"
                f"Provider said: {detail}"
            )
        if status == 404:
            # Some gateways return 404 for an unknown *model* too, so the
            # message covers both cases.
            raise ModelNotFoundError(
                f"The endpoint or model was not found (HTTP 404).\n"
                f"  endpoint: {self._url('chat/completions')}\n"
                f"  model:    {self.model_name}\n"
                "Check BASE_URL and MODEL_NAME with `phoenix setup`."
            )
        if status == 429:
            raise RateLimitError(
                "Rate limited by the provider (HTTP 429). Wait a moment and retry."
            )
        if status >= 500:
            raise ProviderError(
                f"The provider had a server error (HTTP {status}): {detail}"
            )
        if status == 400 and self._looks_like_model_error(detail):
            raise ModelNotFoundError(
                f"The model {self.model_name!r} was rejected by the provider "
                f"(HTTP 400): {detail}"
            )
        raise ProviderError(f"Unexpected response (HTTP {status}): {detail}")

    @staticmethod
    def _extract_error_message(body: str) -> str:
        """Pull a readable message out of an OpenAI-style error body."""
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            return body.strip() or "(empty response body)"
        error = data.get("error")
        if isinstance(error, str):
            return error
        if isinstance(error, dict):
            return str(error.get("message") or error)
        return body.strip() or "(empty response body)"

    @staticmethod
    def _looks_like_model_error(detail: str) -> bool:
        """Heuristic for providers that answer 400 for unknown models."""
        needle = re.compile(r"model", re.IGNORECASE)
        patterns = (
            r"not found", r"does not exist", r"no such", r"doesn't exist",
            r"unknown model", r"invalid model", r"is not available",
        )
        return bool(needle.search(detail)) and any(
            re.search(p, detail, re.IGNORECASE) for p in patterns
        )
