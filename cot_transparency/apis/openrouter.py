import json
import logging
import os
from collections.abc import Sequence
from typing import Any

import httpx
from dotenv import load_dotenv
from retry import retry

from cot_transparency.apis.base import InferenceResponse, ModelCaller
from cot_transparency.apis.util import convert_assistant_if_completion_to_assistant
from cot_transparency.data_models.config import OpenaiInferenceConfig
from cot_transparency.data_models.messages import (
    ChatMessage,
    StrictChatMessage,
    StrictMessageRole,
)
from cot_transparency.util import setup_logger

logger = setup_logger(__name__, logging.INFO)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_PREFIX = "openrouter/"


def _strip_openrouter_prefix(model_name: str) -> str:
    if model_name.startswith(OPENROUTER_PREFIX):
        return model_name[len(OPENROUTER_PREFIX):]
    return model_name


def _format_messages(messages: Sequence[ChatMessage]) -> list[dict[str, str]]:
    strict: list[StrictChatMessage] = convert_assistant_if_completion_to_assistant(messages)
    formatted: list[dict[str, str]] = []
    for msg in strict:
        match msg.role:
            case StrictMessageRole.user:
                formatted.append({"role": "user", "content": msg.content})
            case StrictMessageRole.assistant:
                formatted.append({"role": "assistant", "content": msg.content})
            case StrictMessageRole.system:
                formatted.append({"role": "system", "content": msg.content})
            case StrictMessageRole.none:
                raise ValueError(
                    "OpenRouter chat messages cannot have a None role; got message with StrictMessageRole.none"
                )
    return formatted


class OpenRouterRateLimitError(Exception):
    pass


class OpenRouterTransientError(Exception):
    pass


class OpenRouterCaller(ModelCaller):
    """Caller for OpenRouter's OpenAI-compatible chat completions API.

    Model names passed via OpenaiInferenceConfig.model must be prefixed with
    ``openrouter/`` (e.g. ``openrouter/google/gemma-4-31b-it``). The prefix is
    stripped before sending to the OpenRouter API.
    """

    def __init__(self, timeout: float = 120.0):
        load_dotenv()
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY not set — add it to .env or your environment "
                "before using openrouter/* models."
            )
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self._client = httpx.Client(timeout=timeout)

    def _build_payload(
        self,
        messages: Sequence[ChatMessage],
        config: OpenaiInferenceConfig,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": _strip_openrouter_prefix(config.model),
            "messages": _format_messages(messages),
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "n": config.n,
            "stream": False,
        }
        if config.top_p is not None:
            payload["top_p"] = config.top_p
        if config.frequency_penalty:
            payload["frequency_penalty"] = config.frequency_penalty
        if config.presence_penalty:
            payload["presence_penalty"] = config.presence_penalty
        if config.stop is not None:
            payload["stop"] = [config.stop] if isinstance(config.stop, str) else list(config.stop)
        return payload

    @retry(
        exceptions=(OpenRouterTransientError, httpx.ConnectError, httpx.ReadTimeout),
        tries=10,
        delay=2,
        backoff=2,
        max_delay=60,
        logger=logger,
    )
    @retry(exceptions=(OpenRouterRateLimitError,), tries=-1, delay=5, jitter=(0, 5), max_delay=60, logger=logger)
    def call(
        self,
        messages: Sequence[ChatMessage],
        config: OpenaiInferenceConfig,
        try_number: int = 1,
    ) -> InferenceResponse:
        del try_number
        payload = self._build_payload(messages, config)
        resp = self._client.post(OPENROUTER_BASE_URL, headers=self._headers, json=payload)
        if resp.status_code == 429:
            raise OpenRouterRateLimitError(f"OpenRouter 429: {resp.text[:500]}")
        if 500 <= resp.status_code < 600:
            raise OpenRouterTransientError(f"OpenRouter {resp.status_code}: {resp.text[:500]}")
        if resp.status_code >= 400:
            raise RuntimeError(f"OpenRouter {resp.status_code} for model={payload['model']}: {resp.text[:1000]}")

        try:
            body = resp.json()
        except json.JSONDecodeError as e:
            # OpenRouter can intermittently return non-JSON (SSE keepalive
            # comments, partial responses, provider-side errors). Treat as
            # transient so the outer retry decorator re-issues the request.
            preview = resp.text[:500].replace("\n", "\\n")
            raise OpenRouterTransientError(
                f"OpenRouter returned non-JSON body ({e}); first 500 chars: {preview}"
            )
        choices = body.get("choices")
        if not choices:
            # OpenRouter sometimes returns errors inside a 200 body
            err = body.get("error")
            if err:
                raise OpenRouterTransientError(f"OpenRouter error in body: {err}")
            raise RuntimeError(f"OpenRouter returned no choices: {body}")

        raw_responses: list[str] = []
        for choice in choices:
            message = choice.get("message") or {}
            content = message.get("content")
            if content is None:
                content = ""
            raw_responses.append(content)
        return InferenceResponse(raw_responses=raw_responses)
