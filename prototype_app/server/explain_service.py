"""Server-side proxy to OpenAI-compatible chat-completion backends."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass

from settings import Settings

HF_ENDPOINT = "https://router.huggingface.co/v1/chat/completions"
OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
REQUEST_TIMEOUT_SECONDS = 60
TEMPERATURE = 0.2
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
)


class ExplainUnavailableError(RuntimeError):
    pass


class ExplainBackendError(RuntimeError):
    pass


@dataclass(frozen=True)
class ChatBackend:
    provider: str
    label: str
    endpoint: str
    api_key: str
    model: str


def select_backend(settings: Settings, requested_model: str | None) -> ChatBackend:
    if settings.hf_token:
        use_requested = requested_model and "medgemma" in requested_model.lower()
        return ChatBackend(
            provider="huggingface",
            label="Hugging Face Inference",
            endpoint=HF_ENDPOINT,
            api_key=settings.hf_token,
            model=requested_model if use_requested else settings.hf_medgemma_model,
        )
    if settings.openrouter_api_key:
        return ChatBackend(
            provider="openrouter",
            label="OpenRouter",
            endpoint=OPENROUTER_ENDPOINT,
            api_key=settings.openrouter_api_key,
            model=requested_model or settings.openrouter_model,
        )
    raise ExplainUnavailableError(
        "Server has neither HF_TOKEN nor OPENROUTER_API_KEY configured (.env)"
    )


def complete(backend: ChatBackend, prompt: str) -> str:
    payload = {
        "model": backend.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": TEMPERATURE,
    }
    request = urllib.request.Request(
        backend.endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {backend.api_key}",
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", "ignore")
        raise ExplainBackendError(f"{backend.label} HTTP {error.code}: {detail}") from error
    except Exception as error:
        raise ExplainBackendError(str(error)) from error
    return data["choices"][0]["message"]["content"].strip()
