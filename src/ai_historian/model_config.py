"""Shared, strict model configuration for every AI Historian entry point."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Mapping
from urllib.parse import urlparse

from openai import AsyncOpenAI, OpenAI


@dataclass(frozen=True)
class ProviderSpec:
    api_key_env: str
    base_url_env: str
    default_base_url: str
    supports_json_object: bool = True


@dataclass(frozen=True)
class ChatConfig:
    provider: str
    model: str
    api_key: str
    api_key_env: str
    base_url: str
    base_url_env: str


@dataclass(frozen=True)
class EmbeddingConfig:
    model: str
    api_key: str
    base_url: str


PROVIDER_SPECS: Mapping[str, ProviderSpec] = {
    "openai": ProviderSpec(
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "https://api.openai.com/v1",
    ),
    "anthropic": ProviderSpec(
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_BASE_URL",
        "https://api.anthropic.com/v1/",
        supports_json_object=False,
    ),
    "deepseek": ProviderSpec(
        "DEEPSEEK_API_KEY",
        "DEEPSEEK_BASE_URL",
        "https://api.deepseek.com",
    ),
    "gemini": ProviderSpec(
        "GEMINI_API_KEY",
        "GEMINI_BASE_URL",
        "https://generativelanguage.googleapis.com/v1beta/openai/",
    ),
    "dashscope": ProviderSpec(
        "DASHSCOPE_API_KEY",
        "DASHSCOPE_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    ),
    "compatible": ProviderSpec(
        "AIH_COMPATIBLE_API_KEY",
        "AIH_COMPATIBLE_BASE_URL",
        "",
    ),
}

SUPPORTED_PROVIDERS = tuple(PROVIDER_SPECS)


def _environment(environ: Mapping[str, str] | None = None) -> Mapping[str, str]:
    return os.environ if environ is None else environ


def _value(name: str, environ: Mapping[str, str] | None = None) -> str:
    return str(_environment(environ).get(name, "")).strip()


def chat_provider(environ: Mapping[str, str] | None = None, *, required: bool = False) -> str:
    provider = _value("AIH_CHAT_PROVIDER", environ).lower()
    if not provider:
        if required:
            raise RuntimeError("AIH_CHAT_PROVIDER is required.")
        return ""
    if provider not in PROVIDER_SPECS:
        supported = ", ".join(SUPPORTED_PROVIDERS)
        raise RuntimeError(
            f"Unsupported AIH_CHAT_PROVIDER={provider!r}. Choose one of: {supported}."
        )
    return provider


def chat_model(environ: Mapping[str, str] | None = None) -> str:
    return _value("AIH_CHAT_MODEL", environ)


def chat_base_url(
    environ: Mapping[str, str] | None = None,
    *,
    provider: str | None = None,
) -> str:
    selected = provider or chat_provider(environ)
    if not selected:
        return ""
    spec = PROVIDER_SPECS[selected]
    return _value(spec.base_url_env, environ) or spec.default_base_url


def resolve_chat_config(
    environ: Mapping[str, str] | None = None,
    *,
    require_api_key: bool = True,
    require_model: bool = True,
    require_base_url: bool = True,
) -> ChatConfig:
    selected = chat_provider(environ, required=True)
    spec = PROVIDER_SPECS[selected]
    model = chat_model(environ)
    api_key = _value(spec.api_key_env, environ)
    base_url = chat_base_url(environ, provider=selected)

    if require_model and not model:
        raise RuntimeError("AIH_CHAT_MODEL is required; set the exact model ID for this run.")
    if require_api_key and not api_key:
        raise RuntimeError(
            f"{spec.api_key_env} is required when AIH_CHAT_PROVIDER={selected}."
        )
    if require_base_url and not base_url:
        raise RuntimeError(
            f"{spec.base_url_env} is required when AIH_CHAT_PROVIDER={selected}."
        )

    return ChatConfig(
        provider=selected,
        model=model,
        api_key=api_key,
        api_key_env=spec.api_key_env,
        base_url=base_url,
        base_url_env=spec.base_url_env,
    )


def resolve_chat_api_key(
    required: bool = True,
    environ: Mapping[str, str] | None = None,
) -> tuple[str, str]:
    config = resolve_chat_config(
        environ,
        require_api_key=required,
        require_model=False,
    )
    return config.api_key_env, config.api_key


def chat_client_kwargs(environ: Mapping[str, str] | None = None) -> dict[str, Any]:
    config = resolve_chat_config(environ)
    timeout = float(_value("AIH_REQUEST_TIMEOUT", environ) or "300")
    return {
        "api_key": config.api_key,
        "base_url": config.base_url,
        "timeout": timeout,
    }


def make_sync_chat_client() -> OpenAI:
    return OpenAI(**chat_client_kwargs())


def make_async_chat_client() -> AsyncOpenAI:
    return AsyncOpenAI(**chat_client_kwargs())


def embedding_model(environ: Mapping[str, str] | None = None) -> str:
    return _value("AIH_EMBED_MODEL", environ)


def embedding_base_url(environ: Mapping[str, str] | None = None) -> str:
    return _value("AIH_EMBED_BASE_URL", environ)


def resolve_embedding_config(
    environ: Mapping[str, str] | None = None,
    *,
    required: bool = True,
) -> EmbeddingConfig:
    model = embedding_model(environ)
    api_key = _value("AIH_EMBED_API_KEY", environ)
    base_url = embedding_base_url(environ)
    if required:
        missing = [
            name
            for name, value in (
                ("AIH_EMBED_API_KEY", api_key),
                ("AIH_EMBED_BASE_URL", base_url),
                ("AIH_EMBED_MODEL", model),
            )
            if not value
        ]
        if missing:
            raise RuntimeError(
                "Embedding retrieval requires: " + ", ".join(missing) + "."
            )
    return EmbeddingConfig(model=model, api_key=api_key, base_url=base_url)


def resolve_embedding_api_key(
    required: bool = True,
    environ: Mapping[str, str] | None = None,
) -> tuple[str, str]:
    config = resolve_embedding_config(environ, required=required)
    return "AIH_EMBED_API_KEY", config.api_key


def make_embedding_client() -> OpenAI:
    config = resolve_embedding_config(required=True)
    return OpenAI(api_key=config.api_key, base_url=config.base_url)


def is_local_openai_compatible_base_url(base_url: str) -> bool:
    hostname = (urlparse(str(base_url).strip()).hostname or "").lower()
    return hostname in {"127.0.0.1", "localhost", "0.0.0.0"}


def supports_json_object(provider: str | None = None) -> bool:
    selected = provider or chat_provider(required=True)
    return PROVIDER_SPECS[selected].supports_json_object


def prepare_chat_request_kwargs(**kwargs: Any) -> dict[str, Any]:
    request_kwargs = dict(kwargs)
    provider = chat_provider(required=True)
    if not PROVIDER_SPECS[provider].supports_json_object:
        request_kwargs.pop("response_format", None)
    if provider == "dashscope":
        extra_body = dict(request_kwargs.pop("extra_body", {}) or {})
        extra_body.setdefault("enable_thinking", False)
        request_kwargs["extra_body"] = extra_body
    return request_kwargs


def _is_openai_parameter_error(exc: Exception, *needles: str) -> bool:
    text = str(exc).lower()
    return all(needle.lower() in text for needle in needles)


def _is_gpt5_model(model: Any) -> bool:
    normalized = str(model or "").lower().replace("-", "")
    return normalized.startswith("gpt5")


def _ensure_gpt5_completion_room(kwargs: dict[str, Any]) -> None:
    if not _is_gpt5_model(kwargs.get("model", "")):
        return
    if "max_completion_tokens" in kwargs:
        kwargs["max_completion_tokens"] = max(int(kwargs["max_completion_tokens"]), 1024)


def _adapt_chat_kwargs_after_error(kwargs: dict[str, Any], exc: Exception) -> bool:
    if "max_tokens" in kwargs and _is_openai_parameter_error(
        exc, "max_tokens", "max_completion_tokens"
    ):
        kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")
        _ensure_gpt5_completion_room(kwargs)
        return True
    if "temperature" in kwargs and _is_openai_parameter_error(exc, "temperature", "default"):
        kwargs.pop("temperature", None)
        return True
    if "max_completion_tokens" in kwargs and _is_openai_parameter_error(exc, "output limit"):
        kwargs["max_completion_tokens"] = max(int(kwargs["max_completion_tokens"]) * 2, 512)
        return True
    if "response_format" in kwargs and _is_openai_parameter_error(exc, "response_format"):
        kwargs.pop("response_format", None)
        return True
    return False


def create_chat_completion(client: OpenAI, **kwargs: Any):
    call_kwargs = prepare_chat_request_kwargs(**kwargs)
    if _is_gpt5_model(call_kwargs.get("model", "")):
        if "max_tokens" in call_kwargs:
            call_kwargs["max_completion_tokens"] = call_kwargs.pop("max_tokens")
        call_kwargs.pop("temperature", None)
        call_kwargs.setdefault("max_completion_tokens", 1024)
        _ensure_gpt5_completion_room(call_kwargs)
    for _ in range(4):
        try:
            return client.chat.completions.create(**call_kwargs)
        except Exception as exc:
            if not _adapt_chat_kwargs_after_error(call_kwargs, exc):
                raise
    return client.chat.completions.create(**call_kwargs)


async def create_chat_completion_async(client: AsyncOpenAI, **kwargs: Any):
    call_kwargs = prepare_chat_request_kwargs(**kwargs)
    if _is_gpt5_model(call_kwargs.get("model", "")):
        if "max_tokens" in call_kwargs:
            call_kwargs["max_completion_tokens"] = call_kwargs.pop("max_tokens")
        call_kwargs.pop("temperature", None)
        call_kwargs.setdefault("max_completion_tokens", 1024)
        _ensure_gpt5_completion_room(call_kwargs)
    for _ in range(4):
        try:
            return await client.chat.completions.create(**call_kwargs)
        except Exception as exc:
            if not _adapt_chat_kwargs_after_error(call_kwargs, exc):
                raise
    return await client.chat.completions.create(**call_kwargs)


def extract_json_object(text: str) -> str:
    stripped = (text or "").strip()
    if not stripped:
        raise ValueError("The model returned empty content.")
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    left = stripped.find("{")
    right = stripped.rfind("}")
    if left != -1 and right > left:
        return stripped[left : right + 1]
    raise ValueError(f"Could not extract a JSON object from model output: {stripped[:200]}")


def load_json_object(text: str) -> dict[str, Any]:
    payload = json.loads(extract_json_object(text))
    if not isinstance(payload, dict):
        raise ValueError("The model output is not a JSON object.")
    return payload


def validate_json_text(schema_model, text: str):
    payload = load_json_object(text)
    if hasattr(schema_model, "model_validate"):
        return schema_model.model_validate(payload)
    return schema_model.parse_obj(payload)


CHAT_PROVIDER = _value("AIH_CHAT_PROVIDER").lower()
CHAT_MODEL = chat_model()
CHAT_BASE_URL = (
    chat_base_url(provider=CHAT_PROVIDER) if CHAT_PROVIDER in PROVIDER_SPECS else ""
)
EMBED_MODEL = embedding_model()
EMBED_BASE_URL = embedding_base_url()
