from __future__ import annotations

import base64
import binascii
import json
import re
import zlib
from typing import Any

from comfyui_client import GenerationParams
from core.storage import dict_to_params, params_to_dict

PROMPT_EXCHANGE_VERSION = 1
PROMPT_EXCHANGE_PREFIX = f"CBOT_PROMPT_V{PROMPT_EXCHANGE_VERSION}:"
PROMPT_EXCHANGE_TOKEN_RE = re.compile(
    r"CBOT_PROMPT_V(?P<version>\d+):(?P<payload>[A-Za-z0-9_-]+)"
)
PROMPT_EXCHANGE_MAX_BYTES = 64_000


class PromptExchangeError(ValueError):
    pass


def export_prompt_token(params: GenerationParams) -> str:
    payload = {
        "v": PROMPT_EXCHANGE_VERSION,
        "params": params_to_dict(params),
    }
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    compressed = zlib.compress(raw, level=9)
    encoded = base64.urlsafe_b64encode(compressed).decode("ascii").rstrip("=")
    return f"{PROMPT_EXCHANGE_PREFIX}{encoded}"


def import_prompt_token(text: str) -> GenerationParams:
    payload = _payload_from_text(text)
    params_payload = _extract_params_payload(payload)
    try:
        return dict_to_params(params_payload)
    except Exception as exc:
        raise PromptExchangeError(
            "Не удалось прочитать параметры из кода обмена."
        ) from exc


def _payload_from_text(text: str) -> dict[str, Any]:
    source = (text or "").strip()
    if not source:
        raise PromptExchangeError("Код обмена пустой.")

    token_match = PROMPT_EXCHANGE_TOKEN_RE.search(source)
    if token_match:
        return _decode_token_payload(token_match)

    json_payload = _try_parse_json_payload(source)
    if json_payload is not None:
        return json_payload

    raise PromptExchangeError(
        "Код обмена не найден. Вставьте код вида CBOT_PROMPT_V1:..."
    )


def _decode_token_payload(token_match: re.Match) -> dict[str, Any]:
    try:
        version = int(token_match.group("version"))
    except Exception as exc:
        raise PromptExchangeError("Не удалось распознать версию кода обмена.") from exc

    if version != PROMPT_EXCHANGE_VERSION:
        raise PromptExchangeError(f"Неподдерживаемая версия кода обмена: v{version}.")

    encoded = token_match.group("payload")
    padded = encoded + "=" * (-len(encoded) % 4)

    try:
        compressed = base64.urlsafe_b64decode(padded)
    except (binascii.Error, ValueError) as exc:
        raise PromptExchangeError(
            "Код обмена повреждён: ошибка base64-декодирования."
        ) from exc

    try:
        raw = zlib.decompress(compressed)
    except zlib.error as exc:
        raise PromptExchangeError("Код обмена повреждён: ошибка распаковки.") from exc

    if len(raw) > PROMPT_EXCHANGE_MAX_BYTES:
        raise PromptExchangeError("Код обмена слишком большой.")

    try:
        payload = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise PromptExchangeError("Код обмена повреждён: некорректный JSON.") from exc

    if not isinstance(payload, dict):
        raise PromptExchangeError("Код обмена имеет неверный формат.")

    payload_version = int(payload.get("v") or PROMPT_EXCHANGE_VERSION)
    if payload_version != PROMPT_EXCHANGE_VERSION:
        raise PromptExchangeError(
            f"Неподдерживаемая версия данных: v{payload_version}."
        )

    return payload


def _try_parse_json_payload(source: str) -> dict[str, Any] | None:
    candidate = _extract_json_candidate(source)
    if not candidate:
        return None

    try:
        payload = json.loads(candidate)
    except Exception:
        return None

    return payload if isinstance(payload, dict) else None


def _extract_json_candidate(source: str) -> str:
    text = source.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            text = "\n".join(lines[1:-1]).strip()

    if text.startswith("{") and text.endswith("}"):
        return text

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return text[start : end + 1]
    return ""


def _extract_params_payload(payload: dict[str, Any]) -> dict[str, Any]:
    params_payload = payload.get("params") if isinstance(payload, dict) else None
    if isinstance(params_payload, dict):
        return params_payload

    likely_params_keys = {
        "positive",
        "negative",
        "checkpoint",
        "width",
        "height",
    }
    if any(key in payload for key in likely_params_keys):
        return payload

    raise PromptExchangeError("В коде обмена отсутствуют параметры генерации (params).")
