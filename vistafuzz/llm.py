from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass


@dataclass
class LLMConfig:
    backend: str = "ollama"
    model: str = "qwen2.5-coder:32b"
    host: str = "http://localhost:11434"
    base_url: str = ""
    api_key: str = ""
    temperature: float = 0.0
    num_ctx: int = 8192
    max_tokens: int = 1600
    timeout_sec: float = 120.0

    @classmethod
    def from_env(cls, **overrides) -> "LLMConfig":
        cfg = cls(
            backend=os.environ.get("VISTAFUZZ_LLM_BACKEND", "ollama"),
            model=os.environ.get("VISTAFUZZ_LLM_MODEL", "qwen2.5-coder:32b"),
            host=os.environ.get("VISTAFUZZ_OLLAMA_HOST",
                                os.environ.get("OLLAMA_HOST", "http://localhost:11434")),
            base_url=os.environ.get("VISTAFUZZ_LLM_BASE", ""),
            api_key=os.environ.get("VISTAFUZZ_LLM_KEY", ""),
            timeout_sec=float(os.environ.get("VISTAFUZZ_LLM_TIMEOUT", "120")),
        )
        for key, value in overrides.items():
            if value not in (None, "", 0):
                setattr(cfg, key, value)
        return cfg


class LLMError(RuntimeError):
    pass


def _post(url: str, payload: dict, timeout: float, headers: dict | None = None) -> dict:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    for key, value in (headers or {}).items():
        req.add_header(key, value)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
        raise LLMError(f"{type(exc).__name__}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise LLMError(f"malformed response: {exc}") from exc


def complete(prompt: str, config: LLMConfig) -> str:
    if config.backend == "ollama":
        data = _post(
            config.host.rstrip("/") + "/api/generate",
            {
                "model": config.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": config.temperature,
                    "num_ctx": config.num_ctx,
                    "num_predict": config.max_tokens,
                },
            },
            config.timeout_sec,
        )
        return str(data.get("response", ""))

    if config.backend == "openai":
        base = (config.base_url or "https://api.openai.com").rstrip("/")
        data = _post(
            base + "/v1/chat/completions",
            {
                "model": config.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": config.temperature,
                "max_tokens": config.max_tokens,
            },
            config.timeout_sec,
            {"Authorization": f"Bearer {config.api_key}"} if config.api_key else {},
        )
        try:
            return str(data["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"unexpected response shape: {exc}") from exc

    raise LLMError(f"unknown backend {config.backend!r}")


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def extract_json_object(text: str) -> dict:
    if not text:
        raise LLMError("empty response")
    fenced = _FENCE.search(text)
    if fenced:
        text = fenced.group(1)
    start = text.find("{")
    if start < 0:
        raise LLMError("no JSON object in response")
    depth, in_str, esc = 0, False, False
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:idx + 1])
                except json.JSONDecodeError as exc:
                    raise LLMError(f"malformed JSON object: {exc}") from exc
    raise LLMError("unterminated JSON object")


def probe(config: LLMConfig) -> bool:
    try:
        complete("Reply with the single character: 1", config)
        return True
    except LLMError:
        return False
