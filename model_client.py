"""Provider configuration and transport for the model API."""

import json
import os
import time
from dataclasses import dataclass
from http.client import HTTPException
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import HTTPRedirectHandler, Request, build_opener
from uuid import uuid4

from model_log import ModelCallLog


@dataclass(frozen=True, slots=True)
class Provider:
    label: str
    endpoint: str
    key_env: str


PROVIDERS = {
    "openrouter": Provider("OpenRouter", "https://openrouter.ai/api/v1/chat/completions", "OPENROUTER_API_KEY"),
    "openai": Provider("OpenAI 直连", "https://api.openai.com/v1/chat/completions", "OPENAI_API_KEY"),
}
DECISIONS_ENDPOINT = "https://openrouter.ai/api/alpha/decisions"


def load_local_keys(path: Path) -> None:
    """Fill missing provider environment variables from a local .env file."""
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except FileNotFoundError:
        return

    allowed_keys = {provider.key_env for provider in PROVIDERS.values()}
    for line in lines:
        name, separator, value = line.partition("=")
        name, value = name.strip(), value.strip()
        if separator and name in allowed_keys and value and "\0" not in value:
            os.environ.setdefault(name, value)


@dataclass(frozen=True, slots=True)
class Model:
    provider: str
    model: str
    label: str
    image: bool
    protocol: str = "chat"

    @property
    def id(self) -> str:
        return f"{self.provider}:{self.model}"

    @property
    def available(self) -> bool:
        return bool(os.environ.get(PROVIDERS[self.provider].key_env, "").strip())

    def public(self) -> dict:
        provider = PROVIDERS[self.provider]
        return {
            "id": self.id,
            "model": self.model,
            "label": self.label,
            "provider": provider.label,
            "keyEnv": provider.key_env,
            "image": self.image,
            "protocol": self.protocol,
            "available": self.available,
        }


def load_models(path: Path) -> dict[str, Model]:
    entries = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(entries, list):
        raise ValueError("models.json 应包含模型列表。")
    models = {}
    for entry in entries:
        if (
            not isinstance(entry, dict)
            or not all(isinstance(entry.get(k), str) and entry[k].strip() for k in ("provider", "model", "label"))
            or type(entry.get("image")) is not bool
            or entry["provider"] not in PROVIDERS
        ):
            raise ValueError("models.json 中的模型需要有效的 provider、model、label 和 image。")
        protocol = entry.get("protocol", "chat")
        if protocol not in ("chat", "decisions") or (protocol == "decisions" and entry["provider"] != "openrouter"):
            raise ValueError("models.json 的 protocol 必须为 chat 或 decisions，decisions 需要使用 openrouter。")
        model = Model(entry["provider"], entry["model"], entry["label"], entry["image"], protocol)
        if model.id in models:
            raise ValueError(f"models.json 包含重复模型 {model.id}。")
        models[model.id] = model
    return models


RESPONSE_LIMIT = 1_000_000


@dataclass(slots=True)
class ApiTrace:
    endpoint: str
    request: str
    status: int | None = None
    response: str | None = None
    truncated: bool = False
    call_id: str | None = None


@dataclass(frozen=True, slots=True)
class Completion:
    text: str
    model: str
    elapsed_ms: int
    input_tokens: int | None
    output_tokens: int | None
    trace: ApiTrace | None = None


class ModelError(Exception):
    def __init__(self, message: str, elapsed_ms: int = 0, trace: ApiTrace | None = None):
        super().__init__(message)
        self.elapsed_ms = elapsed_ms
        self.trace = trace


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class ModelClient:
    def __init__(self, open_request=None, *, call_log: ModelCallLog | None = None):
        self.open_request = open_request if open_request is not None else build_opener(NoRedirect()).open
        self.call_log = call_log

    def complete(
        self, model: Model, instructions: str, observation: str, image: str | None, *, context: dict | None = None
    ) -> Completion:
        content = observation
        if image is not None:
            content = [
                {"type": "text", "text": observation},
                {"type": "image_url", "image_url": {"url": image}},
            ]
        payload = {
            "model": model.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": instructions},
                {"role": "user", "content": content},
            ],
        }
        return self._call(model, PROVIDERS[model.provider].endpoint, payload, context)

    def choose(
        self, model: Model, instructions: str, observation: str, image: str | None,
        criteria: dict[str, str], *, context: dict | None = None
    ) -> Completion:
        """Ask for a structured choice and expose the selected key as the result text."""
        if model.provider != "openrouter":
            raise ModelError("Decisions API 需要使用 OpenRouter 入口。")
        state = observation if image is None else [observation, {"type": "image_url", "image_url": {"url": image}}]
        payload = {
            "model": model.model,
            "state": state,
            "questions": {
                "move": {"type": "choice", "instructions": instructions, "criteria": criteria},
            },
        }
        return self._call(model, DECISIONS_ENDPOINT, payload, context, choices=tuple(criteria))

    def _call(
        self, model: Model, endpoint: str, payload: dict, context: dict | None, *,
        choices: tuple[str, ...] | None = None
    ) -> Completion:
        provider = PROVIDERS[model.provider]
        key = os.environ.get(provider.key_env, "").strip()
        if not key:
            raise ModelError(f"缺少 {provider.key_env}。")
        body = json.dumps(payload, ensure_ascii=False, indent=2)
        request = Request(
            endpoint,
            data=body.encode("utf-8"),
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            method="POST",
        )
        trace = ApiTrace(endpoint, redact_api_keys(body))
        if self.call_log is not None:
            trace.call_id = uuid4().hex
            self._record_call(trace, "request", {
                "provider": model.provider,
                "model": redact_api_keys(model.model),
                "context": context,
                "endpoint": trace.endpoint,
                "body": trace.request,
            })

        started = time.monotonic()
        error_message = None
        try:
            return self._request_completion(model, provider, request, trace, started, choices)
        except ModelError as error:
            error_message = str(error)
            raise
        finally:
            if self.call_log is not None:
                duration = elapsed_ms(started)
                self._record_call(trace, "response", {
                    "status": trace.status,
                    "body": trace.response,
                    "truncated": trace.truncated,
                    "elapsed_ms": duration,
                    "error": error_message,
                }, duration)

    def _record_call(self, trace: ApiTrace, event: str, fields: dict, duration: int = 0) -> None:
        try:
            self.call_log.append(trace.call_id, event, fields)
        except OSError:
            detail = "请求未发送。" if event == "request" else "响应正文仍可在走棋记录中查看。"
            raise ModelError(f"无法写入 log 目录的模型日志。{detail}", duration, trace) from None

    def _request_completion(
        self, model: Model, provider: Provider, request: Request, trace: ApiTrace, started: float,
        choices: tuple[str, ...] | None
    ) -> Completion:
        try:
            try:
                response = self.open_request(request, timeout=30)
            except HTTPError as error:
                response = error
            with response:
                trace.status = getattr(response, "status", None)
                data = response.read(RESPONSE_LIMIT + 1)
        except (OSError, HTTPException) as error:
            timeout = isinstance(error, TimeoutError) or isinstance(getattr(error, "reason", None), TimeoutError)
            detail = "请求超时" if timeout else "连接失败"
            raise ModelError(f"{provider.label}{detail}。", elapsed_ms(started), trace) from None

        duration = elapsed_ms(started)
        trace.truncated = len(data) > RESPONSE_LIMIT
        captured = data[:RESPONSE_LIMIT]
        trace.response = redact_api_keys(data.decode("utf-8", errors="replace"), truncated=trace.truncated)
        if trace.status is not None and not 200 <= trace.status < 300:
            detail = {
                400: "模型或输入参数不受支持。",
                401: "请检查此入口的 API 密钥。",
                402: "请检查此入口的余额。",
                403: "此密钥没有访问所选模型的权限。",
                404: "请检查配置中的模型 ID。",
                429: "请求受限，请稍后继续。",
            }.get(trace.status, "可以稍后按单步重试。")
            raise ModelError(f"{provider.label} 返回 HTTP {trace.status}。{detail}", duration, trace)

        try:
            payload = json.loads(captured)
        except (ValueError, RecursionError):
            raise ModelError(f"{provider.label} 返回了无法读取的响应。", duration, trace) from None
        if choices is None:
            try:
                output = payload["choices"][0]["message"]["content"]
            except (IndexError, KeyError, TypeError):
                output = None
            if not isinstance(output, str) or not output.strip():
                raise ModelError(f"{provider.label} 没有返回文本动作。", duration, trace)
            input_key, output_key = "prompt_tokens", "completion_tokens"
        else:
            try:
                answer = payload["answers"]["move"]
                output = answer["choice"] if answer["type"] == "choice" else None
            except (KeyError, TypeError):
                output = None
            if not isinstance(output, str) or output not in choices:
                raise ModelError(f"{provider.label} Decisions 没有返回可选动作。", duration, trace)
            input_key, output_key = "input_tokens", "output_tokens"
        model_used = payload.get("model") or model.model
        usage = payload.get("usage")
        usage = usage if isinstance(usage, dict) else {}
        return Completion(
            output,
            redact_api_keys(model_used) if isinstance(model_used, str) else model.model,
            duration,
            token_count(usage.get(input_key)),
            token_count(usage.get(output_key)),
            trace,
        )


def redact_api_keys(text: str, *, truncated: bool = False) -> str:
    keys = {os.environ.get(provider.key_env, "").strip() for provider in PROVIDERS.values()}
    keys.discard("")
    spellings = keys | {json.dumps(key, ensure_ascii=ascii_only)[1:-1] for key in keys for ascii_only in (True, False)}
    for key in sorted(spellings, key=len, reverse=True):
        text = text.replace(key, "[密钥已隐藏]")
    if truncated:
        # The read limit could split a credential before it can be matched.
        safe_length = max(0, len(text) - max(map(len, spellings), default=1))
        text = text[:safe_length]
    return text


def elapsed_ms(started: float) -> int:
    return round((time.monotonic() - started) * 1000)


def token_count(value) -> int | None:
    return value if type(value) is int and value >= 0 else None
