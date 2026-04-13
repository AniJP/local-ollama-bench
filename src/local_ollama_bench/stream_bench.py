from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass(frozen=True)
class StreamMetrics:
    """Single completed /api/chat streaming run (client-observed)."""

    ttft_ms: float | None
    total_ms: float
    decode_tps: float | None
    e2e_tps: float | None
    eval_count: int | None
    prompt_eval_count: int | None
    done_reason: str | None
    error: str | None = None


def _parse_sse_line(line: str) -> dict[str, Any] | None:
    line = line.strip()
    if not line:
        return None
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return None


def bench_chat_stream(
    *,
    base_url: str,
    model: str,
    user_prompt: str,
    num_predict: int,
    temperature: float = 0.0,
    timeout_s: float = 600.0,
    client: httpx.Client | None = None,
) -> StreamMetrics:
    """
    One streaming chat completion against Ollama.

    TTFT: request start -> first non-empty assistant `message.content` chunk.
    Total: request start -> JSON line with done=true.
    Decode TPS: (eval_count - 1) / (t_done - t_first_token) using Ollama's eval_count.
    """
    url = base_url.rstrip("/") + "/api/chat"
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": user_prompt}],
        "stream": True,
        "options": {
            "num_predict": num_predict,
            "temperature": temperature,
        },
    }

    own_client = client is None
    http = client or httpx.Client(timeout=timeout_s)

    t_start = time.perf_counter()
    t_first_token: float | None = None
    t_done: float | None = None
    eval_count: int | None = None
    prompt_eval_count: int | None = None
    done_reason: str | None = None

    try:
        with http.stream("POST", url, json=payload) as resp:
            if resp.status_code >= 400:
                body = resp.read().decode("utf-8", errors="replace")
                return StreamMetrics(
                    ttft_ms=None,
                    total_ms=(time.perf_counter() - t_start) * 1000,
                    decode_tps=None,
                    e2e_tps=None,
                    eval_count=None,
                    prompt_eval_count=None,
                    done_reason=None,
                    error=f"HTTP {resp.status_code}: {body[:500]}",
                )

            for line in resp.iter_lines():
                data = _parse_sse_line(line)
                if not data:
                    continue
                msg = data.get("message") or {}
                piece = msg.get("content") or ""
                if t_first_token is None and piece:
                    t_first_token = time.perf_counter()
                if data.get("done"):
                    t_done = time.perf_counter()
                    eval_count = data.get("eval_count")
                    prompt_eval_count = data.get("prompt_eval_count")
                    done_reason = data.get("done_reason")
                    break
    except httpx.RequestError as e:
        return StreamMetrics(
            ttft_ms=None,
            total_ms=(time.perf_counter() - t_start) * 1000,
            decode_tps=None,
            e2e_tps=None,
            eval_count=None,
            prompt_eval_count=None,
            done_reason=None,
            error=str(e),
        )
    finally:
        if own_client:
            http.close()

    if t_done is None:
        return StreamMetrics(
            ttft_ms=None,
            total_ms=(time.perf_counter() - t_start) * 1000,
            decode_tps=None,
            e2e_tps=None,
            eval_count=None,
            prompt_eval_count=None,
            done_reason=None,
            error="stream ended without done=true",
        )

    total_ms = (t_done - t_start) * 1000
    ttft_ms = None
    if t_first_token is not None:
        ttft_ms = (t_first_token - t_start) * 1000

    decode_tps: float | None = None
    e2e_tps: float | None = None
    if eval_count is not None and eval_count > 0 and t_done > t_start:
        e2e_tps = eval_count / (t_done - t_start)
    if (
        eval_count is not None
        and eval_count > 1
        and t_first_token is not None
        and t_done > t_first_token
    ):
        decode_tps = (eval_count - 1) / (t_done - t_first_token)

    return StreamMetrics(
        ttft_ms=ttft_ms,
        total_ms=total_ms,
        decode_tps=decode_tps,
        e2e_tps=e2e_tps,
        eval_count=eval_count,
        prompt_eval_count=prompt_eval_count,
        done_reason=done_reason,
        error=None,
    )


def summarize_ms(values: list[float]) -> tuple[float, float, float]:
    """Median, p25, p75 in ms."""
    if not values:
        return float("nan"), float("nan"), float("nan")
    s = sorted(values)
    mid = len(s) // 2
    med = s[mid] if len(s) % 2 else 0.5 * (s[mid - 1] + s[mid])
    q1 = s[len(s) // 4]
    q3 = s[(3 * len(s)) // 4]
    return med, q1, q3


def summarize_tps(values: list[float]) -> tuple[float, float, float]:
    if not values:
        return float("nan"), float("nan"), float("nan")
    s = sorted(values)
    mid = len(s) // 2
    med = s[mid] if len(s) % 2 else 0.5 * (s[mid - 1] + s[mid])
    q1 = s[len(s) // 4]
    q3 = s[(3 * len(s)) // 4]
    return med, q1, q3


def mean_stdev(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.pstdev(values)
