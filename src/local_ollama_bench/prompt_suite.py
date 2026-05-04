from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from local_ollama_bench.stream_bench import StreamMetrics, bench_chat_stream


def default_benchmark_prompts_path() -> Path:
    return Path(__file__).resolve().parent / "prompts" / "benchmark_prompts.txt"


def check_ollama_alive(client: httpx.Client, base_url: str) -> None:
    """Raise RuntimeError with a clear message if Ollama is not reachable."""
    url = base_url.rstrip("/") + "/api/tags"
    try:
        r = client.get(url, timeout=5.0)
    except httpx.RequestError as e:
        raise RuntimeError(
            f"Cannot reach Ollama at {base_url} ({e}).\n"
            "Start the Ollama app or run `ollama serve`. "
            "If Ollama uses another host/port, pass --base-url (e.g. http://127.0.0.1:11434)."
        ) from e
    if r.status_code >= 400:
        raise RuntimeError(
            f"Ollama at {base_url} returned HTTP {r.status_code}: {r.text[:400]}"
        )


def load_prompts(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    out: list[str] = []
    for raw in text.splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        out.append(s)
    return out


def fetch_ollama_ps(client: httpx.Client, base_url: str) -> dict[str, Any] | None:
    url = base_url.rstrip("/") + "/api/ps"
    try:
        r = client.get(url, timeout=15.0)
    except httpx.RequestError:
        return None
    if r.status_code >= 400:
        return {"_error": f"HTTP {r.status_code}", "_body": r.text[:800]}
    try:
        return r.json()
    except json.JSONDecodeError:
        return None


def metrics_record(
    *,
    machine_label: str,
    model: str,
    prompt_id: int,
    prompt: str,
    temperature: float,
    seed: int | None,
    num_predict: int,
    m: StreamMetrics,
    ollama_ps: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "ts": datetime.now(timezone.utc).isoformat(),
        "machine_label": machine_label,
        "model": model,
        "prompt_id": prompt_id,
        "prompt": prompt,
        "temperature": temperature,
        "seed": seed,
        "num_predict": num_predict,
        "response": (m.response_text or "") if not m.error else "",
        "error": m.error or "",
        "ttft_ms": m.ttft_ms,
        "total_ms": m.total_ms,
        "decode_tps": m.decode_tps,
        "e2e_tps": m.e2e_tps,
        "eval_count": m.eval_count,
        "done_reason": m.done_reason,
        "ollama_ps": ollama_ps,
    }


def append_jsonl(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def run_prompt_suite(
    *,
    base_url: str,
    models: list[str],
    prompts: list[str],
    out_jsonl: Path,
    out_csv: Path | None,
    temperature: float,
    seed: int | None,
    num_predict: int,
    warmup_per_model: int,
    memory_snapshot: bool,
    machine_label: str,
) -> int:
    if not models:
        raise ValueError("at least one --model is required")
    if not prompts:
        raise ValueError("no prompts loaded")

    client = httpx.Client(timeout=600.0)
    try:
        check_ollama_alive(client, base_url)
    except RuntimeError as e:
        print(str(e), file=sys.stderr, flush=True)
        client.close()
        return 2

    out_jsonl.unlink(missing_ok=True)
    if out_csv:
        out_csv.unlink(missing_ok=True)

    csv_rows: list[dict[str, Any]] = []
    errors = 0

    try:
        for mi, model in enumerate(models, start=1):
            print(f"\n=== model {mi}/{len(models)}: {model} ===", flush=True)
            for w in range(warmup_per_model):
                wm = bench_chat_stream(
                    base_url=base_url,
                    model=model,
                    user_prompt=prompts[0],
                    num_predict=num_predict,
                    temperature=temperature,
                    seed=seed,
                    client=client,
                )
                tag = "warmup" if w < warmup_per_model - 1 else "warmup (last)"
                if wm.error:
                    print(f"  {tag}: ERROR {wm.error}", flush=True)
                else:
                    print(
                        f"  {tag}: ttft={wm.ttft_ms:.1f} ms total={wm.total_ms:.1f} ms "
                        f"decode_tps={wm.decode_tps} eval={wm.eval_count}",
                        flush=True,
                    )

            for i, prompt in enumerate(prompts, start=1):
                m = bench_chat_stream(
                    base_url=base_url,
                    model=model,
                    user_prompt=prompt,
                    num_predict=num_predict,
                    temperature=temperature,
                    seed=seed,
                    client=client,
                )
                ps: dict[str, Any] | None = None
                if memory_snapshot:
                    ps = fetch_ollama_ps(client, base_url)

                rec = metrics_record(
                    machine_label=machine_label,
                    model=model,
                    prompt_id=i,
                    prompt=prompt,
                    temperature=temperature,
                    seed=seed,
                    num_predict=num_predict,
                    m=m,
                    ollama_ps=ps,
                )
                append_jsonl(out_jsonl, rec)

                if m.error:
                    errors += 1
                    print(f"  prompt {i}/{len(prompts)}: ERROR {m.error}", flush=True)
                else:
                    d = f"{m.decode_tps:.2f}" if m.decode_tps is not None else "n/a"
                    print(
                        f"  prompt {i}/{len(prompts)}: ttft={m.ttft_ms:.1f} ms "
                        f"total={m.total_ms:.1f} ms decode_tps={d} eval={m.eval_count}",
                        flush=True,
                    )

                vram = ""
                if ps and isinstance(ps.get("models"), list) and ps["models"]:
                    first = ps["models"][0]
                    if isinstance(first, dict) and first.get("size_vram") is not None:
                        vram = first.get("size_vram")

                csv_rows.append(
                    {
                        "model": model,
                        "prompt_id": i,
                        "ok": not bool(m.error),
                        "error": m.error or "",
                        "ttft_ms": m.ttft_ms if m.ttft_ms is not None else "",
                        "total_ms": m.total_ms,
                        "decode_tps": m.decode_tps if m.decode_tps is not None else "",
                        "e2e_tps": m.e2e_tps if m.e2e_tps is not None else "",
                        "eval_count": m.eval_count if m.eval_count is not None else "",
                        "response_chars": len(m.response_text or ""),
                        "size_vram": vram,
                    }
                )
    finally:
        client.close()

    if out_csv and csv_rows:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "model",
            "prompt_id",
            "ok",
            "error",
            "ttft_ms",
            "total_ms",
            "decode_tps",
            "e2e_tps",
            "eval_count",
            "response_chars",
            "size_vram",
        ]
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(csv_rows)
        print(f"\nWrote CSV summary to {out_csv}", flush=True)

    print(f"\nWrote JSONL to {out_jsonl} ({len(models) * len(prompts)} rows)", flush=True)
    if errors:
        print(f"Completed with {errors} prompt errors.", flush=True)
    return 1 if errors else 0
