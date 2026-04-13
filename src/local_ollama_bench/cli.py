from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import httpx

from local_ollama_bench.stream_bench import (
    bench_chat_stream,
    mean_stdev,
    summarize_ms,
    summarize_tps,
)


def _preset_prompt(name: str) -> str:
    if name == "short":
        return "Reply with a single short sentence about local inference benchmarks."
    if name == "medium":
        base = (
            "You are helping benchmark a local LLM server. "
            "Summarize what TTFT, decode tokens/s, and total latency mean for UX. "
        )
        return base + ("Keep it under 120 words. " * 8).strip()
    if name == "long":
        filler = (
            "Context padding for prefill measurement. "
            "Repeat the following id: bench-long-7f3a. " * 200
        )
        return filler + "\nNow write exactly three sentences about offline assistants."
    raise ValueError(f"unknown preset {name!r}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Benchmark Ollama /api/chat streaming: TTFT, decode TPS, total latency.",
    )
    p.add_argument(
        "--base-url",
        default="http://127.0.0.1:11434",
        help="Ollama server URL (default: %(default)s)",
    )
    p.add_argument(
        "--model",
        action="append",
        dest="models",
        help="Model tag (repeatable), e.g. --model llama3.2:latest",
    )
    p.add_argument(
        "--num-predict",
        type=int,
        default=128,
        help="Max new tokens per run (Ollama option num_predict). Default: %(default)s",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default: %(default)s)",
    )
    p.add_argument(
        "--preset",
        choices=("short", "medium", "long"),
        default="short",
        help="Built-in prompt length bucket (default: %(default)s)",
    )
    p.add_argument(
        "--prompt",
        default=None,
        help="Override preset with a custom user prompt string",
    )
    p.add_argument("--warmup", type=int, default=1, help="Warmup runs per model (default: %(default)s)")
    p.add_argument("--runs", type=int, default=10, help="Measured runs per model (default: %(default)s)")
    p.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Append one row per measured run to this CSV file",
    )
    args = p.parse_args(argv)

    models = args.models or ["llama3.2:latest"]
    user_prompt = args.prompt if args.prompt is not None else _preset_prompt(args.preset)

    client = httpx.Client(timeout=600.0)
    rows_for_csv: list[dict[str, object]] = []

    try:
        for model in models:
            print(f"\n=== {model} ===", flush=True)
            for w in range(args.warmup):
                m = bench_chat_stream(
                    base_url=args.base_url,
                    model=model,
                    user_prompt=user_prompt,
                    num_predict=args.num_predict,
                    temperature=args.temperature,
                    client=client,
                )
                label = "warmup" if w < args.warmup - 1 else "warmup (last)"
                if m.error:
                    print(f"  {label}: ERROR {m.error}", flush=True)
                else:
                    print(
                        f"  {label}: ttft={m.ttft_ms:.1f} ms  total={m.total_ms:.1f} ms  "
                        f"decode_tps={m.decode_tps}  eval_count={m.eval_count}",
                        flush=True,
                    )

            ttfts: list[float] = []
            totals: list[float] = []
            decodes: list[float] = []
            e2es: list[float] = []
            errors = 0

            for i in range(args.runs):
                m = bench_chat_stream(
                    base_url=args.base_url,
                    model=model,
                    user_prompt=user_prompt,
                    num_predict=args.num_predict,
                    temperature=args.temperature,
                    client=client,
                )
                if m.error:
                    errors += 1
                    print(f"  run {i + 1}/{args.runs}: ERROR {m.error}", flush=True)
                    rows_for_csv.append(
                        {
                            "model": model,
                            "run": i + 1,
                            "ok": False,
                            "error": m.error,
                            "ttft_ms": "",
                            "total_ms": m.total_ms,
                            "decode_tps": "",
                            "e2e_tps": "",
                            "eval_count": "",
                            "prompt_eval_count": "",
                            "done_reason": "",
                        }
                    )
                    continue
                assert m.ttft_ms is not None
                ttfts.append(m.ttft_ms)
                totals.append(m.total_ms)
                if m.decode_tps is not None:
                    decodes.append(m.decode_tps)
                if m.e2e_tps is not None:
                    e2es.append(m.e2e_tps)
                d_tps = f"{m.decode_tps:.2f}" if m.decode_tps is not None else "n/a"
                e_tps = f"{m.e2e_tps:.2f}" if m.e2e_tps is not None else "n/a"
                print(
                    f"  run {i + 1}/{args.runs}: ttft={m.ttft_ms:.1f} ms  "
                    f"total={m.total_ms:.1f} ms  decode_tps={d_tps}  "
                    f"e2e_tps={e_tps}  eval={m.eval_count}",
                    flush=True,
                )
                rows_for_csv.append(
                    {
                        "model": model,
                        "run": i + 1,
                        "ok": True,
                        "error": "",
                        "ttft_ms": m.ttft_ms,
                        "total_ms": m.total_ms,
                        "decode_tps": m.decode_tps if m.decode_tps is not None else "",
                        "e2e_tps": m.e2e_tps if m.e2e_tps is not None else "",
                        "eval_count": m.eval_count if m.eval_count is not None else "",
                        "prompt_eval_count": m.prompt_eval_count
                        if m.prompt_eval_count is not None
                        else "",
                        "done_reason": m.done_reason or "",
                    }
                )

            if errors:
                print(f"  completed with {errors} errors", flush=True)
            if ttfts:
                tt_m, tt_q1, tt_q3 = summarize_ms(ttfts)
                tot_m, tot_q1, tot_q3 = summarize_ms(totals)
                dec_m, dec_q1, dec_q3 = summarize_tps(decodes) if decodes else (float("nan"),) * 3
                e2e_m, e2e_q1, e2e_q3 = summarize_tps(e2es) if e2es else (float("nan"),) * 3
                tt_mean, tt_sd = mean_stdev(ttfts)
                tot_mean, tot_sd = mean_stdev(totals)
                dec_mean, dec_sd = mean_stdev(decodes) if decodes else (float("nan"), float("nan"))
                print(
                    f"  summary: ttft_ms median={tt_m:.1f} (p25={tt_q1:.1f} p75={tt_q3:.1f}) "
                    f"mean={tt_mean:.1f} sd={tt_sd:.1f}",
                    flush=True,
                )
                print(
                    f"           total_ms median={tot_m:.1f} (p25={tot_q1:.1f} p75={tot_q3:.1f}) "
                    f"mean={tot_mean:.1f} sd={tot_sd:.1f}",
                    flush=True,
                )
                print(
                    f"           decode_tps median={dec_m:.2f} (p25={dec_q1:.2f} p75={dec_q3:.2f}) "
                    f"mean={dec_mean:.2f} sd={dec_sd:.2f}",
                    flush=True,
                )
                print(
                    f"           e2e_tps median={e2e_m:.2f} (p25={e2e_q1:.2f} p75={e2e_q3:.2f})",
                    flush=True,
                )
    finally:
        client.close()

    if args.csv and rows_for_csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        new_file = not args.csv.exists()
        fieldnames = [
            "model",
            "run",
            "ok",
            "error",
            "ttft_ms",
            "total_ms",
            "decode_tps",
            "e2e_tps",
            "eval_count",
            "prompt_eval_count",
            "done_reason",
        ]
        with args.csv.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            if new_file:
                w.writeheader()
            for row in rows_for_csv:
                w.writerow(row)
        print(f"\nWrote {len(rows_for_csv)} rows to {args.csv}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
