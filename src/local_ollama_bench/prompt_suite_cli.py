from __future__ import annotations

import argparse
import sys
from pathlib import Path

from local_ollama_bench.prompt_suite import (
    default_benchmark_prompts_path,
    load_prompts,
    run_prompt_suite,
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Run a fixed prompt suite across multiple Ollama models: latency, throughput, "
            "full responses (JSONL), optional VRAM snapshot via GET /api/ps."
        ),
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
        required=True,
        help="Ollama model tag (repeatable), e.g. --model llama3.2:3b --model mistral:7b",
    )
    p.add_argument(
        "--prompt-file",
        type=Path,
        default=None,
        help="One prompt per line (# comments OK). Default: bundled benchmark_prompts.txt (~47 prompts).",
    )
    p.add_argument(
        "--output-jsonl",
        type=Path,
        required=True,
        metavar="PATH",
        help="Write one JSON record per (model, prompt); overwrites if exists.",
    )
    p.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        metavar="PATH",
        help="Optional aggregate CSV (one row per completion).",
    )
    p.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (default: %(default)s)")
    p.add_argument("--seed", type=int, default=None, help="Optional Ollama seed for reproducibility")
    p.add_argument(
        "--num-predict",
        type=int,
        default=256,
        help="Max new tokens per completion (default: %(default)s)",
    )
    p.add_argument(
        "--warmup-per-model",
        type=int,
        default=1,
        help="Warmup completions per model using the first prompt (default: %(default)s)",
    )
    p.add_argument(
        "--no-memory-snapshot",
        action="store_true",
        help="Do not call GET /api/ps after each completion (VRAM / loaded model info).",
    )
    p.add_argument(
        "--machine-label",
        default="",
        help="Stored in each JSONL row for reproducibility notes (e.g. M2 16GB).",
    )
    args = p.parse_args(argv)

    prompt_path = args.prompt_file
    if prompt_path is None:
        prompt_path = default_benchmark_prompts_path()
        if not prompt_path.is_file():
            print(f"Bundled prompts missing at {prompt_path}; pass --prompt-file.", file=sys.stderr)
            return 2

    prompts = load_prompts(prompt_path)
    print(f"Loaded {len(prompts)} prompts from {prompt_path}", flush=True)

    return run_prompt_suite(
        base_url=args.base_url,
        models=args.models,
        prompts=prompts,
        out_jsonl=args.output_jsonl,
        out_csv=args.output_csv,
        temperature=args.temperature,
        seed=args.seed,
        num_predict=args.num_predict,
        warmup_per_model=args.warmup_per_model,
        memory_snapshot=not args.no_memory_snapshot,
        machine_label=args.machine_label or "",
    )


if __name__ == "__main__":
    raise SystemExit(main())
