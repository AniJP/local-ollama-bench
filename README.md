# local-ollama-bench

**Benchmark CLI** for [Ollama](https://ollama.com) streaming chat (TTFT, decode tokens/s, total latency), plus a **FastAPI + browser UI** that proxies Ollama for local chatting.

## Setup

```bash
cd /path/to/local-ollama-bench
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
uv pip install -e .
# or: python -m pip install -e .
```

Requires Ollama running locally (default `http://127.0.0.1:11434`) and models pulled (`ollama pull <model>`).

## Web assistant (FastAPI + simple UI)

Starts a small local server that **proxies** Ollama streaming chat and serves a **single-page chat UI** at `/`.

```bash
source .venv/bin/activate
local-assistant
# open http://127.0.0.1:8765
```

Options:

```bash
local-assistant --host 127.0.0.1 --port 8765 --reload
```

Environment:

| Variable | Meaning |
|----------|--------|
| `OLLAMA_BASE_URL` | Ollama API base (default `http://127.0.0.1:11434`) |

API (same machine as the browser; no CORS needed for the bundled UI):

- `GET /api/models` — model tags from Ollama `GET /api/tags`
- `POST /api/chat/stream` — JSON body `{ "model": "…", "messages": [{ "role": "user"|"assistant"|"system", "content": "…" }] }`; response body is **NDJSON** lines forwarded from Ollama (same format as `/api/chat` with `stream: true`).

## Benchmark CLI

```bash
ollama-bench --help
```

Examples:

```bash
# Single model, short preset prompt, measured runs + warmup
ollama-bench --runs 10 --warmup 2 --model llama3.2:latest --num-predict 128

# Compare two models
ollama-bench --runs 5 --model llama3.2:latest --model llama3:latest --num-predict 64

# Longer input (prefill stress) and CSV log
ollama-bench --preset long --runs 10 --model llama3.2:latest --csv results.csv

# Log full assistant text per run (JSON Lines) + optional reproducible seed
ollama-bench --model llama3.2:latest --preset short --runs 20 --temperature 0.7 \
  --save-responses responses.jsonl --csv runs.csv
```

A worked **temperature / seed** comparison (same preset, with and without `--seed`) is in [`temperature_experiment_notes.md`](temperature_experiment_notes.md).

### CLI flags (short)

| Flag | Meaning |
|------|--------|
| `--base-url` | Ollama server URL (default `http://127.0.0.1:11434`) |
| `--model` | Model tag; repeat for multiple models in one session |
| `--num-predict` | Max **new** tokens per completion (Ollama `num_predict`; model may stop earlier) |
| `--temperature` | Sampling temperature (default `0.0`) |
| `--seed` | Ollama `seed` for reproducible sampling (optional) |
| `--preset` | Built-in prompt: `short` / `medium` / `long` |
| `--prompt` | Custom user message (overrides `--preset`) |
| `--warmup` | Extra runs per model before statistics (not included in summary) |
| `--runs` | Measured runs per model for median / quartiles / mean |
| `--csv` | Append one row per measured run (`temperature`, `seed`, `response_chars`, …) |
| `--save-responses` | Append one JSON object per run (full `prompt` + `response`, JSON Lines) |
| `--save-warmup-responses` | With `--save-responses`, also log warmup runs |

## Metrics (how to read the output)

| Metric | Meaning |
|--------|--------|
| **ttft** (ms) | Wall time from sending the request until the **first non-empty** assistant token arrives on the stream. |
| **total** (ms) | Wall time from request start until the stream ends (`done: true`). |
| **decode_tps** | Decode throughput: \((\mathrm{eval\_count} - 1) / (t_{\mathrm{done}} - t_{\mathrm{first\_token}})\) using Ollama’s `eval_count` on the final chunk. Undefined if fewer than two tokens were generated. |
| **e2e_tps** | End-to-end tokens/s: `eval_count / (t_done - t_start)` (includes prefill + TTFT). |
| **eval** | `eval_count` from Ollama: number of **generated** tokens in that run. |

**Summary line:** median and p25/p75 describe “typical” and spread; mean and sd show central tendency and run-to-run noise.

**Warmup:** Loads weights and stabilizes clocks; the last warmup line is usually closer to steady state than the very first request after idle.

---

## Sample benchmark results

These are **example runs** from a local Apple Silicon Mac with Ollama. Your numbers will differ with hardware, OS power settings, Ollama version, and model tags. Treat this section as **documentation of what a report looks like**, not a universal baseline.

### Run A — `llama3.2:latest`, short preset, `num_predict=128`

Command:

```bash
ollama-bench --runs 10 --warmup 2 --model llama3.2:latest --num-predict 128
```

Observed behavior: each completion used **30 generated tokens** (below the 128 ceiling because the model finished the reply early).

**Summary (measured runs only):**

| Metric | Median | p25 → p75 | Mean ± sd (where shown) |
|--------|--------|-----------|-------------------------|
| TTFT (ms) | 100.1 | 96.9 → 101.6 | 100.2 ± 4.2 |
| Total (ms) | 967.1 | 961.9 → 970.7 | 978.8 ± 35.3 |
| Decode (tok/s) | 33.41 | 33.26 → 33.55 | 33.05 ± 1.10 |
| E2E (tok/s) | 31.02 | 30.91 → 31.19 | — |

Interpretation: **low TTFT**, **~1 s** total for ~30 tokens, **~33 tok/s** steady decode on this machine for this prompt.

---

### Run B — `llama3.2:latest` vs `llama3:latest`, short preset, `num_predict=64`

Command:

```bash
ollama-bench --runs 5 --model llama3.2:latest --model llama3:latest --num-predict 64
```

**`llama3.2:latest`** (30 generated tokens per run):

| Metric | Median | p25 → p75 |
|--------|--------|-----------|
| TTFT (ms) | 105.7 | 103.0 → 117.9 |
| Total (ms) | 977.9 | 971.8 → 978.6 |
| Decode (tok/s) | 33.37 | 33.36 → 33.48 |
| E2E (tok/s) | 30.68 | 30.66 → 30.87 |

**`llama3:latest`** (41 generated tokens per run):

| Metric | Median | p25 → p75 |
|--------|--------|-----------|
| TTFT (ms) | 461.2 | 386.0 → 482.6 |
| Total (ms) | 8263.4 | 8242.1 → 9225.6 |
| Decode (tok/s) | 5.07 | 4.58 → 5.08 |
| E2E (tok/s) | 4.96 | 4.44 → 4.97 |

Interpretation: on this host, **`llama3.2` was an order of magnitude faster** in wall time and tokens/s than **`llama3`** for the same CLI preset. The larger `llama3` run also showed **higher variance** in total latency across runs (thermal scheduling, memory pressure, or model eviction can widen p25–p75).

Warmup for `llama3` can show a **very long** first completion while weights load; measured runs after warmup still reflected the slower steady decode speed (~5 tok/s).

---

## Reproducibility tips

- Use the same **power source** (AC vs battery) and disable heavy background jobs.
- Pin **Ollama version** and **exact model tags** (e.g. `llama3.2:latest` digest) when comparing over time.
- Increase `--runs` (for example 30+) when you need tighter statistics.
- For fair decode comparisons, either fix a prompt that reliably hits **`num_predict`** or report **`eval_count`** alongside TPS.
