# Temperature & seed experiment (worked example)

This document records a small **reproducibility vs sampling** study using `ollama-bench` on the **`short`** preset with **`llama3.2:latest`**. Your hardware and Ollama version will change absolute timings; the **patterns** (seed vs no seed, temperature) are what matter.

## Fixed conditions

| Setting | Value |
|---------|--------|
| Model | `llama3.2:latest` |
| Preset | `short` (user prompt: *“Reply with a single short sentence about local inference benchmarks.”*) |
| Measured runs | `--runs 20` |
| Max new tokens | `--num-predict 128` |
| Warmup | default (`1`) |

## Runs performed

### A — Temperature 0, **with** seed 42

**Command:**

```bash
ollama-bench --model llama3.2:latest --preset short --runs 20 --temperature 0 \
  --seed 42 --csv runs_t0.csv --save-responses responses_t0.jsonl
```

**Artifacts:** `runs_t0.csv`, `responses_t0.jsonl`

| Metric (summary) | Approx. (your run) |
|------------------|---------------------|
| `eval_count` | **30** every run |
| Unique `response` strings (20 runs) | **1** (identical text every time) |
| `total_ms` | Tight spread (e.g. median ~991 ms, sd ~22 ms) |

### B — Temperature 0.7, **with** same seed 42

**Command:**

```bash
ollama-bench --model llama3.2:latest --preset short --runs 20 --temperature 0.7 \
  --seed 42 --csv runs_t07.csv --save-responses responses_t0.7.jsonl
```

**Artifacts:** `runs_t07.csv`, `responses_t0.7.jsonl`

| Metric (summary) | Approx. (your run) |
|------------------|---------------------|
| `eval_count` | **33** every run |
| Unique `response` strings (20 runs) | **1** |
| `total_ms` | Slightly higher median than A; somewhat wider spread than A (e.g. sd ~41 ms) |

**Between A and B:** the assistant text **differs** (two different single-sentence answers), but **within** each condition the seed fixes the sample, so **no within-condition wording variance**.

### C — Temperature 0.7, **no** seed (sampling varies)

**Command:**

```bash
ollama-bench --model llama3.2:latest --preset short --runs 20 --temperature 0.7 \
  --csv runs_t07_noseed.csv --save-responses responses_t07_noseed.jsonl
```

**Artifacts:** `runs_t07_noseed.csv`, `responses_t07_noseed.jsonl`

| Metric (summary) | Approx. (your run) |
|------------------|---------------------|
| `eval_count` | **29–44** across runs (length varies) |
| Unique `response` strings (20 runs) | **20** (all different) |
| `total_ms` | Wider spread (e.g. median ~1109 ms, **sd ~136 ms**) — driven largely by different completion lengths |
| `decode_tps` | Still fairly stable (e.g. median ~32.8 tok/s, sd ~0.65) |

## Representative outputs (seeded runs)

**A — T=0.0, seed 42** (every run):

> Local inference benchmarks, such as the MobileNet and ShuffleNet series, are designed to evaluate the performance of neural networks on small, mobile devices.

**B — T=0.7, seed 42** (every run):

> Local inference benchmarks, such as the ImageNet Local Inference Benchmark (ILIB), evaluate the performance of machine learning models on edge devices and low-power hardware.

**C — T=0.7, no seed:** each line in `responses_t07_noseed.jsonl` has a different `response` (examples: MobileNet/SqueezeNet on mobile; ImageNet/CIFAR-10 domains; MobileNet/ResNet on edge devices, etc.).

## Takeaways

1. **`--seed`** with fixed prompt + temperature makes **repeated completions identical** (for this model/preset), which is good for **latency A/B** but not for studying **phrasing diversity**.
2. **Temperature** still changes **which** completion you get when you move from T=0 to T=0.7 (compare A vs B).
3. **Dropping `--seed` at T=0.7`** produces **different text every run** and **variable length**, which is what you want to document **output variance**; wall time variance tracks length more than “slower decode.”
4. Full text for every run lives in the **JSONL** files (`--save-responses`); CSV rows add **`temperature`**, **`seed`**, **`response_chars`** for spreadsheets.

## Quick checks on JSONL

Count unique responses:

```bash
python3 -c "import json; from pathlib import Path
p=Path('responses_t07_noseed.jsonl')
rs=[json.loads(l)['response'] for l in p.read_text().splitlines() if l.strip()]
print('unique:', len(set(rs)), '/', len(rs))"
```

Inspect one record:

```bash
head -n 1 responses_t07_noseed.jsonl | python3 -m json.tool
```

## Optional: no-seed at T=0

For symmetry you can run temperature **0** without seed; many setups still collapse to one answer often, but it is model-dependent:

```bash
ollama-bench --model llama3.2:latest --preset short --runs 20 --temperature 0 \
  --csv runs_t0_noseed.csv --save-responses responses_t0_noseed.jsonl
```
