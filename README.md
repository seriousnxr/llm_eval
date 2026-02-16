# LLM Evaluation Harness

A lightweight, extensible evaluation system for benchmarking LLMs against GPQA Diamond, MATH500, and LiveCodeBench using an OpenAI-compatible API server.

## Features

- **GPQA Diamond** (198 questions) — Few-shot multiple-choice prompting with randomized answer permutation
- **MATH500** (500 problems) — Multi-strategy answer extraction with `\boxed{}` prompting
- **LiveCodeBench** (~400 problems) — Sandboxed code execution with timeout/memory limits
- **Robust API client** — Retry with exponential backoff for 429/503, truncated JSON detection, per-request timeouts, request throttle (40 req/min)
- **Batched execution** — Samples dispatched in configurable `batch_size` chunks with per-batch disk checkpointing (progress survives crashes)
- **Graceful interruption** — Ctrl+C saves partial results, generates summary report, and exits cleanly
- **Structured reporting** — Per-sample JSONL output and JSON summary report with accuracy, error breakdown, wall-clock timing, and throughput
- **Configurable** — YAML-based configuration for server, retry, concurrency, and output settings
- **Cross-platform** — Works on macOS and Linux without adjustments (probe-based sandbox validation)

## Quick Start

### Prerequisites

- Python 3.10+
- The provided `buggy_server.py` mock server

### Setup

```bash
# Clone the repository
git clone https://github.com/seriousnxr/llm_eval.git
cd llm_eval

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e .
```

### Start the Mock Server

```bash
python3 buggy_server.py --port 8000
```

### Run Evaluations

```bash
# Run all benchmarks
python -m eval_harness.runner

# Run specific tasks
python -m eval_harness.runner --tasks gpqa math500
python -m eval_harness.runner --tasks livecodebench

# Use custom config
python -m eval_harness.runner --config config.yaml --tasks gpqa
```

### Run Tests

```bash
pytest tests/ -v
```

## Configuration

Edit `config.yaml` to customize:

```yaml
server:
  base_url: "http://localhost:8000"
  model: "gpt-4"
  temperature: 0.0
  max_tokens: 2048
  top_p: 1.0
  stop: null
  timeout_seconds: 30

retry:
  max_retries: 10
  base_delay: 1.0
  max_delay: 60.0
  retry_status_codes: [429, 503]

concurrency:
  max_workers: 3
  batch_size: 50

output:
  results_dir: "results"
```

## Architecture

```
eval_harness/
├── client.py          # API client with retry/backoff
├── config.py          # Configuration loading (YAML → dataclasses)
├── runner.py          # Orchestrator with graceful interruption
├── tasks/
│   ├── base.py        # Abstract base task (batched ThreadPoolExecutor)
│   ├── gpqa.py        # GPQA Diamond (few-shot MCQ)
│   ├── math500.py     # MATH500 (multi-strategy extraction)
│   └── livecodebench.py  # LiveCodeBench (sandboxed execution)
├── sandbox/
│   └── executor.py    # Safe subprocess with timeout/memory/network isolation
└── utils/
    ├── extraction.py  # Answer extraction & normalization
    ├── logging.py     # Structured logging & JSONL writer
    └── reporting.py   # Summary report generation
```

## Output

Results are saved to `results/`:

| File | Description |
|------|-------------|
| `gpqa_results.jsonl` | Per-sample GPQA results |
| `math500_results.jsonl` | Per-sample MATH500 results |
| `livecodebench_results.jsonl` | Per-sample LiveCodeBench results |
| `error_log.jsonl` | Structured API error log |
| `summary_report.json` | Aggregate metrics |

### Per-Sample JSONL Fields

Every sample includes: `task`, `sample_id`, `prompt`, `raw_response`, `parsed_answer`, `normalized_answer`, `ground_truth`, `score`, `error_category`, `latency_ms`.

MATH500 additionally: `match_method`, `raw_answer`.

LiveCodeBench additionally: `problem_id`, `generated_code`, `test_results`, `pass_at_1`, `execution_time_ms`.

## Server Behaviors Handled

| Behavior | How We Handle It |
|----------|-----------------|
| Rate limiting (429) | Exponential backoff, respect `Retry-After` header |
| Server errors (503) | Retry with backoff (up to 10 attempts) |
| Truncated JSON (200) | Detect JSON decode failure, auto-retry |
| Slow chunked responses | Per-request timeout (30s default) |

## Sandbox Safety (LiveCodeBench)

- **Timeout**: Hard 5s limit per test case with `SIGKILL` on process group
- **Memory**: 256MB limit via `resource.setrlimit` (RLIMIT_AS on Linux, RLIMIT_DATA fallback on macOS)
- **Network**: Blocked via `sandbox-exec` (macOS) or `unshare -rn` (Linux), probe-validated at init
- **Isolation**: Temp directory per execution, minimal `PATH` env
- **Cleanup**: Process group kill + temp dir removal in `finally` block

## Batching & Checkpointing

Samples are processed in chunks of `batch_size` (default: 50):
- At most `batch_size` futures alive at once → bounded memory
- After each batch, results are flushed to disk (JSONL) → progress survives crashes
- A shared `ThreadPoolExecutor` is reused across batches → no straggler stall
- On Ctrl+C: pending futures are cancelled, partial results + summary report are saved
