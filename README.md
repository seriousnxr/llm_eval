# LLM Evaluation Harness

A lightweight, extensible evaluation system for benchmarking LLMs against GPQA Diamond, MATH500, and LiveCodeBench using an OpenAI-compatible API server.

## Features

- **GPQA Diamond** (198 questions) — Few-shot multiple-choice prompting with randomized answer permutation
- **MATH500** (500 problems) — Multi-strategy answer extraction with `\boxed{}` prompting
- **LiveCodeBench** (~400 problems) — Sandboxed code execution with timeout/memory limits
- **Robust API client** — Retry with exponential backoff for 429/503, truncated JSON detection, per-request timeouts
- **Structured reporting** — Per-sample JSONL output and JSON summary report
- **Configurable** — YAML-based configuration for server, retry, concurrency, and output settings

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
│   ├── base.py        # Abstract base task (ThreadPoolExecutor)
│   ├── gpqa.py        # GPQA Diamond (few-shot MCQ)
│   ├── math500.py     # MATH500 (multi-strategy extraction)
│   └── livecodebench.py  # LiveCodeBench (sandboxed execution)
├── sandbox/
│   └── executor.py    # Safe subprocess with timeout/memory limits
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
- **Memory**: 256MB limit via `resource.setrlimit`
- **Network**: Blocked via `sandbox-exec` (macOS) or `unshare -rn` (Linux)
- **Isolation**: Temp directory per execution, minimal `PATH` env
- **Cleanup**: Process group kill + temp dir removal in `finally` block
