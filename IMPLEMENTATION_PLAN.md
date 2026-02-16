# Implementation Plan: LLM Evaluation Harness

## Research Summary

### Best Practices from Reference Implementations

**OpenAI simple-evals** (reference-quality, simple approach):
- Zero-shot, chain-of-thought prompting — "Solve step by step" style
- GPQA: Randomized answer permutation, regex `(?i)Answer\s*:\s*\$?([A-D])\$?`
- MATH: "Answer: $ANSWER" extraction with LLM-based equality checking fallback
- ThreadPool for concurrent execution with tqdm progress bars
- Per-sample result objects aggregated into final metrics

**EleutherAI lm-evaluation-harness** (industry standard):
- Modular architecture: tasks, models, evaluators in separate modules
- Config-driven task definitions (YAML)
- API model support with retries, caching, batching
- `local-chat-completions` model type for OpenAI-compatible servers

**LiveCodeBench** (code evaluation reference):
- `multiprocessing.Process` for sandboxed code execution with hard kill on timeout
- `ProcessPoolExecutor` for parallel evaluation of problems
- `run_test` pipes stdin to generated code, captures stdout, compares to expected
- Dataset: `load_dataset("livecodebench/code_generation_lite", version_tag="release_v1")` — 400 problems

### Server Behaviors to Handle (from buggy_server.py analysis)

| Behavior | Trigger | Strategy |
|---|---|---|
| **Rate limit (429)** | 50 req/min window, 5s cooldown at 0 remaining | Exponential backoff, respect `Retry-After` header |
| **Server overload (503)** | ~8% random chance | Retry with backoff |
| **Truncated JSON (200)** | ~2% (MD5 hash-based) | Detect JSON decode error on 200, retry |
| **Slow chunked response** | ~5% chance, 2s+ per chunk | Per-request timeout (e.g. 30s), retry on timeout |

---

## Architecture

```
goaly_final/
├── config.yaml                    # Global configuration
├── buggy_server.py                # PROVIDED — do not modify
├── README.md                      # Setup instructions
├── IMPLEMENTATION_PLAN.md         # This document
│
├── eval_harness/
│   ├── __init__.py
│   ├── client.py                  # API client with retry/backoff
│   ├── config.py                  # Configuration loading & dataclass
│   ├── runner.py                  # Orchestrator: load tasks, run, report
│   │
│   ├── tasks/
│   │   ├── __init__.py
│   │   ├── base.py                # Abstract base task class
│   │   ├── gpqa.py                # GPQA Diamond evaluator
│   │   ├── math500.py             # MATH500 evaluator
│   │   └── livecodebench.py       # LiveCodeBench evaluator
│   │
│   ├── sandbox/
│   │   ├── __init__.py
│   │   └── executor.py            # Safe subprocess execution
│   │
│   └── utils/
│       ├── __init__.py
│       ├── extraction.py          # Answer extraction & normalization
│       ├── reporting.py           # JSONL & summary report generation
│       └── logging.py             # Structured error logging
│
├── tests/
│   ├── test_client.py
│   ├── test_extraction.py
│   ├── test_gpqa.py
│   ├── test_math500.py
│   ├── test_sandbox.py
│   └── test_reporting.py
│
└── results/                       # Generated at runtime
    ├── gpqa_results.jsonl
    ├── math500_results.jsonl
    ├── livecodebench_results.jsonl
    └── summary_report.json
```

---

## Phase-by-Phase Implementation

### Phase 1: Foundation (Commits 1-2)

**Commit 1: Project scaffold & configuration**
- Initialize git repo, `.gitignore`, `pyproject.toml` with dependencies
- Create `config.yaml` with all configurable parameters:
  ```yaml
  server:
    base_url: "http://localhost:8000"
    model: "gpt-4"
    temperature: 0.0
    max_tokens: 2048
    top_p: 1.0
    timeout_seconds: 30

  retry:
    max_retries: 5
    base_delay: 1.0
    max_delay: 60.0
    retry_status_codes: [429, 503]

  concurrency:
    max_workers: 10
    batch_size: 50

  output:
    results_dir: "results"
  ```
- Create `config.py` dataclass to load/validate config
- Create package structure with `__init__.py` files

**Commit 2: API client with robust error handling**
- `client.py`: `EvalClient` class
  - Async-capable HTTP client using `aiohttp` (or threaded with `requests`)
  - OpenAI-compatible request format: `POST /v1/chat/completions`
  - **Retry logic**: Exponential backoff with jitter for 429/503
    - Respect `Retry-After` header from 429 responses
    - Cap backoff at `max_delay` seconds
  - **Truncated JSON detection**: Catch `json.JSONDecodeError` on 200 responses → retry
  - **Timeout enforcement**: Per-request timeout to handle slow chunked responses
  - **Structured error logging**: Log every error to JSONL with timestamp, error type, status code, retry count
  - Return typed response dataclass with `content`, `latency_ms`, `error_category`

### Phase 2: Task Framework & GPQA (Commits 3-4)

**Commit 3: Base task abstraction & answer extraction utilities**
- `tasks/base.py`: Abstract `BaseTask` class
  ```python
  class BaseTask(ABC):
      def load_dataset(self) -> list[dict]: ...
      def format_prompt(self, sample: dict) -> list[dict]: ...
      def extract_answer(self, response: str) -> str | None: ...
      def score(self, predicted: str, ground_truth: str) -> float: ...
      def run(self, client: EvalClient) -> list[SampleResult]: ...
  ```
- `utils/extraction.py`:
  - `extract_mcq_answer(text)`: Multi-pattern regex for A/B/C/D extraction
    - Pattern 1: `(?i)Answer\s*:\s*\$?([A-D])\$?` (OpenAI style)
    - Pattern 2: `(?i)(?:correct answer|choose)\s+(?:is\s+)?([A-D])`
    - Pattern 3: `^([A-D])$` on last non-empty line
    - Fallback: first standalone A-D letter in response
  - `extract_boxed_answer(text)`: `\boxed{...}` with nested brace handling
  - `extract_answer_is(text)`: "the answer is X" pattern
  - `normalize_math_answer(text)`: Fraction normalization, LaTeX cleanup, trailing zeros

**Commit 4: GPQA Diamond evaluator (198 questions)**
- `tasks/gpqa.py`:
  - Load dataset: `datasets.load_dataset("Idavidrein/gpqa", "gpqa_diamond")`
    - Fields: `Question`, `Correct Answer`, `Incorrect Answer 1/2/3`
  - Prompt template (following OpenAI simple-evals):
    ```
    Answer the following multiple choice question. The last line of your
    response should be of the following format: 'Answer: $LETTER' (without
    quotes) where LETTER is one of ABCD. Think step by step before answering.

    {Question}

    A) {choice_A}
    B) {choice_B}
    C) {choice_C}
    D) {choice_D}
    ```
  - **Randomized answer permutation** per question (seeded RNG for reproducibility)
  - Parse answer letter using cascading regex extractors
  - Handle ambiguous responses: if no clear letter found → `error_category: "parse_failure"`
  - JSONL output per sample with all required fields

### Phase 3: MATH500 (Commit 5)

**Commit 5: MATH500 evaluator (500 problems)**
- `tasks/math500.py`:
  - Load dataset: `datasets.load_dataset("lighteval/MATH", split="test")` → first 500
    - Or from `openai/prm800k` math_splits (MATH-500 IID variant)
    - Fields: `problem`, `solution`, `answer`, `level`, `type`
  - Prompt template:
    ```
    Solve the following math problem step by step. The last line of your
    response should be of the form Answer: $ANSWER (without quotes) where
    $ANSWER is the answer to the problem.

    {problem}

    Remember to put your answer on its own line after "Answer:".
    ```
  - **Multi-strategy answer extraction** (ordered by reliability):
    1. `\boxed{...}` extraction (handle nested braces)
    2. `Answer:\s*(.+)` pattern
    3. "the answer is" pattern
    4. Last mathematical expression fallback
  - **Answer normalization pipeline**:
    - LaTeX command cleanup: `\frac{a}{b}` → `a/b`, `\text{}` removal
    - Fraction handling: convert to canonical form, compare `1/2 == 0.5`
    - Trailing zeros: `3.0` == `3`
    - Whitespace/dollar sign stripping
  - **Scoring**: Exact string match after normalization
  - JSONL includes: `raw_answer`, `parsed_answer`, `normalized_answer`, `ground_truth`, `match_method`

### Phase 4: LiveCodeBench (Commits 6-7)

**Commit 6: Sandbox execution infrastructure**
- `sandbox/executor.py`: `SafeExecutor` class
  - **Timeout enforcement**: `subprocess.run(..., timeout=5)` per test case + process-level `multiprocessing.Process` with `p.join(timeout)` and `p.kill()`
  - **Memory limits**: `resource.setrlimit(resource.RLIMIT_AS, ...)` via `preexec_fn` (256MB default)
  - **Network isolation**: On macOS, use `sandbox-exec` or simply rely on subprocess isolation; on Linux, use `unshare` for network namespace
  - **Temporary file isolation**: `tempfile.mkdtemp()` per execution, cleanup in `finally` block
  - **Stdout/stderr capture**: `subprocess.PIPE` for both, decode with error handling
  - **Process cleanup**: Kill process group `os.killpg()` to catch child processes
  - Return `ExecutionResult(stdout, stderr, exit_code, timed_out, error_type)`

**Commit 7: LiveCodeBench evaluator (~400 problems)**
- `tasks/livecodebench.py`:
  - Load dataset: `load_dataset("livecodebench/code_generation_lite", version_tag="release_v1")`
    - Fields: `question_title`, `question_content`, `input_output` (JSON with `inputs`/`outputs`), `difficulty`
  - Prompt: Ask model to generate a complete Python program reading stdin, writing stdout
  - Send prompt to model, extract code from response (strip markdown fences if present)
  - For each test case in `input_output`:
    - Execute code via `SafeExecutor` with stdin piped in
    - Compare stdout to expected output (strip trailing whitespace)
  - **pass@1**: 1 if ALL test cases pass, 0 otherwise
  - Per-problem JSONL: `problem_id`, `generated_code`, `test_results[]`, `pass_at_1`, `execution_time_ms`, `error_category`
  - Error categories: `timeout`, `runtime_error`, `wrong_answer`, `compilation_error`, `memory_limit`

### Phase 5: Orchestration & Reporting (Commit 8)

**Commit 8: Runner, reporting, tests, README**
- `runner.py`: Main orchestrator
  - Load config from YAML
  - Run each benchmark with configurable worker pool
  - **Concurrent requests**: `ThreadPoolExecutor` with `max_workers` from config
  - **Rate-limit aware scheduling**: Track 429 responses, dynamically throttle
  - **Graceful interruption**: `signal.signal(SIGINT, ...)` handler that saves partial results
  - Progress bar via `tqdm`
- `utils/reporting.py`:
  - Write per-sample JSONL with all required fields
  - Generate `summary_report.json`
- **Unit tests** (minimum 5):
  1. `test_extraction.py`: MCQ answer extraction with edge cases
  2. `test_extraction.py`: Math answer normalization (`\frac`, decimals, etc.)
  3. `test_client.py`: Retry logic, truncated JSON detection
  4. `test_sandbox.py`: Timeout enforcement, stdout capture
  5. `test_reporting.py`: JSONL field completeness validation
  6. `test_gpqa.py`: Answer permutation correctness
- `README.md`: Setup, run instructions, architecture overview
- Final full run against `buggy_server.py` → commit results

---

## JSONL Output Schema

Every sample across all benchmarks includes:

```json
{
  "task": "gpqa|math500|livecodebench",
  "sample_id": "unique_id",
  "prompt": "full prompt sent to model",
  "raw_response": "raw model output text",
  "parsed_answer": "extracted answer before normalization",
  "normalized_answer": "answer after normalization pipeline",
  "ground_truth": "expected correct answer",
  "score": 1.0,
  "error_category": "none|parse_failure|api_error|timeout|truncated_json|rate_limit|server_error",
  "latency_ms": 245,
  "timestamp": "2026-02-16T10:30:00Z"
}
```

MATH500 additionally includes: `match_method` (boxed|answer_pattern|fallback)

LiveCodeBench additionally includes: `generated_code`, `test_results`, `pass_at_1`, `execution_time_ms`

---

## Dependencies

```
aiohttp>=3.9           # Async HTTP client
datasets>=2.14         # HuggingFace datasets loading
pyyaml>=6.0            # Config file parsing
tqdm>=4.65             # Progress bars
numpy>=1.24            # Metrics computation
pandas>=2.0            # Data manipulation
pytest>=7.0            # Testing
```

---

## Git Commit Plan (8+ commits)

| # | Message | Content |
|---|---------|---------|
| 1 | `feat: project scaffold, config system, and package structure` | Dirs, config.yaml, config.py, pyproject.toml, .gitignore |
| 2 | `feat: robust API client with retry, backoff, and error handling` | client.py with retry logic, timeout, truncated JSON detection |
| 3 | `feat: base task abstraction and answer extraction utilities` | base.py, extraction.py, logging.py |
| 4 | `feat: GPQA Diamond evaluator with full 198-question support` | gpqa.py, test_gpqa.py |
| 5 | `feat: MATH500 evaluator with multi-strategy answer extraction` | math500.py, test_extraction.py |
| 6 | `feat: sandboxed code execution with timeout and memory limits` | sandbox/executor.py, test_sandbox.py |
| 7 | `feat: LiveCodeBench evaluator with pass@1 reporting` | livecodebench.py |
| 8 | `feat: orchestrator, reporting, README, and full run results` | runner.py, reporting.py, README.md, results/ |

---

## Key Design Decisions

1. **Sync + ThreadPool over full async**: Simpler to implement and debug; `ThreadPoolExecutor` gives adequate concurrency for I/O-bound API calls against a single local server. Rate-limiting at 50 req/min means massive concurrency isn't needed.

2. **Cascading regex extraction over LLM-based judging**: The mock server returns random answers, so we don't have a second LLM to use as an equality judge. We rely purely on deterministic extraction + normalization.

3. **subprocess.Process + os.killpg for sandboxing**: Following LiveCodeBench's pattern, use process-level isolation with hard kill on timeout. More reliable than threading for untrusted code.

4. **Answer permutation for GPQA**: Following OpenAI simple-evals, randomize answer order per question with a seeded RNG for reproducibility. This prevents position bias.

5. **Retry truncated JSON on 200**: The server sometimes truncates responses mid-JSON. Detect via `json.JSONDecodeError` when status is 200, and retry — unique behavior of the buggy server that standard retry logic wouldn't catch.

---

## Risk Mitigations

| Risk | Mitigation |
|---|---|
| Rate limit exhaustion stalls progress | Dynamic throttling: if seeing 429s, reduce concurrency and add delays |
| GPQA dataset requires HF auth/agreement | Accept terms on HuggingFace, use `HF_TOKEN` env var |
| LiveCodeBench code execution may be dangerous | Strict sandbox: timeout, memory limit, process group kill, temp dir isolation |
| Mock server returns random answers → near-chance accuracy | Expected behavior; the eval harness correctness is validated by the infrastructure, not model accuracy |
| Slow chunked responses block workers | Per-request timeout prevents indefinite hangs; worker returns and retries |
