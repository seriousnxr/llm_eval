================================================================================
              LLM EVALUATION INFRASTRUCTURE: TAKE-HOME PROJECT
================================================================================

OVERVIEW
--------
Build a lightweight, extensible LLM evaluation system. Run two complete 
benchmarks against a model served via an OpenAI-compatible API, handle the 
provided mock server's real-world behaviors, and produce structured results.

⚠️ IMPORTANT SETUP:
   • Download buggy_server.py (see download section below) - this is your MODEL SERVER
   • Run it with: python3 buggy_server.py --port 8000
   • It provides an OpenAI-compatible endpoint at localhost:8000/v1/chat/completions
   • DO NOT modify this server - treat it as a black box (like vLLM/SGLang)
   • Your job is to build the EVALUATION CLIENT that talks to this server


================================================================================
                            REQUIRED DELIVERABLES
================================================================================

You must complete ALL of the following:

1. Full run of GPQA Diamond (all 198 questions)
   Report accuracy with per-sample JSONL output

2. Full run of MATH500 (all 500 problems)
   Report accuracy with per-sample JSONL output

3. Per-sample JSONL fields must include:
   • prompt
   • raw_response
   • parsed_answer
   • normalized_answer
   • ground_truth
   • score
   • error_category
   • latency_ms

4. Summary report (JSON or Markdown format):
   • Aggregate accuracy per eval
   • Error breakdown by category
   • Wall-clock timing
   • Throughput (requests/sec)

5. README with setup instructions

6. Git repository with 8+ commits showing incremental development

7. All results from a single complete run against the provided mock server


================================================================================
                         IMPLEMENTATION (3-4 HOURS)
================================================================================

WHAT YOU'RE BUILDING
--------------------
You are building an EVALUATION HARNESS - the client-side system that:
  1. Loads benchmark datasets (GPQA, MATH500, LiveCodeBench)
  2. Sends prompts to the model server (localhost:8000)
  3. Parses and evaluates responses
  4. Generates accuracy reports

You are NOT building the model server - that's provided (buggy_server.py).


API CLIENT
----------
• Build a client that calls http://localhost:8000/v1/chat/completions
• Support configurable parameters: model, temperature, max_tokens, top_p, stop
• Retry logic with exponential backoff for 429 and 503 errors
• Per-request timeout configuration
• Structured error logging in JSONL format


GPQA DIAMOND (198 questions, full run)
---------------------------------------
• Load the full diamond split dataset
• Use few-shot multiple-choice prompting
• Parse answer letter (A/B/C/D) from model response
• Handle ambiguous responses gracefully
• Report final accuracy


MATH500 (500 problems, full run)
---------------------------------
• Prompt with \boxed{} instruction for answer formatting
• Multi-strategy answer extraction:
  1. \boxed{} extraction
  2. "the answer is" pattern matching
  3. Last expression fallback
  
• Answer normalization:
  • Fractions (e.g., 1/2 vs 0.5)
  • LaTeX cleanup
  • Trailing zeros
  
• JSONL output must include:
  raw_answer, parsed_answer, normalized_answer, ground_truth, match_method


ORCHESTRATION & REPORTING
--------------------------
• Concurrent request handling (async or threads)
• Configurable batch size
• YAML or JSON configuration file
• Graceful interruption with partial result saving
• Generate summary report with:
  - Per-eval accuracy
  - Error breakdown by category
  - Wall-clock timing
  - Throughput metrics


CODE QUALITY REQUIREMENTS
--------------------------
• Separate modules for different concerns
• Type hints throughout
• Minimum 5 unit tests
• Git repository with 8+ incremental commits
• Clear commit messages showing development progression


================================================================================
                    ADD-ON: LIVECODEБENCH (REQUIRED)
================================================================================

⚠️ This is the hardest part of the interview and demonstrates real systems
   capability. You MUST implement this.

OBJECTIVE
---------
Build sandboxed code execution infrastructure for evaluating code generation
models against test cases.


IMPLEMENTATION REQUIREMENTS
----------------------------

1. LiveCodeBenchTask Class
   • Send coding problem to model
   • Receive generated code
   • Execute code in sandbox against test cases
   • Report pass@1 accuracy

2. Dataset
   Use: livecodebench/code_generation_lite, release_v1
   • Full dataset: ~400 problems
   • Easy-only subset: ~130 problems (acceptable for initial implementation)

3. Safe Subprocess Execution (Critical)
   Your sandbox MUST handle:
   
   ✓ Timeout enforcement
     - Hard limit per test case (e.g., 5 seconds)
     - Clean process termination on timeout
   
   ✓ Memory limits
     - Prevent out-of-memory attacks
   
   ✓ Network isolation
     - Block network access during execution
   
   ✓ Temporary file isolation
     - Create temporary directory per execution
     - Clean up after completion or failure
   
   ✓ Stdout/stderr capture
     - Capture all output for comparison
   
   ✓ Process cleanup
     - Kill parent AND child processes
     - Prevent process leaks

4. Test Case Execution
   • Read stdin input from test case
   • Pipe to generated code
   • Capture stdout output
   • Compare against expected output
   • Handle encoding issues (binary vs text mode)

5. Reporting
   Per-problem JSONL output:
   • problem_id
   • generated_code
   • test_results (array of pass/fail per test case)
   • pass@1 (1 if all tests pass, 0 otherwise)
   • execution_time_ms
   • error_category (timeout, runtime_error, wrong_answer, etc.)



ESTIMATED TIME
--------------
45-60 minutes for working implementation with basic safety


================================================================================
                              DELIVERABLES
================================================================================

Submit your EVALUATION HARNESS (client-side code):

  • Working Implementation: Complete evaluation client with tests, documentation,
    and results (GPQA Diamod, MATH500 and LiveCodeBench) from a full run against the provided buggy_server.py

Note: You do NOT submit server code - only the evaluation client.


================================================================================
                               SUBMISSION
================================================================================

Email your submission to: recruiting@goaly.ai

Include:
  • Link to private GitHub repository
  • README with setup instructions
  • Summary report with results


================================================================================
                    DATASETS & SERVER INFORMATION
================================================================================

DATASETS
--------
• GPQA Diamond: 
  https://huggingface.co/datasets/Idavidrein/gpqa (diamond split)

• MATH500:
  https://huggingface.co/datasets/lighteval/MATH (first 500 problems)
  https://huggingface.co/datasets/math-ai/math500
• LiveCodeBench:
  https://huggingface.co/datasets/livecodebench/code_generation_lite


MOCK SERVER (PROVIDED - DO NOT MODIFY)
--------------------------------------
You will receive buggy_server.py - a standalone Python script that acts as
your model server. It replaces the need for GPU/vLLM/SGLang.

To start the server:
    python3 buggy_server.py --port 8000

Once running, it provides:
    Endpoint: http://localhost:8000/v1/chat/completions
    Format: OpenAI-compatible (same as GPT API)
    Responses: Returns mock model outputs for GPQA/MATH500/code prompts

⚠️ CRITICAL: DO NOT modify buggy_server.py
   • Treat it as a production black box (like a real vLLM endpoint)
   • The server may exhibit realistic production behaviors
   • Your evaluation client must be robust enough to handle any issues

Your job is to build the CLIENT CODE that talks to this server and handles
whatever behaviors it exhibits during testing.


================================================================================
                              GOOD LUCK!
================================================================================

This is a challenging project that tests real production engineering skills:
  • Building correct evaluation infrastructure
  • Handling unreliable APIs gracefully  
  • Implementing safe code execution
  • Writing maintainable, testable code

Questions? Email recruiting@goaly.ai


================================================================================
                              © Goaly AI
================================================================================