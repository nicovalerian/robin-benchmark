# ROBIN Benchmark — Claude Code Guide

## Python Interpreter
Always use `C:\Users\nico\AppData\Local\Microsoft\WindowsApps\python3.13.exe` for this project.

## Project Overview
ROBIN generates a 4-level Indonesian-English code-mixing benchmark dataset and evaluates LLM instruction-following robustness. The pipeline has four phases:
- **Phase 1** (`scripts/run_phase1.py`): Dataset generation — perturbs 750 base instructions into L0–L3 variants via DigitalOcean Serverless Inference (minimax-m2.5)
- **Phase 2** (`scripts/run_phase2.py`): Inference — runs all configured models on the 2,996 prompts
- **Phase 3** (`scripts/run_phase3.py`): Evaluation — constraint checker, semantic scorer, LLM-as-judge
- **Phase 4** (`scripts/run_phase4.py`): Analysis — PDR calculation, visualizations

## Key Architecture Decisions

### Constraints are evaluation metadata only
Constraints (keyword inclusion, word-count range) are derived from the gold response and stored in the JSONL `constraints` field. They are **never embedded in the instruction text** sent to models. Phase 3 checks model outputs against these constraints programmatically.

### Perturbation is split: LLM + post-processing
The LLM handles structural transformations (clause switching, morphological fusion, phonological respelling). Post-processing applies deterministic colloquial substitutions:
- **L2**: Indonesian discourse particles injected if absent (65% chance)
- **L3**: IndoCollex word substitutions (45%/word) + discourse particles (80%) + random lowercase on non-first sentence starts (25%)

### minimax-m2.5 is a reasoning model
It burns tokens on internal `reasoning_content` before writing output. Always use `max_tokens=2048` minimum. The httpx-level timeout does not reliably fire because partial HTTP responses reset the read timer — use `asyncio.wait_for(coro, timeout=120.0)` for a hard event-loop cancellation.

## Running Phase 1
```bash
python scripts/run_phase1.py --samples 750 --output data/processed/robin_dataset.jsonl
```
Config: `configs/full_config.yaml` — `max_concurrency: 20`, `rate_limit_per_minute: 120`, `max_retries: 6`.

Logs go to `logs/phase1_v3.log` when redirected. Stats written to `data/processed/dataset_stats.json`.

## Dataset Stats (current)
- 749 base instructions × 4 levels = **2,996 prompts**
- Category split: logical_reasoning 25%, mathematical_reasoning 20%, creative_writing 20%, information_extraction 20%, coding 15%
- Average instruction length: L0 12.7 words → L3 21.1 words
- Output: `data/processed/robin_dataset.jsonl`

## Common Pitfalls
- **Length constraint inversion**: `target_max = max(target_min + 10, int(word_count * 1.3))` ensures min < max for short gold responses. The bug was patched in `src/phase1/constraint_injector.py`.
- **Windows asyncio**: `asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())` is set in `run_phase1.py` main.
- **Unicode on Windows**: Always open files with `encoding='utf-8'` and wrap stdout with `io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')` when printing Indonesian text in scripts.
- **IndoCollex path**: `data/raw/indocollex.json` — 97 formal→colloquial entries. Missing file is handled gracefully (returns empty dict).
