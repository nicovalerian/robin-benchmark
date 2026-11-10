---
license: apache-2.0
language:
- id
- en
tags:
- code-mixing
- code-switching
- instruction-following
- robustness
- indonesian
- jaksel
- benchmark
pretty_name: ROBIN — Robust Instruction Benchmark for Indonesian Noise
size_categories:
- n<1K
- 10K<n<100K
task_categories:
- text-generation
configs:
- config_name: benchmark
  default: true
  data_files:
  - split: test
    path: data/robin_dataset.jsonl
- config_name: inference
  data_files:
  - split: test
    path: results/robin_phase2_final_15000.jsonl
- config_name: evaluation
  data_files:
  - split: test
    path: results/robin_phase3_final_15000.jsonl
---

# ROBIN: Benchmark for evaluating LLM instruction robustness under Indonesian code-mixing

**ROBIN** (Robust Instruction Benchmark for Indonesian Noise) represents **750
constraint-augmented base instructions** under one formal-Indonesian control and
three synthetic Indonesian-English code-mixed conditions, yielding **3,000
condition-indexed prompt records**. This repository also releases the **15,000
model outputs** (5 models × 3,000 prompts) and their scores used in the paper.

- Paper: *ROBIN: Benchmark for evaluating LLM instruction robustness under Indonesian code-mixing* (MethodsX, manuscript MEX-D-26-02459)
- Code: <https://github.com/nicovalerian/robin-benchmark>
- Upstream source: [`ilhamfadheel/alpaca-cleaned-indonesian`](https://huggingface.co/datasets/ilhamfadheel/alpaca-cleaned-indonesian) (Stanford Alpaca provenance)

```python
from datasets import load_dataset

robin = load_dataset("nicovalerian/robin-benchmark", "benchmark", split="test")     # 750 base instructions
outputs = load_dataset("nicovalerian/robin-benchmark", "inference", split="test")   # 15,000 raw responses
scores = load_dataset("nicovalerian/robin-benchmark", "evaluation", split="test")   # 15,000 scored records
```

## Files

| Path | Rows | Description |
|------|------|-------------|
| `data/robin_dataset.jsonl` | 750 | Frozen benchmark artifact (one row per base instruction, all four conditions). SHA-256 `66c584aecef676eec029d448674177315c392ffe35ba8bcbcdf72dc5d84b8720` |
| `data/robin_dataset_stats.json` | – | Generation statistics (category split, generator) |
| `results/robin_phase2_final_15000.jsonl` | 15,000 | Raw model responses, one per model × instruction × condition, with finish reason, usage and effective request parameters |
| `results/robin_phase3_final_15000.jsonl` | 15,000 | Per-response scores: `cpr`, `constraint_results`, `rouge_l`, `bert_f1`, `semantic`, `composite` |
| `results/summary.json` | – | WLR ranking with mean-level score, per-type CPR and PDR-derived score |
| `results/pdr_analysis.json` | – | Per-model condition-level scores, per-level CPR by type, PDR by level and component |
| `results/skill_profiles.json` | – | Scores by task category |
| `results/figures/` | – | Paper figures (robustness comparison, constraint breakdown, trend, heatmap, examples) |
| `manifests/` | – | Phase 2 run manifests (dataset hash, code commit, per-model protocol) |

## Benchmark structure (`data/robin_dataset.jsonl`)

| Field | Description |
|-------|-------------|
| `id` | `robin_00000` … `robin_00749` |
| `category` | `logical_reasoning` (188), `mathematical_reasoning` (150), `creative_writing` (150), `information_extraction` (150), `coding` (112) |
| `original_instruction` | Source Indonesian Alpaca instruction |
| `original_input` | Optional source input (nonempty for 310 items); appended unchanged to all four prompts at inference |
| `gold_response` | One shared reference per base instruction, generated from the constraint-augmented source instruction (not from the L0 rewrite) |
| `constraints` | Three verifiable constraints: `keyword`, `length` (inclusive word range), `format` (`json`, `numbered_list`, `bullet_list`, `table`) with `requirement`, `verification_regex`, `target_value` |
| `perturbations` | `level_0_clean`, `level_1_mild`, `level_2_jaksel`, `level_3_adversarial` |
| `perturbation_meta` | Generator, deployment, assigned L2/L3 style plans, effective parameters |

The three formal constraint lines are appended verbatim to every condition; all
2,250 L1–L3 records retain them exactly.

### Conditions

| Level | Name | Operationalization |
|-------|------|--------------------|
| **L0** | Formal Indonesian (control) | Generated formal-Indonesian rendering |
| **L1** | Lexical Borrowing | Eligible Indonesian content nouns replaced with common English forms, no morphological fusion. 8 L1 bodies equal L0 because no eligible replacement was realized (3,000 records, 2,992 unique prompt strings) |
| **L2** | Morphological Fusion | Item-specific Indonesian-affix + English-root plan (`me-/meng-`, `di-`, `nge-`) and exactly one English connector (`actually`, `by the way`, `I mean`, `literally`, `basically`). 717 hyphenated and 33 concatenated affix-root forms |
| **L3** | Intra-Sentential Switching | One English clause, one assigned respelling (`rait`, `rili`, `litereli`, `greit`, `plis`), and an English suffix on an Indonesian root (`-ly` in 461 items; author-designed `-ness` in 289) |

Example (`robin_00333`, information extraction; constraints: *Pastikan jawaban mengandung kata: jumlah, paling. / Jawab dalam 109-202 kata. / Gunakan format poin-poin.*):

| Level | Instruction body |
|-------|------------------|
| L0 | Jelaskan konsep pendapatan marjinal. |
| L1 | Jelaskan konsep revenue marjinal. |
| L2 | Meng-explain konsep pendapatan marjinal, basically. |
| L3 | Konsep pendapatan marjinal, jelasly — plis, explain marginal revenue. |

### Construction

- 750 instructions sampled from `ilhamfadheel/alpaca-cleaned-indonesian` with a keyword-and-heuristic category classifier (25/20/20/20/15%); empty, sub-20-character and duplicate instructions excluded.
- Keyword and length constraints derived from the upstream response; format constraint chosen by category.
- L0–L3 rewrites **and** gold references generated with the same OpenAI-compatible `gpt-5.6-luna` deployment; temperature omitted (the endpoint rejects zero), `max_completion_tokens=2048`.
- Author-defined post-processing: L2 particle insertion (p=0.65 when absent); L3 IndoCollex substitution (p=0.45 per mapping), particle insertion (p=0.80) and non-opening-sentence lowercasing (p=0.25). Restoring the exact constraint block leaves detected particles in only 15 L2 and 21 L3 bodies.
- 679 of 750 gold references satisfy all three constraints (733 keyword, 700 length, 745 format); all are retained.

## Evaluation protocol and results

Five models were queried through OpenRouter without the reference: `openai/gpt-5.4-nano`,
`deepseek/deepseek-v4-flash-0731`, `qwen/qwen3.7-flash`, `z-ai/glm-4.7-flash`,
`inclusionai/ling-3.0-flash`. Reasoning off, output capped at 1,024 tokens,
temperature omitted for GPT-5.4 Nano and 0.0 otherwise. One record per
model × instruction × condition; no missing or duplicate keys. 45 nonempty
responses reached the token limit and one DeepSeek L1 response has
`finish_reason=error`; all are retained (see paper for leave-out sensitivity).

Scoring per response *i*: constraint pass fraction *cᵢ* ∈ {0, ⅓, ⅔, 1};
semantic *Mᵢ* = 0.5·ROUGE-L F1 + 0.5·IndoBERT BERTScore-style F1
(`indobenchmark/indobert-base-p1`, layer 9, greedy, no IDF/rescaling, 512-token
truncation); composite *sᵢ* = 0.6·*cᵢ* + 0.4·*Mᵢ*.
**Primary metric: Worst-Level Robustness** WLR = min over L0–L3 of the
condition-level mean composite. PDR is a secondary diagnostic only and does not
rank models.

| Model | S₀ | **WLR** | Mean-Lvl | PDR-R |
|-------|----|---------|----------|-------|
| Ling 3.0 Flash | 0.7618 | **0.7470** | 0.7551 | 98.54 |
| DeepSeek V4 Flash 0731 | 0.7554 | 0.7425 | 0.7485 | 98.48 |
| GPT-5.4 Nano | 0.7462 | 0.7333 | 0.7397 | 98.37 |
| Qwen 3.7 Flash | 0.7021 | 0.6923 | 0.6982 | 98.96 |
| GLM 4.7 Flash | 0.6568 | 0.6544 | 0.6560 | 99.41 |

| Model | Keyword | Length | Format |
|-------|---------|--------|--------|
| Ling 3.0 Flash | 0.9643 | 0.8450 | 0.9857 |
| DeepSeek V4 Flash 0731 | 0.9830 | 0.8157 | 0.9950 |
| GPT-5.4 Nano | 0.9543 | 0.7500 | 0.9937 |
| Qwen 3.7 Flash | 0.9397 | 0.7073 | 0.9157 |
| GLM 4.7 Flash | 0.8990 | 0.5377 | 0.9240 |

The Ling–DeepSeek WLR difference (0.0045, paired bootstrap 95% CI [−0.0036, 0.0135])
is not statistically decisive. Length compliance is the lowest constraint rate
for every model.

## Limitations

- L1–L3 are **synthetic** rewrites by `gpt-5.6-luna`, not naturally occurring Jaksel utterances; semantic equivalence and naturalness have not been validated by native speakers. The L3 `-ness` construction is author-designed.
- Content fidelity uses ROUGE-L and an IndoBERT-based measure against one generated reference; these are proxies, not substitutes for human judgment. Response language is not audited.
- Hyphenated vs. concatenated forms and tokenizer fragmentation are uncontrolled, so differences cannot be attributed causally to individual code-mixing mechanisms.
- Results reflect five API-served models under one OpenRouter inference protocol and may depend on aliases, provider routing and time of access.
- The upstream source revision, pre-sampling pool size and dataset-construction seed were not retained; the frozen artifact is identified by its SHA-256 above.
- Category labels are heuristic.

## License

Apache-2.0. Source instructions derive from `ilhamfadheel/alpaca-cleaned-indonesian` (Stanford Alpaca); downstream use should also respect those terms.

## Citation

```bibtex
@article{marcello2026robin,
  title   = {ROBIN: Benchmark for evaluating LLM instruction robustness under Indonesian code-mixing},
  author  = {Marcello, Nico Valerian and Grizha, Jecelyn and Lakaoni, Bryan and Lucky, Henry},
  journal = {MethodsX},
  note    = {Manuscript MEX-D-26-02459, under review},
  year    = {2026}
}
```
