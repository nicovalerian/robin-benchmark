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
pretty_name: ROBIN (Robust Instruction Benchmark for Indonesian Noise)
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

# ROBIN: Robust Instruction Benchmark for Indonesian Noise

ROBIN tests how well language models follow instructions written in Indonesian-English code-mixed text.

It has 750 Indonesian instructions. Each one carries three checkable constraints and comes in four versions: one formal Indonesian version and three increasingly code-mixed ones. That gives 3,000 prompts. This repo also has the 15,000 responses (five models × 3,000 prompts) and scores from the paper.

- Paper: *ROBIN: Benchmark for evaluating LLM instruction robustness under Indonesian code-mixing* (MethodsX, manuscript MEX-D-26-02459, under review)
- Code: <https://github.com/nicovalerian/robin-benchmark>
- Source instructions: [`ilhamfadheel/alpaca-cleaned-indonesian`](https://huggingface.co/datasets/ilhamfadheel/alpaca-cleaned-indonesian), an Indonesian translation of Stanford Alpaca

```python
from datasets import load_dataset

robin = load_dataset("nicovalerian/robin-benchmark", "benchmark", split="test")     # 750 instructions
outputs = load_dataset("nicovalerian/robin-benchmark", "inference", split="test")   # 15,000 model responses
scores = load_dataset("nicovalerian/robin-benchmark", "evaluation", split="test")   # 15,000 scored responses
```

## Files

| Path | Rows | Contents |
|------|------|----------|
| `data/robin_dataset.jsonl` | 750 | The benchmark, one row per instruction with all four versions. SHA-256 `66c584aecef676eec029d448674177315c392ffe35ba8bcbcdf72dc5d84b8720` |
| `data/robin_dataset_stats.json` | | Category counts and generator info |
| `results/robin_phase2_final_15000.jsonl` | 15,000 | Raw responses, one per model, instruction and version, with finish reason, token usage and request settings |
| `results/robin_phase3_final_15000.jsonl` | 15,000 | Scores for each response: `cpr`, `constraint_results`, `rouge_l`, `bert_f1`, `semantic`, `composite` |
| `results/summary.json` | | Models ranked by WLR, with the four-version average, per-constraint rates and PDR score |
| `results/pdr_analysis.json` | | Per-version scores, per-version constraint rates and PDR details |
| `results/skill_profiles.json` | | Scores by task category |
| `results/figures/` | | Figures from the paper |
| `manifests/` | | Run manifests for the inference run (dataset hash, code commit, model settings) |

## Fields in `data/robin_dataset.jsonl`

| Field | Contents |
|-------|----------|
| `id` | `robin_00000` to `robin_00749` |
| `category` | `logical_reasoning` (188), `mathematical_reasoning` (150), `creative_writing` (150), `information_extraction` (150) or `coding` (112) |
| `original_instruction` | The source Indonesian Alpaca instruction |
| `original_input` | Extra input from the source, present for 310 items. It's appended unchanged to all four versions when a model is run. |
| `gold_response` | One reference answer per instruction, written from the constrained original instruction (not the L0 rewrite) and shared by all four versions |
| `constraints` | Three constraints: `keyword`, `length` (word range, both ends allowed) and `format` (`json`, `numbered_list`, `bullet_list` or `table`). Each has `requirement`, `verification_regex` and `target_value`. |
| `perturbations` | The four versions: `level_0_clean`, `level_1_mild`, `level_2_jaksel`, `level_3_adversarial` |
| `perturbation_meta` | Generator, deployment, the style plans assigned for L2 and L3, and request settings |

The three constraint lines are copied unchanged onto the end of every version, and all 2,250 code-mixed prompts keep them exactly.

## The four versions

| Level | Name | What changes |
|-------|------|--------------|
| L0 | Formal Indonesian | Nothing. This is the control. |
| L1 | Lexical borrowing | A few Indonesian nouns become English (`teks` to `text`, `daftar` to `list`). In 8 items there was nothing to swap, so L1 equals L0 and the 3,000 prompts contain 2,992 distinct strings. |
| L2 | Morphological fusion | An Indonesian prefix goes on an English verb (`me-`, `di-` or `nge-`), plus one English filler such as `actually`, `by the way`, `I mean`, `literally` or `basically`. 717 of these words are hyphenated and 33 are written joined. |
| L3 | Intra-sentential switching | One clause switches to English, one English word is respelled the Indonesian way (`rait`, `rili`, `litereli`, `greit`, `plis`), and an Indonesian word gets an English suffix (`-ly` in 461 items, `-ness` in 289). |

Here's `robin_00137` (creative writing), copied exactly. Every version ends with the same three lines: *Pastikan jawaban mengandung kata: virusnya, dokter. / Jawab dalam 6-24 kata. / Gunakan format poin-poin.*

| Level | Instruction |
|-------|-------------|
| L0 | Sampaikan sebuah lelucon mengenai komputer. |
| L1 | Sampaikan sebuah lelucon mengenai computer. |
| L2 | Nge-share sebuah lelucon mengenai komputer, I mean, satu lelucon. |
| L3 | Topik komputer, litereli luculy, tell a joke about computers. |

## How it was built

- We sampled 750 instructions and sorted them into five categories with keyword rules (25% logical reasoning, 20% each for math, creative writing and information extraction, 15% coding). Empty, very short (under 20 characters) and duplicate instructions were dropped.
- The keyword and length constraints come from the original Alpaca answer. The format is chosen by category.
- `gpt-5.6-luna` wrote both the four versions and the reference answers. Temperature was left unset because the endpoint rejects 0, and output was capped at 2,048 tokens.
- A seeded script then tries to add some colloquial touches. At L2 it adds a discourse particle 65% of the time when none is present. At L3 it applies IndoCollex word swaps (45% per mapping), adds particles (80%) and lowercases some sentence starts (25%). Restoring the constraint lines undoes most of this, so only 15 L2 and 21 L3 prompts end up with a particle.
- 679 of the 750 references pass all three of their own constraints (733 pass keyword, 700 length, 745 format). We kept all of them.

## Evaluation and results

Five models were called through OpenRouter: `openai/gpt-5.4-nano`, `deepseek/deepseek-v4-flash-0731`, `qwen/qwen3.7-flash`, `z-ai/glm-4.7-flash` and `inclusionai/ling-3.0-flash`. None of them saw the reference answer. Reasoning was off and output was capped at 1,024 tokens. Temperature was 0.0, except for GPT-5.4 Nano, where it was left unset. Every model, instruction and version has exactly one record. 45 responses hit the token limit and one DeepSeek L1 response ended with `finish_reason=error`. We kept all of them.

Each response gets a constraint score (the fraction of its three constraints it meets: 0, 1/3, 2/3 or 1) and a similarity score (the average of ROUGE-L F1 and a BERTScore-style F1 from `indobenchmark/indobert-base-p1`, using layer 9, greedy matching, a 512-token cutoff and no IDF weighting or rescaling). The final score is 0.6 × constraint score + 0.4 × similarity.

Models are ranked by Worst-Level Robustness (WLR), the lowest of their four per-version average scores. PDR measures relative change from L0 and is only a side diagnostic. We don't rank on it.

| Model | S0 | WLR | Mean-Lvl | PDR-R |
|-------|----|-----|----------|-------|
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

Ling scores highest, but its lead over DeepSeek (0.0045) is within noise. The paired bootstrap 95% CI is [-0.0036, 0.0135]. Length is the hardest constraint for every model.

## Limitations

- The code-mixed prompts are synthetic. No native speaker has checked whether they read naturally or keep the original meaning. The L3 `-ness` suffix is our own invention.
- Answer quality is measured against a single generated reference with automatic metrics, which can't stand in for human judgment. We don't check what language a response is written in.
- Spelling choices (hyphenated vs. joined prefixes) and how each tokenizer splits words weren't controlled, so differences between versions can't be pinned on a single feature.
- Results come from five hosted models under one setup. They may change with provider routing, model versions or timing.
- We didn't record the exact upstream dataset revision, pool size or sampling seed. The released file is identified by its hash.
- Task categories come from keyword matching and are rough.

## License

Apache 2.0. The source instructions come from `ilhamfadheel/alpaca-cleaned-indonesian` (based on Stanford Alpaca), so please respect those terms too.

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
