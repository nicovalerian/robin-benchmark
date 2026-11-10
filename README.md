# ROBIN: Robust Instruction Benchmark for Indonesian Noise

> Benchmark for evaluating LLM instruction robustness under Indonesian code-mixing

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![HuggingFace Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-ROBIN-yellow.svg)](https://huggingface.co/datasets/nicovalerian/robin-benchmark)

## What is ROBIN?

Instruction-following benchmarks commonly use standardized monolingual prompts, although users may communicate informally and mix languages. ROBIN represents **750 constraint-augmented base instructions** under one formal-Indonesian control and three synthetic Indonesian-English code-mixed conditions (Lexical Borrowing, Morphological Fusion, Intra-Sentential Switching), yielding **3,000 condition-indexed prompt records**. The paper evaluates five language models on all variants (**15,000 outputs**).

Responses are scored with directly verifiable constraint checks (keyword, length, format) and reference-based content-fidelity metrics (ROUGE-L F1 and an IndoBERT-based BERTScore-style F1). **Worst-Level Robustness (WLR)**, the lowest condition-level mean composite score, is the primary metric; **Performance Drop Rate (PDR)** is a secondary relative-change diagnostic and never ranks models.

The frozen artifact and all 15,000 model outputs and scores are on the Hugging Face Hub at [**`nicovalerian/robin-benchmark`**](https://huggingface.co/datasets/nicovalerian/robin-benchmark):

```python
from datasets import load_dataset

robin = load_dataset("nicovalerian/robin-benchmark", "benchmark", split="test")     # 750 base instructions
outputs = load_dataset("nicovalerian/robin-benchmark", "inference", split="test")   # 15,000 raw responses
scores = load_dataset("nicovalerian/robin-benchmark", "evaluation", split="test")   # 15,000 scored records
```

The same benchmark file is committed here as `data/processed/robin_dataset.jsonl` (SHA-256 `66c584aecef676eec029d448674177315c392ffe35ba8bcbcdf72dc5d84b8720`), with the paper's Phase 4 outputs in `results/robin_final/`.

## Results (paper, Table 2–3)

| Model (OpenRouter) | S₀ | **WLR** | Mean-Lvl | PDR-R |
|--------------------|----|---------|----------|-------|
| `inclusionai/ling-3.0-flash` | 0.7618 | **0.7470** | 0.7551 | 98.54 |
| `deepseek/deepseek-v4-flash-0731` | 0.7554 | 0.7425 | 0.7485 | 98.48 |
| `openai/gpt-5.4-nano` | 0.7462 | 0.7333 | 0.7397 | 98.37 |
| `qwen/qwen3.7-flash` | 0.7021 | 0.6923 | 0.6982 | 98.96 |
| `z-ai/glm-4.7-flash` | 0.6568 | 0.6544 | 0.6560 | 99.41 |

| Model | Keyword | Length | Format |
|-------|---------|--------|--------|
| Ling 3.0 Flash | 0.9643 | 0.8450 | 0.9857 |
| DeepSeek V4 Flash 0731 | 0.9830 | 0.8157 | 0.9950 |
| GPT-5.4 Nano | 0.9543 | 0.7500 | 0.9937 |
| Qwen 3.7 Flash | 0.9397 | 0.7073 | 0.9157 |
| GLM 4.7 Flash | 0.8990 | 0.5377 | 0.9240 |

Ling 3.0 Flash has the highest WLR point estimate, but its difference from DeepSeek V4 Flash 0731 (0.0045, paired bootstrap 95% CI [−0.0036, 0.0135]) is not statistically decisive. Length compliance is the lowest constraint-specific rate for every model. GLM has the highest PDR-derived score but the lowest WLR, which is why PDR does not rank models.

![Robustness comparison](results/robin_final/figures/robustness_comparison.png)

## Perturbation Methodology

Each instruction is rewritten into one formal-Indonesian control and three synthetic code-mixed conditions. The L0–L3 rewrites and the gold references use the same OpenAI-compatible `gpt-5.6-luna` deployment (temperature omitted because the endpoint rejects zero; `max_completion_tokens=2048`). The three formal constraint lines are appended verbatim to every condition (all 2,250 L1–L3 records retain them exactly). The prompt templates are in `configs/perturbation_prompts.yaml` and reproduced in the paper's Appendix.

| Level | Name | Operationalization |
|-------|------|--------------------|
| **L0** | Formal Indonesian (control) | Generated formal-Indonesian rendering |
| **L1** | Lexical Borrowing | Eligible Indonesian content nouns replaced by common English forms, no morphological fusion. 8 L1 bodies equal L0 (no eligible replacement), so 3,000 records contain 2,992 unique prompt strings |
| **L2** | Morphological Fusion | Item-specific style plan: an Indonesian affix family (`me-/meng-`, `di-`, `nge-`) on an English root plus exactly one English connector (`actually`, `by the way`, `I mean`, `literally`, `basically`). 717 hyphenated and 33 concatenated affix-root forms |
| **L3** | Intra-Sentential Switching | One English clause, one assigned respelling (`rait`, `rili`, `litereli`, `greit`, `plis`), and an English suffix on an Indonesian root (`-ly` in 461 items; author-designed `-ness` in 289) |

Author-defined post-processing (seeded) then attempts L2 particle insertion (p=0.65 when absent) and, at L3, IndoCollex substitution (p=0.45 per mapping), particle insertion (p=0.80) and non-opening-sentence lowercasing (p=0.25). Because the exact constraint block is restored afterwards, detected particles remain in only 15 L2 and 21 L3 bodies.

These conditions draw on documented patterns (Wijaya & Bram 2021; Budiarti & Faris 2023; Suswandi 2022; IndoCollex, Wibowo et al. 2021) but are **not validated as representative Jaksel**; ROBIN is a controlled synthetic stress test.

### Dataset examples

Drawn verbatim from `data/processed/robin_dataset.jsonl` (constraint lines omitted; they follow each body unchanged).

**`robin_00029`** (coding) — *Pastikan jawaban mengandung kata: switch, jauh. / Jawab dalam 15-28 kata. / Gunakan format daftar bernomor.*

| Level | Instruction body |
|-------|------------------|
| L0 | Susunlah kalimat yang menjelaskan fungsi produk. |
| L1 | Susunlah sentence yang menjelaskan fungsi produk. |
| L2 | Susunlah kalimat yang nge-describe fungsi produk, I mean, untuk menjelaskan kegunaannya. |
| L3 | Fungsi produk litereli jelasly, compose a sentence explaining it. |

**`robin_00333`** (information extraction) — *Pastikan jawaban mengandung kata: jumlah, paling. / Jawab dalam 109-202 kata. / Gunakan format poin-poin.*

| Level | Instruction body |
|-------|------------------|
| L0 | Jelaskan konsep pendapatan marjinal. |
| L1 | Jelaskan konsep revenue marjinal. |
| L2 | Meng-explain konsep pendapatan marjinal, basically. |
| L3 | Konsep pendapatan marjinal, jelasly — plis, explain marginal revenue. |

## Quick Start

### 1. Install

```bash
git clone https://github.com/nicovalerian/robin-benchmark.git
cd robin-benchmark

python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate # Linux/Mac

pip install -r requirements.txt
```

> Requires **Python 3.10+** (developed and tested on Python 3.13).

### 2. Configure API keys

```bash
cp .env.example .env
```

```env
OPENROUTER_API_KEY=...   # Phase 2 inference (paper protocol)
ELICE_API_KEY=...        # only needed to regenerate Phase 1 with gpt-5.6-luna
```

### 3. Reproduce the paper evaluation

The committed dataset is the canonical artifact — **do not re-run Phase 1** to reproduce the paper (LLM rewrites are not deterministic). Run Phases 2–4 against it with the paper's OpenRouter configuration:

```bash
python scripts/run_phase2.py --config configs/openrouter_phase2.yaml --output data/output/inference_results.jsonl
python scripts/run_phase3.py --input data/output/inference_results.jsonl --output data/output/evaluation_results.jsonl
python scripts/run_phase4.py --input data/output/evaluation_results.jsonl --output results/ --visualize
```

To re-score the released outputs without any API calls, download `results/robin_phase2_final_15000.jsonl` from the Hugging Face repo and pass it to Phase 3, or pass `results/robin_phase3_final_15000.jsonl` directly to Phase 4.

Failed cells can be re-queried exactly with `scripts/retry_failed_phase2.py --config <cfg> --model <name> --keys <keys.jsonl> --output <out.jsonl>`. The paper run retried 118 DeepSeek cells (`configs/deepseek_retry.yaml`, `retry_keys/deepseek_failed_118.jsonl`) and 3 GLM cells (`configs/glm_retry.yaml`, `retry_keys/glm_failed3_retry_keys.jsonl`); every retained record carries its own `protocol_hash` and `effective_parameters`.

### One-command pipeline

```bash
python scripts/run_pipeline.py --config configs/openrouter_phase2.yaml           # Phases 2-4 (Phase 1 only if the dataset is missing)
python scripts/run_pipeline.py --models ling-3.0-flash,glm-4.7-flash             # subset of models
python scripts/run_pipeline.py --limit 50 -y                                     # first 50 samples, no prompt
python scripts/run_pipeline.py --smoke -y                                        # 1 sample, isolated outputs
```

## Models Evaluated

All five models were queried through OpenRouter without the reference. Reasoning was requested off, output was capped at 1,024 tokens, and temperature was omitted for GPT-5.4 Nano (unsupported) and set to 0.0 otherwise. Requests used a 240-second timeout, up to four attempts, eight concurrent requests per model and 120 requests per minute. 45 nonempty responses reached the token limit and one DeepSeek L1 response has `finish_reason=error`; all are retained. The `gpt-5.6-luna` reference generator is not evaluated.

| Name | OpenRouter `model_id` |
|------|-----------------------|
| `gpt5.4-nano` | `openai/gpt-5.4-nano` |
| `deepseek-v4-flash-0731` | `deepseek/deepseek-v4-flash-0731` |
| `qwen3.7-flash` | `qwen/qwen3.7-flash` |
| `glm-4.7-flash` | `z-ai/glm-4.7-flash` |
| `ling-3.0-flash` | `inclusionai/ling-3.0-flash` |

### Adding models / Bring Your Own Key

ROBIN is model-agnostic. Each `inference.models` entry may set its own `base_url` and `api_key_env` (the env var *name*, never the key), so one run can mix OpenRouter, OpenAI-compatible gateways and local servers. Phase 2 validates every model's key up front and aborts naming any missing variable.

```yaml
inference:
  models:
    - name: "ling-3.0-flash"
      provider: "openrouter"
      base_url: "https://openrouter.ai/api/v1"
      api_key_env: "OPENROUTER_API_KEY"
      model_id: "inclusionai/ling-3.0-flash"
      temperature: 0.0
      max_output_tokens: 1024
      extra_body:
        reasoning: {enabled: false}
        provider: {require_parameters: true}
```

### Using a local LLM

Any OpenAI-compatible server works (Ollama, vLLM, LM Studio). Set `base_url` on the model entry; a loopback server needs no key.

```yaml
inference:
  models:
    - name: "llama3.2-local"
      model_id: "llama3.2"
      base_url: "http://localhost:11434/v1/"
```

Default base URLs: Ollama `http://localhost:11434/v1/`, LM Studio `http://localhost:1234/v1/`, vLLM `http://localhost:8000/v1/`. For a remote server, also set `api_key_env`.

## Pipeline Phases

| Phase | What it does | Output |
|-------|--------------|--------|
| **Phase 1** | Samples instructions, injects constraints, generates gold references and L0–L3 rewrites | `data/processed/robin_dataset.jsonl` |
| **Phase 2** | Runs inference for every model × instruction × condition (`original_input` appended unchanged) | `data/output/inference_results.jsonl` |
| **Phase 3** | Constraint checking + ROUGE-L + IndoBERT BERTScore-style scoring | `data/output/evaluation_results.jsonl` (flat, one record per model × sample × level) |
| **Phase 4** | WLR (primary), mean-level, per-type CPR, PDR (secondary), figures | `results/*.json`, `results/figures/` |

**Phase 1.** 750 instructions are sampled from `ilhamfadheel/alpaca-cleaned-indonesian` with a keyword-and-heuristic classifier (logical reasoning 25%, mathematical reasoning / creative writing / information extraction 20% each, coding 15%), giving 188/150/150/150/112 items after excluding empty, sub-20-character and duplicate instructions. Each item receives three programmatically verifiable constraints: keywords and a word-count range derived from the upstream response, and a category-specific format (JSON, numbered list, bullet list, or table). One gold reference per base instruction is generated from the constraint-augmented source instruction (not from the L0 rewrite) and shared across L0–L3. 679 of 750 references satisfy all three constraints (733 keyword, 700 length, 745 format); all are retained, and the 679-item subset preserves the WLR order.

**Phase 3.** For response *i*, cᵢ ∈ {0, ⅓, ⅔, 1} is the fraction of constraints satisfied. Keyword terms must match outside longer Unicode words; length is an inclusive whitespace-delimited count; JSON must parse; list formats need a matching line marker; tables need two consecutive pipe-delimited rows with stable column counts. The semantic score is Mᵢ = 0.5·ROUGE-L F1 (unstemmed) + 0.5·BERTScore-style F1 with `indobenchmark/indobert-base-p1` (greedy cosine matching at hidden layer 9, 512-token truncation, no IDF, no baseline rescaling). The composite is sᵢ = 0.6·cᵢ + 0.4·Mᵢ.

## Robustness Metrics

For model *m* and condition ℓ ∈ {0, 1, 2, 3}, let S_ℓ be the mean composite over that condition's responses.

### Worst-Level Robustness (WLR) — primary

```
WLR_m = min_{ℓ ∈ {0,1,2,3}} S_ℓ
```

WLR includes L0, so it reflects the weakest condition-level performance on the absolute [0, 1] scale; a high WLR needs both baseline capability and consistent performance across code-mixed conditions. It is reported beside the mean-level score S̄ = ¼ Σ S_ℓ.

### Per-constraint-type compliance

Phase 4 reports keyword / length / format compliance separately (`cpr_by_type` in `summary.json`, `cpr_by_type_by_level` in `pdr_analysis.json`, `figures/constraint_breakdown.png`). Length has the largest between-model range (0.3073 vs 0.0840 keyword and 0.0793 format).

### Performance Drop Rate (PDR) — secondary diagnostic

For each component q ∈ {C (CPR), M (semantic), S (composite)} and ℓ ∈ {1, 2, 3}:

```
PDR^(q)_ℓ = (q_0 − q_ℓ) / q_0 × 100%      (0 if q_0 = 0)
PDR_ℓ     = mean over q of PDR^(q)_ℓ        (component PDRs rounded to 2 decimals first)
R_m       = max(0, 100 − mean(PDR_1, PDR_2, PDR_3))
```

Negative values (improvement over L0) allow R_m > 100. PDR describes relative change only: normalization can make a model that is weak at L0 appear stable, so it does not rank models.

## Output Files

```
data/processed/
├── robin_dataset.jsonl          # Frozen 750-item benchmark (paper artifact)
└── robin_dataset_stats.json

results/robin_final/             # Paper Phase 4 outputs (15,000 scored records)
├── summary.json                 # WLR ranking + mean-level + cpr_by_type + PDR
├── pdr_analysis.json            # Per-level scores, per-level CPR by type, PDR details
├── skill_profiles.json          # Scores by task category
└── figures/                     # robustness_comparison, constraint_breakdown, perturbation_trend, pass_rate_heatmap, perturbation_examples

artifacts/                       # Phase 2 run manifests for the paper run (dataset hash, code commit, per-model protocol)
```

The full 15,000-record Phase 2 and Phase 3 JSONL files (`robin_phase2_final_15000.jsonl`, `robin_phase3_final_15000.jsonl`) are hosted on the Hugging Face dataset repo under `results/`.

## Limitations

- L1–L3 are synthetic `gpt-5.6-luna` rewrites, not naturally occurring Jaksel; bodies have not been human-validated for semantic equivalence or naturalness. The L3 `-ness` construction is author-designed.
- Content fidelity is approximated by ROUGE-L and an IndoBERT-based measure against one generated reference; response language is not audited.
- Orthographic form and tokenizer fragmentation are uncontrolled, so condition differences cannot be attributed causally to a single mechanism.
- Results cover five API-served models under one OpenRouter protocol and may depend on aliases, provider routing and time of access.
- The upstream source revision, pre-sampling pool size and dataset-construction seed were not retained; the frozen artifact is identified by its hash.
- Category labels are keyword-heuristic.

## References

- Wijaya, A. D. & Bram, B. (2021). A sociolinguistic analysis of Indoglish phenomenon in South Jakarta. *PROJECT*, 4(4). https://doi.org/10.22460/project.v4i4.p672-684
- Budiarti, D. & Faris, I. N. I. (2023). "Jangan Di-Read!": Indonesian affixation to English words on social media. *LITE*, 19(1). https://doi.org/10.33633/lite.v19i1.7492
- Suswandi, I. (2022). Analysis of affix -ly as modifier of adverb of manner in social media posts. *Jurnal Kata*, 6(2). https://doi.org/10.22216/kata.v6i2.1569
- Wibowo, H. A., et al. (2021). IndoCollex: A testbed for morphological transformation of Indonesian colloquial words. *Findings of ACL-IJCNLP 2021*. https://doi.org/10.18653/v1/2021.findings-acl.280

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

## License

Apache License 2.0
