# ROBIN: Robust Instruction Benchmark for Indonesian Noise

> Benchmark for Evaluating LLM Instruction Robustness Under Indonesian Code-Mixing

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-green.svg)](https://opensource.org/licenses/Apache-2.0)

## What is ROBIN?

ROBIN evaluates how well LLMs follow instructions when text contains Indonesian code-mixing (mixing Indonesian with English, slang, typos). It measures **Performance Drop Rate (PDR)** - how much worse a model performs on noisy text vs clean text.

**Example perturbations:**
| Level | Example |
|-------|---------|
| Clean | "Jelaskan konsep machine learning dalam 50 kata" |
| Mild | "Explain concept machine learning dalam 50 kata" |
| Jaksel | "Jelaskan dong konsep machine learning gitu, dalam 50 kata ya" |
| Adversarial | "Jelasin dong gmn machine learning dlm 50 kata" |

## Perturbation Methodology

ROBIN implements a 4-level perturbation framework based on Indonesian-English code-mixing patterns documented in linguistic research.

### Perturbation Levels

| Level | Name | Description | Replacement Rate |
|-------|------|-------------|------------------|
| **0** | Clean | Standard Indonesian (baseline control) | 0% |
| **1** | Mild | English loanword substitution | 40% |
| **2** | Jaksel | Jakarta Selatan code-switching style | 50% patterns, 40% suffixes |
| **3** | Adversarial | Slang + typos combined | 50% slang + 8% typo |

### Level 0: Clean (Control)
Normalized Indonesian text serving as the performance baseline. All perturbation levels are compared against Level 0 to calculate PDR.

### Level 1: Mild English Mixing
Replaces Indonesian words with English cognates commonly used in formal/semi-formal Indonesian contexts. Based on loanword patterns in academic and instructional text.

**Example substitutions:**
| Indonesian | English | Indonesian | English |
|------------|---------|------------|---------|
| jelaskan | explain | teks | text |
| tuliskan | write | kalimat | sentence |
| buatlah | create | daftar | list |
| identifikasi | identify | hasil | result |
| analisis | analyze | masalah | problem |

**Linguistic basis:** Educated Indonesians frequently use English terms in instruction-following contexts, particularly for technical and academic vocabulary (Fauzi, 2015).

### Level 2: Jaksel Code-Switching
Simulates casual bilingual speech patterns common in Jakarta Selatan (South Jakarta) urban youth. Characterized by:

1. **Function word switching** (50% rate, up to 2 occurrences each): Replacing Indonesian conjunctions/prepositions with English
   - `yang` → `which`, `dan` → `and`, `untuk` → `for`, `dengan` → `with`
   - `karena` → `because`, `bisa` → `can`, `harus` → `must`, `mau` → `want to`

2. **Sentence prefixes** (30% rate): English discourse markers at sentence start
   - `So`, `Anyway`, `By the way`, `I mean`, `You know`, `Okay so`

3. **Discourse suffixes** (40% rate): Informal particles typical of Jakartan speech
   - `sih`, `dong`, `deh`, `nih`, `gitu`, `kan`, `lho`, `ya`, `tuh`

**Example:** "Ini sangat penting untuk bisnis" → "So ini so important for bisnis sih"

### Level 3: Adversarial Noise
Combines informal slang with keyboard typos to simulate real-world noisy input:

1. **Slang substitution** (50% rate): Uses 60+ colloquial forms from IndoCollex dataset
   - `tidak` → `gak`, `bagaimana` → `gimana`, `yang` → `yg`
   - `adalah` → `adlh`, `dalam` → `dlm`, `untuk` → `utk`, `karena` → `krn`
   - `sangat` → `bgt`, `banyak` → `bnyk`, `tersebut` → `tsb`

2. **Typo injection** (8% rate): Simulates keyboard errors on 26-character map
   - Character swap: `kata` → `kaat`
   - Character deletion: `kata` → `kta`
   - Adjacent key replacement: Full QWERTY keyboard mapping

### Academic References

The perturbation methodology is grounded in peer-reviewed linguistic research:

1. **Fauzi, I. (2015).** *English Borrowings in Indonesian Newspapers.* Journal on English as a Foreign Language.
   - Documents noun borrowing patterns and morphological adaptation of English verbs with Indonesian prefixes.

2. **Azizah, N. (2018).** *Anglicism in Indonesian.* Ethical Lingua: Journal of Language Teaching and Literature.
   - Analyzes informal code-mixing patterns including `ng-/nge-` prefix phenomenon.

3. **Wibowo, H. A., et al. (2021).** *IndoCollex: A Testbed for Morphological Transformation of Indonesian Word Colloquialism.* ACL-IJCNLP.
   - Provides colloquial transformation data used for Level 3 slang mappings.

## Quick Start (5 minutes)

### 1. Install

```bash
git clone https://github.com/nicovalerian/robin-benchmark.git
cd robin-benchmark

python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate # Linux/Mac

pip install -r requirements.txt
```

### 2. Configure API Keys

**Recommended: Use OpenRouter** (single API key for all models)

```bash
cp .env.example .env
```

Edit `.env`:
```env
OPENROUTER_API_KEY=sk-or-v1-...  # Get at https://openrouter.ai/keys
```

OpenRouter gives you access to GPT-4o, Claude, Gemini, Llama, Qwen, and more with one API key. Free tier available.

### 3. Run the Full Pipeline

```bash
# Phase 1: Generate benchmark dataset (default: 30 samples)
python scripts/run_phase1.py

# Generate more samples
python scripts/run_phase1.py --samples 100
python scripts/run_phase1.py --samples 750

# Phase 2: Run inference on all configured models
python scripts/run_phase2.py --input data/processed/robin_dataset.jsonl -y

# Or limit to fewer samples for testing
python scripts/run_phase2.py --input data/processed/robin_dataset.jsonl -y --limit 50

# Phase 3: Evaluate model responses
python scripts/run_phase3.py --input data/output/inference_results.jsonl

# Phase 4: Generate final report
python scripts/run_phase4.py --input data/output/evaluation_results.jsonl

# Phase 4 with automatic visualization generation
python scripts/run_phase4.py --input data/output/evaluation_results.jsonl --visualize

# Skip visualization prompt
python scripts/run_phase4.py --input data/output/evaluation_results.jsonl --no-visualize
```

Results are saved to `results/` folder. Visualizations are saved to `results/figures/`.

## Example Output

After running all 4 phases:

```
==================================================
ROBIN Benchmark Results Summary
==================================================
Total samples evaluated: 3
Models: gemini-2.0-flash, gemma-3-27b, gemma-3-4b, llama-3.3-70b, llama-3.1-8b

Robustness Ranking (higher = more robust):
  1. llama-3.1-8b: 106.06
  2. gemini-2.0-flash: 101.29
  3. gemma-3-4b: 100.43
  4. gemma-3-27b: 99.29
  5. llama-3.3-70b: 96.09
==================================================
```

**Performance Drop Rate by Perturbation Level:**

| Model | Level 1 (Mild) | Level 2 (Jaksel) | Level 3 (Adversarial) |
|-------|----------------|------------------|----------------------|
| llama-3.1-8b | 1.60% | 9.40% | 10.40% |
| gemini-2.0-flash | 0.70% | 2.96% | 0.22% |
| gemma-3-4b | 0.00% | 0.00% | 0.00% |
| gemma-3-27b | 0.00% | 0.00% | 0.00% |
| llama-3.3-70b | 3.90% | 10.80% | 10.80% |

## Using Your Own Local LLM

ROBIN supports Ollama, vLLM, LM Studio, or any OpenAI-compatible server.

### Step 1: Start your local server

```bash
# Ollama
ollama serve && ollama pull llama3.2

# vLLM
python -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-3.2-3B-Instruct

# LM Studio - just enable "Local Server" in the app
```

### Step 2: Add your model to config

Edit `configs/full_config.yaml`:

```yaml
inference:
  models:
    # Add your local model
    - name: "my-llama"
      provider: "ollama"           # or "vllm", "lmstudio", "local"
      model_id: "llama3.2"         # model name in your server
      # base_url: "http://localhost:11434"  # optional, uses default ports
```

### Step 3: Run the pipeline

```bash
python scripts/run_phase1.py
python scripts/run_phase2.py --input data/processed/robin_dataset.jsonl -y
python scripts/run_phase3.py --input data/output/inference_results.jsonl
python scripts/run_phase4.py --input data/output/evaluation_results.jsonl
```

Your local model will appear in the results alongside any cloud models.

## Pipeline Phases Explained

| Phase | What it does | Input | Output |
|-------|--------------|-------|--------|
| **Phase 1** | Creates perturbed prompts from source dataset | HuggingFace dataset | `data/processed/robin_dataset.jsonl` |
| **Phase 2** | Runs inference on all models | Phase 1 output | `data/output/inference_results.jsonl` |
| **Phase 3** | Evaluates responses (constraints, semantic, judge) | Phase 2 output | `data/output/evaluation_results.jsonl` |
| **Phase 4** | Calculates PDR and generates reports | Phase 3 output | `results/*.json` |

## Configuration Reference

### Dataset Size

Edit `configs/full_config.yaml`:
```yaml
dataset:
  target_size: 100  # Number of samples (default: 750)
```

### Rate Limiting (for free API tiers)

Global rate limiting (applies to all models without specific delays):
```yaml
inference:
  rate_limit_delay: 2.0  # Seconds between requests (default)
  max_concurrent: 2      # Parallel requests
```

Per-model rate limiting (recommended for mixed providers):
```yaml
inference:
  models:
    - name: "llama-3.1-8b"
      provider: "groq"
      model_id: "llama-3.1-8b-instant"
      rate_limit_delay: 2.5  # 30 RPM = 2s + 0.5s buffer
    
    - name: "gemini-2.0-flash"
      provider: "google"
      model_id: "gemini-2.0-flash"
      rate_limit_delay: 0.5  # Google has generous limits
```

### Adding Models

```yaml
inference:
  models:
    # OpenRouter (recommended - single API for all models)
    - name: "gpt-4o"
      provider: "openrouter"
      model_id: "openai/gpt-4o"
      
    - name: "claude-3.5-sonnet"
      provider: "openrouter"
      model_id: "anthropic/claude-3.5-sonnet"
    
    # Direct provider
    - name: "gemini-2.0-flash"
      provider: "google"
      model_id: "gemini-2.0-flash"
    
    # Local models
    - name: "my-local-model"
      provider: "ollama"
      model_id: "llama3.2"
```

## Supported Providers

| Provider | Models | API Key | Free Tier Limits |
|----------|--------|---------|------------------| 
| **OpenRouter** | DeepSeek V3, Grok 4.1, GLM 4.6/4.7, GPT-4o/4o-mini | `OPENROUTER_API_KEY` | **Recommended** - varies by model |
| Google AI Studio | Gemini 2.0/2.5, Gemma 3 | `GOOGLE_API_KEY` | 15 RPM |
| Groq | Llama 3.1, Kimi K2, Qwen3, GPT-OSS | `GROQ_API_KEY` | 30-60 RPM |
| **Ollama** | Any local model | None | **Free** |
| **vLLM** | Any local model | None | **Free** |
| **LM Studio** | Any local model | None | **Free** |

### Verified OpenRouter Models (Tested Feb 2026)

| Model | Model ID | Status |
|-------|----------|--------|
| DeepSeek V3 | `deepseek/deepseek-chat` | Working |
| Grok 4.1 Fast | `x-ai/grok-4.1-fast` | Working |
| GLM 4.6 | `z-ai/glm-4.6` | Working |
| GLM 4.7 | `z-ai/glm-4.7` | Working |
| GPT-4o Mini | `openai/gpt-4o-mini` | Working |
| GPT-4o | `openai/gpt-4o` | Working |

## Troubleshooting

### "No API keys available"

1. Check `.env` file exists: `ls -la .env`
2. Remove placeholder text like `sk-...` 
3. Restart your terminal after editing `.env`

### Rate limit errors (429)

Increase delays in `configs/full_config.yaml`:
```yaml
inference:
  rate_limit_delay: 5.0
  max_concurrent: 1
```

### Local LLM: "Cannot connect to server"

1. Verify server is running: `curl http://localhost:11434/v1/models`
2. Check the port matches your config
3. For Ollama: `ollama list` to see available models

### Phase 2 hangs

The script waits for confirmation. Use `-y` flag to skip:
```bash
python scripts/run_phase2.py --input data/processed/robin_dataset.jsonl -y
```

### "Event loop is closed" error on Windows

If you encounter asyncio errors on Windows, the script now automatically sets the correct event loop policy. If you still see issues:
```python
# Add this before running Phase 2
import asyncio
asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
```

### Model decommissioned errors

If you see errors like "model has been decommissioned":
1. Check [Groq's rate limits page](https://console.groq.com/docs/rate-limits) for current available models
2. Update `configs/full_config.yaml` with working model IDs
3. Working models as of Feb 2026: `llama-3.3-70b-versatile`, `llama-3.1-8b-instant`

## Output Files

After running all phases:

```
results/
├── summary.json              # Models ranked by robustness score
├── pdr_analysis.json         # Detailed PDR by level and metric
├── skill_profiles.json       # Performance by category/constraint type
└── figures/                  # Generated visualizations
    ├── pdr_comparison.png    # Performance Drop Rate comparison by model
    ├── pdr_comparison.pdf    # (PDF version)
    ├── pass_rate_heatmap.png # Constraint pass rate heatmap
    ├── pass_rate_heatmap.pdf # (PDF version)
    ├── perturbation_trend.png # Performance trend across perturbation levels
    └── perturbation_trend.pdf # (PDF version)

data/output/
├── inference_results.jsonl   # Raw model responses
└── evaluation_results.jsonl  # Scored responses
```

### Visualization Details

**PDR Comparison Chart** (`pdr_comparison.png/pdf`):
- Grouped bar chart showing Performance Drop Rate for each model
- Compares Level 1 (Mild), Level 2 (Jaksel), and Level 3 (Adversarial) perturbations
- Helps identify which models are most robust to Indonesian code-mixing

**Pass Rate Heatmap** (`pass_rate_heatmap.png/pdf`):
- Color-coded heatmap showing constraint pass rates
- Rows = Models, Columns = Perturbation levels (Clean, Mild, Jaksel, Adversarial)
- Green = 100% pass rate, Red = Lower pass rates

**Perturbation Trend** (`perturbation_trend.png/pdf`):
- Line chart showing performance scores across all perturbation levels
- Tracks how each model's performance changes from Clean → Adversarial
- Identifies models with stable vs. degrading performance

## Citation

```bibtex
@inproceedings{marcello2026robin,
  title={ROBIN: Benchmark for Evaluating LLM Instruction Robustness Under Indonesian Code-Mixing},
  author={Nico Valerian Marcello and Jecelyn Grizha and Bryan Lakaoni and Henry Lucky},
  booktitle={IEEE Conference},
  year={2026}
}
```

## License

Apache License 2.0
