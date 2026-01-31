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
| Mild | "Jelaskan concept machine learning dalam 50 kata" |
| Jaksel | "Explain dong konsep machine learning gitu, dalam 50 kata ya" |
| Adversarial | "Jelasin dong gmn machine learning dlm 50 kata" |

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
# Phase 1: Generate benchmark dataset (creates data/processed/robin_dataset.jsonl)
python scripts/run_phase1.py

# Phase 2: Run inference on all configured models
python scripts/run_phase2.py --input data/processed/robin_dataset.jsonl -y

# Phase 3: Evaluate model responses
python scripts/run_phase3.py --input data/output/inference_results.jsonl

# Phase 4: Generate final report
python scripts/run_phase4.py --input data/output/evaluation_results.jsonl
```

Results are saved to `results/` folder.

## Example Output

After running all 4 phases:

```
==================================================
ROBIN Benchmark Results Summary
==================================================
Total samples evaluated: 9
Models: gemini-2.0-flash, gemma-3-27b

Robustness Ranking (higher = more robust):
  1. gemma-3-27b: 95.65
  2. gemini-2.0-flash: 94.92
==================================================
```

**Performance Drop Rate by Perturbation Level:**

| Model | Level 1 (Mild) | Level 2 (Jaksel) | Level 3 (Adversarial) |
|-------|----------------|------------------|----------------------|
| gemini-2.0-flash | 1.84% | 3.89% | 9.51% |
| gemma-3-27b | 4.41% | 1.35% | 7.29% |

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

```yaml
inference:
  rate_limit_delay: 2.0  # Seconds between requests
  max_concurrent: 2      # Parallel requests
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

| Provider | Models | API Key | Free Tier |
|----------|--------|---------|-----------| 
| **OpenRouter** | GPT-4o, Claude, Gemini, Llama, Qwen | `OPENROUTER_API_KEY` | **Recommended** |
| Google AI Studio | Gemini 2.0, Gemma 3 | `GOOGLE_API_KEY` | 15 RPM |
| Groq | Llama 3.3, Mixtral | `GROQ_API_KEY` | 14.4k req/day |
| Together AI | Llama 4, Qwen | `TOGETHER_API_KEY` | $1 credit |
| OpenAI | GPT-4o | `OPENAI_API_KEY` | No |
| Anthropic | Claude 3.5 | `ANTHROPIC_API_KEY` | No |
| **Ollama** | Any local model | None | **Free** |
| **vLLM** | Any local model | None | **Free** |
| **LM Studio** | Any local model | None | **Free** |

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

## Output Files

After running all phases:

```
results/
├── summary.json         # Models ranked by robustness score
├── pdr_analysis.json    # Detailed PDR by level and metric
└── skill_profiles.json  # Performance by category/constraint type

data/output/
├── inference_results.jsonl    # Raw model responses
└── evaluation_results.jsonl   # Scored responses
```

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
