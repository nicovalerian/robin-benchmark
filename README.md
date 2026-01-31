# ROBIN: Robust Instruction Benchmark for Indonesian Noise

> Benchmark for Evaluating LLM Instruction Robustness Under Indonesian Code-Mixing

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-green.svg)](https://opensource.org/licenses/Apache-2.0)

## Overview

ROBIN is a comprehensive benchmark designed to evaluate how Large Language Models (LLMs) handle instruction-following tasks when exposed to Indonesian code-mixing patterns. The benchmark introduces four levels of linguistic perturbation while maintaining verifiable constraints.

### Key Features

- **4 Perturbation Levels**: Clean Indonesian -> Mild Mixing -> Jaksel Code-Switching -> Adversarial Noise
- **Verifiable Constraints**: Keyword inclusion, length requirements (objectively measurable)
- **Hybrid Evaluation**: Objective checks + Semantic similarity + LLM-as-judge
- **Multi-Provider Support**: OpenAI, Anthropic, Google, Together AI, Groq, HuggingFace
- **Local LLM Support**: Ollama, vLLM, LM Studio, or any OpenAI-compatible endpoint
- **Reproducible Pipeline**: End-to-end automation with configuration files

## Quick Start

### Prerequisites

- Python 3.10+
- API keys for cloud providers (optional) OR local LLM server (Ollama, vLLM, etc.)

### Installation

```bash
# Clone the repository
git clone https://github.com/nicovalerian/robin-benchmark.git
cd robin-benchmark

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Configuration

#### Option A: Cloud API Providers

1. Copy the example environment file:
```bash
cp .env.example .env
```

2. Add your API keys (only add the ones you have):
```env
GOOGLE_API_KEY=...           # Google AI Studio (Gemini, Gemma) - FREE TIER
GROQ_API_KEY=gsk_...         # Groq (Llama, Mixtral) - FREE TIER
OPENAI_API_KEY=sk-...        # OpenAI (GPT-4o)
ANTHROPIC_API_KEY=sk-ant-... # Anthropic (Claude)
TOGETHER_API_KEY=...         # Together AI (Llama, Qwen)
HUGGINGFACE_API_KEY=hf_...   # HuggingFace (Cendol)
```

The script will automatically detect available API keys and only run models you have access to.

#### Option B: Local LLM (Ollama, vLLM, LM Studio)

No API keys needed! Add your local model to `configs/full_config.yaml`:

```yaml
inference:
  models:
    # Ollama (default: http://localhost:11434)
    - name: "my-local-llama"
      provider: "ollama"
      model_id: "llama3.2"
    
    # vLLM (default: http://localhost:8000)
    - name: "my-vllm-model"
      provider: "vllm"
      model_id: "meta-llama/Llama-3.2-3B-Instruct"
    
    # LM Studio (default: http://localhost:1234)
    - name: "lmstudio-model"
      provider: "lmstudio"
      model_id: "local-model"
    
    # Custom OpenAI-compatible endpoint
    - name: "custom-server"
      provider: "local"
      model_id: "my-model"
      base_url: "http://my-server:8080"
```

**Start your local server first:**
```bash
# Ollama
ollama serve
ollama pull llama3.2

# vLLM
python -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-3.2-3B-Instruct

# LM Studio
# Just start the app and enable "Local Server"
```

### Running the Pipeline

```bash
# Phase 1: Generate perturbed dataset (no API keys needed)
python scripts/run_phase1.py --config configs/full_config.yaml

# Phase 2: Run model inference
python scripts/run_phase2.py --config configs/full_config.yaml --input data/processed/robin_dataset.jsonl

# Phase 3: Evaluate responses
python scripts/run_phase3.py --config configs/full_config.yaml

# Phase 4: Generate analysis report
python scripts/run_phase4.py --output results/

# Or run all phases:
python scripts/run_all.py --config configs/full_config.yaml

# Quick test with sample data (no API keys needed):
python scripts/generate_sample.py
```

### Testing Your Local LLM

1. Create a minimal config file `configs/local_only.yaml`:
```yaml
dataset:
  source: "FreedomIntelligence/alpaca-gpt4-indonesian"
  target_size: 10  # Small test

inference:
  temperature: 0.0
  max_tokens: 512
  max_concurrent: 1
  rate_limit_delay: 0.1
  
  models:
    - name: "my-local-model"
      provider: "ollama"
      model_id: "llama3.2"
```

2. Run the pipeline:
```bash
# Generate test dataset
python scripts/run_phase1.py --config configs/local_only.yaml --output data/processed/test_10.jsonl

# Run inference on your local model
python scripts/run_phase2.py --config configs/local_only.yaml --input data/processed/test_10.jsonl --output data/output/local_results.jsonl -y

# Evaluate results
python scripts/run_phase3.py --config configs/local_only.yaml --input data/output/local_results.jsonl
```

## Project Structure

```
robin-benchmark/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment template
├── configs/                     # Configuration files
│   └── full_config.yaml         # All phases configuration
├── data/
│   ├── raw/                     # Source datasets
│   ├── processed/               # Generated benchmark data
│   └── output/                  # Inference results
├── src/
│   ├── phase1/                  # Dataset construction
│   ├── phase2/                  # Model inference
│   ├── phase3/                  # Evaluation
│   ├── phase4/                  # Analysis
│   └── utils/                   # Shared utilities
├── scripts/                     # CLI entry points
└── results/                     # Final reports
```

## Benchmark Design

### Task Categories (Natural Distribution)

| Category | Description | Constraint Types |
|----------|-------------|------------------|
| Logical Reasoning | Syllogisms, deduction | Keyword, format |
| Mathematical Reasoning | Word problems, calculations | Numeric, length |
| Creative Writing | Stories, poetry, dialogue | Length, keyword |
| Information Extraction | Summarization, QA | Length, keyword |
| Coding | Code generation, debugging | Format, keyword |

### Perturbation Levels

| Level | Name | Description | Example |
|-------|------|-------------|---------|
| 0 | Clean | Formal Indonesian | "Jelaskan konsep..." |
| 1 | Mild | English loanwords | "Jelaskan concept..." |
| 2 | Jaksel | Code-switching | "Explain dong konsep ini..." |
| 3 | Adversarial | Typos + slang | "Jelasin dong gmn..." |

### Evaluation Metrics

1. **Constraint Compliance** (Binary): Pass/Fail via regex
2. **Semantic Fidelity**: ROUGE-L, BERTScore vs gold reference
3. **Intention Score** (1-5): LLM-as-judge with rubrics
4. **Performance Drop Rate (PDR)**: `(Score_clean - Score_noisy) / Score_clean * 100`

## Supported Providers

| Provider | Models | API Key Env Var | Free Tier |
|----------|--------|-----------------|-----------|
| Google AI Studio | Gemini 2.0, Gemma 3 | `GOOGLE_API_KEY` | Yes (15 RPM) |
| Groq | Llama 3.3, Mixtral | `GROQ_API_KEY` | Yes (14.4k/day) |
| Together AI | Llama 4, Qwen, Mistral | `TOGETHER_API_KEY` | $1 credit |
| OpenAI | GPT-4o, GPT-4-turbo | `OPENAI_API_KEY` | No |
| Anthropic | Claude 3.5 Sonnet | `ANTHROPIC_API_KEY` | No |
| HuggingFace | Cendol, Indonesian LLMs | `HUGGINGFACE_API_KEY` | Limited |
| **Ollama** | Any local model | None needed | **Free** |
| **vLLM** | Any local model | None needed | **Free** |
| **LM Studio** | Any local model | None needed | **Free** |

## Troubleshooting

### Rate Limit Errors (429)
Increase `rate_limit_delay` in config:
```yaml
inference:
  rate_limit_delay: 5.0  # Wait 5 seconds between requests
  max_concurrent: 1      # One request at a time
```

### Local LLM Connection Failed
1. Check your server is running: `curl http://localhost:11434/v1/models`
2. Verify the `base_url` in config matches your server
3. For Ollama, ensure model is pulled: `ollama pull llama3.2`

### API Key Not Detected
1. Check `.env` file exists in project root
2. Ensure no placeholder text (remove `sk-...` examples)
3. Restart terminal after editing `.env`

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

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Bina Nusantara University, School of Computer Science
- IndoNLP community for Indonesian NLP resources
- FreedomIntelligence for alpaca-gpt4-indonesian dataset
