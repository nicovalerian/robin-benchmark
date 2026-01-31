# ROBIN: Robust Instruction Benchmark for Indonesian Noise

> Benchmark for Evaluating LLM Instruction Robustness Under Indonesian Code-Mixing

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-green.svg)](https://opensource.org/licenses/Apache-2.0)

## Overview

ROBIN is a comprehensive benchmark designed to evaluate how Large Language Models (LLMs) handle instruction-following tasks when exposed to Indonesian code-mixing patterns. The benchmark introduces four levels of linguistic perturbation while maintaining verifiable constraints.

### Key Features

- **4 Perturbation Levels**: Clean Indonesian → Mild Mixing → Jaksel Code-Switching → Adversarial Noise
- **Verifiable Constraints**: Keyword inclusion, length requirements (objectively measurable)
- **Hybrid Evaluation**: Objective checks + Semantic similarity + LLM-as-judge
- **Multi-Provider Support**: OpenAI, Anthropic, Google, Together AI, HuggingFace Inference
- **Reproducible Pipeline**: End-to-end automation with configuration files

## Quick Start

### Prerequisites

- Python 3.10+
- API keys for inference providers (see Configuration)

### Installation

```bash
# Clone the repository
cd robin-benchmark

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Configuration

1. Copy the example environment file:
```bash
cp .env.example .env
```

2. Add your API keys:
```env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
TOGETHER_API_KEY=...
HUGGINGFACE_API_KEY=hf_...
```

### Running the Pipeline

```bash
# Phase 1: Generate perturbed dataset
python scripts/run_phase1.py --config configs/full_config.yaml

# Phase 2: Run model inference
python scripts/run_phase2.py --config configs/full_config.yaml

# Phase 3: Evaluate responses
python scripts/run_phase3.py --config configs/full_config.yaml

# Phase 4: Generate analysis report
python scripts/run_phase4.py --output results/

# Or run all phases:
python scripts/run_all.py --config configs/full_config.yaml

# Quick test with sample data (no API keys needed):
python scripts/generate_sample.py
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
│   └── output/                  # Evaluation results
├── src/
│   ├── phase1/                  # Dataset construction
│   │   ├── constraint_injector.py
│   │   ├── perturbation_engine.py
│   │   └── task_classifier.py
│   ├── phase2/                  # Model inference
│   │   ├── inference_runner.py
│   │   └── providers/
│   ├── phase3/                  # Evaluation
│   │   ├── constraint_checker.py
│   │   ├── semantic_scorer.py
│   │   └── llm_judge.py
│   ├── phase4/                  # Analysis
│   │   ├── pdr_calculator.py
│   │   └── skill_mapper.py
│   └── utils/                   # Shared utilities
├── scripts/                     # CLI entry points
└── tests/                       # Unit tests
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

## Data Sources

### Primary Dataset
- **alpaca-gpt4-indonesian** (49.9k samples): [HuggingFace Link](https://huggingface.co/datasets/FreedomIntelligence/alpaca-gpt4-indonesian)

### Supplementary Resources
- **IndoCollex**: Formal→slang morphology mappings
- **evol-instruct-indonesian**: Coding tasks
- **math-olympiad-indonesian**: Math reasoning

## Target Models

| Model | Provider | Access |
|-------|----------|--------|
| Llama-4 (Scout/Maverick) | Together AI / Groq | API |
| Gemma-4 | Google AI Studio | API |
| Cendol | HuggingFace Inference | API |
| GPT-4o | OpenAI | API |
| Claude 3.5 | Anthropic | API |
| Qwen-2.5 | Together AI | API |

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
