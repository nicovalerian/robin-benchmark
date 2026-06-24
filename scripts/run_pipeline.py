#!/usr/bin/env python3
"""
ROBIN Pipeline — one command for Phases 2 -> 3 -> 4.

Runs inference, evaluation, and analysis back to back on the canonical 750
dataset. If the dataset is missing it is generated first (Phase 1), so a fresh
checkout can go from nothing to results with a single command.

  python scripts/run_pipeline.py                 # all configured models
  python scripts/run_pipeline.py --models a,b     # only these models
  python scripts/run_pipeline.py --add-model x=y  # add an ad-hoc DO model
  python scripts/run_pipeline.py --yes            # skip the confirmation prompt

The dataset path and sample count are configurable (--dataset, --samples); when
the dataset already exists it is reused as-is (resumable), never regenerated.
"""
import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from phase2 import select_models  # noqa: E402
from utils import load_config  # noqa: E402

SCRIPTS = Path(__file__).parent
PYTHON = sys.executable


def _split_csv(value):
    if not value:
        return None
    return [t.strip() for t in value.split(",") if t.strip()]


def _run(label, cmd):
    print(f"\n{'='*64}\n[{label}]\n{'='*64}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\n{label} failed (exit {result.returncode}). Stopping.")
        sys.exit(result.returncode)


def _ensure_dataset(args):
    """Return the dataset path, generating it via Phase 1 if absent."""
    dataset = Path(args.dataset)
    if dataset.exists():
        n = sum(1 for _ in open(dataset, encoding="utf-8"))
        print(f"Dataset: {dataset} (exists, {n} samples) - reusing")
        return dataset

    print(f"Dataset: {dataset} not found - generating {args.samples} samples (Phase 1)")
    _run(
        "Phase 1: Dataset Construction",
        [PYTHON, str(SCRIPTS / "run_phase1.py"),
         "--config", args.config,
         "--samples", str(args.samples),
         "--output", str(dataset)],
    )
    return dataset


def _resolve_models(args):
    config = load_config(args.config)
    models = config.get("inference", {}).get("models", [])
    return select_models(
        models,
        include=_split_csv(args.models),
        exclude=_split_csv(args.exclude_models),
        add=args.add_model,
    )


def _confirm(models, args):
    """Show the plan and (unless --yes) let the user confirm or pick a subset."""
    print(f"\n{'='*64}\nROBIN Pipeline plan\n{'='*64}")
    print(f"  Dataset        : {args.dataset} (samples: {args.samples} if generated)")
    print(f"  Limit          : {args.limit if args.limit else 'none (full dataset)'}")
    print(f"  Visualizations : {'no' if args.no_visualize else 'yes'}")
    print(f"  Models ({len(models)}):")
    for i, m in enumerate(models, 1):
        print(f"    {i:>2}. {m['name']}  ({m['model_id']})")

    if args.yes:
        return models

    raw = input(
        "\nProceed with all models? [Enter=yes / numbers like 1,3 to select / q=quit]: "
    ).strip().lower()
    if raw in ("q", "n", "no"):
        print("Aborted.")
        sys.exit(0)
    if not raw or raw in ("y", "yes"):
        return models

    try:
        picks = sorted({int(x) for x in raw.replace(" ", "").split(",")})
        chosen = [models[i - 1] for i in picks if 1 <= i <= len(models)]
    except (ValueError, IndexError):
        print("Could not parse selection. Aborting.")
        sys.exit(1)
    if not chosen:
        print("No valid models selected. Aborting.")
        sys.exit(1)
    print("Selected: " + ", ".join(m["name"] for m in chosen))
    return chosen


def main():
    p = argparse.ArgumentParser(
        description="ROBIN Pipeline: run Phases 2->4 (auto-generates the dataset if missing)"
    )
    p.add_argument("--config", default="configs/full_config.yaml")
    p.add_argument("--dataset", default="data/processed/robin_dataset.jsonl",
                   help="Phase 1 dataset; generated if missing")
    p.add_argument("--samples", type=int, default=750,
                   help="Sample count when the dataset must be generated (default: 750)")
    p.add_argument("--limit", type=int, default=None,
                   help="Limit samples sent to inference (smoke testing)")
    p.add_argument("--models", default=None,
                   help="Comma-separated names/model_ids to keep (default: all)")
    p.add_argument("--exclude-models", default=None,
                   help="Comma-separated names/model_ids to drop")
    p.add_argument("--add-model", action="append", default=None,
                   help="Ad-hoc DO model 'name=model_id' (or just model_id); repeatable")
    p.add_argument("--inference-output", default="data/output/inference_results.jsonl")
    p.add_argument("--eval-output", default="data/output/evaluation_results.jsonl")
    p.add_argument("--results-dir", default="results/")
    p.add_argument("--bert-batch-size", type=int, default=64)
    p.add_argument("--no-visualize", action="store_true",
                   help="Skip Phase 4 figures")
    p.add_argument("--smoke", action="store_true",
                   help="1-sample end-to-end check on isolated paths (first model only)")
    p.add_argument("-y", "--yes", action="store_true",
                   help="Skip the confirmation prompt (non-interactive)")
    args = p.parse_args()

    if args.smoke:
        # Self-contained 1-sample run on separate files so it never touches the
        # real dataset or results. Only the first configured model unless the
        # caller pinned --models explicitly.
        args.dataset = "data/processed/smoke_pipeline.jsonl"
        args.samples = 1
        args.limit = 1
        args.inference_output = "data/output/smoke_inference.jsonl"
        args.eval_output = "data/output/smoke_eval.jsonl"
        args.results_dir = "results/smoke"
        args.no_visualize = True
        if not args.models:
            first = load_config(args.config).get("inference", {}).get("models", [])
            if first:
                args.models = first[0].get("name") or first[0].get("model_id")

    try:
        models = _resolve_models(args)
    except ValueError as exc:
        print(f"Model selection error: {exc}")
        sys.exit(1)

    models = _confirm(models, args)
    model_names = ",".join(m["name"] for m in models)

    dataset = _ensure_dataset(args)

    # Phase 2 — inference. Forward the resolved selection: any ad-hoc models via
    # --add-model, then --models pins the exact final set.
    phase2 = [
        PYTHON, str(SCRIPTS / "run_phase2.py"),
        "--config", args.config,
        "--input", str(dataset),
        "--output", args.inference_output,
        "--models", model_names,
    ]
    for spec in args.add_model or []:
        phase2 += ["--add-model", spec]
    if args.limit:
        phase2 += ["--limit", str(args.limit)]
    _run("Phase 2: Model Inference", phase2)

    # Phase 3 — evaluation.
    _run(
        "Phase 3: Evaluation",
        [PYTHON, str(SCRIPTS / "run_phase3.py"),
         "--config", args.config,
         "--dataset", str(dataset),
         "--input", args.inference_output,
         "--output", args.eval_output,
         "--bert-batch-size", str(args.bert_batch_size)],
    )

    # Phase 4 — analysis. Pass an explicit visualize flag so it never blocks on a
    # prompt inside the pipeline.
    phase4 = [
        PYTHON, str(SCRIPTS / "run_phase4.py"),
        "--config", args.config,
        "--input", args.eval_output,
        "--dataset", str(dataset),
        "--output", args.results_dir,
        "--no-visualize" if args.no_visualize else "--visualize",
    ]
    _run("Phase 4: Analysis & Reporting", phase4)

    print(f"\n{'='*64}\nPipeline complete. Results in {args.results_dir}\n{'='*64}")


if __name__ == "__main__":
    main()
