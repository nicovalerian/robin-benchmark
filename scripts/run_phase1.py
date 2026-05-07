#!/usr/bin/env python3
"""
ROBIN Phase 1 — Dataset Construction

Loads the Indonesian Alpaca-cleaned dataset, picks a balanced subset,
injects verifiable constraints, and uses DigitalOcean Serverless
Inference to generate the four perturbation levels (L0-L3) for each
base instruction.

Output: 1 JSONL row per base instruction containing all four levels.
With --samples 750 this yields 750 base × 4 levels = 3000 prompts.

Checkpointing: completed samples are appended to the output file
immediately. Restarting the script resumes from where it left off.
"""
import argparse
import asyncio
import io
import json
import random
import sys
import time
from pathlib import Path

from datasets import load_dataset
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from phase1 import TaskClassifier, ConstraintInjector, PerturbationEngine  # noqa: E402
from utils import load_config, setup_logger  # noqa: E402

_CAT_LABEL = {
    "logical_reasoning":    "logical_reason",
    "mathematical_reasoning": "mathematical ",
    "creative_writing":     "creative_writ",
    "information_extraction": "info_extract ",
    "coding":               "coding       ",
}


def _append_jsonl(row: dict, path: Path) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_checkpoint(path: Path) -> tuple[list[dict], set[str]]:
    """Load existing output; return (rows, set-of-original-instructions)."""
    if not path.exists():
        return [], set()
    rows = []
    keys = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows.append(row)
            keys.add(row["original_instruction"])
    return rows, keys


async def _process_one(
    engine, sample, classifier, injector, idx, lock, output_path, completed_keys,
    on_done=None,
):
    instruction = (sample.get("instruction") or "").strip()
    input_text = (sample.get("input") or "").strip()
    output_text = (sample.get("output") or "").strip()

    if not instruction or len(instruction) < 20:
        return None

    if instruction in completed_keys:
        return None

    category = classifier.classify(instruction, input_text)
    full_instruction = f"{instruction}\n{input_text}".strip() if input_text else instruction

    constrained = injector.inject_constraints(
        instruction=full_instruction,
        category=category,
        gold_response=output_text,
    )

    try:
        perturbed = await engine.perturb_async(constrained.constrained_instruction)
    except Exception as exc:
        err_str = str(exc)
        err_type = (
            "timeout" if "timed out" in err_str.lower() else
            "empty_completion" if "empty completion" in err_str.lower() else
            "rate_limit" if "429" in err_str or "rate_limit" in err_str.lower() else
            "api_error"
        )
        if on_done:
            on_done(category, False)
        return {"_error": err_str, "_error_type": err_type, "_id": idx, "_category": category}

    row = {
        "id": f"robin_{idx:05d}",
        "category": category,
        "original_instruction": instruction,
        "original_input": input_text,
        "gold_response": output_text,
        "constraints": [
            {
                "constraint_type": c.constraint_type,
                "requirement": c.requirement,
                "verification_regex": c.verification_regex,
                "target_value": c.target_value,
            }
            for c in constrained.constraints
        ],
        "perturbations": {
            "level_0_clean": perturbed.level_0_clean,
            "level_1_mild": perturbed.level_1_mild,
            "level_2_jaksel": perturbed.level_2_jaksel,
            "level_3_adversarial": perturbed.level_3_adversarial,
        },
        "perturbation_meta": perturbed.metadata,
    }

    async with lock:
        _append_jsonl(row, output_path)
        completed_keys.add(instruction)

    if on_done:
        on_done(category, True)
    return row


def _make_progress() -> Progress:
    return Progress(
        SpinnerColumn(spinner_name="dots"),
        TextColumn("[bold]{task.description:<16}", justify="left"),
        BarColumn(bar_width=28),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )


async def _run(args, logger):
    config = load_config(args.config)
    dataset_config = config.get("dataset", {})
    pert_config = config.get("perturbation", {})
    llm_cfg = pert_config.get("llm", {})

    random.seed(args.seed)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    robin_samples, completed_keys = _load_checkpoint(output_path)
    if completed_keys:
        logger.info(f"Resuming: {len(completed_keys)} samples already completed")

    source = dataset_config.get("source", "ilhamfadheel/alpaca-cleaned-indonesian")
    logger.info(f"Loading dataset: {source}")
    source_dataset = load_dataset(source, split="train")

    classifier = TaskClassifier(dataset_config.get("categories"))
    injector = ConstraintInjector(
        constraint_types=config.get("constraints", {}).get("types", ["keyword", "length"]),
        seed=args.seed,
    )
    engine = PerturbationEngine(
        model=llm_cfg.get("model", "gemma-4-31B-it"),
        base_url=llm_cfg.get("base_url", "https://inference.do-ai.run/v1/"),
        prompts_path=pert_config.get("prompts_path"),
        indocollex_path=config.get("paths", {}).get("data_raw", "data/raw") + "/indocollex.json",
        max_concurrency=llm_cfg.get("max_concurrency", 8),
        temperature=llm_cfg.get("temperature", 0.4),
        max_retries=llm_cfg.get("max_retries", 3),
        rate_limit_per_minute=llm_cfg.get("rate_limit_per_minute", 120),
        max_tokens=llm_cfg.get("max_tokens", 1024),
        seed=args.seed,
    )

    target_size = args.samples or dataset_config.get("target_size", 750)
    weights = classifier.get_distribution_weights()
    category_targets = {cat: max(1, int(target_size * w)) for cat, w in weights.items()}
    remainder = target_size - sum(category_targets.values())
    if remainder > 0:
        largest = max(category_targets, key=category_targets.get)
        category_targets[largest] += remainder
    logger.info(f"Target distribution: {category_targets}")

    category_counts = {cat: 0 for cat in weights}
    for row in robin_samples:
        cat = row.get("category")
        if cat in category_counts:
            category_counts[cat] += 1

    indices = list(range(len(source_dataset)))
    random.shuffle(indices)

    by_category: dict[str, list] = {cat: [] for cat in weights}
    for idx in indices:
        sample = source_dataset[idx]
        instruction = (sample.get("instruction") or "").strip()
        input_text = (sample.get("input") or "").strip()
        if not instruction or len(instruction) < 20:
            continue
        if instruction in completed_keys:
            continue
        cat = classifier.classify(instruction, input_text)
        if cat in by_category:
            by_category[cat].append(sample)

    failures: list[dict] = []
    cat_cursors = {cat: 0 for cat in weights}
    lock = asyncio.Lock()

    # ---------------------------------------------------------------- rich UI
    console = Console(file=sys.stdout)
    progress = _make_progress()

    overall_task = progress.add_task(
        "OVERALL",
        total=target_size,
        completed=sum(category_counts.values()),
    )
    cat_tasks = {
        cat: progress.add_task(
            _CAT_LABEL.get(cat, cat[:14]),
            total=category_targets[cat],
            completed=category_counts[cat],
        )
        for cat in weights
    }

    _t0 = time.monotonic()
    _done = [sum(category_counts.values())]
    _fails = [0]
    _round_label = ["—"]

    def on_done(cat: str, success: bool) -> None:
        if success:
            _done[0] += 1
            progress.advance(overall_task)
            if cat in cat_tasks:
                progress.advance(cat_tasks[cat])
        else:
            _fails[0] += 1
        elapsed = time.monotonic() - _t0
        rate = _done[0] / elapsed * 60 if elapsed > 0 else 0.0
        label = (
            f"OVERALL  [green]{rate:.1f}/min[/green]"
            + (f"  [red]{_fails[0]} fail[/red]" if _fails[0] else "")
            + f"  [dim]round {_round_label[0]}[/dim]"
        )
        progress.update(overall_task, description=label)

    # ---------------------------------------------------------------- rounds
    max_rounds = args.max_rounds
    with progress:
        for round_idx in range(1, max_rounds + 1):
            if all(category_counts[c] >= category_targets[c] for c in weights):
                logger.info("All category targets met.")
                break

            batch = []
            for cat, target in category_targets.items():
                need = target - category_counts[cat]
                if need <= 0:
                    continue
                ask = int(need * 1.25) if round_idx == 1 else need
                available = by_category[cat][cat_cursors[cat]:cat_cursors[cat] + ask]
                cat_cursors[cat] += ask
                for s in available:
                    batch.append((cat, s))

            if not batch:
                logger.info("No more candidates available.")
                break

            _round_label[0] = f"{round_idx}/{max_rounds}"
            gaps = {c: category_targets[c] - category_counts[c] for c in weights}
            logger.info(f"Round {round_idx}/{max_rounds}: {len(batch)} candidates (gaps: {gaps})")

            start_id = len(robin_samples) + len(failures)
            coros = [
                _process_one(
                    engine, s, classifier, injector, start_id + i, lock,
                    output_path, completed_keys, on_done=on_done,
                )
                for i, (_cat, s) in enumerate(batch)
            ]
            results = await asyncio.gather(*coros, return_exceptions=True)

            for (cat, _s), r in zip(batch, results):
                if r is None or isinstance(r, BaseException):
                    continue
                if "_error" in r:
                    failures.append(r)
                    continue
                if category_counts[cat] < category_targets[cat]:
                    robin_samples.append(r)
                    category_counts[cat] += 1

    # ---------------------------------------------------------------- summary
    if failures:
        fail_path = output_path.parent / (output_path.stem + "_failures.jsonl")
        with open(fail_path, "w", encoding="utf-8") as f:
            for row in failures:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        error_types: dict[str, int] = {}
        for r in failures:
            t = r.get("_error_type", "unknown")
            error_types[t] = error_types.get(t, 0) + 1
        logger.warning(f"{len(failures)} failures — breakdown: {error_types}")
        logger.warning(f"Logged to {fail_path}")

    stats = {
        "total_samples": len(robin_samples),
        "total_prompts": len(robin_samples) * 4,
        "failures": len(failures),
        "category_distribution": category_counts,
        "config_used": str(args.config),
        "source_dataset": source,
        "perturbation_model": llm_cfg.get("model", "gemma-4-31B-it"),
    }
    stats_path = output_path.parent / (output_path.stem + "_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    elapsed_total = time.monotonic() - _t0
    console.print(
        f"\n[bold green]Done.[/bold green]  "
        f"{len(robin_samples)} samples  "
        f"({len(robin_samples) * 4} prompts)  "
        f"{len(failures)} failures  "
        f"[dim]{elapsed_total / 60:.1f} min[/dim]"
    )
    logger.info(f"Generated {len(robin_samples)} samples ({len(robin_samples) * 4} prompts)")
    logger.info(f"Saved to: {output_path}")
    await engine.aclose()


def main():
    parser = argparse.ArgumentParser(description="ROBIN Phase 1: Dataset Construction")
    parser.add_argument("--config", type=str, default="configs/full_config.yaml")
    parser.add_argument("--output", type=str, default="data/processed/robin_dataset.jsonl")
    parser.add_argument("--samples", type=int, default=None,
                        help="Override target_size from config (default: use config = 750)")
    parser.add_argument("--max-rounds", type=int, default=8,
                        help="Max fill rounds (default: 8)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    load_dotenv()
    logger = setup_logger("robin-phase1", level="INFO")
    logger.info("Starting ROBIN Phase 1: Dataset Construction")

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(_run(args, logger))
    logger.info("Phase 1 complete!")


if __name__ == "__main__":
    main()
