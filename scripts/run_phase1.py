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
import logging
import random
import re
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

# Shares the name configured by setup_logger("robin-phase1") in main(), so DEBUG
# records from the fuzzy strip pass route through the same handler.
_logger = logging.getLogger("robin-phase1")

# Patterns that identify constraint requirement sentences produced by ConstraintInjector.
# Used to strip mangled versions from perturbed text before re-appending originals.
_CONSTRAINT_LINE_RE = re.compile(
    r"(?i)^("
    r"pastikan\s+jawaban\s+mengandung|"  # keyword constraint
    r"jawab\s+\S+\s+\d|"               # length: "Jawab dalam/dlm X-Y kata"
    r"jawab\s+\S+\s+format|"           # format (JSON): "Jawab dalam format JSON"
    r"gunakan\s+format|"               # format: "Gunakan format daftar bernomor"
    r"tampilkan\s+jawaban"             # format: "Tampilkan jawaban dalam format tabel"
    r")"
)

# Opener phrases that begin a constraint sentence, including the L3 colloquial /
# Jaksel approximations the exact-match _CONSTRAINT_LINE_RE misses (e.g.
# "jwb dlm 32-59 kata", "pastiin jawaban ada kata: X", "pake format daftar
# bernomor deh"). Unlike the clean re-appended lines, these mangled versions are
# emitted INLINE at the tail of the body sentence — on the same line as the real
# instruction — so a line-level strip would delete the instruction itself. We
# instead truncate the body at the first opener occurrence. Word boundaries keep
# "menggunakan format" / legitimate substrings from triggering a false truncation.
_CONSTRAINT_OPENER_RE = re.compile(
    r"""(?xi)\b(
    # keyword constraint (formal + Jaksel)
    pastikan\s+jawaban\s+mengandung
    | pastiin\s+(?:jawaban|ada\s+kata)
    | (?:make|meik)\s+sure\s+jawaban
    # length constraint: "Jawab/jwb dalam/dlm <N> ..."
    | (?:jawab|jwb)\s+(?:dalam|dlm)\b
    # format constraint (formal + Jaksel)
    | gunakan\s+format
    | (?:pake|pakai|gunain|use)\s+format
    | tampilkan\s+jawaban
    )"""
)


def _enforce_constraint_lines(
    text: str, requirements: list[str], sample_id: str | None = None
) -> str:
    """Guarantee constraint sentences appear verbatim at the end of a perturbed text.

    The perturbation LLM may code-mix or abbreviate constraint lines despite
    R3 in perturbation_prompts.yaml (e.g. "Jawab dlm" instead of "Jawab dalam",
    or random-lowercased "gunakan format"). This strips any mangled constraint
    sentences and re-appends the originals so Phase 3 always evaluates against
    the correct requirements.

    Two passes clean the body before the originals are re-appended:
      1. Line-level exact-match (_CONSTRAINT_LINE_RE): drops standalone constraint
         lines (formal phrasing on their own line).
      2. Inline truncation (_CONSTRAINT_OPENER_RE): the LLM often appends a
         mangled/Jaksel copy of the constraints INLINE at the end of the body
         sentence (e.g. "...biar hasilnya benerly. jawab dlm 32-59 kata. Gunakan
         format daftar bernomor deh."). We cut the body at the first opener so the
         real instruction survives but the trailing mangled constraints do not.
    """
    lines = text.splitlines()
    core = [l for l in lines if not _CONSTRAINT_LINE_RE.match(l.strip())]
    body = "\n".join(core).strip()

    m = _CONSTRAINT_OPENER_RE.search(body)
    if m and body[: m.start()].strip():
        # Guard: only truncate when real content precedes the opener, so we never
        # blank out a body that is entirely a (mangled) constraint block.
        removed = body[m.start():]
        body = body[: m.start()].rstrip()
        _logger.debug(
            "inline constraint strip [%s]: removed %r", sample_id or "?", removed
        )

    restored = body
    for req in requirements:
        restored += "\n" + req
    return restored


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
    """Load existing output; return (rows, set-of-original-instructions).

    Skips lines that fail to parse (e.g. truncated or doubled writes from a
    crash) so a corrupted checkpoint doesn't abort the entire run.
    """
    if not path.exists():
        return [], set()
    rows = []
    keys = set()
    bad = 0
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                print(f"[checkpoint] skipping malformed line {lineno}", file=sys.stderr)
                continue
            rows.append(row)
            keys.add(row["original_instruction"])
    if bad:
        print(f"[checkpoint] {bad} malformed line(s) skipped — those samples will be re-generated", file=sys.stderr)
    return rows, keys


async def _process_one(
    engine, sample, classifier, injector, idx, lock, output_path, completed_keys,
    category_counts, category_targets,
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

    # Constraints are derived from the Alpaca gold (for keyword/length estimation)
    # then embedded into the constrained_instruction text so the LLM-generated
    # gold and all four perturbation levels see them explicitly.
    constrained = injector.inject_constraints(
        instruction=full_instruction,
        category=category,
        gold_response=output_text,
    )

    try:
        # Generate gold and all four perturbation levels in parallel.
        # generate_gold uses the constrained instruction so the reference
        # response satisfies the embedded format/keyword/length requirements.
        gold_response, perturbed = await asyncio.gather(
            engine.generate_gold(constrained.constrained_instruction),
            engine.perturb_async(constrained.constrained_instruction),
        )
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

    constraint_reqs = [c.requirement for c in constrained.constraints]
    row = {
        "id": f"robin_{idx:05d}",
        "category": category,
        "original_instruction": instruction,
        "original_input": input_text,
        "gold_response": gold_response,
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
            "level_0_clean": _enforce_constraint_lines(perturbed.level_0_clean, constraint_reqs, f"robin_{idx:05d}/L0"),
            "level_1_mild": _enforce_constraint_lines(perturbed.level_1_mild, constraint_reqs, f"robin_{idx:05d}/L1"),
            "level_2_jaksel": _enforce_constraint_lines(perturbed.level_2_jaksel, constraint_reqs, f"robin_{idx:05d}/L2"),
            "level_3_adversarial": _enforce_constraint_lines(perturbed.level_3_adversarial, constraint_reqs, f"robin_{idx:05d}/L3"),
        },
        "perturbation_meta": perturbed.metadata,
    }

    # Enforce category cap atomically — prevents overshooting target_size
    # when many coroutines finish near-simultaneously.
    accepted = False
    async with lock:
        if category_counts.get(category, 0) < category_targets.get(category, 0):
            _append_jsonl(row, output_path)
            completed_keys.add(instruction)
            category_counts[category] = category_counts.get(category, 0) + 1
            accepted = True

    if not accepted:
        return None

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
                    output_path, completed_keys, category_counts, category_targets,
                    on_done=on_done,
                )
                for i, (_cat, s) in enumerate(batch)
            ]
            results = await asyncio.gather(*coros, return_exceptions=True)

            for (_cat, _s), r in zip(batch, results):
                if r is None or isinstance(r, BaseException):
                    continue
                if "_error" in r:
                    failures.append(r)
                    continue
                robin_samples.append(r)
                # category_counts already updated atomically inside _process_one

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
