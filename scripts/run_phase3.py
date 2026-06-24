#!/usr/bin/env python3
"""
ROBIN Phase 3 — Evaluation.

Reads Phase 2 flat JSONL (one record per model x sample x level), joins with
Phase 1 dataset for constraints and gold_response, then scores each response:

  Stream A: Rule-based constraint checker (keyword / length / format)
  Stream B: Semantic scorer — ROUGE-L + BERTScore (indobert-base-p1)

  Composite score = 0.6 * CPR + 0.4 * (0.5 * ROUGE-L + 0.5 * BERTScore)

Output is a flat JSONL with one evaluation record per input record.
Already-evaluated records are skipped — safe to resume after interruption.
BERTScore runs in a single batched call across all pending records.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from phase3 import ConstraintChecker, SemanticScorer
from utils import load_config, setup_logger


# Composite score weights (paper Sec. III-C)
W_CONSTRAINT = 0.6
W_SEMANTIC = 0.4


def _load_dataset(path: Path) -> dict[str, dict]:
    index: dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            index[row["id"]] = row
    return index


def _load_inference(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _load_completed(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
                done.add(f"{r['model_id']}|{r['sample_id']}|{r['level']}")
            except Exception:
                pass
    return done


def main() -> None:
    parser = argparse.ArgumentParser(description="ROBIN Phase 3: Evaluation")
    parser.add_argument("--config", default="configs/full_config.yaml")
    parser.add_argument("--dataset", default="data/processed/robin_dataset.jsonl",
                        help="Phase 1 output (constraints + gold_response)")
    parser.add_argument("--input", default="data/output/inference_results.jsonl",
                        help="Phase 2 output (flat inference records)")
    parser.add_argument("--output", default="data/output/evaluation_results.jsonl")
    parser.add_argument("--bert-batch-size", type=int, default=64,
                        help="Batch size for BERTScore inference (lower if OOM)")
    parser.add_argument("--chunk-size", type=int, default=500,
                        help="Records scored+written per checkpoint flush "
                             "(bounds memory; smaller = more frequent progress/resume points)")
    args = parser.parse_args()

    logger = setup_logger("robin-phase3", level="INFO")
    logger.info("Starting ROBIN Phase 3: Evaluation")

    config = load_config(args.config)
    bert_model = (
        config.get("evaluation", {})
            .get("semantic_scorer", {})
            .get("bert_model", SemanticScorer.DEFAULT_MODEL)
    )

    dataset = _load_dataset(Path(args.dataset))
    inference = _load_inference(Path(args.input))
    logger.info(f"Loaded {len(inference)} inference records across {len(dataset)} samples")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = _load_completed(output_path)
    logger.info(f"Already evaluated: {len(completed)} — skipping")

    # Only process successful inference records not yet evaluated
    pending = [
        r for r in inference
        if r.get("success") and r.get("response")
        and f"{r['model_id']}|{r['sample_id']}|{r['level']}" not in completed
    ]
    logger.info(f"Pending: {len(pending)} records")

    if not pending:
        logger.info("Nothing to evaluate. Phase 3 complete.")
        return

    checker = ConstraintChecker()
    scorer = SemanticScorer(bert_model=bert_model, bert_batch_size=args.bert_batch_size)

    # Score and write in chunks rather than one giant batch. This (1) checkpoints
    # progress incrementally so a crash/interrupt only loses the current chunk
    # instead of the whole run, and (2) bounds peak memory to one chunk's worth of
    # BERT embeddings instead of all pending records at once. bert_batch_size
    # still controls the inner GPU batching within each chunk.
    total = len(pending)
    chunk_size = max(1, args.chunk_size)
    logger.info(
        f"Running semantic scoring (ROUGE-L + BERTScore) on {total} records "
        f"in chunks of {chunk_size}..."
    )

    written = 0
    with open(output_path, "a", encoding="utf-8") as out:
        for start in range(0, total, chunk_size):
            chunk = pending[start:start + chunk_size]
            responses = [r["response"] for r in chunk]
            references = [dataset.get(r["sample_id"], {}).get("gold_response", "") for r in chunk]

            semantic_scores = scorer.score_batch(responses, references)

            for record, sem in zip(chunk, semantic_scores):
                sample = dataset.get(record["sample_id"], {})
                constraints = sample.get("constraints", [])

                all_results = checker.check_all(record["response"], constraints)
                cpr = all_results.cpr
                composite = W_CONSTRAINT * cpr + W_SEMANTIC * sem.combined

                row = {
                    "model_name": record["model_name"],
                    "model_id": record["model_id"],
                    "sample_id": record["sample_id"],
                    "level": record["level"],
                    "category": record["category"],
                    # Stream A
                    "cpr": round(cpr, 4),
                    "constraint_results": [
                        {
                            "type": r.constraint_type,
                            "passed": r.passed,
                            "expected": r.expected,
                            "actual": r.actual,
                            "details": r.details,
                        }
                        for r in all_results.results
                    ],
                    # Stream B
                    "rouge_l": round(sem.rouge_l, 4),
                    "bert_f1": round(sem.bert_f1, 4),
                    "semantic": round(sem.combined, 4),
                    # Composite (used by Phase 4 PDR)
                    "composite": round(composite, 4),
                }
                out.write(json.dumps(row, ensure_ascii=False) + "\n")

            # Flush each chunk so the on-disk checkpoint advances and a resume
            # after interruption picks up from the last completed chunk.
            out.flush()
            written += len(chunk)
            logger.info(f"  Evaluated {written}/{total} records")

    logger.info(f"Phase 3 complete. Total evaluated: {len(completed) + written}")


if __name__ == "__main__":
    main()
