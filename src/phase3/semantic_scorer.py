"""
ROBIN Phase 3 — Stream B: Semantic scorer.

Two metrics computed against a fixed gold_response reference:
  - ROUGE-L F1  (via rouge-score, pure Python, no model)
  - BERTScore F1 (via bert-score, using indobenchmark/indobert-base-p1)

Both measure semantic drift: the reference is the same across all four
perturbation levels, so a drop in score at L1-L3 vs L0 indicates that
prompt-side code-mixing caused the model to deviate from the intended content.

BERTScore is run in a single batched call per score_batch() invocation
to minimise model load/unload overhead.
"""
from __future__ import annotations

from dataclasses import dataclass

try:
    from rouge_score import rouge_scorer as _rouge_lib
    _ROUGE_AVAILABLE = True
except ImportError:
    _ROUGE_AVAILABLE = False

try:
    from bert_score import score as _bert_score_fn
    _BERT_AVAILABLE = True
except ImportError:
    _BERT_AVAILABLE = False


@dataclass
class SemanticScore:
    rouge_l: float
    bert_f1: float
    combined: float  # simple average: 0.5 * rouge_l + 0.5 * bert_f1


class SemanticScorer:
    """
    Batched semantic scorer. Call score_batch() with all (response, reference)
    pairs at once so BERTScore runs a single forward pass batch rather than
    one-at-a-time inference.
    """

    DEFAULT_MODEL = "indobenchmark/indobert-base-p1"

    def __init__(
        self,
        bert_model: str = DEFAULT_MODEL,
        bert_batch_size: int = 64,
    ):
        self.bert_model = bert_model
        self.bert_batch_size = bert_batch_size
        # use_stemmer=False: stemmer is English-only; Indonesian morphology
        # is handled by the BERTScore model, not by ROUGE preprocessing.
        self._rouge = _rouge_lib.RougeScorer(["rougeL"], use_stemmer=False) if _ROUGE_AVAILABLE else None

    def score_batch(
        self, responses: list[str], references: list[str]
    ) -> list[SemanticScore]:
        """
        Score all pairs in one call. BERTScore batches internally at
        bert_batch_size. Returns one SemanticScore per pair.
        """
        n = len(responses)
        rouge_l_scores = [0.0] * n
        bert_f1_scores = [0.0] * n

        # ROUGE-L: pure Python, fast per-pair
        if self._rouge:
            for i, (resp, ref) in enumerate(zip(responses, references)):
                if resp and ref:
                    rouge_l_scores[i] = self._rouge.score(ref, resp)["rougeL"].fmeasure

        # BERTScore: single batched call
        if _BERT_AVAILABLE and responses:
            _, _, F1 = _bert_score_fn(
                responses,
                references,
                model_type=self.bert_model,
                lang="id",
                batch_size=self.bert_batch_size,
                verbose=False,
            )
            bert_f1_scores = [f.item() for f in F1]

        return [
            SemanticScore(
                rouge_l=rl,
                bert_f1=bf,
                combined=(rl + bf) / 2,
            )
            for rl, bf in zip(rouge_l_scores, bert_f1_scores)
        ]

    def score(self, response: str, reference: str) -> SemanticScore:
        return self.score_batch([response], [reference])[0]
