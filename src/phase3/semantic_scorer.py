"""
ROBIN Phase 3 -- Stream B: Semantic scorer.

Two metrics computed against a fixed gold_response reference:
  - ROUGE-L F1  (via rouge-score, pure Python, no model)
  - BERTScore F1 (computed directly with transformers to avoid bert_score
                  library's OverflowError on models whose tokenizer has no
                  explicit model_max_length, e.g. indobert-base-p1)

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
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer
    _BERT_AVAILABLE = True
except ImportError:
    _BERT_AVAILABLE = False


@dataclass
class SemanticScore:
    rouge_l: float
    bert_f1: float
    combined: float  # 0.5 * rouge_l + 0.5 * bert_f1


class SemanticScorer:
    """
    Batched semantic scorer. Call score_batch() with all (response, reference)
    pairs at once so BERTScore runs a single forward pass batch rather than
    one-at-a-time inference.
    """

    DEFAULT_MODEL = "indobenchmark/indobert-base-p1"
    # Layer index for embedding extraction (BERT-base convention: layer 9).
    # bert_score's model2layers map does not include indobert-base-p1, so we
    # specify it explicitly and load the model ourselves.
    _BERT_LAYER = 9

    def __init__(
        self,
        bert_model: str = DEFAULT_MODEL,
        bert_batch_size: int = 64,
    ):
        self.bert_model = bert_model
        self.bert_batch_size = bert_batch_size
        self._rouge = _rouge_lib.RougeScorer(["rougeL"], use_stemmer=False) if _ROUGE_AVAILABLE else None
        self._tokenizer = None
        self._model = None
        self._device = None

    def _load_bert(self) -> None:
        if self._model is not None:
            return
        self._tokenizer = AutoTokenizer.from_pretrained(self.bert_model)
        # Prevent OverflowError: indobert-base-p1 has no model_max_length set,
        # which defaults to a huge sentinel value that overflows the Rust tokenizer.
        self._tokenizer.model_max_length = 512
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = AutoModel.from_pretrained(
            self.bert_model, output_hidden_states=True
        ).eval().to(self._device)

    def _embed(self, texts: list[str]) -> list[torch.Tensor]:
        """Return per-text token embeddings from layer _BERT_LAYER (no padding tokens)."""
        self._load_bert()
        per_text: list[torch.Tensor] = []
        for i in range(0, len(texts), self.bert_batch_size):
            batch = texts[i : i + self.bert_batch_size]
            enc = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            enc = {k: v.to(self._device) for k, v in enc.items()}
            with torch.no_grad():
                out = self._model(**enc)
            layer = out.hidden_states[self._BERT_LAYER]  # (B, T, H)
            layer = F.normalize(layer, dim=-1)
            for j in range(layer.shape[0]):
                mask = enc["attention_mask"][j].bool()
                per_text.append(layer[j][mask].cpu())
        return per_text

    @staticmethod
    def _greedy_f1(ref_emb: torch.Tensor, cand_emb: torch.Tensor) -> float:
        """BERTScore F1 via greedy token matching (standard formulation)."""
        # Drop [CLS] and [SEP]
        r = ref_emb[1:-1] if len(ref_emb) > 2 else ref_emb
        c = cand_emb[1:-1] if len(cand_emb) > 2 else cand_emb
        if len(r) == 0 or len(c) == 0:
            return 0.0
        sim = r @ c.T  # (len_r, len_c), cosine sim (already L2-normed)
        recall = sim.max(dim=1).values.mean().item()
        precision = sim.max(dim=0).values.mean().item()
        if precision + recall <= 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

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

        if self._rouge:
            for i, (resp, ref) in enumerate(zip(responses, references)):
                if resp and ref:
                    rouge_l_scores[i] = self._rouge.score(ref, resp)["rougeL"].fmeasure

        if _BERT_AVAILABLE and responses:
            ref_embs = self._embed(references)
            cand_embs = self._embed(responses)
            bert_f1_scores = [
                self._greedy_f1(r, c) for r, c in zip(ref_embs, cand_embs)
            ]

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
