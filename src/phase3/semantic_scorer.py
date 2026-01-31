from dataclasses import dataclass

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False

try:
    from bert_score import score as bert_score_fn
    BERT_AVAILABLE = True
except ImportError:
    BERT_AVAILABLE = False


@dataclass
class SemanticScore:
    rouge_l_precision: float
    rouge_l_recall: float
    rouge_l_f1: float
    bert_precision: float
    bert_recall: float
    bert_f1: float
    combined_score: float


class SemanticScorer:
    def __init__(
        self,
        bert_model: str = "indolem/indobert-base-uncased",
        use_rouge: bool = True,
        use_bert: bool = True,
    ):
        self.bert_model = bert_model
        self.use_rouge = use_rouge and ROUGE_AVAILABLE
        self.use_bert = use_bert and BERT_AVAILABLE
        
        if self.use_rouge:
            self.rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    
    def score(
        self,
        response: str,
        reference: str,
    ) -> SemanticScore:
        rouge_l_p, rouge_l_r, rouge_l_f1 = 0.0, 0.0, 0.0
        bert_p, bert_r, bert_f1 = 0.0, 0.0, 0.0
        
        if self.use_rouge and reference:
            scores = self.rouge.score(reference, response)
            rouge_l_p = scores["rougeL"].precision
            rouge_l_r = scores["rougeL"].recall
            rouge_l_f1 = scores["rougeL"].fmeasure
        
        if self.use_bert and reference:
            try:
                P, R, F1 = bert_score_fn(
                    [response],
                    [reference],
                    model_type=self.bert_model,
                    lang="id",
                    verbose=False,
                )
                bert_p = P[0].item()
                bert_r = R[0].item()
                bert_f1 = F1[0].item()
            except Exception:
                pass
        
        combined = (rouge_l_f1 + bert_f1) / 2 if (rouge_l_f1 and bert_f1) else max(rouge_l_f1, bert_f1)
        
        return SemanticScore(
            rouge_l_precision=rouge_l_p,
            rouge_l_recall=rouge_l_r,
            rouge_l_f1=rouge_l_f1,
            bert_precision=bert_p,
            bert_recall=bert_r,
            bert_f1=bert_f1,
            combined_score=combined,
        )
    
    def score_batch(
        self,
        responses: list[str],
        references: list[str],
    ) -> list[SemanticScore]:
        return [
            self.score(resp, ref)
            for resp, ref in zip(responses, references)
        ]
