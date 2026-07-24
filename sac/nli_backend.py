"""DeBERTa-MNLI entailment scorer (score S4) — a model *separate* from the generator.

Premise = the model's generated answer, hypothesis = the atomic claim. The score
is P(entailment): does the answer entail the claim? Mechanistically unrelated to
the generator's own confidence, which is what makes it a strong ablation point.
The softmax lives in the tested `sac.scores.softmax_entail`.
"""
import torch
from sac.scores import softmax_entail


class NLIBackend:
    def __init__(self, model_id="microsoft/deberta-large-mnli"):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id, device_map="auto")
        # resolve the entailment column from the model config rather than assuming
        self.entail_index = self.model.config.label2id.get("ENTAILMENT", 2)

    @torch.no_grad()
    def entail_prob(self, premise, hypothesis):
        inputs = self.tok(premise, hypothesis, return_tensors="pt",
                          truncation=True, max_length=512).to(self.model.device)
        logits = self.model(**inputs).logits[0].tolist()
        return softmax_entail(logits, self.entail_index)
