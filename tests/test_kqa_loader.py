import os
from sac.kqa_loader import load_kqa, load_physician_nli, gold_statements

FIX = os.path.join(os.path.dirname(__file__), "fixtures")

def test_load_kqa_normalizes_fields():
    items = load_kqa(os.path.join(FIX, "kqa_sample.jsonl"))
    assert len(items) == 1
    it = items[0]
    assert it.qid == "q1"
    assert "ibuprofen" in it.question.lower()
    assert len(it.statements) == 3
    importances = {s.importance for s in it.statements}
    assert importances == {"must_have", "nice_to_have"}

def test_gold_statements_returns_all_texts():
    items = load_kqa(os.path.join(FIX, "kqa_sample.jsonl"))
    gs = gold_statements(items[0])
    assert len(gs) == 3
    assert all(isinstance(s, str) for s in gs)

def test_load_nli_pairs():
    pairs = load_physician_nli(os.path.join(FIX, "nli_sample.jsonl"))
    assert len(pairs) == 2
    assert pairs[0].label == "contradiction"
    assert pairs[1].label == "entailment"
