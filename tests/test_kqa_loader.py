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

def test_load_kqa_tolerates_secondary_aliases():
    items = load_kqa(os.path.join(FIX, "kqa_aliases.jsonl"))
    assert len(items) == 1
    it = items[0]
    # Test that secondary aliases are mapped correctly
    assert it.qid == "q2"
    assert it.question == "What are the side effects of metformin?"
    assert it.reference_answer == "Metformin may cause gastrointestinal distress..."
    assert len(it.statements) == 3
    # Check statement importances
    must_haves = [s for s in it.statements if s.importance == "must_have"]
    nice_to_haves = [s for s in it.statements if s.importance == "nice_to_have"]
    assert len(must_haves) == 2
    assert len(nice_to_haves) == 1
    assert "nausea and diarrhea" in must_haves[0].text
    assert "diarrhea" in must_haves[0].text or "Monitor" in must_haves[1].text

def test_load_nli_lowercases_and_maps_aliases():
    pairs = load_physician_nli(os.path.join(FIX, "nli_aliases.jsonl"))
    assert len(pairs) == 1
    pair = pairs[0]
    # Test that secondary aliases are mapped correctly
    assert pair.premise == "Metformin may cause gastrointestinal distress."
    assert pair.hypothesis == "Metformin causes nausea."
    # Test that uppercase label is lowercased
    assert pair.label == "contradiction"
