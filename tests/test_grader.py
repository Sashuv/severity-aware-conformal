from sac.grader import (build_grade_prompt, parse_verdict, grade_claim,
                        build_severity_prompt, parse_severity, tag_severity)
from sac.backends import MockJudge

def test_grade_prompt_includes_evidence_and_claim():
    p = build_grade_prompt("claim X", ["stmt A", "stmt B"])
    assert "claim X" in p and "stmt A" in p and "stmt B" in p

def test_parse_verdict_maps_three_ways():
    assert parse_verdict('{"verdict": "entailed", "rationale": "ok"}')[0] == 0
    assert parse_verdict('{"verdict": "contradicted", "rationale": "no"}')[0] == 1
    assert parse_verdict('{"verdict": "neutral", "rationale": "n/a"}')[0] == -1

def test_parse_verdict_tolerates_extra_text():
    raw = 'Here is my judgment:\n{"verdict":"CONTRADICTED","rationale":"dose wrong"}\nThanks'
    label, rationale = parse_verdict(raw)
    assert label == 1 and "dose" in rationale

def test_grade_claim_uses_judge():
    judge = MockJudge(['{"verdict":"contradicted","rationale":"unsafe dose"}'])
    label, rationale = grade_claim("Take 5000 mg.", ["Max safe dose is 3200 mg."], judge)
    assert label == 1

def test_parse_severity():
    assert parse_severity('{"severity":"dangerous","rationale":"dose"}')[0] == "dangerous"
    assert parse_severity('{"severity":"BENIGN","rationale":"fact"}')[0] == "benign"

def test_tag_severity_uses_judge():
    judge = MockJudge(['{"severity":"dangerous","rationale":"dosing"}'])
    tier, _ = tag_severity("Take 5000 mg.", judge)
    assert tier == "dangerous"

def test_parse_verdict_no_json_block_is_unverifiable():
    """parse_verdict with no JSON block must return label -1 (unverifiable)."""
    result = parse_verdict("I think this claim is mostly fine, no JSON here.")
    assert result[0] == -1

def test_parse_verdict_malformed_json_is_unverifiable():
    """parse_verdict with broken JSON must return label -1 (unverifiable)."""
    result = parse_verdict('{"verdict": "entailed"')
    assert result[0] == -1

def test_parse_verdict_missing_verdict_key_is_unverifiable():
    """parse_verdict with missing verdict key must return label -1 (unverifiable)."""
    result = parse_verdict('{"rationale": "no verdict field"}')
    assert result[0] == -1

def test_parse_verdict_unknown_verdict_is_unverifiable():
    """parse_verdict with unknown verdict value must return label -1 (unverifiable)."""
    result = parse_verdict('{"verdict": "partially true", "rationale": "x"}')
    assert result[0] == -1

def test_parse_severity_malformed_defaults_benign():
    """parse_severity with no JSON defaults to tier benign."""
    result = parse_severity("no json at all")
    assert result[0] == "benign"
