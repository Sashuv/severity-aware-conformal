from sac.decompose import build_decompose_prompt, parse_claims, decompose
from sac.backends import MockGen

def test_parse_strips_markers_and_blanks():
    raw = "1. Ibuprofen is an NSAID.\n- It can raise blood pressure.\n\n  * Acetaminophen is an alternative.  \n"
    claims = parse_claims(raw)
    assert claims == [
        "Ibuprofen is an NSAID.",
        "It can raise blood pressure.",
        "Acetaminophen is an alternative.",
    ]

def test_prompt_contains_question_and_answer():
    p = build_decompose_prompt("Q?", "A text")
    assert "Q?" in p and "A text" in p

def test_decompose_uses_backend():
    backend = MockGen(scripted={"A text": "1. claim one\n2. claim two"})
    claims = decompose("Q?", "A text", backend)
    assert claims == ["claim one", "claim two"]
