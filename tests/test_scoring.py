import math
from sac.scoring import softmax_true_false, build_ptrue_prompt, score_claim
from sac.backends import MockGen

def test_softmax_equal_logits_is_half():
    assert abs(softmax_true_false(1.0, 1.0) - 0.5) < 1e-9

def test_softmax_higher_true_logit_above_half():
    assert softmax_true_false(3.0, 0.0) > 0.9

def test_prompt_contains_claim_and_true_false():
    p = build_ptrue_prompt("Ibuprofen is an NSAID.")
    assert "Ibuprofen is an NSAID." in p
    assert "True" in p and "False" in p

def test_score_claim_uses_backend_true_prob():
    backend = MockGen(true_probs={"5000 mg": 0.85})
    assert score_claim("Take 5000 mg per day.", backend) == 0.85
