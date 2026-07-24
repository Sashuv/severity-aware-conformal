import pytest
from sac.crc import Claim
from sac.scores import random_scores, swap_confidence, softmax_entail, mean_logprob


def test_random_scores_reproducible_and_in_range():
    a = random_scores(["x", "y", "z"], seed=0)
    b = random_scores(["x", "y", "z"], seed=0)
    assert a == b
    assert set(a) == {"x", "y", "z"}
    assert all(0.0 <= v < 1.0 for v in a.values())


def test_random_scores_seed_changes_values():
    assert random_scores(["x", "y"], seed=0) != random_scores(["x", "y"], seed=1)


def test_swap_confidence_replaces_score_preserves_label_and_tier():
    claims = [Claim("a", 0.1, "dangerous", 1, claim_id="c0"),
              Claim("b", 0.2, "benign", 0, claim_id="c1")]
    out = swap_confidence(claims, {"c0": 0.9, "c1": 0.3})
    assert [c.confidence for c in out] == [0.9, 0.3]
    assert [c.tier for c in out] == ["dangerous", "benign"]
    assert [c.label for c in out] == [1, 0]
    # originals untouched
    assert claims[0].confidence == 0.1


def test_swap_confidence_missing_id_raises():
    claims = [Claim("a", 0.1, "benign", 0, claim_id="c0")]
    with pytest.raises(KeyError):
        swap_confidence(claims, {})


def test_softmax_entail_high_entailment_logit_near_one():
    # logits ordered [contradiction, neutral, entailment]
    assert softmax_entail([0.0, 0.0, 10.0]) > 0.99


def test_softmax_entail_high_contradiction_near_zero():
    assert softmax_entail([10.0, 0.0, 0.0]) < 0.01


def test_softmax_entail_uniform_is_one_third():
    assert abs(softmax_entail([1.0, 1.0, 1.0]) - 1 / 3) < 1e-9


def test_mean_logprob_is_length_normalized():
    assert mean_logprob([-1.0, -3.0]) == -2.0


def test_mean_logprob_empty_is_neg_inf():
    assert mean_logprob([]) == float("-inf")
