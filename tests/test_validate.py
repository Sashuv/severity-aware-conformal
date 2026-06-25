from sac.crc import Claim
from sac.kqa_loader import NLIPair
from sac.validate import (labeler_agreement, score_auroc, gate2_pass,
                          unverifiable_count)
from sac.backends import MockJudge

def test_score_auroc_excludes_unverifiable():
    claims = [Claim("a",0.9,label=0), Claim("b",0.1,label=1),
              Claim("c",0.5,label=-1)]   # the -1 must be ignored
    assert score_auroc(claims) == 1.0

def test_gate2_threshold():
    assert gate2_pass(0.71) is True
    assert gate2_pass(0.69) is False

def test_unverifiable_count():
    claims = [Claim("a",0.9,label=0), Claim("b",0.5,label=-1), Claim("c",0.5,label=-1)]
    assert unverifiable_count(claims) == 2

def test_labeler_agreement_perfect():
    pairs = [NLIPair("evidence", "hypo1", "contradiction"),
             NLIPair("evidence", "hypo2", "entailment")]
    judge = MockJudge(['{"verdict":"contradicted","rationale":""}',
                       '{"verdict":"entailed","rationale":""}'])
    out = labeler_agreement(pairs, judge)
    assert out["accuracy"] == 1.0
    assert out["n"] == 2
