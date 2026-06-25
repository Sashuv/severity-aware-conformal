import numpy as np
from sac.crc import (Claim, crc_calibrate, crc_calibrate_stratified,
                     apply_stratified, auroc, retention, realized_risk_marginal)

def test_calibrate_picks_threshold_meeting_budget():
    # 10 claims: confidences 0.1..1.0; the 3 lowest-confidence are hallucinations.
    conf = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    lab  = [1,1,1,0,0,0,0,0,0,0]
    lam = crc_calibrate(conf, lab, alpha=0.10)
    # Keeping conf>=0.4 retains zero hallucinations -> corrected risk = 1/11 <= 0.10
    assert lam == 0.4

def test_calibrate_infeasible_returns_inf():
    # cushion 1/(n+1) alone exceeds alpha for tiny n
    lam = crc_calibrate([0.9, 0.8], [1, 0], alpha=0.01)
    assert not np.isfinite(lam)

def test_stratified_independent_per_tier():
    claims = ([Claim("d", 0.9, "dangerous", 0) for _ in range(50)] +
              [Claim("d", 0.2, "dangerous", 1) for _ in range(5)] +
              [Claim("b", 0.6, "benign", 0) for _ in range(50)] +
              [Claim("b", 0.2, "benign", 1) for _ in range(5)])
    thr = crc_calibrate_stratified(claims, {"dangerous": 0.05, "benign": 0.20})
    assert thr["dangerous"] <= 0.9 and np.isfinite(thr["dangerous"])
    assert np.isfinite(thr["benign"])

def test_auroc_perfect_separation():
    assert auroc([0.9,0.8,0.7,0.2,0.1], [0,0,0,1,1]) == 1.0

def test_retention_and_marginal_risk():
    claims = [Claim("a",0.9,"benign",0), Claim("b",0.9,"benign",1),
              Claim("c",0.1,"benign",0)]
    kept = np.array([True, True, False])
    assert retention(claims, kept) == 0.5            # 1 of 2 true claims kept
    assert realized_risk_marginal(claims, kept) == 1/3  # 1 kept halluc / 3 total
