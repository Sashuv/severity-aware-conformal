from sac.crc import Claim
from sac.ablation import run_ablation


def _claims(spec):
    # spec: list of (tier, label, confidence)
    return [Claim(f"t{i}", conf, tier, lab, claim_id=f"c{i}")
            for i, (tier, lab, conf) in enumerate(spec)]


def test_run_ablation_perfect_score_controls_danger_and_retains():
    # A perfect score: true claims high confidence, hallucinations low.
    spec = ([("dangerous", 0, 0.9)] * 40 + [("dangerous", 1, 0.1)] * 10 +
            [("benign", 0, 0.9)] * 40 + [("benign", 1, 0.1)] * 10)
    res = run_ablation(_claims(spec), 0.10, 0.05, 0.15, n_splits=20)
    assert res["s_d_risk"] <= 0.05 + 1e-9          # danger budget held
    assert res["g_marg"] <= 0.10 + 1e-9            # global marginal held
    assert res["s_d_ret"] > 0.9                    # near-all true dangerous kept
    assert 0.99 <= res["auroc"] <= 1.0


def test_run_ablation_returns_all_metrics():
    spec = ([("dangerous", 0, 0.5)] * 30 + [("dangerous", 1, 0.5)] * 5 +
            [("benign", 0, 0.5)] * 30 + [("benign", 1, 0.5)] * 5)
    res = run_ablation(_claims(spec), 0.10, 0.05, 0.15, n_splits=10)
    for k in ("auroc", "g_marg", "g_d_risk", "g_b_risk", "g_d_ret", "g_b_ret",
              "s_d_risk", "s_b_risk", "s_d_ret", "s_b_ret",
              "g_violation", "s_violation"):
        assert k in res
        assert res[k] == res[k]  # not NaN
