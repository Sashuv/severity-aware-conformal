"""Run the global-vs-severity-aware headline for one score, averaged over splits.

The score ablation calls this once per scorer on the *same* fixed claims (with
`.confidence` swapped to that scorer's value) and compares the resulting tables.
"""
import numpy as np
from sac.crc import (crc_calibrate, crc_calibrate_stratified, apply_global,
                     apply_stratified, realized_risk_marginal, tier_risk_marginal,
                     tier_retention)
from sac.validate import score_auroc


def run_ablation(claims, alpha_marginal, alpha_dangerous, alpha_benign, n_splits=300):
    """Average global and severity-aware CRC metrics over `n_splits` cal/test splits.

    `claims` must be verifiable (label in {0,1}) with `.confidence` set to the
    score under test. Returns a dict of mean metrics plus danger-budget violation
    frequencies, computed on the held-out test half of each split.
    """
    G = {k: [] for k in ("marg", "d_risk", "b_risk", "d_ret", "b_ret")}
    S = {k: [] for k in ("d_risk", "b_risk", "d_ret", "b_ret")}
    for seed in range(n_splits):
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(claims)); k = len(claims) // 2
        cal = [claims[i] for i in idx[:k]]
        te = [claims[i] for i in idx[k:]]

        lam = crc_calibrate([c.confidence for c in cal], [c.label for c in cal], alpha_marginal)
        kg = apply_global(te, lam)
        G["marg"].append(realized_risk_marginal(te, kg))
        G["d_risk"].append(tier_risk_marginal(te, kg, "dangerous"))
        G["b_risk"].append(tier_risk_marginal(te, kg, "benign"))
        G["d_ret"].append(tier_retention(te, kg, "dangerous"))
        G["b_ret"].append(tier_retention(te, kg, "benign"))

        th = crc_calibrate_stratified(cal, {"dangerous": alpha_dangerous, "benign": alpha_benign})
        ks = apply_stratified(te, th)
        S["d_risk"].append(tier_risk_marginal(te, ks, "dangerous"))
        S["b_risk"].append(tier_risk_marginal(te, ks, "benign"))
        S["d_ret"].append(tier_retention(te, ks, "dangerous"))
        S["b_ret"].append(tier_retention(te, ks, "benign"))

    m = np.mean
    return {
        "auroc":       score_auroc(claims),
        "g_marg":      m(G["marg"]),
        "g_d_risk":    m(G["d_risk"]),  "g_b_risk": m(G["b_risk"]),
        "g_d_ret":     m(G["d_ret"]),   "g_b_ret":  m(G["b_ret"]),
        "s_d_risk":    m(S["d_risk"]),  "s_b_risk": m(S["b_risk"]),
        "s_d_ret":     m(S["d_ret"]),   "s_b_ret":  m(S["b_ret"]),
        "g_violation": m(np.array(G["d_risk"]) > alpha_dangerous),
        "s_violation": m(np.array(S["d_risk"]) > alpha_dangerous),
    }
