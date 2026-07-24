"""Alternative claim scores for the score-ablation study.

Every score obeys the CRC convention *higher = more likely true*, so each can be
dropped straight into `crc_calibrate` (which thresholds with `confidence >= lam`).
We hold the claims, labels and tiers fixed and swap only the score, so the
ablation isolates the score as the single variable.
"""
import numpy as np
from dataclasses import replace


def random_scores(claim_ids, seed=0):
    """Deterministic uniform[0,1) score per claim id — the noise-floor baseline.

    A useless score still yields valid risk control (CRC is distribution-free);
    only retention collapses. That contrast is the cleanest proof of the claim.
    """
    rng = np.random.default_rng(seed)
    return {cid: float(rng.random()) for cid in claim_ids}


def swap_confidence(claims, scores_by_id):
    """Return copies of `claims` with `.confidence` replaced by the given score.

    Label and tier are preserved; originals are left untouched. Raises KeyError
    if any claim has no score, so a partial scoring pass can't silently skew.
    """
    return [replace(c, confidence=float(scores_by_id[c.claim_id])) for c in claims]


def softmax_entail(logits, entail_index=2):
    """P(entailment) from 3-way NLI logits ordered [contradiction, neutral, entailment]."""
    a = np.asarray(logits, dtype=float)
    e = np.exp(a - a.max())
    return float((e / e.sum())[entail_index])


def mean_logprob(token_logprobs):
    """Length-normalized sequence log-prob (mean of token log-probs).

    Higher (closer to 0) = the model finds the claim more probable. Empty = -inf.
    """
    a = np.asarray(token_logprobs, dtype=float)
    return float(a.mean()) if a.size else float("-inf")
