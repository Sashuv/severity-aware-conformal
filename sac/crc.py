from dataclasses import dataclass
import numpy as np

@dataclass
class Claim:
    text: str
    confidence: float
    tier: str = "benign"
    label: int = -1          # 0=true, 1=hallucination, -1=unverifiable/unknown
    answer_id: str = ""
    claim_id: str = ""
    grader_rationale: str = ""
    severity_rationale: str = ""

def crc_calibrate(confidences, labels, alpha, B=1.0):
    confidences = np.asarray(confidences, dtype=float)
    labels = np.asarray(labels, dtype=int)
    n = len(confidences)
    if n == 0:
        return np.inf
    for lam in np.sort(np.unique(confidences)):
        kept = confidences >= lam
        rhat = np.sum(kept & (labels == 1)) / n
        corrected = (n / (n + 1)) * rhat + B / (n + 1)
        if corrected <= alpha:
            return float(lam)
    return np.inf

def crc_calibrate_stratified(claims, alphas, B=1.0):
    thresholds = {}
    for tier, alpha in alphas.items():
        conf = np.array([c.confidence for c in claims if c.tier == tier])
        lab = np.array([c.label for c in claims if c.tier == tier])
        thresholds[tier] = crc_calibrate(conf, lab, alpha, B=B)
    return thresholds

def apply_global(claims, lam):
    return np.array([c.confidence >= lam for c in claims])

def apply_stratified(claims, thresholds):
    return np.array([c.confidence >= thresholds[c.tier] for c in claims])

def auroc(confidences, labels):
    confidences = np.asarray(confidences); labels = np.asarray(labels)
    pos = confidences[labels == 0]; neg = confidences[labels == 1]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty_like(order, dtype=float); ranks[order] = np.arange(1, len(order)+1)
    return (ranks[:len(pos)].sum() - len(pos)*(len(pos)+1)/2) / (len(pos)*len(neg))

def retention(claims, kept):
    kept = np.asarray(kept); labels = np.array([c.label for c in claims])
    nt = np.sum(labels == 0)
    return np.sum(kept & (labels == 0)) / nt if nt else 0.0

def realized_risk_marginal(claims, kept):
    kept = np.asarray(kept); labels = np.array([c.label for c in claims])
    return np.sum(kept & (labels == 1)) / len(claims) if len(claims) else 0.0

def tier_risk_marginal(claims, kept, tier):
    kept = np.asarray(kept); idx = np.array([c.tier == tier for c in claims])
    labels = np.array([c.label for c in claims]); nt = idx.sum()
    return np.sum(kept & idx & (labels == 1)) / nt if nt else 0.0

def tier_retention(claims, kept, tier):
    kept = np.asarray(kept); idx = np.array([c.tier == tier for c in claims])
    labels = np.array([c.label for c in claims]); nt = np.sum(idx & (labels == 0))
    return np.sum(kept & idx & (labels == 0)) / nt if nt else 0.0
