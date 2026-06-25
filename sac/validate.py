import numpy as np
from sklearn.metrics import cohen_kappa_score
from sac.crc import auroc
from sac.grader import build_grade_prompt, parse_verdict

_LABEL_TO_NLI = {0: "entailment", 1: "contradiction", -1: "neutral"}

def labeler_agreement(pairs, backend):
    gold, pred = [], []
    for p in pairs:
        label, _ = parse_verdict(backend.judge(build_grade_prompt(p.hypothesis, [p.premise])))
        pred.append(_LABEL_TO_NLI[label])
        gold.append(p.label)
    n = len(gold)
    acc = float(np.mean([g == q for g, q in zip(gold, pred)])) if n else 0.0
    try:
        kappa = float(cohen_kappa_score(gold, pred))
    except ValueError:
        kappa = float("nan")
    return {"accuracy": acc, "kappa": kappa, "n": n}

def score_auroc(claims):
    conf = [c.confidence for c in claims if c.label in (0, 1)]
    lab = [c.label for c in claims if c.label in (0, 1)]
    return auroc(conf, lab)

def gate2_pass(auroc_value, threshold=0.7):
    return bool(auroc_value >= threshold)

def unverifiable_count(claims):
    return sum(1 for c in claims if c.label == -1)
