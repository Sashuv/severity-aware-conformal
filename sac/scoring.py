import math

_PROMPT = """Consider this medical claim and decide whether it is factually true.
Claim: {claim}
Answer with a single word, True or False.
Answer:"""

def softmax_true_false(logit_true, logit_false):
    m = max(logit_true, logit_false)
    et = math.exp(logit_true - m); ef = math.exp(logit_false - m)
    return et / (et + ef)

def build_ptrue_prompt(claim):
    return _PROMPT.format(claim=claim)

def score_claim(claim, backend):
    return backend.true_prob(build_ptrue_prompt(claim))
