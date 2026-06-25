import re

_PROMPT = """You are decomposing a medical answer into atomic claims.
An atomic claim is a single, self-contained, verifiable statement of fact.
Output ONE claim per line, no numbering, no commentary.

Question: {question}
Answer: {answer}

Atomic claims:"""

def build_decompose_prompt(question, answer):
    return _PROMPT.format(question=question, answer=answer)

def parse_claims(raw):
    claims = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"^\s*(\d+[\.\)]|[-*•])\s*", "", line).strip()
        if line:
            claims.append(line)
    return claims

def decompose(question, answer, backend):
    raw = backend.generate(build_decompose_prompt(question, answer))
    return parse_claims(raw)
