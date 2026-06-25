import json
from dataclasses import dataclass

@dataclass
class KQAStatement:
    text: str
    importance: str   # "must_have" | "nice_to_have"

@dataclass
class KQAItem:
    qid: str
    question: str
    reference_answer: str
    statements: list

@dataclass
class NLIPair:
    premise: str
    hypothesis: str
    label: str        # "entailment" | "neutral" | "contradiction"

def _first(d, keys, default=""):
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]
    return default

def load_kqa(path):
    items = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            qid = str(_first(row, ["id", "qid", "question_id"], default=f"q{i}"))
            question = _first(row, ["Question", "question"])
            answer = _first(row, ["Answer", "answer", "reference_answer"])
            stmts = []
            for s in _first(row, ["Must_have", "must_have", "must_haves"], default=[]):
                stmts.append(KQAStatement(str(s), "must_have"))
            for s in _first(row, ["Nice_to_have", "nice_to_have", "nice_to_haves"], default=[]):
                stmts.append(KQAStatement(str(s), "nice_to_have"))
            items.append(KQAItem(qid, question, answer, stmts))
    return items

def load_physician_nli(path):
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            pairs.append(NLIPair(
                premise=_first(row, ["premise", "statement", "evidence"]),
                hypothesis=_first(row, ["hypothesis", "claim", "generated"]),
                label=_first(row, ["label", "gold_label"]).lower(),
            ))
    return pairs

def gold_statements(item):
    return [s.text for s in item.statements]
