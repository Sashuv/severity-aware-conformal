import json, os
from dataclasses import asdict
from sac.crc import Claim

def append_claims(path, claims):
    with open(path, "a", encoding="utf-8") as f:
        for c in claims:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")

def load_claims(path):
    if not os.path.exists(path):
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(Claim(**json.loads(line)))
    return out

def existing_claim_ids(path):
    return {c.claim_id for c in load_claims(path) if c.claim_id}
