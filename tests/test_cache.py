from sac.crc import Claim
from sac.cache import append_claims, load_claims, existing_claim_ids

def test_roundtrip(tmp_path):
    p = tmp_path / "claims.jsonl"
    claims = [Claim("a claim", 0.8, "dangerous", 1, answer_id="q1", claim_id="q1_c0")]
    append_claims(str(p), claims)
    out = load_claims(str(p))
    assert len(out) == 1
    assert out[0].claim_id == "q1_c0"
    assert out[0].confidence == 0.8
    assert out[0].tier == "dangerous"

def test_append_is_resumable(tmp_path):
    p = tmp_path / "claims.jsonl"
    append_claims(str(p), [Claim("a", 0.5, claim_id="q1_c0")])
    append_claims(str(p), [Claim("b", 0.6, claim_id="q1_c1")])
    assert existing_claim_ids(str(p)) == {"q1_c0", "q1_c1"}
    assert len(load_claims(str(p))) == 2

def test_existing_ids_empty_when_missing(tmp_path):
    assert existing_claim_ids(str(tmp_path / "none.jsonl")) == set()
