from typing import Protocol

class GenBackend(Protocol):
    def generate(self, prompt: str, **kw) -> str: ...
    def true_prob(self, prompt: str) -> float: ...

class JudgeBackend(Protocol):
    def judge(self, prompt: str) -> str: ...

class MockGen:
    """Deterministic generator for tests. `match` substrings map to outputs."""
    def __init__(self, scripted=None, true_probs=None, default_text="", default_prob=0.5):
        self.scripted = scripted or {}
        self.true_probs = true_probs or {}
        self.default_text = default_text
        self.default_prob = default_prob

    def generate(self, prompt, **kw):
        for key, val in self.scripted.items():
            if key in prompt:
                return val
        return self.default_text

    def true_prob(self, prompt):
        for key, val in self.true_probs.items():
            if key in prompt:
                return val
        return self.default_prob

class MockJudge:
    """Returns canned judge outputs in sequence; repeats the last one."""
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def judge(self, prompt):
        i = min(self.calls, len(self.responses) - 1)
        self.calls += 1
        return self.responses[i]
