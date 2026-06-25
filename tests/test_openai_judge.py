from sac.openai_judge import OpenAIJudge

class _Msg:
    def __init__(self, content): self.message = type("m", (), {"content": content})
class _Resp:
    def __init__(self, content): self.choices = [_Msg(content)]

class FlakyClient:
    """Raises `fails` times, then returns the canned content."""
    def __init__(self, content, fails=0):
        self.content = content; self.fails = fails; self.calls = 0
        self.chat = type("c", (), {"completions": self})()
    def create(self, **kw):
        self.calls += 1
        if self.calls <= self.fails:
            raise RuntimeError("transient")
        return _Resp(self.content)

def test_returns_content_on_success():
    client = FlakyClient('{"verdict":"entailed"}')
    judge = OpenAIJudge(client=client)
    assert judge.judge("prompt") == '{"verdict":"entailed"}'

def test_retries_then_succeeds():
    client = FlakyClient('{"verdict":"neutral"}', fails=2)
    judge = OpenAIJudge(client=client, max_retries=4, backoff_base=0)
    assert judge.judge("prompt") == '{"verdict":"neutral"}'
    assert client.calls == 3

def test_raises_after_exhausting_retries():
    client = FlakyClient("x", fails=99)
    judge = OpenAIJudge(client=client, max_retries=2, backoff_base=0)
    try:
        judge.judge("prompt"); assert False
    except RuntimeError:
        assert client.calls == 2
