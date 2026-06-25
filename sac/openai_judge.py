import os, time

class OpenAIJudge:
    def __init__(self, model="gpt-4o", max_retries=4, backoff_base=1.0,
                 temperature=0.0, client=None):
        self.model = model
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.temperature = temperature
        if client is None:
            from openai import OpenAI
            client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.client = client

    def judge(self, prompt):
        last = None
        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    messages=[{"role": "user", "content": prompt}],
                )
                return resp.choices[0].message.content
            except Exception as e:                       # noqa: BLE001 (retry any transient)
                last = e
                if self.backoff_base:
                    time.sleep(self.backoff_base * (2 ** attempt))
        raise last
