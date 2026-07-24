import torch
from sac.scoring import softmax_true_false
from sac.scores import mean_logprob

class HFBackend:
    def __init__(self, model_id="meta-llama/Meta-Llama-3-8B-Instruct"):
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                                 bnb_4bit_quant_type="nf4")
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=bnb, device_map="auto")
        # token ids for " True" / " False" (leading space = mid-sentence variant)
        self.id_true = self.tok(" True", add_special_tokens=False).input_ids[-1]
        self.id_false = self.tok(" False", add_special_tokens=False).input_ids[-1]

    @torch.no_grad()
    def generate(self, prompt, max_new_tokens=512):
        inputs = self.tok(prompt, return_tensors="pt").to(self.model.device)
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens,
                                  do_sample=False, pad_token_id=self.tok.eos_token_id)
        text = self.tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return text.strip()

    @torch.no_grad()
    def true_prob(self, prompt):
        inputs = self.tok(prompt, return_tensors="pt").to(self.model.device)
        logits = self.model(**inputs).logits[0, -1, :]   # next-token logits
        return softmax_true_false(float(logits[self.id_true]), float(logits[self.id_false]))

    @torch.no_grad()
    def seq_logprob(self, context, claim):
        """Length-normalized log-prob of `claim` tokens given `context` (score S2).

        Teacher-forced: each claim token is scored by the distribution at the
        previous position. Higher (closer to 0) = the model finds the claim more
        probable in context. Aggregation lives in the tested `mean_logprob`.
        """
        ctx_ids = self.tok(context, return_tensors="pt").input_ids[0]
        claim_ids = self.tok(claim, add_special_tokens=False).input_ids
        full = torch.cat([ctx_ids, torch.tensor(claim_ids)]).unsqueeze(0).to(self.model.device)
        logprobs = torch.log_softmax(self.model(full).logits[0], dim=-1)
        start = ctx_ids.shape[0]
        tok_lp = [float(logprobs[start + i - 1, claim_ids[i]]) for i in range(len(claim_ids))]
        return mean_logprob(tok_lp)
