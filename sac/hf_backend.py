import torch
from sac.scoring import softmax_true_false

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
