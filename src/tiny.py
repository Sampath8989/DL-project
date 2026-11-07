"""A randomly initialised LlamaForCausalLM plus a byte tokenizer.

Used to exercise the extraction path offline, where no pretrained
checkpoint can be downloaded. It produces no meaningful language, only
correctly shaped hidden states, so it validates plumbing, not science.
"""
import torch
from transformers import LlamaConfig, LlamaForCausalLM


class Enc(dict):
    """dict with a .to(device) so it can stand in for a BatchEncoding."""

    def to(self, device):
        return Enc({k: (v.to(device) if torch.is_tensor(v) else v)
                    for k, v in self.items()})


class CharTokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def __call__(self, text, return_tensors=None):
        ids = [min(b, 255) for b in text.encode("utf-8")][:256] or [0]
        return Enc(input_ids=torch.tensor([ids]),
                   attention_mask=torch.ones(1, len(ids), dtype=torch.long))

    def decode(self, ids, skip_special_tokens=True):
        return bytes(int(i) for i in ids if int(i) not in (0,)).decode(
            "utf-8", errors="ignore")

    def encode(self, text, add_special_tokens=False):
        """Byte ids, so src/baselines.py can look up its Yes/No tokens.

        A byte tokenizer gives " Yes" and " No" the same first token, the
        space; yes_no_ids() drops that overlap and keeps the distinct
        letters, which is enough to exercise the p(true) path offline."""
        return [min(b, 255) for b in text.encode("utf-8")][:256] or [0]


def build_tiny(seed=0, hidden=64, layers=4):
    torch.manual_seed(seed)
    cfg = LlamaConfig(vocab_size=256, hidden_size=hidden,
                      intermediate_size=hidden * 2, num_hidden_layers=layers,
                      num_attention_heads=4, num_key_value_heads=4,
                      max_position_embeddings=512)
    model = LlamaForCausalLM(cfg).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, CharTokenizer()
