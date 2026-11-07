"""The one place decoding is decided.

Why this module exists
    `model.generate(do_sample=False)` does not mean greedy decoding with
    nothing else applied. Hugging Face starts from the checkpoint's own
    generation_config.json, and a repetition penalty is a logits
    processor, not a sampling warper, so it is applied even when
    sampling is off. The checkpoints in this project ship:

        Qwen2.5-0.5B-Instruct   repetition_penalty 1.10, top_k 20
        Qwen2.5-1.5B-Instruct   repetition_penalty 1.10, top_k 20
        Qwen2.5-3B-Instruct     repetition_penalty 1.05, top_k 20
        Llama-3.2-1B/3B         neither

    So the first extraction run decoded Qwen under a penalty and Llama
    without one, and did not even hold the penalty fixed across the three
    Qwen sizes. Every cross-family comparison was confounded by decoding
    rather than by the model, the within-Qwen size trend was partly a
    1.10 -> 1.05 change, and because the penalty also pushes probability
    away from tokens already in the prompt, Qwen's answers were pushed
    away from the vocabulary of the very question being asked.

    The fix is not to match the penalty in the scorer. That would make
    the consistency check pass and leave the confound in place. Decoding
    is pinned here, explicitly, for every model, and recorded.

The split this module enforces
    token plumbing  (bos/eos/pad ids)   comes from the checkpoint
    decoding policy (penalty, warpers)  comes from us, never the checkpoint

Nothing here is tuned. These are fixed choices, recorded in every output
file so a result can always be traced to the decoding that produced it.
"""
from dataclasses import dataclass

# Pinned sampling policy for the baselines that need samples.
# temperature 1.0 rather than 0.7: 0.7 was inherited from Qwen's shipped
# config by accident, which is no reason for a research choice, and the
# published semantic-entropy protocol samples near temperature 1.
SAMPLE_TEMPERATURE = 1.0
SAMPLE_TOP_P = 0.9
SAMPLE_TOP_K = 0           # 0 disables it; HF would otherwise default to 50
SAMPLE_REPETITION_PENALTY = 1.0

# decoding-relevant keys worth recording from whatever a checkpoint ships
AUDIT_KEYS = ("do_sample", "temperature", "top_p", "top_k",
              "repetition_penalty", "no_repeat_ngram_size",
              "encoder_repetition_penalty", "length_penalty", "num_beams",
              "min_length", "min_new_tokens", "renormalize_logits")


@dataclass
class DecodingRecord:
    """What was actually used, for the run record."""
    shipped: dict
    pinned: dict
    overridden: dict

    def as_dict(self):
        return dict(shipped=self.shipped, pinned=self.pinned,
                    overridden=self.overridden)


def _token_ids(model):
    """bos/eos/pad from the checkpoint -- these are plumbing, not policy.

    Dropping them would stop generation from terminating: Qwen2.5 ends on
    two different ids and only its own config knows that.
    """
    gc = getattr(model, "generation_config", None)
    cfg = model.config
    out = {}
    for k in ("bos_token_id", "eos_token_id", "pad_token_id"):
        v = getattr(gc, k, None) if gc is not None else None
        if v is None:
            v = getattr(cfg, k, None)
        if v is not None:
            out[k] = v
    if "pad_token_id" not in out and "eos_token_id" in out:
        e = out["eos_token_id"]
        out["pad_token_id"] = e[0] if isinstance(e, (list, tuple)) else e
    return out


def audit_shipped(model):
    """The checkpoint's own decoding settings, as a plain dict."""
    gc = getattr(model, "generation_config", None)
    if gc is None:
        return {}
    return {k: getattr(gc, k) for k in AUDIT_KEYS
            if getattr(gc, k, None) is not None}


def greedy_config(model, max_new_tokens):
    """True greedy decoding, identical for every model in the grid.

    A fresh GenerationConfig is built rather than the checkpoint's being
    edited, because `generate(generation_config=cfg)` uses the object
    given instead of merging the checkpoint's in. Every field that could
    alter the distribution is set here even where it equals the library
    default, so the record says what was intended rather than what a
    future version of transformers happens to default to.
    """
    from transformers import GenerationConfig
    cfg = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        repetition_penalty=1.0,
        encoder_repetition_penalty=1.0,
        no_repeat_ngram_size=0,
        length_penalty=1.0,
        min_new_tokens=None,
        renormalize_logits=False,
        **_token_ids(model),
    )
    return cfg


def sampling_config(model, max_new_tokens, k, temperature=None, top_p=None,
                    top_k=None):
    """Pinned sampling for the semantic-entropy and self-consistency
    baselines. Identical across models, for the same reason as above:
    Qwen would otherwise contribute top_k 20 and Llama the library's
    default of 50, and the two entropies would not be comparable."""
    from transformers import GenerationConfig
    cfg = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=True,
        num_beams=1,
        num_return_sequences=k,
        temperature=SAMPLE_TEMPERATURE if temperature is None else temperature,
        top_p=SAMPLE_TOP_P if top_p is None else top_p,
        top_k=SAMPLE_TOP_K if top_k is None else top_k,
        repetition_penalty=SAMPLE_REPETITION_PENALTY,
        encoder_repetition_penalty=1.0,
        no_repeat_ngram_size=0,
        renormalize_logits=False,
        **_token_ids(model),
    )
    return cfg


def record(model, cfg):
    """Build the record that goes into every output json."""
    shipped = audit_shipped(model)
    pinned = {k: getattr(cfg, k) for k in AUDIT_KEYS
              if getattr(cfg, k, None) is not None}
    pinned["max_new_tokens"] = cfg.max_new_tokens
    overridden = {k: dict(was=shipped[k], now=pinned.get(k))
                  for k in shipped
                  if k in pinned and shipped[k] != pinned[k]}
    return DecodingRecord(shipped, pinned, overridden)


def describe(rec, prefix="  "):
    lines = [f"{prefix}decoding pinned: "
             + ", ".join(f"{k}={v}" for k, v in sorted(rec.pinned.items()))]
    if rec.overridden:
        lines.append(f"{prefix}overrode the checkpoint's own settings: "
                     + ", ".join(f"{k} {d['was']} -> {d['now']}"
                                 for k, d in sorted(rec.overridden.items())))
    else:
        lines.append(f"{prefix}the checkpoint shipped nothing that needed "
                     f"overriding")
    return "\n".join(lines)
