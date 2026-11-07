"""Published uncertainty baselines, computed from the same frozen LM.

These are the methods a probe has to beat before "read the internal
states" is worth anything. All of them use only the model's output
distribution or its own sampled text -- never its hidden activations --
so they are exactly the comparison a reviewer should demand.

  predictive entropy      mean/max per-token entropy of the greedy answer
                          (Malinin & Gales; Kuhn et al. 2023 baseline)
  p(true)                 the model is shown its own answer and asked
                          whether it is correct; read P(Yes)
                          (Kadavath et al. 2022, "self-evaluation")
  semantic entropy        entropy over meaning-clusters of k sampled
                          answers. The discrete variant is used: clusters
                          are exact matches after normalisation, which
                          needs no entailment model and therefore fits in
                          6 GB alongside the generator
                          (Farquhar et al., Nature 2024)
  self-consistency        fraction of k samples that disagree with the
                          greedy answer (Wang et al.; Manakul et al.)

Nothing here is trained and nothing here receives a gradient. The LM is
frozen, as everywhere else in this project.

Orientation
    Every feature is stored raw. BASE_ORIENT says which sign points at
    hallucination, so a single-feature AUROC can be computed without
    silently flipping a baseline to make it look better. Feature sets fed
    to logistic regression do not need it -- the fit finds the sign.
"""
import math

import numpy as np
import torch

from .base_features import BASE_ORIENT, BASE_NAMES  # noqa: F401

YES = ["Yes", " Yes", "yes", " yes", "YES", "True", " True"]
NO = ["No", " No", "no", " no", "NO", "False", " False"]

P_TRUE_TEMPLATE = (
    "Question: {q}\n"
    "Proposed answer: {a}\n"
    "Is the proposed answer correct? Reply Yes or No.\n"
    "Answer:"
)


# ------------------------------------------------------------------ helpers
def _first_ids(tok, words):
    """First-token ids for each candidate spelling, de-duplicated.

    Scoring a single next-token position means only the first token of
    "Yes" can be read, so several spellings are summed rather than one
    being guessed at."""
    ids = set()
    for w in words:
        try:
            enc = tok.encode(w, add_special_tokens=False)
        except TypeError:
            enc = tok.encode(w)
        if len(enc):
            ids.add(int(enc[0]))
    return sorted(ids)


def yes_no_ids(tok):
    y, n = _first_ids(tok, YES), _first_ids(tok, NO)
    overlap = set(y) & set(n)
    if overlap:                      # a tokenizer that cannot tell them apart
        y = [i for i in y if i not in overlap]
        n = [i for i in n if i not in overlap]
    if not y or not n:
        raise ValueError("tokenizer gives no distinct Yes/No first tokens")
    return y, n


def _enc(tok, text, device):
    enc = tok(text, return_tensors="pt")
    return enc.to(device) if hasattr(enc, "to") else enc


# ------------------------------------------------- single-pass token scores
def eos_ids_of(model):
    """Every id the checkpoint terminates on (Qwen2.5 has two)."""
    from .decoding import _token_ids
    e = _token_ids(model).get("eos_token_id")
    if e is None:
        return []
    return [int(x) for x in (e if isinstance(e, (list, tuple)) else [e])]


@torch.no_grad()
def teacher_forced_scores(model, tok, prompt, answer, device,
                          n_generated=None, eos_ids=None):
    """Per-token logprob and entropy of the greedy answer, in one forward.

    Greedy decoding is deterministic given its prefix, so the distribution
    at answer position t under teacher forcing is the same distribution the
    generator saw at step t. One forward pass therefore recovers what
    generation would have reported, which is why extraction does not have
    to be re-run to add these features.

    `n_generated` is the number of steps generation actually took. The
    cached answer is decoded with skip_special_tokens, so a generation that
    stopped on EOS has one more step than the text has tokens, and the
    generation-time mean includes that EOS step. When n_generated equals
    n_ans + 1 the EOS step is scored here too, so both routes average over
    the same steps; without this every short answer disagreed by one EOS
    logprob. Greedy emitted the argmax, so with several EOS ids the
    generated one is the most probable of them. Any other mismatch in
    token count means the re-tokenised text is not what was generated, and
    the scores are NaN rather than silently misaligned.
    """
    p_ids = _enc(tok, prompt, device)["input_ids"]
    full_ids = _enc(tok, prompt + answer, device)["input_ids"]
    n_p = p_ids.shape[1]
    n_ans = full_ids.shape[1] - n_p
    nan = dict(mean_logprob_tf=float("nan"), min_logprob_tf=float("nan"),
               mean_entropy=float("nan"), max_entropy=float("nan"))
    score_eos = False
    if n_generated is not None:
        if n_generated == n_ans + 1 and eos_ids:
            score_eos = True
        elif n_generated != n_ans:
            return dict(nan, n_scored=0)
    if n_ans <= 0 and not score_eos:
        return dict(nan, n_scored=0)

    logits = model(input_ids=full_ids).logits[0].float()
    lps, ents = [], []
    for t in range(n_ans + int(score_eos)):
        pos = n_p + t - 1                      # distribution that produced token n_p+t
        logp = torch.log_softmax(logits[pos], dim=-1)
        if t < n_ans:
            lps.append(float(logp[int(full_ids[0, n_p + t])]))
        else:                                  # the EOS step generation took
            lps.append(float(logp[eos_ids].max()))
        p = logp.exp()
        ents.append(float(-(p * logp).sum()))
    return dict(mean_logprob_tf=float(np.mean(lps)),
                min_logprob_tf=float(np.min(lps)),
                mean_entropy=float(np.mean(ents)),
                max_entropy=float(np.max(ents)),
                n_scored=len(lps))


@torch.no_grad()
def p_true(model, tok, question, answer, device, ids=None):
    """Kadavath-style self-verification: P(Yes) at the verdict position."""
    y_ids, n_ids = ids if ids is not None else yes_no_ids(tok)
    text = P_TRUE_TEMPLATE.format(q=question, a=answer.strip() or "(no answer)")
    enc = _enc(tok, text, device)
    logits = model(**enc).logits[0, -1].float()
    logp = torch.log_softmax(logits, dim=-1)
    py = float(torch.logsumexp(logp[y_ids], dim=0).exp())
    pn = float(torch.logsumexp(logp[n_ids], dim=0).exp())
    tot = py + pn
    # renormalised over the two verdicts: the absolute mass on "Yes" is
    # dominated by formatting, the ratio is the actual judgement
    return float(py / tot) if tot > 0 else float("nan")


# --------------------------------------------------------------- sampling
@torch.no_grad()
def sample_answers(model, tok, prompt, k=5, temperature=None, top_p=None,
                   max_new_tokens=24, device="cuda", seed=None,
                   gen_cfg=None):
    """k sampled continuations in one batched generate call.

    Hidden states are deliberately not requested: these samples feed the
    text-level baselines only, so the pass is cheap in both time and VRAM
    even where the hidden-state capture of extraction is not.

    Decoding is pinned by src.decoding rather than left to the
    checkpoint. Without that, Qwen contributes top_k 20 from its own
    config and Llama gets the library default of 50, and the resulting
    semantic entropies are not comparable across families -- the same
    class of bug as the repetition penalty in extraction.
    """
    if seed is not None:
        torch.manual_seed(seed)
    if gen_cfg is None:
        from .decoding import sampling_config
        gen_cfg = sampling_config(model, max_new_tokens, k,
                                  temperature=temperature, top_p=top_p)
    enc = _enc(tok, prompt, device)
    out = model.generate(**enc, generation_config=gen_cfg)
    n_p = enc["input_ids"].shape[1]
    texts = []
    for i in range(out.shape[0]):
        texts.append(tok.decode(out[i, n_p:], skip_special_tokens=True))
    return texts


def discrete_semantic_entropy(samples, norm):
    """Entropy over exact-match meaning clusters of the sampled answers.

    Reported in nats and NOT normalised by log k, so the number is
    comparable to the published quantity. Range is [0, log k].
    """
    forms = [norm(s) for s in samples]
    forms = [f for f in forms if f != ""]
    if not forms:
        return float("nan"), float("nan")
    counts = {}
    for f in forms:
        counts[f] = counts.get(f, 0) + 1
    n = len(forms)
    h = -sum((c / n) * math.log(c / n) for c in counts.values())
    return float(h), float(len(counts) / n)


def self_consistency(greedy, samples, norm):
    """Disagreement rate between the sampled answers and the greedy one.

    Stored as disagreement (not agreement) so that, like every other
    feature here, a larger number leans towards hallucinated.
    """
    g = norm(greedy)
    forms = [norm(s) for s in samples]
    forms = [f for f in forms if f != ""]
    if not forms or g == "":
        return float("nan")
    agree = sum(1 for f in forms if f == g) / len(forms)
    return float(1.0 - agree)


# ------------------------------------------------------------------ driver
def item_baselines(model, tok, question, prompt, greedy_text, norm, device,
                   k=5, temperature=None, max_new_tokens=24, yn_ids=None,
                   seed=None, gen_cfg=None, n_generated=None, eos_ids=None):
    """Every baseline feature for one item, as a name -> float dict."""
    r = teacher_forced_scores(model, tok, prompt, greedy_text, device,
                              n_generated=n_generated, eos_ids=eos_ids)
    n_scored = r.pop("n_scored")
    try:
        r["p_true"] = p_true(model, tok, question, greedy_text, device, yn_ids)
    except Exception:
        r["p_true"] = float("nan")
    samples = sample_answers(model, tok, prompt, k=k, temperature=temperature,
                             max_new_tokens=max_new_tokens, device=device,
                             seed=seed, gen_cfg=gen_cfg)
    se, distinct = discrete_semantic_entropy(samples, norm)
    r["semantic_entropy"] = se
    r["distinct_rate"] = distinct
    r["self_consistency"] = self_consistency(greedy_text, samples, norm)
    return r, samples, n_scored


def to_matrix(rows, names=None):
    """Stack per-item dicts into [N, F] in a fixed, recorded column order."""
    names = list(names) if names else BASE_NAMES
    M = np.full((len(rows), len(names)), np.nan, dtype=np.float32)
    for i, r in enumerate(rows):
        for j, nm in enumerate(names):
            v = r.get(nm, float("nan"))
            M[i, j] = float(v) if v is not None else float("nan")
    return M, names
