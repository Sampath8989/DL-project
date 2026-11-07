"""Stage 1: run the frozen LM once per item and cache its internal states.

Nothing here is trained. The LM's weights never receive a gradient; the
forward pass is the expensive step, so its output is written to disk and
every probe experiment afterwards reads the cache instead of the GPU.
"""
import json
import os

import numpy as np
import torch


def load_model(cfg):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    dtype = dict(float16=torch.float16, bfloat16=torch.bfloat16,
                 float32=torch.float32)[cfg.dtype]
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name, dtype=dtype)
    model.to(cfg.device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok


@torch.no_grad()
def extract_item(model, tok, prompt, cfg, gen_cfg=None):
    """One generation pass; returns the cached tensors for a single item.

    `gen_cfg` must be the pinned config from src.decoding. Passing none
    falls back to the old behaviour, which inherits the checkpoint's own
    repetition penalty and is the bug this signature exists to prevent --
    so it is only reachable from the offline tiny-model path.
    """
    enc = tok(prompt, return_tensors="pt").to(cfg.device)
    kw = dict(return_dict_in_generate=True, output_hidden_states=True,
              output_scores=True)
    try:                      # raw, pre-processor logits where available
        out = model.generate(**enc, generation_config=gen_cfg,
                             output_logits=True, **kw)
    except TypeError:         # transformers too old for output_logits
        out = model.generate(**enc, generation_config=gen_cfg, **kw)

    # hidden_states: tuple over generated steps; each is a tuple over layers
    steps = out.hidden_states
    n_new = len(steps)

    # prompt-final position, every layer -> [L+1, d]
    h_prompt_last = torch.stack([l[0, -1, :] for l in steps[0]]).float().cpu()

    # per-generated-token, every layer -> [T, L+1, d]
    traj = []
    for t in range(n_new):
        traj.append(torch.stack([l[0, -1, :] for l in steps[t]]).float().cpu())
    traj = torch.stack(traj)
    h_answer_mean = traj.mean(0)

    if cfg.keep_token_trajectory:
        traj = traj[: cfg.max_traj_tokens]
    else:
        traj = traj[:1]
    if cfg.layer_stride > 1:
        traj = traj[:, :: cfg.layer_stride, :]

    # Teacher-free confidence features, taken from the RAW logits when the
    # installed transformers exposes them. `out.scores` are post-processor,
    # so if any logits processor is ever active again they would no longer
    # be the model's own distribution, and would stop agreeing with the
    # teacher-forced route in src/baselines.py. Raw logits make the two
    # definitionally the same quantity.
    seq = getattr(out, "logits", None) or out.scores
    lp = []
    for t, sc in enumerate(seq):
        logp = torch.log_softmax(sc[0].float(), dim=-1)
        tokid = out.sequences[0, enc["input_ids"].shape[1] + t]
        lp.append(float(logp[tokid]))

    text = tok.decode(out.sequences[0, enc["input_ids"].shape[1]:],
                      skip_special_tokens=True)
    return dict(h_prompt_last=h_prompt_last.numpy(),
                h_answer_mean=h_answer_mean.numpy(),
                traj=traj.numpy(),
                token_logprobs=np.array(lp, dtype=np.float32),
                n_new_tokens=n_new, text=text)


def run(items, cfg, label_fn, model=None, tok=None, progress_every=25,
        gen_cfg=None, dec_record=None):
    """Extract over a list of items and write one .npz plus a meta json.

    Progress is printed as it goes. This matters more than it looks: the
    .npz is written once at the very end, so without these lines a stalled
    extraction is indistinguishable from a working one, and an overnight
    run can sit dead for hours looking busy.
    """
    import time
    if model is None:
        model, tok = load_model(cfg)
    os.makedirs(cfg.out_dir, exist_ok=True)

    # Decoding is pinned here, never inherited from the checkpoint. See
    # src/decoding.py for what went wrong when it was inherited.
    if gen_cfg is None:
        from .decoding import greedy_config, record
        gen_cfg = greedy_config(model, cfg.max_new_tokens)
        dec_record = record(model, gen_cfg)
    if dec_record is not None:
        from .decoding import describe
        print(describe(dec_record), flush=True)

    H_p, H_a, TR, LP, Y, META = [], [], [], [], [], []
    n = len(items) if cfg.max_items is None else min(cfg.max_items, len(items))
    T = cfg.max_traj_tokens
    t0 = time.time()
    print(f"extracting {n} items -> {cfg.out_dir}", flush=True)
    for i in range(n):
        it = items[i]
        r = extract_item(model, tok, it["prompt"], cfg, gen_cfg)
        if progress_every and ((i + 1) % progress_every == 0 or i + 1 == n):
            el = time.time() - t0
            rate = (i + 1) / max(el, 1e-6)
            eta = (n - i - 1) / max(rate, 1e-9)
            print(f"  {i+1}/{n} items  {rate*60:.1f}/min  "
                  f"eta {eta/60:.1f} min  "
                  f"hallucinated so far {np.mean(Y) if Y else float('nan'):.1%}",
                  flush=True)
        pad = np.zeros((T, *r["traj"].shape[1:]), dtype=np.float32)
        m = np.zeros(T, dtype=np.float32)
        k = min(T, r["traj"].shape[0])
        pad[:k] = r["traj"][:k]
        m[:k] = 1.0
        H_p.append(r["h_prompt_last"])
        H_a.append(r["h_answer_mean"])
        TR.append(pad)
        LP.append([float(np.mean(r["token_logprobs"])),
                   float(np.min(r["token_logprobs"])),
                   float(r["n_new_tokens"]), float(len(r["text"])),
                   float(it.get("log_pop", float("nan")))])
        Y.append(int(label_fn(it, r["text"])))
        META.append(dict(idx=i, task=it.get("task", "na"), text=r["text"],
                         question=it.get("question", ""),
                         gold=it.get("gold", []), mask=m.tolist()))

    conf_names = ["mean_logprob", "min_logprob", "n_new_tokens",
                  "answer_chars", "log_popularity"]
    path = os.path.join(cfg.out_dir, "acts.npz")
    print(f"writing cache ({n} items) -- this is the slow single write, "
          f"do not interrupt it", flush=True)
    np.savez_compressed(path,
                        h_prompt=np.stack(H_p), h_answer=np.stack(H_a),
                        traj=np.stack(TR),
                        mask=np.stack([np.array(m["mask"], np.float32) for m in META]),
                        conf=np.array(LP, np.float32),
                        y=np.array(Y, np.int64),
                        task=np.array([m["task"] for m in META]),
                        conf_names=np.array(conf_names))
    with open(os.path.join(cfg.out_dir, "acts_meta.json"), "w") as f:
        json.dump(dict(model=cfg.model_name, n=n, config=vars(cfg),
                       decoding=(dec_record.as_dict() if dec_record else None),
                       conf_names=conf_names, items=META), f, indent=2)
    return path
