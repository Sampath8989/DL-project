"""Pre-flight: is decoding actually pinned, and does it match the scorer?

    python scripts/10_check_decoding.py --model Qwen/Qwen2.5-0.5B-Instruct \
        --device cuda --n 16

Two minutes per model, and it answers the question that cost this project
a night and five caches:

  1. What does this checkpoint ship in generation_config.json, and what
     does pinning override? Qwen2.5 ships a repetition_penalty, which
     Hugging Face applies even when do_sample=False, so "greedy" was not
     greedy and was not the same across families.
  2. Under pinned decoding, does the logprob recorded during generation
     equal the logprob recomputed by teacher forcing? These are the same
     quantity by two routes. If they disagree, the scorer is not looking
     at the distribution that produced the text.
  3. How much does pinning change the answers? If the text moves, the
     labels move, and any cache extracted before pinning has to go.

Run this before committing hours of GPU to an extraction. Exit 0 if
decoding is pinned and the two routes agree.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from src.config import ExtractConfig
from src.extract import load_model
from src.decoding import greedy_config, record as dec_record_of, describe
from src.baselines import teacher_forced_scores, eos_ids_of
from data.datasets import PROMPT_TEMPLATE, load_working_set, normalize
from src.paths import DATA


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", default="popqa")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default=None)
    p.add_argument("--n", type=int, default=16)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--compare-shipped", action="store_true", default=True,
                   help="also decode with the checkpoint's own settings, to "
                        "measure what pinning changes")
    p.add_argument("--out", default=None)
    return p.parse_args()


@torch.no_grad()
def decode(model, tok, prompt, device, gen_cfg=None, max_new_tokens=32):
    """Generate once; return (text, mean generation-time logprob, steps)."""
    enc = tok(prompt, return_tensors="pt")
    enc = enc.to(device) if hasattr(enc, "to") else enc
    kw = dict(return_dict_in_generate=True, output_scores=True)
    if gen_cfg is None:                      # the checkpoint's own settings
        kw["max_new_tokens"] = max_new_tokens
        kw["do_sample"] = False
    try:
        out = model.generate(**enc, generation_config=gen_cfg,
                             output_logits=True, **kw)
    except TypeError:
        out = model.generate(**enc, generation_config=gen_cfg, **kw)
    seq = getattr(out, "logits", None) or out.scores
    n_p = enc["input_ids"].shape[1]
    lp = []
    for t, sc in enumerate(seq):
        logp = torch.log_softmax(sc[0].float(), dim=-1)
        lp.append(float(logp[int(out.sequences[0, n_p + t])]))
    text = tok.decode(out.sequences[0, n_p:], skip_special_tokens=True)
    return text, (float(np.mean(lp)) if lp else float("nan")), len(lp)


def main():
    a = parse()
    ws = DATA / f"working_{a.dataset}.jsonl"
    if not ws.exists():
        sys.exit(f"no {ws}; run scripts/00_prepare_data.py first")
    items = load_working_set(ws)[: a.n]

    dtype = a.dtype or ("float32" if a.device == "cpu" else "float16")
    cfg = ExtractConfig(model_name=a.model, device=a.device, dtype=dtype,
                        max_new_tokens=a.max_new_tokens)
    print(f"loading {a.model} on {a.device} ({dtype}) ...")
    model, tok = load_model(cfg)

    gen_cfg = greedy_config(model, a.max_new_tokens)
    drec = dec_record_of(model, gen_cfg)
    print()
    print(describe(drec))
    print()

    eos_ids = eos_ids_of(model)
    rows = []
    for it in items:
        prompt = PROMPT_TEMPLATE.format(q=it["question"])
        text_p, lp_gen, n_gen = decode(model, tok, prompt, a.device, gen_cfg,
                                       a.max_new_tokens)
        tf = teacher_forced_scores(model, tok, prompt, text_p, a.device,
                                   n_generated=n_gen, eos_ids=eos_ids)
        row = dict(question=it["question"], pinned_text=text_p,
                   lp_generation=lp_gen, lp_teacher_forced=tf["mean_logprob_tf"])
        if a.compare_shipped:
            text_s, lp_s, _ = decode(model, tok, prompt, a.device, None,
                                  a.max_new_tokens)
            row.update(shipped_text=text_s, shipped_lp=lp_s,
                       text_changed=normalize(text_s) != normalize(text_p))
        rows.append(row)

    g = np.array([r["lp_generation"] for r in rows], float)
    t = np.array([r["lp_teacher_forced"] for r in rows], float)
    ok = np.isfinite(g) & np.isfinite(t)
    r = float(np.corrcoef(g[ok], t[ok])[0, 1]) if ok.sum() > 2 else float("nan")
    mad = float(np.mean(np.abs(g[ok] - t[ok]))) if ok.sum() else float("nan")

    print(f"two logprob routes under pinned decoding, {int(ok.sum())} items:")
    print(f"  r = {r:.6f}   mean|diff| = {mad:.6f}")
    routes_ok = np.isfinite(r) and r >= 0.999 and mad <= 0.01
    print("  " + ("AGREE -- the scorer sees the distribution that produced "
                  "the text" if routes_ok else
                  "DISAGREE -- something is still altering the distribution"))

    changed = None
    if a.compare_shipped:
        changed = sum(1 for x in rows if x.get("text_changed"))
        shifts = np.array([abs(x["shipped_lp"] - x["lp_generation"])
                           for x in rows], float)
        print()
        print(f"what pinning changes, vs the checkpoint's own settings:")
        print(f"  answers changed: {changed}/{len(rows)}")
        print(f"  mean |logprob shift|: {np.nanmean(shifts):.4f}")
        if changed:
            print("  -> any cache extracted before pinning has different "
                  "answers, so different labels. It must be re-extracted.")
            for x in rows[:3]:
                if x.get("text_changed"):
                    print(f"     q: {x['question'][:58]}")
                    print(f"       shipped: {x['shipped_text'].strip()[:58]!r}")
                    print(f"       pinned : {x['pinned_text'].strip()[:58]!r}")
        else:
            print("  -> pinning does not change this model's answers; an "
                  "existing cache may still be usable if the routes agree.")

    out = dict(model=a.model, dataset=a.dataset, n=len(rows),
               decoding=drec.as_dict(), routes_r=r, routes_mad=mad,
               routes_ok=bool(routes_ok), answers_changed=changed,
               items=rows)
    dest = Path(a.out) if a.out else Path("out") / (
        "decoding_check__" + "".join(
            c if c.isalnum() or c in "-_." else "_" for c in a.model)
        + f"__{a.dataset}.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {dest}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return 0 if routes_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
