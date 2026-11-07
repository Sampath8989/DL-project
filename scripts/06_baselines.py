"""Stage 1b: compute the published uncertainty baselines for one cache.

    python scripts/06_baselines.py --model Qwen/Qwen2.5-3B-Instruct \
        --dataset popqa --device cuda

Requires that scripts/03_run_experiment.py has already written
out/<model>__<dataset>/acts_meta.json, because the baselines are computed
for exactly the greedy answers that are already cached and labelled. Run
03 first, then this.

Cost per item: two single forward passes plus one batched sample of k
continuations. No hidden states are captured here, so this pass is far
cheaper in VRAM than extraction and can use a larger k than it could if
it were folded into extraction.

The run is resumable. Every item is appended to baselines_progress.jsonl
as it completes, so an overnight run that dies at item 612 restarts at
item 612 instead of at zero.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from src.paths import OUT, ensure_dirs
from src.config import ExtractConfig
from src.extract import load_model
from src.baselines import (BASE_NAMES, item_baselines, to_matrix, yes_no_ids,
                           eos_ids_of)
from src.decoding import sampling_config, record as dec_record_of, describe
from data.datasets import PROMPT_TEMPLATE, normalize


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", default="popqa")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default=None)
    p.add_argument("--k", type=int, default=5,
                   help="samples per item for semantic entropy")
    p.add_argument("--temperature", type=float, default=None,
                   help="default comes from src/decoding.py; 0.7 used to be "
                        "the default only because Qwen's shipped config "
                        "contributed it by accident")
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--max-new-tokens", type=int, default=24)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    p.add_argument("--limit", type=int, default=None,
                   help="debug: stop after this many items")
    p.add_argument("--force", action="store_true",
                   help="ignore an existing baselines.npz and start over")
    p.add_argument("--allow-inconsistent", action="store_true",
                   help="write the baselines even when the two logprob "
                        "routes disagree. Only for inspecting a known-bad "
                        "cache; never for a result that will be reported.")
    p.add_argument("--check-first", type=int, default=16,
                   help="score this many items, then verify the two logprob "
                        "routes agree before committing to the rest. 0 "
                        "disables the early check.")
    p.add_argument("--tiny", action="store_true")
    return p.parse_args()


def slug(s):
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)


R_MIN = 0.999          # the two routes compute the same quantity
MAD_MAX = 0.01         # ... so they must agree to about two decimals


def logprob_consistency(out_dir, idx_list, rows, names):
    """Compare the teacher-forced logprob against the one extraction
    recorded, for the items scored so far.

    These are the same number by two routes: the log-probability the model
    assigned to each token of its own greedy answer. They can only
    disagree if the scorer is not looking at the distribution that
    produced the text -- which is exactly what happened when extraction
    inherited Qwen's repetition_penalty and the scorer did not. Returns
    (r, mean_abs_diff, n) or None when there is nothing to compare.
    """
    acts = out_dir / "acts.npz"
    if not acts.exists() or not rows:
        return None
    z = np.load(acts, allow_pickle=True)
    cn = [str(x) for x in z["conf_names"]]
    if "mean_logprob" not in cn:
        return None
    gen_col = z["conf"][:, cn.index("mean_logprob")]
    j = names.index("mean_logprob_tf")
    gen, tf = [], []
    for i, r in zip(idx_list, rows):
        v = r.get("mean_logprob_tf", float("nan"))
        if i < len(gen_col) and np.isfinite(v) and np.isfinite(gen_col[i]):
            gen.append(float(gen_col[i]))
            tf.append(float(v))
    if len(gen) < 8:
        return None
    gen, tf = np.array(gen), np.array(tf)
    r = float(np.corrcoef(gen, tf)[0, 1]) if np.std(gen) > 0 else float("nan")
    return r, float(np.mean(np.abs(gen - tf))), len(gen)


def consistency_verdict(res):
    if res is None:
        return True, "no cached logprob column to compare against"
    r, mad, n = res
    ok = (np.isfinite(r) and r >= R_MIN and mad <= MAD_MAX)
    msg = (f"teacher-forced vs recorded logprob over {n} items: "
           f"r={r:.4f} mean|diff|={mad:.4f} "
           f"(need r>={R_MIN} and mean|diff|<={MAD_MAX})")
    return ok, msg


def explain_inconsistency():
    return (
        "\nThese two routes compute the same quantity and must agree.\n"
        "They disagree when the cache was decoded under settings the\n"
        "scorer is not reproducing. The known cause in this project is a\n"
        "checkpoint's own generation_config.json: Qwen2.5 ships a\n"
        "repetition_penalty, which Hugging Face applies even with\n"
        "do_sample=False, so a cache extracted before src/decoding.py\n"
        "existed was decoded under a penalty that no longer applies.\n"
        "\nThe fix is to re-extract that cache with decoding pinned, not\n"
        "to reapply the penalty here -- that would make this check pass\n"
        "and leave the cross-family confound in place. Check\n"
        "acts_meta.json: if it has no 'decoding' block it predates the\n"
        "fix and must be re-extracted.\n")


def main():
    a = parse()
    ensure_dirs()
    model_name = "tiny-random-llama" if a.tiny else a.model
    out_dir = Path(a.out) if a.out else OUT / f"{slug(model_name)}__{a.dataset}"
    meta_path = out_dir / "acts_meta.json"
    if not meta_path.exists():
        sys.exit(f"no cache at {meta_path}\n"
                 f"run scripts/03_run_experiment.py for this model and "
                 f"dataset first -- the baselines are scored on the greedy "
                 f"answers it cached, not on fresh generations.")

    meta = json.loads(meta_path.read_text())
    items = meta["items"]
    if a.limit:
        items = items[: a.limit]
    dest = out_dir / "baselines.npz"
    if dest.exists() and not a.force:
        z = np.load(dest, allow_pickle=True)
        if len(z["base"]) == len(items):
            print(f"baselines already complete for {len(items)} items: {dest}")
            print("pass --force to recompute")
            return
        print(f"existing baselines cover {len(z['base'])}/{len(items)} items; "
              f"continuing")

    prog_path = out_dir / "baselines_progress.jsonl"
    done = {}
    if prog_path.exists() and not a.force:
        for line in prog_path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
                done[int(r["idx"])] = r
            except (json.JSONDecodeError, KeyError):
                continue
        print(f"resuming: {len(done)} of {len(items)} items already scored")
    elif a.force and prog_path.exists():
        prog_path.unlink()

    todo = [it for it in items if int(it["idx"]) not in done]
    if todo:
        dtype = a.dtype or ("float32" if a.device == "cpu" else "float16")
        if a.tiny:
            from src.tiny import build_tiny
            model, tok = build_tiny()
            device = "cpu"
        else:
            cfg = ExtractConfig(model_name=a.model, device=a.device,
                                dtype=dtype)
            print(f"loading {a.model} on {a.device} ({dtype}) ...")
            model, tok = load_model(cfg)
            device = a.device
        try:
            yn = yes_no_ids(tok)
        except ValueError as e:
            print(f"  p(true) unavailable: {e}")
            yn = None

        gen_cfg = sampling_config(model, a.max_new_tokens, a.k,
                                  temperature=a.temperature, top_p=a.top_p)
        drec = dec_record_of(model, gen_cfg)
        print(describe(drec))

        # Steps each greedy answer took during extraction, so the scorer can
        # include the EOS step exactly when generation did.
        n_gen = {}
        za = np.load(out_dir / "acts.npz", allow_pickle=True)
        zcn = [str(x) for x in za["conf_names"]]
        if "n_new_tokens" in zcn:
            col = za["conf"][:, zcn.index("n_new_tokens")]
            n_gen = {int(m["idx"]): int(col[i])
                     for i, m in enumerate(meta["items"])}
        try:
            eos_ids = eos_ids_of(model)
        except Exception:
            eos_ids = []

        t0 = time.time()
        checked = not a.check_first
        with open(prog_path, "a", encoding="utf-8") as log:
            for c, it in enumerate(todo):
                idx = int(it["idx"])
                q = it.get("question", "")
                prompt = PROMPT_TEMPLATE.format(q=q)
                greedy = it.get("text", "")
                feats, samples, n_scored = item_baselines(
                    model, tok, q, prompt, greedy, normalize, device,
                    k=a.k, temperature=a.temperature,
                    max_new_tokens=a.max_new_tokens, yn_ids=yn,
                    seed=a.seed + idx, gen_cfg=gen_cfg,
                    n_generated=n_gen.get(idx), eos_ids=eos_ids)
                rec = dict(idx=idx, feats=feats, samples=samples,
                           n_scored=int(n_scored))
                log.write(json.dumps(rec, ensure_ascii=False) + "\n")
                log.flush()
                done[idx] = rec
                # Early consistency gate. Checking after 16 items costs
                # under a minute; checking only at the end cost 48 minutes
                # of GPU on a cache that could not be used.
                if not checked and (c + 1) >= a.check_first:
                    checked = True
                    ids = [int(it2["idx"]) for it2 in todo[: c + 1]]
                    rws = [done[i]["feats"] for i in ids]
                    res = logprob_consistency(out_dir, ids, rws, BASE_NAMES)
                    ok, msg = consistency_verdict(res)
                    print(f"  early check -- {msg}")
                    if not ok and not a.allow_inconsistent:
                        print(explain_inconsistency())
                        print(f"  stopping after {c+1} items rather than "
                              f"spending the GPU on a cache that cannot be "
                              f"reported. Pass --allow-inconsistent only to "
                              f"inspect it.")
                        return 2
                    if not ok:
                        print("  continuing because --allow-inconsistent "
                              "was passed. These numbers must not be "
                              "reported.")

                if (c + 1) % 25 == 0 or c + 1 == len(todo):
                    el = time.time() - t0
                    rate = (c + 1) / max(el, 1e-6)
                    left = (len(todo) - c - 1) / max(rate, 1e-9)
                    print(f"  {c+1}/{len(todo)} items  "
                          f"{rate*60:.1f}/min  eta {left/60:.1f} min")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    rows = [done[int(it["idx"])]["feats"] for it in items]
    B, names = to_matrix(rows, BASE_NAMES)

    # Final gate on the full set. Same comparison as the early check; a
    # cache that fails here is not written as a usable result.
    idx_all = [int(it["idx"]) for it in items]
    res = logprob_consistency(out_dir, idx_all, rows, names)
    ok, note = consistency_verdict(res)
    print(f"  {note}")
    if not ok:
        print(explain_inconsistency())
        if not a.allow_inconsistent:
            bad = out_dir / "baselines_FAILED_CONSISTENCY.json"
            bad.write_text(json.dumps(dict(
                model=model_name, dataset=a.dataset, n=len(items),
                check=note, reason="logprob routes disagree; cache decoding "
                                   "does not match the scorer"), indent=2))
            print(f"  wrote {bad} and no baselines.npz. Re-extract this "
                  f"cache with decoding pinned, then run this again.")
            return 2
        print("  --allow-inconsistent: writing anyway. Do not report these.")

    cov = {n: int(np.isfinite(B[:, j]).sum()) for j, n in enumerate(names)}
    np.savez_compressed(dest, base=B, base_names=np.array(names),
                        idx=np.array([int(it["idx"]) for it in items]))
    (out_dir / "baselines_meta.json").write_text(json.dumps(dict(
        model=model_name, dataset=a.dataset, n=len(items), k=a.k,
        temperature=a.temperature, top_p=a.top_p,
        max_new_tokens=a.max_new_tokens, seed=a.seed,
        base_names=names, coverage=cov, consistency=note,
        consistency_ok=bool(ok),
        decoding=(drec.as_dict() if todo else "unchanged (resumed run)"),
        samples={str(it["idx"]): done[int(it["idx"])]["samples"]
                 for it in items[:25]}), indent=2))
    print(f"\ncoverage (finite values per feature, n={len(items)}):")
    for n in names:
        print(f"  {n:20s} {cov[n]}")
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
