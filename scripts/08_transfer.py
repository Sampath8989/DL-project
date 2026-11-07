"""Does the probe learn hallucination, or does it learn this dataset?

    python scripts/08_transfer.py

For each model, trains the probe on one dataset's activations and tests
it on the other's, in both directions. The language model is the same in
both halves, so the hidden size matches and the probe transfers without
any adaptation.

Why this is the deciding experiment
    A probe that has found an internal "this is made up" signal should
    carry across question sources: the signal is a property of the model,
    not of PopQA. A probe that has instead fitted the shape of PopQA --
    its entity distribution, its question template, its answer lengths --
    collapses to chance the moment the source changes.

    In-domain AUROC therefore cannot settle the project's question on its
    own, and the transfer number is reported beside it everywhere.

Fitted on the source only: the layer, the standardisation statistics and
the calibration temperature. Nothing is tuned on the target.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.paths import OUT
from src import experiment as E
from src.base_features import ALL_ORIENT as BASE_ORIENT
from src.train import fit_one, fit_temperature, predict, _logits
from src.evaluate import (auroc, auroc_se, auprc, tpr_at_fpr, label_balance,
                          baseline_ladder, bootstrap_auroc_delta, cv_oof,
                          select_feats, LADDER)



def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--arches", default="linear,mlp,layer_attn")
    p.add_argument("--rep", default="h_answer")
    p.add_argument("--pairs", default="popqa>triviaqa,triviaqa>popqa")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--only", default=None)
    p.add_argument("--out", default=str(OUT / "review3_transfer.json"))
    return p.parse_args()


def models_with(datasets):
    """Models that have a cache for every dataset named."""
    have = {}
    for d in sorted(OUT.glob("*__*")):
        if not (d / "acts.npz").exists():
            continue
        model, _, ds = d.name.rpartition("__")
        have.setdefault(model, {})[ds] = d
    return {m: v for m, v in have.items() if all(ds in v for ds in datasets)}


def rep_for(z, arch, rep, layer):
    """Same representation choice as in-domain, with the layer forced."""
    if arch == "traj":
        layer = min(layer, z["traj"].shape[2] - 1)
        return z["traj"][:, :, layer, :], z["mask"]
    if arch == "layer_attn":
        return z[rep], None
    return z[rep][:, layer, :], None


def features(d, z, n):
    names = [str(c) for c in z["conf_names"]]
    F = np.asarray(z["conf"], float)[:n]
    bp = d / "baselines.npz"
    if bp.exists():
        b = np.load(bp, allow_pickle=True)
        B = np.asarray(b["base"], float)
        if len(B) >= n:
            F = np.hstack([F, B[:n]])
            names = names + [str(x) for x in b["base_names"]]
    return F, names


def one_direction(model, src_dir, tgt_dir, src, tgt, a):
    zs = np.load(src_dir / "acts.npz", allow_pickle=True)
    zt = np.load(tgt_dir / "acts.npz", allow_pickle=True)
    ys, yt = np.asarray(zs["y"]), np.asarray(zt["y"])
    if zs["h_answer"].shape[1:] != zt["h_answer"].shape[1:]:
        return dict(error=f"shape mismatch {zs['h_answer'].shape[1:]} vs "
                           f"{zt['h_answer'].shape[1:]}")
    bs, bt = label_balance(ys), label_balance(yt)
    if bs["n_pos"] in (0, len(ys)) or bt["n_pos"] in (0, len(yt)):
        return dict(error="single-class split")

    tr, va, te, _ = E.make_splits(len(ys), a.seed)
    out = dict(model=model, source=src, target=tgt,
               source_balance=bs, target_balance=bt, probes={})
    Ft, names_t = features(tgt_dir, zt, len(yt))

    Xb, _ = select_feats(Ft, names_t, LADDER[-1][1])
    b2 = cv_oof(Xb, yt, seed=a.seed) if Xb is not None else None
    out["target_auroc_B2_oof"] = (auroc(yt, b2) if b2 is not None
                                  else float("nan"))

    for arch in a.arches.split(","):
        t0 = time.time()
        # layer chosen on the SOURCE validation split only
        Xs_all, layer, masks = E.select_rep(zs, arch, a.rep, -1, ys, tr, va)
        Xt_all, mask_t = rep_for(zt, arch, a.rep, layer)

        # standardisation from the source training split only
        axes = tuple(range(Xs_all.ndim - 1))
        mu = Xs_all[tr].mean(axis=axes, keepdims=True)
        sd = Xs_all[tr].std(axis=axes, keepdims=True) + 1e-6
        Xtr, Xva = (Xs_all[tr] - mu) / sd, (Xs_all[va] - mu) / sd
        Xte_src = (Xs_all[te] - mu) / sd
        Xtgt = (Xt_all - mu) / sd

        cfg = E.probe_config(arch)
        ms = masks[0]
        probs_src, probs_tgt = [], []
        for s in cfg.seeds:
            m, _ = fit_one(Xtr, ys[tr], Xva, ys[va], cfg, seed=s,
                           mask_tr=None if ms is None else ms[tr],
                           mask_va=None if ms is None else ms[va],
                           device="cpu")
            t = 1.0
            if cfg.calibrate:
                t = fit_temperature(_logits(
                    m, Xva, None if ms is None else ms[va], "cpu"), ys[va])
            probs_src.append(predict(m, Xte_src, t,
                                     None if ms is None else ms[te], "cpu"))
            probs_tgt.append(predict(m, Xtgt, t, mask_t, "cpu"))
        ps = np.stack(probs_src).mean(0)
        pt = np.stack(probs_tgt).mean(0)

        r = dict(layer=int(layer),
                 in_domain_auroc=auroc(ys[te], ps),
                 in_domain_auroc_se=auroc_se(ys[te], ps),
                 transfer_auroc=auroc(yt, pt),
                 transfer_auroc_se=auroc_se(yt, pt),
                 transfer_auprc=auprc(yt, pt),
                 transfer_tpr_at_fpr10=tpr_at_fpr(yt, pt, 0.10),
                 n_target=int(len(yt)))
        r["transfer_drop"] = r["in_domain_auroc"] - r["transfer_auroc"]
        r.update({f"target_{k}": v for k, v in
                  baseline_ladder(yt, pt, Ft, names_t).items()})
        if b2 is not None:
            r["transfer_vs_target_B2"] = bootstrap_auroc_delta(
                yt, pt, b2, seed=a.seed)
        r["seconds"] = round(time.time() - t0, 1)
        out["probes"][arch] = r
        print(f"    {arch:11s} in-domain {r['in_domain_auroc']:.3f} -> "
              f"transfer {r['transfer_auroc']:.3f} "
              f"(drop {r['transfer_drop']:+.3f})  "
              f"gain over target B2 "
              f"{r.get('target_unique_gain_over_B2', float('nan')):+.3f}"
              f"  [{r['seconds']}s]")
    return out


def main():
    a = parse()
    pairs = [tuple(p.split(">")) for p in a.pairs.split(",")]
    needed = sorted({d for p in pairs for d in p})
    have = models_with(needed)
    if a.only:
        have = {m: v for m, v in have.items() if a.only in m}
    if not have:
        sys.exit(f"no model has a cache for all of {needed}. "
                 f"Extract the second dataset first.")
    print(f"{len(have)} model(s) with both datasets: {', '.join(have)}")

    rows = []
    for model, dirs in have.items():
        for src, tgt in pairs:
            print(f"\n=== {model}   {src} -> {tgt}")
            try:
                r = one_direction(model, dirs[src], dirs[tgt], src, tgt, a)
            except Exception as e:
                import traceback
                print(f"    FAILED: {type(e).__name__}: {e}")
                traceback.print_exc(limit=3)
                r = dict(model=model, source=src, target=tgt,
                         error=f"{type(e).__name__}: {e}")
            rows.append(r)
            Path(a.out).write_text(json.dumps(dict(
                created=time.strftime("%Y-%m-%d %H:%M:%S"),
                seed=a.seed, results=rows), indent=2))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
