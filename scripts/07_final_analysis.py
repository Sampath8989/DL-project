"""Stage 2: the Review 3 table. One json, every number in the deck.

    python scripts/07_final_analysis.py                 # everything cached
    python scripts/07_final_analysis.py --crossfit      # stronger, slower

Reads every out/<model>__<dataset>/ that extraction and 06_baselines.py
have filled, trains all four probes on the CPU, and scores each one
against the nested baseline ladder. Writes out/review3_results.json.

No number in the presentation is typed by hand. If a figure in the deck
disagrees with this file, the file is right.

What the headline means
    unique_gain_over_B2 > 0 and its bootstrap interval excluding 0
        reading the activations beats the published activation-free
        methods. That is the positive result.
    unique_gain_over_B2 ~ 0
        the probe is re-deriving uncertainty the model already exposes
        through its output distribution. That is the negative result, and
        at this scale it is the more likely one. It is still a result.
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
from src.evaluate import (report, artefact_probe, label_balance, auroc,
                          auroc_se, baseline_ladder, single_feature_aurocs,
                          bootstrap_auroc_delta, cv_auc, cv_oof, select_feats,
                          LADDER, LADDER_DESC)

GRID_ORDER = ["Qwen2.5-0.5B", "Qwen2.5-1.5B", "Llama-3.2-1B",
              "Qwen2.5-3B", "Llama-3.2-3B"]


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--arches", default="linear,mlp,layer_attn,traj")
    p.add_argument("--rep", default="h_answer")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--only", default=None,
                   help="substring filter on the cache directory name")
    p.add_argument("--crossfit", action="store_true",
                   help="also score the ladder on all N items using "
                        "cross-fitted probe scores instead of the 20%% test "
                        "split. Much tighter intervals, roughly 5x the CPU "
                        "time.")
    p.add_argument("--clusters", type=int, default=20)
    p.add_argument("--out", default=str(OUT / "review3_results.json"))
    return p.parse_args()


def find_caches(only=None):
    rows = []
    for d in sorted(OUT.glob("*__*")):
        if not (d / "acts.npz").exists():
            continue
        if only and only not in d.name:
            continue
        model, _, dataset = d.name.rpartition("__")
        rows.append((d, model, dataset))
    return rows


def load_features(d, z, n):
    """Surface + confidence columns from extraction, plus the baseline
    columns from 06 when they exist. Returns [n, F] and the names."""
    names = [str(c) for c in z["conf_names"]]
    F = np.asarray(z["conf"], float)[:n]
    bp = d / "baselines.npz"
    if bp.exists():
        b = np.load(bp, allow_pickle=True)
        B = np.asarray(b["base"], float)
        bn = [str(x) for x in b["base_names"]]
        if len(B) >= n:
            F = np.hstack([F, B[:n]])
            names = names + bn
        else:
            print(f"    baselines cover only {len(B)}/{n} items; ignoring "
                  f"them rather than padding")
    else:
        print("    no baselines.npz -- B2 rung unavailable, run "
              "scripts/06_baselines.py for this cache")
    return F, names


def crossfit_probe(X, y, masks, arch, seed, folds=5):
    """One probe score per item, from folds that never saw that item.

    The ladder otherwise has to be fitted on 160 test items, where a
    twelve-feature logistic regression is barely identified. Cross-fitting
    buys the full 800 at the cost of training the probe five times.
    """
    from sklearn.model_selection import StratifiedKFold
    cfg = E.probe_config(arch)
    out = np.full(len(y), np.nan)
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for tr_idx, te_idx in cv.split(np.zeros(len(y)), y):
        cut = int(0.75 * len(tr_idx))
        tr, va = tr_idx[:cut], tr_idx[cut:]
        ens = E.fit(X, y, tr, va, te_idx, cfg, masks)
        out[te_idx] = ens["prob"]
    return out


def analyse_one(d, model, dataset, a):
    z = np.load(d / "acts.npz", allow_pickle=True)
    y = np.asarray(z["y"])
    n = len(y)
    task = z["task"]
    bal = label_balance(y)
    print(f"\n=== {model}  /  {dataset}   n={n}  "
          f"{bal['pos_rate']:.1%} hallucinated")
    if bal["n_pos"] in (0, n):
        print("    single class: skipped. This is a data problem.")
        return None

    F, names = load_features(d, z, n)
    tr, va, te, rng = E.make_splits(n, a.seed)

    groups, k_used = E.cluster_groups(z["h_answer"], tr, k=a.clusters,
                                      seed=a.seed)
    ctrl = E.control_labels(y, groups, rng, seed=a.seed)

    # baselines alone, on the test split, each on its own
    singles = single_feature_aurocs(y[te], F[te], names, BASE_ORIENT)
    best_name, best_auc = None, -1.0
    for nm, v in singles.items():
        if np.isfinite(v["auroc"]) and v["auroc"] > best_auc:
            best_name, best_auc = nm, v["auroc"]
    print(f"    best single baseline: {best_name} "
          f"AUROC {best_auc:.3f}" if best_name else "    no usable baseline")

    out = dict(model=model, dataset=dataset, n=n, clusters=k_used,
               label_balance=bal, single_baselines=singles,
               best_single_baseline=best_name,
               best_single_baseline_auroc=float(best_auc),
               feature_names=names, probes={})

    # the B2 rung as a score, for the head-to-head bootstrap
    X_b2, b2_used = select_feats(F[te], names, LADDER[-1][1])
    b2_oof = cv_oof(X_b2, y[te], seed=a.seed) if X_b2 is not None else None
    out["auroc_B2_oof"] = (auroc(y[te], b2_oof) if b2_oof is not None
                           else float("nan"))

    for arch in a.arches.split(","):
        t0 = time.time()
        X, layer, masks = E.select_rep(z, arch, a.rep, -1, y, tr, va)
        cfg = E.probe_config(arch)
        ens = E.fit(X, y, tr, va, te, cfg, masks)
        p = ens["prob"]

        r = report(y[te], p, spread=ens["spread"])
        r["auroc_se"] = auroc_se(y[te], p)
        r["layer"] = layer
        r["n_params"] = ens["n_params"]

        for cname, cy in ctrl.items():
            ce = E.fit(X, cy, tr, va, te, cfg, masks)
            ca = E.safe_auc(cy[te], ce["prob"])
            r[f"auroc_control_{cname}"] = ca
            r[f"selectivity_{cname}"] = (r["auroc"] - ca
                                         if np.isfinite(ca) else float("nan"))

        r.update(baseline_ladder(y[te], p, F[te], names))

        if b2_oof is not None:
            r["vs_B2"] = bootstrap_auroc_delta(y[te], p, b2_oof,
                                               seed=a.seed)
        if best_name is not None:
            j = names.index(best_name)
            sgn = BASE_ORIENT.get(best_name, +1)
            r["vs_best_single"] = bootstrap_auroc_delta(
                y[te], p, sgn * F[te, j], seed=a.seed)

        if a.crossfit:
            pc = crossfit_probe(X, y, masks, arch, a.seed)
            ok = np.isfinite(pc)
            r["crossfit"] = dict(n=int(ok.sum()),
                                 auroc=auroc(y[ok], pc[ok]),
                                 auroc_se=auroc_se(y[ok], pc[ok]))
            r["crossfit"].update(baseline_ladder(y[ok], pc[ok], F[ok], names))
            Xb, _ = select_feats(F[ok], names, LADDER[-1][1])
            if Xb is not None:
                bo = cv_oof(Xb, y[ok], seed=a.seed)
                r["crossfit"]["auroc_B2_oof"] = auroc(y[ok], bo)
                r["crossfit"]["vs_B2"] = bootstrap_auroc_delta(
                    y[ok], pc[ok], bo, seed=a.seed)

        r["artefact_task_acc"] = artefact_probe(
            z["h_answer"][te].reshape(len(te), -1), task[te])
        r["seconds"] = round(time.time() - t0, 1)
        out["probes"][arch] = r

        g2 = r.get("unique_gain_over_B2", float("nan"))
        g0 = r.get("unique_gain_over_B0", float("nan"))
        print(f"    {arch:11s} AUROC {r['auroc']:.3f}+-{r['auroc_se']:.3f}  "
              f"gain/B0 {g0:+.3f}  gain/B2 {g2:+.3f}  "
              f"sel(group) {r.get('selectivity_group', float('nan')):+.3f}  "
              f"[{r['seconds']}s]")
    return out


def main():
    a = parse()
    caches = find_caches(a.only)
    if not caches:
        sys.exit(f"no caches with acts.npz under {OUT}. Run extraction "
                 f"(scripts/03_run_experiment.py) first.")
    print(f"{len(caches)} cache(s): " +
          ", ".join(d.name for d, _, _ in caches))

    results = []
    for d, model, dataset in caches:
        try:
            r = analyse_one(d, model, dataset, a)
        except Exception as e:                       # one bad cache must not
            import traceback                         # lose the other nineteen
            print(f"    FAILED: {type(e).__name__}: {e}")
            traceback.print_exc(limit=3)
            r = dict(model=model, dataset=dataset, error=f"{type(e).__name__}: {e}")
        if r:
            results.append(r)
            Path(a.out).write_text(json.dumps(
                dict(created=time.strftime("%Y-%m-%d %H:%M:%S"),
                     seed=a.seed, crossfit=bool(a.crossfit),
                     ladder={n: dict(describes=LADDER_DESC[n], features=f)
                              for n, f in LADDER},
                     results=results), indent=2))

    print(f"\nwrote {a.out}  ({len(results)} model-dataset pairs)")


if __name__ == "__main__":
    main()
