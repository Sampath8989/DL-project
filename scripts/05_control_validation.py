"""Does the random-label control actually work?

A reviewer challenged the +0.242 selectivity we reported in the
confound-only condition: if the control reads positive where there is
nothing to find, is the control broken?

This script answers it by running the control in four conditions whose
correct answers we know in advance. A control that is working must read
near zero in the conditions where nothing is learnable, and must collapse
when a probe is merely memorising.

    truth        a real truth direction is planted in the hidden states
    confound     only a length direction; length predicts the label
    noise        the hidden states carry nothing at all
    memorisable  the states carry a strong group fingerprint, and labels
                 are a deterministic function of the group, so a probe can
                 memorise the mapping and still generalise to held-out items

Two control-label schemes are compared:

    permuted     labels shuffled across items. The variant used in
                 scripts/02. Cheap, but unlearnable by construction, so it
                 can only ever detect within-split memorisation.
    group        a random label assigned per group, in the manner of
                 Hewitt and Liang's control tasks (EMNLP 2019). Groups
                 recur across train and test, so a probe that has latched
                 onto the group fingerprint can carry the random mapping
                 to held-out data - which is exactly what selectivity is
                 meant to expose.

Expected, if the control is sound:
    truth        selectivity clearly positive, unique gain positive
    confound     selectivity positive, unique gain ~0   <- the reported finding
    noise        selectivity ~0, unique gain ~0          <- reads zero on nothing
    memorisable  selectivity ~0 despite a high raw score <- catches memorising
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src.config import ProbeConfig
from src.train import fit_ensemble, _standardize
from src.evaluate import report

N, L, D, T = 1400, 13, 64, 16
N_GROUPS = 40
CONDITIONS = ("truth", "confound", "noise", "memorisable")


def layer_weight(l, peak, width):
    return float(np.exp(-0.5 * ((l - peak) / width) ** 2))


def make(condition, rng):
    group = rng.integers(0, N_GROUPS, size=N)
    task = rng.integers(0, 3, size=N)

    if condition == "memorisable":
        # the label is a deterministic property of the group, so the
        # mapping is learnable and transfers to held-out items
        group_label = rng.integers(0, 2, size=N_GROUPS)
        y = group_label[group].astype(np.int64)
    else:
        p_h = np.array([0.20, 0.45, 0.70])[task]
        y = (rng.random(N) < p_h).astype(np.int64)

    length = 8.0 + 2.5 * y + rng.normal(0, 2.0, N)
    mean_lp = -0.40 - 0.35 * y + rng.normal(0, 0.55, N)
    min_lp = mean_lp - np.abs(rng.normal(0.6, 0.35, N))
    conf = np.stack([length, mean_lp, min_lp], 1)

    u_truth = rng.normal(0, 1, D); u_truth /= np.linalg.norm(u_truth)
    u_len = rng.normal(0, 1, D); u_len /= np.linalg.norm(u_len)
    u_group = rng.normal(0, 1, (N_GROUPS, D))
    u_group /= np.linalg.norm(u_group, axis=1, keepdims=True)

    z_len = (length - length.mean()) / length.std()
    s_truth = (2 * y - 1).astype(float)
    peak = int(0.62 * L)

    H = rng.normal(0, 1.0, (N, L, D)).astype(np.float32)
    for l in range(L):
        if condition in ("truth", "confound"):
            H[:, l, :] += (0.55 * layer_weight(l, L - 2, 4.0) * z_len)[:, None] * u_len
        if condition == "truth":
            w = 0.95 * layer_weight(l, peak, 1.6)
            H[:, l, :] += (w * s_truth)[:, None] * u_truth
        if condition == "memorisable":
            H[:, l, :] += 1.6 * u_group[group]

    traj = rng.normal(0, 1.0, (N, T, D)).astype(np.float32)
    mask = np.ones((N, T), np.float32)
    n_tok = np.clip(np.round(length).astype(int), 3, T)
    for i in range(N):
        mask[i, n_tok[i]:] = 0.0
    ramp = np.clip((np.arange(T) - 3) / max(T - 4, 1), 0, 1)
    for t in range(T):
        if condition in ("truth", "confound"):
            traj[:, t, :] += (0.55 * z_len)[:, None] * u_len
        if condition == "truth":
            traj[:, t, :] += (0.95 * ramp[t] * s_truth)[:, None] * u_truth
        if condition == "memorisable":
            traj[:, t, :] += 1.6 * u_group[group]
    traj *= mask[:, :, None]

    return dict(H=H, traj=traj, mask=mask, conf=conf.astype(np.float32),
                y=y, group=group, task=task, peak_layer=peak)


def split(n, rng):
    idx = rng.permutation(n)
    a, b = int(0.6 * n), int(0.8 * n)
    return idx[:a], idx[a:b], idx[b:]


def best_layer(H, y, tr, va):
    best, best_a = 0, -1.0
    for l in range(H.shape[1]):
        m = LogisticRegression(max_iter=1000).fit(H[tr, l], y[tr])
        a = roc_auc_score(y[va], m.predict_proba(H[va, l])[:, 1])
        if a > best_a:
            best, best_a = l, a
    return best


def fit(X, labels, tr, va, te, cfg, masks):
    Xtr, Xva, Xte = _standardize(X[tr], X[va], X[te])
    mtr, mva, mte = masks
    ens = fit_ensemble(Xtr, labels[tr], Xva, labels[va], Xte, cfg,
                       mtr, mva, mte, device="cpu")
    return ens


def run_arch(arch, data, tr, va, te, rng):
    cfg = ProbeConfig(arch=arch, epochs=40, patience=8, seeds=[0, 1, 2, 3, 4])
    H, y, group = data["H"], data["y"], data["group"]

    masks = (None, None, None)
    if arch in ("linear", "mlp"):
        X = H[:, best_layer(H, y, tr, va), :]
    elif arch == "layer_attn":
        X = H
    else:
        X = data["traj"]
        masks = (data["mask"][tr], data["mask"][va], data["mask"][te])

    ens = fit(X, y, tr, va, te, cfg, masks)

    # control 1: labels permuted across items (unlearnable by construction)
    y_perm = y[rng.permutation(len(y))]
    auc_perm = _safe_auc(y_perm[te], fit(X, y_perm, tr, va, te, cfg, masks)["prob"])

    # control 2: a random label per group, learnable and generalisable
    g_label = rng.integers(0, 2, size=N_GROUPS)
    y_group = g_label[group].astype(np.int64)
    auc_group = _safe_auc(y_group[te], fit(X, y_group, tr, va, te, cfg, masks)["prob"])

    r = report(y[te], ens["prob"], spread=ens["spread"],
               confounds=data["conf"][te], auc_random=auc_group)
    r.update(arch=arch, n_params=ens["n_params"],
             auroc_control_group=auc_group, auroc_control_permuted=auc_perm,
             selectivity_group=r["auroc"] - auc_group,
             selectivity_permuted=r["auroc"] - auc_perm)
    return r


def _safe_auc(y, p):
    return float(roc_auc_score(y, p)) if len(set(y.tolist())) > 1 else float("nan")


def main():
    t0 = time.time()
    out = {}
    for cond in CONDITIONS:
        data = make(cond, np.random.default_rng(11))
        rng = np.random.default_rng(7)
        tr, va, te = split(N, rng)
        rows = []
        print(f"\n=== {cond} ===  positive rate {data['y'].mean():.3f}")
        for arch in ("linear", "mlp", "layer_attn", "traj"):
            r = run_arch(arch, data, tr, va, te, rng)
            rows.append(r)
            print(f"  {arch:11s} AUROC {r['auroc']:.3f} | control(group) "
                  f"{r['auroc_control_group']:.3f} | selectivity "
                  f"{r['selectivity_group']:+.3f} | gain {r['unique_gain']:+.3f}")
        out[cond] = dict(rows=rows, pos_rate=float(data["y"].mean()))
    out["runtime_s"] = round(time.time() - t0, 1)
    out["n_groups"] = N_GROUPS
    with open("out/control_validation.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote out/control_validation.json in {out['runtime_s']} s")


if __name__ == "__main__":
    main()
