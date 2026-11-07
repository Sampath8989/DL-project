"""Stage-2 validation: does the evaluation harness tell genuine internal
signal apart from a surface confound?

Two synthetic conditions are generated with identical labels, identical
surface statistics and identical noise, differing in exactly one respect:

  truth      hidden states carry a truth direction in the middle layers
             *in addition to* a length direction
  confound   hidden states carry only the length direction

A probe should score well in both, because length is predictive of the
label in both. Only the deconfounded metric should separate them. If the
harness reports a large unique gain under `confound`, the harness is
broken and every downstream number would be untrustworthy.

This is a instrument-calibration experiment on synthetic data. It makes
no claim about any real language model.
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
from src.evaluate import report, artefact_probe

N, L, D, T = 1400, 13, 64, 16
RNG = np.random.default_rng(20260928)


def layer_weight(l, peak, width):
    return float(np.exp(-0.5 * ((l - peak) / width) ** 2))


def make_synthetic(condition, rng):
    """Return activations, trajectory, confounds and labels."""
    task = rng.integers(0, 3, size=N)                   # 0 easy 1 medium 2 hard
    p_h = np.array([0.20, 0.45, 0.70])[task]
    y = (rng.random(N) < p_h).astype(np.int64)

    # surface features, correlated with the label in BOTH conditions
    length = 8.0 + 2.5 * y + rng.normal(0, 2.0, N)
    mean_lp = -0.40 - 0.35 * y + rng.normal(0, 0.55, N)
    min_lp = mean_lp - np.abs(rng.normal(0.6, 0.35, N))
    conf = np.stack([length, mean_lp, min_lp], 1)

    u_truth = rng.normal(0, 1, D); u_truth /= np.linalg.norm(u_truth)
    u_len = rng.normal(0, 1, D);   u_len /= np.linalg.norm(u_len)
    u_task = rng.normal(0, 1, (3, D)); u_task /= np.linalg.norm(u_task, axis=1, keepdims=True)

    z_len = (length - length.mean()) / length.std()
    s_truth = (2 * y - 1).astype(float)

    H = rng.normal(0, 1.0, (N, L, D)).astype(np.float32)
    for l in range(L):
        w_len = 0.55 * layer_weight(l, peak=L - 2, width=4.0)
        w_tru = 0.95 * layer_weight(l, peak=int(0.62 * L), width=1.6)
        H[:, l, :] += (w_len * z_len)[:, None] * u_len[None, :]
        H[:, l, :] += 0.45 * u_task[task]                     # dataset identity
        if condition == "truth":
            H[:, l, :] += (w_tru * s_truth)[:, None] * u_truth[None, :]

    # token trajectory at the peak truth layer, signal ramping in over time
    peak = int(0.62 * L)
    traj = rng.normal(0, 1.0, (N, T, D)).astype(np.float32)
    mask = np.ones((N, T), np.float32)
    n_tok = np.clip(np.round(length).astype(int), 3, T)
    for i in range(N):
        mask[i, n_tok[i]:] = 0.0
    ramp = np.clip((np.arange(T) - 3) / max(T - 4, 1), 0, 1)
    for t in range(T):
        traj[:, t, :] += (0.55 * z_len)[:, None] * u_len[None, :]
        if condition == "truth":
            traj[:, t, :] += (0.95 * ramp[t] * s_truth)[:, None] * u_truth[None, :]
    traj *= mask[:, :, None]

    return dict(H=H, traj=traj, mask=mask, conf=conf.astype(np.float32),
                y=y, task=task, peak_layer=peak)


def split(n, rng):
    idx = rng.permutation(n)
    a, b = int(0.6 * n), int(0.8 * n)
    return idx[:a], idx[a:b], idx[b:]


def best_layer(H, y, tr, va):
    """Cheap validation-set layer selection for the fixed-layer probes."""
    best, best_a = 0, -1.0
    for l in range(H.shape[1]):
        m = LogisticRegression(max_iter=1000).fit(H[tr, l], y[tr])
        a = roc_auc_score(y[va], m.predict_proba(H[va, l])[:, 1])
        if a > best_a:
            best, best_a = l, a
    return best, best_a


def run_arch(arch, data, tr, va, te, rng):
    cfg = ProbeConfig(arch=arch, epochs=40, patience=8, seeds=[0, 1, 2, 3, 4])
    H, y = data["H"], data["y"]
    mask_tr = mask_va = mask_te = None
    chosen = None

    if arch in ("linear", "mlp"):
        chosen, _ = best_layer(H, y, tr, va)
        X = H[:, chosen, :]
    elif arch == "layer_attn":
        X = H
    else:
        X = data["traj"]
        mask_tr, mask_va, mask_te = data["mask"][tr], data["mask"][va], data["mask"][te]

    Xtr, Xva, Xte = _standardize(X[tr], X[va], X[te])
    ens = fit_ensemble(Xtr, y[tr], Xva, y[va], Xte, cfg,
                       mask_tr, mask_va, mask_te, device="cpu")

    # random-label control: same probe, labels permuted
    perm = rng.permutation(len(y))
    yr = y[perm]
    ens_r = fit_ensemble(Xtr, yr[tr], Xva, yr[va], Xte, cfg,
                         mask_tr, mask_va, mask_te, device="cpu")
    auc_random = roc_auc_score(yr[te], ens_r["prob"])

    r = report(y[te], ens["prob"], spread=ens["spread"],
               confounds=data["conf"][te], auc_random=auc_random)
    r.update(arch=arch, n_params=ens["n_params"], layer=chosen,
             temps=[round(t, 3) for t in ens["temps"]])
    return r, ens


def main():
    out = {}
    t0 = time.time()
    for condition in ("truth", "confound"):
        rng = np.random.default_rng(7)
        data = make_synthetic(condition, np.random.default_rng(11))
        tr, va, te = split(N, rng)
        rows = []
        for arch in ("linear", "mlp", "layer_attn", "traj"):
            r, ens = run_arch(arch, data, tr, va, te, rng)
            rows.append(r)
            print(f"[{condition:8s}] {arch:11s} "
                  f"AUROC {r['auroc']:.3f}  AUPRC {r['auprc']:.3f}  "
                  f"ECE {r['ece']:.3f}  sel {r['selectivity']:+.3f}  "
                  f"gain {r['unique_gain']:+.3f}  params {r['n_params']}")
            if arch == "layer_attn":
                m = ens["members"][0]
                out[f"{condition}_layer_attention"] = (
                    m.last_attn.mean(0).cpu().numpy().round(4).tolist())
        art = artefact_probe(data["H"][te][:, data["peak_layer"], :], data["task"][te])
        print(f"[{condition:8s}] artefact probe (task identity) acc = {art:.3f}")
        out[condition] = dict(rows=rows, artefact_task_acc=art,
                              confound_only_auroc=rows[0]["auroc_confounds_only"],
                              peak_layer=data["peak_layer"],
                              pos_rate=float(data["y"].mean()))
    out["runtime_s"] = round(time.time() - t0, 1)
    os.makedirs("out", exist_ok=True)
    with open("out/harness_validation.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote out/harness_validation.json in", out["runtime_s"], "s")


if __name__ == "__main__":
    main()
