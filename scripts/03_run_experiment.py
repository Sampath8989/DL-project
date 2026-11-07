"""Driver for a real run: extract from a pretrained model, then probe.

    python scripts/03_run_experiment.py --dataset popqa \
        --model Qwen/Qwen2.5-3B-Instruct --arch layer_attn --device cuda

Extraction is cached. Re-running with a different --arch against the same
model and dataset reuses the cache and touches no GPU, so the probe sweep
after the first extraction costs minutes on a CPU.

Shape contract
    h_prompt, h_answer : [N, L+1, d]     layer_attn uses every layer
                                          linear / mlp select one layer
    traj               : [N, T, L', d]   traj selects one layer -> [N, T, d]
Label convention
    1 = hallucinated
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src.paths import DATA, OUT, ensure_dirs
from src.config import ExtractConfig, ProbeConfig
from src.extract import run as extract_run
from src.train import fit_ensemble, _standardize
from src.evaluate import (report, artefact_probe, label_balance,
                          build_confounds)
from data.datasets import (LOADERS, build_working_set, load_working_set,
                           save_working_set, label)


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--dataset", default="popqa", choices=list(LOADERS))
    p.add_argument("--arch", default="layer_attn",
                   choices=["linear", "mlp", "layer_attn", "traj"])
    p.add_argument("--rep", default="h_answer",
                   choices=["h_prompt", "h_answer", "traj"])
    p.add_argument("--layer", type=int, default=-1,
                   help="layer for linear/mlp/traj; -1 selects on validation")
    p.add_argument("--n", type=int, default=800)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default=None)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--layer-stride", type=int, default=1,
                   help="store every k-th layer of the token trajectory")
    p.add_argument("--out", default=None)
    p.add_argument("--cache", default=None, help="reuse an existing acts.npz")
    p.add_argument("--tiny", action="store_true", help="offline plumbing test")
    return p.parse_args()


def slug(s):
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)


def get_items(a):
    """Use the saved working set when present so every model in the grid
    sees exactly the same items, in the same order."""
    ws = DATA / f"working_{a.dataset}.jsonl"
    if ws.exists():
        items = load_working_set(ws)
        print(f"working set: {ws.name} ({len(items)} items)")
    else:
        print(f"no saved working set for {a.dataset}; building one "
              f"(run scripts/00_prepare_data.py to fix this ordering)")
        items = build_working_set(a.dataset, n=a.n, seed=a.seed)
        save_working_set(items, ws)
    return items[: a.n] if a.n else items


def get_cache(a, items, out_dir):
    if a.cache and Path(a.cache).exists():
        print(f"reusing cache: {a.cache}")
        return a.cache
    default = Path(out_dir) / "acts.npz"
    if default.exists():
        print(f"reusing cache: {default}")
        return str(default)

    dtype = a.dtype or ("float32" if a.device == "cpu" else "float16")
    cfg = ExtractConfig(model_name=a.model, device=a.device, dtype=dtype,
                        max_items=a.n, max_new_tokens=a.max_new_tokens,
                        layer_stride=a.layer_stride, out_dir=str(out_dir))
    if a.tiny:
        from src.tiny import build_tiny
        m, t = build_tiny()
        cfg.model_name, cfg.device, cfg.dtype = "tiny-random-llama", "cpu", "float32"
        return extract_run(items, cfg, label, model=m, tok=t)
    print(f"extracting from {a.model} on {a.device} ({dtype}) ...")
    return extract_run(items, cfg, label)


def pick_layer(X, y, tr, va):
    best, best_a = 0, -1.0
    for l in range(X.shape[1]):
        m = LogisticRegression(max_iter=1000).fit(X[tr, l], y[tr])
        s = roc_auc_score(y[va], m.predict_proba(X[va, l])[:, 1])
        if s > best_a:
            best, best_a = l, s
    return best


def main():
    a = parse()
    ensure_dirs()
    if a.tiny:
        a.model = "tiny-random-llama"
    out_dir = Path(a.out) if a.out else OUT / f"{slug(a.model)}__{a.dataset}"
    out_dir.mkdir(parents=True, exist_ok=True)

    items = get_items(a)
    path = get_cache(a, items, out_dir)
    z = np.load(path, allow_pickle=True)
    y, conf, task = z["y"], z["conf"], z["task"]
    conf_names = ([str(c) for c in z["conf_names"]] if "conf_names" in z
                  else None)

    bal = label_balance(y)
    print("label balance:", bal)
    if bal["n_pos"] in (0, bal["n"]):
        print("WARNING: one class only. Probes cannot be trained on this "
              "cache. Change the question mix or the model before "
              "interpreting anything below.")
    elif not 0.1 <= bal["pos_rate"] <= 0.9:
        print(f"WARNING: heavily imbalanced ({bal['pos_rate']:.1%} "
              f"hallucinated). Metrics will be noisy.")

    rng = np.random.default_rng(a.seed)
    idx = rng.permutation(len(y))
    n1, n2 = int(0.6 * len(y)), int(0.8 * len(y))
    tr, va, te = idx[:n1], idx[n1:n2], idx[n2:]

    mask_tr = mask_va = mask_te = None
    layer = a.layer
    if a.arch == "traj":
        if layer < 0:
            layer = pick_layer(z["h_answer"], y, tr, va)
        layer = min(layer, z["traj"].shape[2] - 1)   # traj may be strided
        X = z["traj"][:, :, layer, :]
        mask_tr, mask_va, mask_te = z["mask"][tr], z["mask"][va], z["mask"][te]
    elif a.arch == "layer_attn":
        X = z[a.rep]
    else:
        H = z[a.rep]
        if layer < 0:
            layer = pick_layer(H, y, tr, va)
        X = H[:, layer, :]

    cfg = ProbeConfig(arch=a.arch)
    Xtr, Xva, Xte = _standardize(X[tr], X[va], X[te])
    print(f"training {a.arch} probe on {X.shape[1:]} ...")
    ens = fit_ensemble(Xtr, y[tr], Xva, y[va], Xte, cfg,
                       mask_tr, mask_va, mask_te, device="cpu")

    yr = y[rng.permutation(len(y))]
    ens_r = fit_ensemble(Xtr, yr[tr], Xva, yr[va], Xte, cfg,
                         mask_tr, mask_va, mask_te, device="cpu")
    auc_random = (roc_auc_score(yr[te], ens_r["prob"])
                  if len(set(yr[te].tolist())) > 1 else float("nan"))

    C, used = build_confounds(conf[te], conf_names, verbose=True)
    r = report(y[te], ens["prob"], spread=ens["spread"], confounds=C,
               auc_random=auc_random)
    r.update(model=a.model, dataset=a.dataset, arch=a.arch, rep=a.rep,
             layer=int(layer), n_params=ens["n_params"], cache=str(path),
             confounds_used=used, seed=a.seed,
             artefact_task_acc=artefact_probe(
                 z["h_answer"][te].reshape(len(te), -1), task[te]))

    dest = out_dir / f"{a.arch}_{a.rep}.json"
    dest.write_text(json.dumps(r, indent=2))
    print()
    for k, v in r.items():
        print(f"  {k:22s} {v}")
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
