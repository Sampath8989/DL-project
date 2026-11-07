"""Shared probe-fitting logic, so every driver scores it the same way.

scripts/03_run_experiment.py predates this module and still carries its
own copy; the split construction here reproduces it exactly, including
the order in which the random generator is drawn from, so Review 3
numbers are comparable to the Review 2 ones rather than merely similar.

Nothing in this module touches a GPU or a language model. It reads the
cache that extraction wrote and trains small classifiers on the CPU.
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from .config import ProbeConfig
from .train import fit_ensemble, _standardize

ARCHES = ["linear", "mlp", "layer_attn", "traj"]


def make_splits(n, seed=0):
    """60/20/20 by permutation. Identical construction to 03."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n1, n2 = int(0.6 * n), int(0.8 * n)
    return idx[:n1], idx[n1:n2], idx[n2:], rng


def pick_layer(X, y, tr, va):
    """Choose a layer on the validation split only. See CLAUDE.md
    invariant 2: moving this to the test split inflates everything."""
    best, best_a = 0, -1.0
    for l in range(X.shape[1]):
        m = LogisticRegression(max_iter=1000).fit(X[tr, l], y[tr])
        s = roc_auc_score(y[va], m.predict_proba(X[va, l])[:, 1])
        if s > best_a:
            best, best_a = l, s
    return best


def select_rep(z, arch, rep, layer, y, tr, va):
    """Pull the representation an architecture consumes out of the cache."""
    masks = (None, None, None)
    if arch == "traj":
        if layer < 0:
            layer = pick_layer(z["h_answer"], y, tr, va)
        layer = min(layer, z["traj"].shape[2] - 1)
        X = z["traj"][:, :, layer, :]
        masks = (z["mask"], z["mask"], z["mask"])
    elif arch == "layer_attn":
        X = z[rep]
    else:
        H = z[rep]
        if layer < 0:
            layer = pick_layer(H, y, tr, va)
        X = H[:, layer, :]
    return X, int(layer), masks


def fit(X, labels, tr, va, te, cfg, masks=(None, None, None)):
    Xtr, Xva, Xte = _standardize(X[tr], X[va], X[te])
    mtr, mva, mte = masks
    return fit_ensemble(Xtr, labels[tr], Xva, labels[va], Xte, cfg,
                        None if mtr is None else mtr[tr],
                        None if mva is None else mva[va],
                        None if mte is None else mte[te], device="cpu")


def cluster_groups(H, tr, k=20, seed=0):
    """Groups the representation can trivially identify.

    Hewitt and Liang assign a random label per *word type*; there is no
    word type here, so the analogue used is a k-means cluster of the
    answer-state. Fitted on the training split only, then applied to all
    items, so the clustering cannot peek at test labels.

    This is an operationalisation, not their exact control: it asks
    whether the probe can fit an arbitrary mapping over structure the
    representation really encodes, which is the property that makes a
    control informative. Permuted labels are reported alongside it,
    because scripts/05_control_validation.py showed permuted labels have
    a blind spot exactly where a probe is memorising.
    """
    F = np.asarray(H, float).reshape(len(H), -1)
    mu, sd = F[tr].mean(0), F[tr].std(0) + 1e-8
    F = (F - mu) / sd
    k = max(2, min(k, len(tr) // 10))
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(F[tr])
    return km.predict(F), k


def control_labels(y, groups, rng, seed=0):
    """The two control-label schemes, as a name -> labels dict."""
    out = {}
    out["permuted"] = y[rng.permutation(len(y))]
    g = np.asarray(groups)
    lab = np.random.default_rng(seed + 17).integers(0, 2, size=int(g.max()) + 1)
    out["group"] = lab[g].astype(np.int64)
    return out


def safe_auc(y, p):
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def probe_config(arch):
    return ProbeConfig(arch=arch)
