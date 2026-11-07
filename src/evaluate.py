"""Evaluation and deconfounding.

Every metric degrades to NaN on a degenerate split (single class) rather
than raising, so a sweep never dies half way through.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from sklearn.model_selection import cross_val_score


def _single_class(y):
    return len(np.unique(np.asarray(y))) < 2


def auroc(y, p):
    if _single_class(y):
        return float("nan")
    return float(roc_auc_score(y, p))


def auprc(y, p):
    """Area under precision-recall. Reported because hallucination is the
    minority class and PR is sensitive to the positive rate."""
    if _single_class(y):
        return float("nan")
    return float(average_precision_score(y, p))


def tpr_at_fpr(y, p, target_fpr=0.10):
    """Operating-point metric: how much hallucination is caught when only
    `target_fpr` of correct answers may be flagged."""
    if _single_class(y):
        return float("nan")
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.interp(target_fpr, fpr, tpr))


def ece(y, p, bins=15):
    """Expected calibration error, equal-width bins."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    if len(y) == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    tot = 0.0
    for b in range(bins):
        m = idx == b
        if m.sum() == 0:
            continue
        tot += (m.sum() / len(y)) * abs(y[m].mean() - p[m].mean())
    return float(tot)


def brier(y, p):
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def risk_coverage(y, p, spread, points=20):
    """Selective prediction: sort by ensemble disagreement, report error at
    each coverage level. Low spread should mean low error."""
    y = np.asarray(y, float)
    order = np.argsort(spread)
    err = (np.asarray(p) >= 0.5).astype(float) != y
    err = err[order]
    out = []
    for c in np.linspace(1.0 / points, 1.0, points):
        k = max(int(round(c * len(y))), 1)
        out.append((float(c), float(err[:k].mean())))
    return out


def selectivity(auc_real, auc_random):
    """Hewitt-Liang style control: real-task skill minus random-label skill.
    A probe that fits shuffled labels as well as real ones has learned
    nothing about truth."""
    if np.isnan(auc_real) or np.isnan(auc_random):
        return float("nan")
    return float(auc_real - auc_random)


def build_confounds(conf, names=None, verbose=False):
    """Clean the confound matrix before it is used as a baseline.

    Drops columns that are entirely missing (popularity is undefined for
    TriviaQA and for the hand-written control set) and median-imputes
    partial gaps, adding a missingness indicator so the imputation itself
    cannot be mistaken for signal.
    """
    C = np.asarray(conf, dtype=float)
    if C.ndim == 1:
        C = C[:, None]
    names = list(names) if names is not None else [f"c{i}" for i in range(C.shape[1])]
    keep, kept_names = [], []
    for j in range(C.shape[1]):
        col = C[:, j]
        if np.all(~np.isfinite(col)):
            continue                                   # nothing to use
        if np.nanstd(col[np.isfinite(col)]) == 0:
            continue                                   # constant, carries nothing
        if np.any(~np.isfinite(col)):
            med = float(np.nanmedian(col[np.isfinite(col)]))
            miss = (~np.isfinite(col)).astype(float)
            col = np.where(np.isfinite(col), col, med)
            keep.append(col); kept_names.append(names[j])
            keep.append(miss); kept_names.append(names[j] + "_missing")
        else:
            keep.append(col); kept_names.append(names[j])
    if not keep:
        raise ValueError("every confound column was empty or constant")
    out = np.stack(keep, axis=1)
    if verbose:
        print("  confounds used:", kept_names)
    return out, kept_names


def cv_auc(X, y):
    """Cross-validated AUROC of a logistic fit on X.

    The single place a feature set is scored, so the confound baseline,
    the benchmark baselines and the probe are never compared across two
    slightly different fitting procedures.
    """
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    y = np.asarray(y)
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    m = LogisticRegression(max_iter=2000)
    sc = cross_val_score(m, X, y, cv=min(5, int(np.bincount(y).min())),
                         scoring="roc_auc")
    return float(sc.mean())


def deconfound(y, probe_score, confounds):
    """Does the probe add anything over surface features?

    Fit logistic regression on the confounds alone, then on confounds plus
    the probe score. `unique_gain` is the AUROC the probe contributes that
    the confounds could not already supply.
    """
    y = np.asarray(y)
    C = np.asarray(confounds, float)
    if C.ndim == 1:
        C = C[:, None]
    if _single_class(y):
        return dict(auroc_confounds_only=float("nan"),
                    auroc_with_probe=float("nan"), unique_gain=float("nan"))
    s = np.asarray(probe_score, float)[:, None]
    both = np.hstack([C, s])

    a_c = cv_auc(C, y)
    a_b = cv_auc(both, y)
    return dict(auroc_confounds_only=a_c, auroc_with_probe=a_b,
                unique_gain=a_b - a_c)


def artefact_probe(X, task_ids):
    """Can a classifier recover which *task* an item came from, from the
    same activations? High accuracy means the representation carries
    dataset identity that a truth probe could exploit instead of truth."""
    task_ids = np.asarray(task_ids)
    if len(np.unique(task_ids)) < 2:
        return float("nan")
    # a class with a single member makes cross-validation undefined
    _, counts = np.unique(task_ids, return_counts=True)
    if counts.min() < 3:
        return float("nan")
    F = np.asarray(X, float).reshape(len(task_ids), -1)
    F = (F - F.mean(0)) / (F.std(0) + 1e-8)
    m = LogisticRegression(max_iter=2000)
    return float(cross_val_score(m, F, task_ids, cv=3, scoring="accuracy").mean())


def label_balance(y):
    y = np.asarray(y)
    return dict(n=int(len(y)), n_pos=int(y.sum()),
                pos_rate=float(y.mean()) if len(y) else float("nan"))


def report(y, prob, spread=None, confounds=None, auc_random=None):
    r = dict(auroc=auroc(y, prob), auprc=auprc(y, prob),
             tpr_at_fpr10=tpr_at_fpr(y, prob, 0.10),
             ece=ece(y, prob), brier=brier(y, prob), **label_balance(y))
    if spread is not None:
        r["mean_spread"] = float(np.mean(spread))
    if auc_random is not None:
        r["selectivity"] = selectivity(r["auroc"], auc_random)
        r["auroc_random_labels"] = float(auc_random)
    if confounds is not None:
        r.update(deconfound(y, prob, confounds))
    return r


# ===================================================================
# Review 3: the baseline ladder
#
# Review 2 asked "does the probe beat surface features". That was the
# wrong bar, because nobody detecting hallucination in practice uses
# surface features -- they use the model's own uncertainty. So the
# baselines are nested, cheapest first, and the probe is charged against
# each rung:
#
#   B0 surface     length, characters, entity popularity
#   B1 confidence  B0 + token logprob and predictive entropy (one pass)
#   B2 benchmark   B1 + p(true), semantic entropy, self-consistency
#                  (the published methods; needs extra sampling)
#
# unique_gain_over_B2 is the Review 3 headline. It is the only number
# that answers "is reading the activations worth anything that the
# published, activation-free methods cannot already do".
# ===================================================================
SURFACE_FEATS = ["n_new_tokens", "answer_chars", "log_popularity"]
CONFIDENCE_FEATS = ["mean_logprob", "min_logprob", "mean_logprob_tf",
                    "min_logprob_tf", "mean_entropy", "max_entropy"]
BENCHMARK_FEATS = ["p_true", "semantic_entropy", "self_consistency",
                   "distinct_rate"]

LADDER = [("B0", SURFACE_FEATS),
          ("B1", SURFACE_FEATS + CONFIDENCE_FEATS),
          ("B2", SURFACE_FEATS + CONFIDENCE_FEATS + BENCHMARK_FEATS)]

# kept short in the result keys (unique_gain_over_B2) and spelled out here,
# so a label can be changed for a slide without renaming a metric
LADDER_DESC = {"B0": "surface", "B1": "confidence",
               "B2": "benchmark (published methods)"}


def select_feats(F, names, wanted, verbose=False):
    """Slice the feature matrix by column name, then clean it.

    Missing names are skipped silently: popularity is undefined for
    TriviaQA and the baseline columns are absent until 06_baselines.py has
    run, and a rung of the ladder that cannot be built should degrade to
    the rungs below it rather than crash a sweep at 3 a.m.
    """
    names = [str(n) for n in names]
    idx = [names.index(w) for w in wanted if w in names]
    if not idx:
        return None, []
    sub = np.asarray(F, float)[:, idx]
    sub_names = [names[i] for i in idx]
    try:
        return build_confounds(sub, sub_names, verbose=verbose)
    except ValueError:
        return None, []


def baseline_ladder(y, probe_score, F, names, verbose=False):
    """Score every rung alone and every rung plus the probe."""
    y = np.asarray(y)
    out = {}
    if _single_class(y):
        return out
    s = np.asarray(probe_score, float)[:, None]
    for rung, wanted in LADDER:
        X, used = select_feats(F, names, wanted, verbose=verbose)
        if X is None:
            out[f"auroc_{rung}"] = float("nan")
            out[f"auroc_{rung}_plus_probe"] = float("nan")
            out[f"unique_gain_over_{rung}"] = float("nan")
            out[f"feats_{rung}"] = []
            continue
        a_b = cv_auc(X, y)
        a_p = cv_auc(np.hstack([X, s]), y)
        out[f"auroc_{rung}"] = a_b
        out[f"auroc_{rung}_plus_probe"] = a_p
        out[f"unique_gain_over_{rung}"] = a_p - a_b
        out[f"feats_{rung}"] = used
    return out


def single_feature_aurocs(y, F, names, orient=None):
    """AUROC of each baseline feature on its own, correctly oriented.

    `orient` gives the sign that points at hallucination, so a baseline is
    never quietly flipped to flatter it. Items missing that feature are
    dropped and the coverage recorded, because imputing a confidence score
    would invent confidence the model never expressed.
    """
    from .base_features import ALL_ORIENT
    orient = orient or ALL_ORIENT
    y = np.asarray(y)
    F = np.asarray(F, float)
    names = [str(n) for n in names]
    out = {}
    for j, nm in enumerate(names):
        col = F[:, j]
        ok = np.isfinite(col)
        if ok.sum() < 20 or _single_class(y[ok]):
            out[nm] = dict(auroc=float("nan"), n=int(ok.sum()))
            continue
        sgn = orient.get(nm, +1)
        a = auroc(y[ok], sgn * col[ok])
        out[nm] = dict(auroc=float(a), n=int(ok.sum()),
                       se=auroc_se(y[ok], sgn * col[ok]))
    return out


def auroc_se(y, p):
    """Hanley-McNeil standard error of an AUROC.

    Used instead of a bare point estimate because at n=160 test items a
    difference of 0.03 is inside the noise, and the conclusion of this
    project turns on exactly such differences.
    """
    y = np.asarray(y)
    if _single_class(y):
        return float("nan")
    a = auroc(y, p)
    n1 = float((y == 1).sum())
    n0 = float((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    q1 = a / (2.0 - a)
    q2 = 2.0 * a * a / (1.0 + a)
    var = (a * (1 - a) + (n1 - 1) * (q1 - a * a)
           + (n0 - 1) * (q2 - a * a)) / (n1 * n0)
    return float(np.sqrt(max(var, 0.0)))


def bootstrap_auroc_delta(y, p_better, p_worse, n_boot=2000, seed=0):
    """Paired bootstrap on AUROC(p_better) - AUROC(p_worse).

    Paired, because both scores are computed on the same items: resampling
    them independently would overstate the uncertainty of the difference.
    `p_gt_zero` is the fraction of resamples where the first score wins,
    which is what licenses or refuses the word "better".
    """
    y = np.asarray(y)
    a = np.asarray(p_better, float)
    b = np.asarray(p_worse, float)
    ok = np.isfinite(a) & np.isfinite(b)
    y, a, b = y[ok], a[ok], b[ok]
    if _single_class(y):
        return dict(delta=float("nan"), lo=float("nan"), hi=float("nan"),
                    p_gt_zero=float("nan"), n_boot=0)
    rng = np.random.default_rng(seed)
    n = len(y)
    deltas = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if len(np.unique(y[i])) < 2:
            continue
        deltas.append(roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]))
    if not deltas:
        return dict(delta=float("nan"), lo=float("nan"), hi=float("nan"),
                    p_gt_zero=float("nan"), n_boot=0)
    d = np.asarray(deltas)
    return dict(delta=float(auroc(y, a) - auroc(y, b)),
                lo=float(np.percentile(d, 2.5)),
                hi=float(np.percentile(d, 97.5)),
                p_gt_zero=float((d > 0).mean()), n_boot=int(len(d)))


def cv_oof(X, y, folds=5, seed=0):
    """Out-of-fold probabilities for a feature set.

    `cv_auc` averages AUROC across folds, which is what Review 2 reported
    and is kept for comparability. A paired bootstrap needs one score per
    item instead, so this returns pooled out-of-fold predictions. The two
    AUROCs will differ slightly; both are reported rather than one being
    quietly substituted for the other.
    """
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    y = np.asarray(y)
    if _single_class(y):
        return np.full(len(y), np.nan)
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    k = min(folds, int(np.bincount(y).min()))
    if k < 2:
        return np.full(len(y), np.nan)
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    m = LogisticRegression(max_iter=2000)
    return cross_val_predict(m, X, y, cv=cv, method="predict_proba")[:, 1]
