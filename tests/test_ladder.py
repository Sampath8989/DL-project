"""Does the Review 3 scoring machinery say the right thing when we
already know the answer?

Runs the baseline ladder on synthetic data in three conditions whose
correct verdict is fixed in advance, so a wiring mistake shows up as a
wrong verdict rather than as a plausible number.

    null        the probe score is noise. Nothing should be credited.
    redundant   the probe score is a near-copy of p(true). The baselines
                already have it, so again nothing should be credited --
                this is the case a naive "probe AUROC is 0.78!" headline
                gets wrong.
    extra       the probe score carries a second cause of hallucination
                that no baseline can see. Here, and only here, the gain
                should be clearly positive.

Needs numpy and sklearn only. No torch, no GPU, no model download.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.base_features import ALL_ORIENT
from src.evaluate import (baseline_ladder, single_feature_aurocs, auroc,
                          auroc_se, bootstrap_auroc_delta, cv_oof,
                          select_feats, LADDER)

N = 800
NAMES = ["mean_logprob", "min_logprob", "n_new_tokens", "answer_chars",
         "log_popularity", "mean_logprob_tf", "min_logprob_tf",
         "mean_entropy", "max_entropy", "p_true", "semantic_entropy",
         "self_consistency", "distinct_rate"]


def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def make(rng):
    """Two independent causes of hallucination; the baselines see one."""
    u = rng.normal(size=N)            # visible in the output distribution
    v = rng.normal(size=N)            # visible only inside the model
    y = ((u + v) > 0).astype(np.int64)

    e = lambda s: rng.normal(scale=s, size=N)
    p_true_col = sig(-1.3 * u + e(0.5))
    F = np.stack([
        -0.8 * u + e(0.6),                       # mean_logprob
        -0.9 * u + e(0.9),                       # min_logprob
        8 + 0.4 * u + e(1.0),                    # n_new_tokens
        30 + 2.0 * u + e(5.0),                   # answer_chars
        3.5 - 0.3 * u + e(0.8),                  # log_popularity
        -0.8 * u + e(0.6),                       # mean_logprob_tf
        -0.9 * u + e(0.9),                       # min_logprob_tf
        np.maximum(0.2 + 0.5 * u + e(0.3), 0),   # mean_entropy
        np.maximum(0.6 + 0.7 * u + e(0.4), 0),   # max_entropy
        p_true_col,                              # p_true
        np.maximum(0.8 * u + e(0.4), 0),         # semantic_entropy
        np.clip(0.4 + 0.25 * u + e(0.15), 0, 1),  # self_consistency
        np.clip(0.3 + 0.2 * u + e(0.12), 0, 1),  # distinct_rate
    ], axis=1).astype(np.float32)

    probes = dict(null=rng.normal(size=N),
                  redundant=1.0 - p_true_col + e(0.02),
                  extra=sig(2.2 * v + e(0.3)))
    return y, F, probes


def main():
    rng = np.random.default_rng(0)
    y, F, probes = make(rng)
    print(f"n={N}  hallucinated={y.mean():.1%}")

    singles = single_feature_aurocs(y, F, NAMES, ALL_ORIENT)
    print("\nbaselines alone (oriented):")
    for n in NAMES:
        print(f"  {n:20s} {singles[n]['auroc']:.3f}")
    bad = [n for n in NAMES if singles[n]["auroc"] < 0.5]
    assert not bad, f"orientation is wrong for {bad}: every synthetic " \
                    f"baseline was built to point at hallucination"

    Xb, used = select_feats(F, NAMES, LADDER[-1][1])
    b2 = cv_oof(Xb, y, seed=0)
    print(f"\nB2 pooled out-of-fold AUROC {auroc(y, b2):.3f} "
          f"on {len(used)} columns")

    verdicts = {}
    for cond, p in probes.items():
        r = baseline_ladder(y, p, F, NAMES)
        d = bootstrap_auroc_delta(y, p, b2, n_boot=400, seed=0)
        g0 = r["unique_gain_over_B0"]
        g1 = r["unique_gain_over_B1"]
        g2 = r["unique_gain_over_B2"]
        print(f"\n{cond}")
        print(f"  probe alone AUROC      {auroc(y, p):.3f} "
              f"+-{auroc_se(y, p):.3f}")
        print(f"  gain over B0 surface   {g0:+.3f}")
        print(f"  gain over B1 confid.   {g1:+.3f}")
        print(f"  gain over B2 benchmark {g2:+.3f}")
        print(f"  probe - B2             {d['delta']:+.3f} "
              f"[{d['lo']:+.3f}, {d['hi']:+.3f}]  P(>0)={d['p_gt_zero']:.2f}")
        verdicts[cond] = (g0, g1, g2)

    g_null, g_red, g_extra = (verdicts[k][2] for k in
                              ("null", "redundant", "extra"))
    print("\nchecks")
    assert abs(g_null) < 0.03, f"null probe credited {g_null:+.3f}"
    print(f"  null probe earns nothing            {g_null:+.3f}  ok")
    assert abs(g_red) < 0.03, f"redundant probe credited {g_red:+.3f}"
    print(f"  redundant probe earns nothing       {g_red:+.3f}  ok")
    assert g_extra > 0.05, f"genuine signal missed, gain only {g_extra:+.3f}"
    print(f"  genuine extra signal is found       {g_extra:+.3f}  ok")

    # the ladder must be monotone in difficulty: a harder rung cannot
    # credit the probe with more than an easier one, for real signal
    g0e, g1e, g2e = verdicts["extra"]
    assert g0e >= g2e - 0.02, (
        f"B0 gain {g0e:+.3f} below B2 gain {g2e:+.3f}: the rungs are not "
        f"nested, so select_feats is picking the wrong columns")
    print(f"  rungs are nested ({g0e:+.3f} >= {g2e:+.3f})     ok")

    d = bootstrap_auroc_delta(y, b2, b2, n_boot=200, seed=0)
    assert abs(d["delta"]) < 1e-9, "a score does not beat itself"
    print(f"  a score does not beat itself        {d['delta']:+.1e}  ok")

    print("\nall ladder checks passed")


if __name__ == "__main__":
    main()
