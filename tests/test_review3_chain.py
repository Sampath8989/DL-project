"""End-to-end plumbing test for the Review 3 scripts, offline.

Builds two synthetic caches in exactly the shape extraction writes, with
a truth signal planted in one layer so the probes have something real to
find, then runs 06_baselines.py, 07_final_analysis.py and 08_transfer.py
against them as subprocesses, the same way the laptop will.

This validates wiring, not science: a randomly initialised model and a
planted signal. What it catches is the class of bug that only appears
when the four scripts are chained -- a column name that does not match,
an index that is off by one between the cache and the baselines, a
transfer that silently standardises with the target's own statistics.

    python tests/test_review3_chain.py

Takes a few minutes on a CPU and needs no checkpoint download.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from data.seed_questions import SEED

N, LAYERS, D, T = 160, 5, 48, 8
MODEL = "tiny-random-llama"
PEAK = 3


def build_cache(out_dir, dataset, seed):
    """A cache with a real truth signal at layer PEAK and nowhere else."""
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    w = rng.normal(size=D)
    w /= np.linalg.norm(w)
    strength = rng.normal(size=N)
    y = (strength > 0).astype(np.int64)

    H = rng.normal(0, 1, (N, LAYERS, D)).astype(np.float32)
    for l in range(LAYERS):
        # the signal is strongest at PEAK and fades either side of it, so
        # validation-split layer selection has a real choice to make
        amp = 1.8 * np.exp(-((l - PEAK) ** 2) / 1.2)
        H[:, l, :] += amp * strength[:, None] * w[None, :]
    H_prompt = rng.normal(0, 1, (N, LAYERS, D)).astype(np.float32)

    traj = rng.normal(0, 1, (N, T, LAYERS, D)).astype(np.float32)
    traj[:, :, PEAK, :] += 1.2 * strength[:, None, None] * w[None, None, :]
    mask = np.ones((N, T), dtype=np.float32)

    # surface features: weakly informative, as on real data
    conf = np.stack([
        -0.5 * strength + rng.normal(0, 0.8, N),      # mean_logprob
        -0.6 * strength + rng.normal(0, 1.0, N),      # min_logprob
        8 + rng.normal(0, 2, N),                      # n_new_tokens
        30 + rng.normal(0, 8, N),                     # answer_chars
        (3.5 - 0.2 * strength + rng.normal(0, 0.9, N)
         if dataset == "popqa" else np.full(N, np.nan)),
    ], axis=1).astype(np.float32)

    qs = [SEED[i % len(SEED)] for i in range(N)]
    task = np.array([f"{dataset}_pop{i % 3}" for i in range(N)])
    items = []
    for i, (q, g, t) in enumerate(qs):
        # a wrong answer is another item's gold, not a constant string. A
        # constant makes the answer text perfectly predictable from its own
        # token logprobs, the B2 rung hits AUROC 1.000, and every "does the
        # probe add anything over B2" check becomes vacuous at the ceiling.
        wrong = SEED[(i + 37) % len(SEED)][1][0]
        items.append(dict(idx=i, task=task[i], question=q,
                          gold=list(g),
                          text=" " + (g[0] if y[i] == 0 else wrong),
                          mask=mask[i].tolist()))

    np.savez_compressed(
        out_dir / "acts.npz", h_prompt=H_prompt, h_answer=H, traj=traj,
        mask=mask, conf=conf, y=y, task=task,
        conf_names=np.array(["mean_logprob", "min_logprob", "n_new_tokens",
                             "answer_chars", "log_popularity"]))
    (out_dir / "acts_meta.json").write_text(json.dumps(
        dict(model=MODEL, n=N, config={}, items=items), indent=2))
    return y


def run(name, args):
    print(f"\n--- {name}")
    p = subprocess.run([sys.executable] + args, cwd=ROOT,
                       capture_output=True, text=True)
    tail = (p.stdout or "").strip().splitlines()
    for line in tail[-14:]:
        print("   ", line)
    if p.returncode != 0:
        print((p.stderr or "")[-2500:])
        raise SystemExit(f"{name} failed with exit {p.returncode}")
    return p.stdout


def main():
    out = ROOT / "out"
    dirs = {}
    for ds, seed in (("popqa", 1), ("triviaqa", 2)):
        d = out / f"{MODEL}__{ds}"
        if d.exists():
            shutil.rmtree(d)
        build_cache(d, ds, seed)
        dirs[ds] = d
    print(f"built two synthetic caches, N={N}, signal at layer {PEAK}")

    # The guard first. This synthetic cache's mean_logprob column is a
    # fabricated number, not a real recorded logprob, so the two routes
    # cannot agree -- which is exactly the shape of the Qwen
    # repetition-penalty bug. 06 must refuse to write baselines for it.
    print("\n--- 06 must REFUSE an inconsistent cache")
    pr = subprocess.run(
        [sys.executable, "scripts/06_baselines.py", "--tiny", "--model",
         MODEL, "--dataset", "popqa", "--device", "cpu", "--k", "4",
         "--max-new-tokens", "8", "--force", "--check-first", "12"],
        cwd=ROOT, capture_output=True, text=True)
    assert pr.returncode == 2, (
        f"06 exited {pr.returncode}; it must exit 2 when the logprob "
        f"routes disagree.\n{(pr.stdout or '')[-1500:]}")
    assert "early check" in (pr.stdout or ""), "the early gate did not run"
    assert not (dirs["popqa"] / "baselines.npz").exists(), \
        "06 wrote baselines.npz despite failing the consistency check"
    assert (dirs["popqa"] / "baselines_FAILED_CONSISTENCY.json").exists() or \
        "stopping after" in pr.stdout, "no refusal record was left behind"
    print("    refused, exit 2, nothing written  ok")

    for ds in ("popqa", "triviaqa"):
        run(f"06_baselines {ds}",
            ["scripts/06_baselines.py", "--tiny", "--model", MODEL,
             "--dataset", ds, "--device", "cpu", "--k", "4",
             "--max-new-tokens", "8", "--force", "--allow-inconsistent"])
        b = np.load(dirs[ds] / "baselines.npz", allow_pickle=True)
        assert len(b["base"]) == N, f"{ds}: {len(b['base'])} rows, expected {N}"
        names = [str(x) for x in b["base_names"]]
        assert "p_true" in names and "semantic_entropy" in names
        finite = np.isfinite(b["base"]).mean(0)
        print(f"    {ds}: {len(names)} baseline columns, "
              f"finite fraction min {finite.min():.2f}")
        assert finite.max() > 0.5, "every baseline column is NaN"

    res = ROOT / "out" / "chain_results.json"
    tr = ROOT / "out" / "chain_transfer.json"
    run("07_final_analysis",
        ["scripts/07_final_analysis.py", "--only", MODEL,
         "--arches", "linear,mlp,layer_attn,traj", "--out", str(res)])
    run("08_transfer",
        ["scripts/08_transfer.py", "--only", MODEL,
         "--arches", "linear,mlp", "--out", str(tr)])

    R = json.loads(res.read_text())["results"]
    assert len(R) == 2, f"expected two model-dataset pairs, got {len(R)}"
    print("\nchecks on 07")
    for row in R:
        assert "error" not in row, row["error"]
        for arch, r in row["probes"].items():
            for k in ("auroc", "auroc_se", "unique_gain_over_B2",
                      "auroc_control_permuted", "auroc_control_group",
                      "selectivity_group", "vs_B2"):
                assert k in r, f"{row['dataset']}/{arch} missing {k}"
            assert 0.0 <= r["auroc"] <= 1.0
        a = row["probes"]["layer_attn"]["auroc"]
        lin = row["probes"]["linear"]
        assert row["best_single_baseline_auroc"] < 0.99, (
            f"{row['dataset']}: a baseline reaches "
            f"{row['best_single_baseline_auroc']:.3f}, so the ladder is at "
            f"its ceiling and the gain checks below prove nothing")
        print(f"  {row['dataset']:9s} layer_attn AUROC {a:.3f}  "
              f"linear layer {lin['layer']} (planted {PEAK})  "
              f"best baseline {row['best_single_baseline']}")
        assert a > 0.70, (f"planted signal not recovered on "
                          f"{row['dataset']}: AUROC {a:.3f}")
        assert abs(lin["layer"] - PEAK) <= 1, (
            f"layer selection picked {lin['layer']}, planted at {PEAK}")

    Tr = json.loads(tr.read_text())["results"]
    assert len(Tr) == 2, f"expected both directions, got {len(Tr)}"
    print("\nchecks on 08")
    for row in Tr:
        assert "error" not in row, row["error"]
        for arch, r in row["probes"].items():
            assert "transfer_auroc" in r and "transfer_drop" in r
            print(f"  {row['source']}->{row['target']} {arch:11s} "
                  f"in-domain {r['in_domain_auroc']:.3f} "
                  f"transfer {r['transfer_auroc']:.3f}")
        # the two caches plant the signal on independent random directions,
        # so a transferred probe SHOULD fail. If it succeeds, the target's
        # own statistics are leaking into the fit.
        t = row["probes"]["linear"]["transfer_auroc"]
        assert t < 0.70, (f"transfer AUROC {t:.3f} across independently "
                          f"planted directions -- the target is leaking "
                          f"into training")
    print("  transfer correctly fails across unrelated signals  ok")

    print("\nchain test passed")


def cleanup():
    """Remove the synthetic caches. Left in out/, they match the *__* glob
    that 07, 08 and 09 use to find real caches, so 09 fails the Review 3
    driver on them and 07/08 would pull them into the analysis."""
    for ds in ("popqa", "triviaqa"):
        d = ROOT / "out" / f"{MODEL}__{ds}"
        if d.exists():
            shutil.rmtree(d)


if __name__ == "__main__":
    try:
        main()
    finally:
        cleanup()
