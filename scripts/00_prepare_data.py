"""Check the downloaded datasets and build the fixed working sets.

Run this once after downloading, before any extraction. It prints what it
found and what it decided, so a wrong column or a half-finished download
is visible immediately rather than three hours into a GPU run.

    python scripts/00_prepare_data.py                  # everything available
    python scripts/00_prepare_data.py --source popqa --n 800
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.paths import RAW, DATA, ensure_dirs
from data.datasets import LOADERS, build_working_set, save_working_set


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="all",
                   choices=["all"] + list(LOADERS))
    p.add_argument("--n", type=int, default=800,
                   help="working-set size; 0 means keep everything")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    a = parse()
    ensure_dirs()
    print(f"looking for raw data in: {RAW}")
    files = sorted(p for p in RAW.rglob("*") if p.is_file())
    if files:
        for p in files[:30]:
            print(f"   {p.relative_to(RAW)}   {p.stat().st_size / 1e6:.1f} MB")
    else:
        print("   (empty) - put the downloaded PopQA / TriviaQA files here")
    print()

    sources = list(LOADERS) if a.source == "all" else [a.source]
    built = []
    for src in sources:
        print(f"[{src}]")
        try:
            n = None if a.n == 0 else a.n
            items = build_working_set(src, n=n, seed=a.seed)
        except FileNotFoundError as e:
            print(f"  skipped: {e}\n")
            continue
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}\n")
            continue

        path = save_working_set(items, DATA / f"working_{src}.jsonl")
        gold_n = [len(i["gold"]) for i in items]
        qlen = [len(i["question"].split()) for i in items]
        lp = [i["log_pop"] for i in items if np.isfinite(i["log_pop"])]
        print(f"  saved {len(items)} items -> {path.relative_to(path.parent.parent)}")
        print(f"  gold answers per item: median {int(np.median(gold_n))}, "
              f"max {max(gold_n)}")
        print(f"  question length: median {int(np.median(qlen))} words")
        if lp:
            print(f"  log10 popularity: min {min(lp):.2f}  median "
                  f"{np.median(lp):.2f}  max {max(lp):.2f}")
        else:
            print("  popularity: not available for this source "
                  "(the confound vector will drop that column)")
        print(f"  popularity bands: {dict(Counter(i['pop_bucket'] for i in items))}")
        print(f"  example: {items[0]['question']!r} -> {items[0]['gold'][:3]}")
        print()
        built.append(src)

    if not built:
        print("Nothing was built. Download the datasets into data/raw/ "
              "(any layout) or set HALLU_DATA to the folder holding them.")
        return 1
    print("ready:", ", ".join(built))
    print("next:  python scripts/03_run_experiment.py --dataset popqa "
          "--model <checkpoint> --arch layer_attn --device cuda")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
