"""Download PopQA and TriviaQA into data/raw/ with one command.

Neither dataset ships inside a Python package the way sklearn ships iris
-- iris is 150 rows and a few kilobytes, these are research benchmarks.
The equivalent convenience is the Hugging Face `datasets` library, which
this script wraps so the files land where the loader expects them.

    pip install datasets
    python scripts/00_fetch_data.py                  # both, default configs
    python scripts/00_fetch_data.py --source popqa
    python scripts/00_fetch_data.py --stream --limit 3000   # low bandwidth

SIZES -- read this before starting a download
    PopQA                     ~14,300 rows, a few MB. Single `test` split.
    TriviaQA unfiltered.nocontext   ~633 MB   <- the one you want
    TriviaQA rc / rc.nocontext      ~2.7 GB   <- includes evidence documents
                                                 that this project never uses

If Hugging Face is unreachable from your network, download the files by
hand instead and drop them anywhere under data/raw/ -- the loader searches
recursively and sniffs the columns, so the folder layout does not matter.
    PopQA     https://huggingface.co/datasets/akariasai/PopQA
    TriviaQA  https://huggingface.co/datasets/mandarjoshi/trivia_qa
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.paths import RAW, ensure_dirs

POPQA = "akariasai/PopQA"
TRIVIA = "mandarjoshi/trivia_qa"
TRIVIA_DEFAULT_CONFIG = "unfiltered.nocontext"
HEAVY = {"rc", "rc.nocontext", "rc.web", "rc.web.nocontext",
         "rc.wikipedia", "rc.wikipedia.nocontext", "unfiltered"}


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="all",
                   choices=["all", "popqa", "triviaqa"])
    p.add_argument("--trivia-config", default=TRIVIA_DEFAULT_CONFIG)
    p.add_argument("--trivia-split", default="validation")
    p.add_argument("--limit", type=int, default=None,
                   help="keep only the first N rows (with --stream, also "
                        "downloads only that much)")
    p.add_argument("--stream", action="store_true",
                   help="stream instead of downloading the whole file. Saves "
                        "bandwidth but takes rows in file order, not at "
                        "random, so prefer a full download when you can.")
    return p.parse_args()


def need_datasets():
    try:
        import datasets  # noqa: F401
    except ImportError:
        print("The `datasets` library is not installed.\n"
              "    pip install datasets\n"
              "Or download the files by hand and put them under data/raw/ "
              "- see the header of this file for the links.")
        raise SystemExit(1)
    from datasets import load_dataset
    return load_dataset


def fetch_popqa(load_dataset, a):
    print(f"[popqa] {POPQA} (test split, ~14.3k rows, a few MB)")
    if a.stream:
        ds = load_dataset(POPQA, split="test", streaming=True)
        rows = []
        for i, r in enumerate(ds):
            if a.limit and i >= a.limit:
                break
            rows.append(r)
    else:
        ds = load_dataset(POPQA, split="test")
        if a.limit:
            ds = ds.select(range(min(a.limit, len(ds))))
        rows = list(ds)

    import pandas as pd
    df = pd.DataFrame(rows)
    out = RAW / "popqa_test.tsv"
    df.to_csv(out, sep="\t", index=False)
    print(f"[popqa] wrote {len(df)} rows -> {out}")
    print(f"[popqa] columns: {list(df.columns)}")
    for c in ("question", "possible_answers", "s_pop"):
        if c not in df.columns:
            print(f"[popqa] WARNING: expected column {c!r} is missing. "
                  f"Tell Claude the real column names before running "
                  f"00_prepare_data.py.")
    return out


def fetch_triviaqa(load_dataset, a):
    cfg = a.trivia_config
    if cfg in HEAVY:
        print(f"[triviaqa] WARNING: config {cfg!r} bundles evidence documents "
              f"and is 2.7-3.3 GB. This project never reads them. "
              f"{TRIVIA_DEFAULT_CONFIG!r} is ~633 MB and has the same "
              f"questions and answers.")
    print(f"[triviaqa] {TRIVIA} [{cfg}] split={a.trivia_split}")

    keep = ("question", "question_id", "answer")
    rows = []
    if a.stream:
        ds = load_dataset(TRIVIA, cfg, split=a.trivia_split, streaming=True)
        for i, r in enumerate(ds):
            if a.limit and i >= a.limit:
                break
            rows.append({k: r[k] for k in keep if k in r})
    else:
        ds = load_dataset(TRIVIA, cfg, split=a.trivia_split)
        if a.limit:
            ds = ds.select(range(min(a.limit, len(ds))))
        drop = [c for c in ds.column_names if c not in keep]
        if drop:
            ds = ds.remove_columns(drop)     # evidence pages are large
        rows = list(ds)

    out = RAW / f"triviaqa_{cfg.replace('.', '_')}_{a.trivia_split}.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    print(f"[triviaqa] wrote {len(rows)} rows -> {out}")
    if rows:
        print(f"[triviaqa] keys: {list(rows[0])}")
    return out


def main():
    a = parse()
    ensure_dirs()
    print(f"target folder: {RAW}\n")
    load_dataset = need_datasets()

    done = []
    for src, fn in (("popqa", fetch_popqa), ("triviaqa", fetch_triviaqa)):
        if a.source not in ("all", src):
            continue
        try:
            done.append(fn(load_dataset, a))
        except Exception as e:
            print(f"[{src}] FAILED: {type(e).__name__}: {e}")
            print(f"[{src}] download it by hand instead and drop the file "
                  f"anywhere under {RAW}")
        print()

    if not done:
        return 1
    print("next:  python scripts/00_prepare_data.py --n 800")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
