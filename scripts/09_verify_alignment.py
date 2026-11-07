"""Do the caches on disk still line up with the working sets?

    python scripts/09_verify_alignment.py

Extraction writes acts_meta.json in working-set order, so cached item i
must be working-set item i. Every label, every baseline and every probe
score downstream assumes it. If the working set were ever rebuilt with a
different sample or a different row order -- a re-run of
00_prepare_data.py after a pandas upgrade would do it -- the labels in
acts.npz would quietly belong to different questions, and nothing else in
the pipeline would notice.

This check costs a second and is the reason the Review 3 driver verifies
the working sets instead of regenerating them.

Exit 0 if every cache lines up, 1 otherwise.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.paths import DATA, OUT
from data.datasets import load_working_set


VERDICT = OUT / "alignment_verdict.json"


def write_verdict(ok, checked, bad, note=""):
    """Record the verdict in a file, not only in an exit code.

    The driver gates the whole run on this check, and an exit code read
    back through a PowerShell pipeline turned out not to be trustworthy:
    this script printed "all 3 cache(s) line up" and the driver still
    halted the run on a misread status. A gate that can fail closed on a
    false alarm can also fail open on a real mismatch, so the verdict is
    written where it can be read unambiguously.
    """
    OUT.mkdir(parents=True, exist_ok=True)
    VERDICT.write_text(json.dumps(dict(
        ok=bool(ok), checked=int(checked),
        bad=[{"cache": n, "why": m} for n, m in bad],
        note=note, when=time.strftime("%Y-%m-%d %H:%M:%S")), indent=2))


def main():
    caches = sorted(p for p in OUT.glob("*__*") if (p / "acts_meta.json").exists())
    if not caches:
        print("no caches on disk yet; nothing to verify")
        write_verdict(True, 0, [], "no caches to verify")
        return 0

    ws_cache = {}
    bad, ok = [], []
    for d in caches:
        model, _, dataset = d.name.rpartition("__")
        if dataset not in ws_cache:
            p = DATA / f"working_{dataset}.jsonl"
            ws_cache[dataset] = load_working_set(p) if p.exists() else None
        ws = ws_cache[dataset]
        if ws is None:
            bad.append((d.name, f"no working_{dataset}.jsonl to compare against"))
            continue

        meta = json.loads((d / "acts_meta.json").read_text())
        items = meta["items"]
        if len(items) > len(ws):
            bad.append((d.name, f"cache has {len(items)} items, working set "
                                f"only {len(ws)}"))
            continue

        mism = [i for i, it in enumerate(items)
                if it.get("question", "") != ws[i]["question"]]
        if mism:
            first = mism[0]
            bad.append((d.name,
                        f"{len(mism)}/{len(items)} questions differ; first at "
                        f"index {first}\n      cache: "
                        f"{items[first].get('question','')!r}\n      "
                        f"working set: {ws[first]['question']!r}"))
            continue

        # the labels must also match what the current rule says about the
        # cached answer text, or the labelling rule changed after extraction
        note = ""
        z = d / "acts.npz"
        if z.exists():
            from data.datasets import label
            y = np.load(z, allow_pickle=True)["y"]
            relabel = np.array([label(ws[i], items[i].get("text", ""))
                                for i in range(len(items))])
            diff = int((y[: len(items)] != relabel).sum())
            if diff:
                bad.append((d.name, f"{diff}/{len(items)} labels disagree with "
                                    f"the current labelling rule applied to the "
                                    f"cached answers. The rule changed after "
                                    f"extraction; re-extract or revert it."))
                continue
            note = f", {len(items)} labels reproduce"
        ok.append((d.name, f"{len(items)} items in order{note}"))

    for name, msg in ok:
        print(f"  ok    {name}: {msg}")
    for name, msg in bad:
        print(f"  BAD   {name}: {msg}")

    if bad:
        print(f"\n{len(bad)} cache(s) do not line up with the working sets.")
        print("Do NOT run the analysis on these. Either restore the working "
              "set they were built from, or delete the cache and re-extract.")
        write_verdict(False, len(ok) + len(bad), bad)
        print(f"wrote {VERDICT}")
        return 1
    print(f"\nall {len(ok)} cache(s) line up with their working sets")
    write_verdict(True, len(ok), [])
    print(f"wrote {VERDICT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
