"""Move aside caches whose decoding no longer matches what we pin.

    python scripts/11_quarantine_stale.py            # report only
    python scripts/11_quarantine_stale.py --apply    # actually move them

A cache is stale when BOTH of these hold:

  1. its acts_meta.json has no "decoding" block, so it was extracted
     before src/decoding.py existed and therefore inherited whatever the
     checkpoint's generation_config.json said; and
  2. the decoding check for that model (scripts/10_check_decoding.py)
     found that pinning changes the answers.

Both conditions matter. Qwen2.5 ships a repetition_penalty, so its old
caches hold penalised answers and different labels: stale. Llama-3.2
ships none, so pinning changes nothing and its old caches are still
exactly true-greedy decoding: keep them. The decision is made from
measurement, not from an assumption about which family is affected.

Nothing is deleted. Stale caches move to out/_quarantine_predecoding/,
which keeps them available for the before-and-after comparison -- the
size of this bug is itself a result worth reporting -- and out of the
glob that the analysis scripts walk.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.paths import OUT

QUARANTINE = OUT / "_quarantine_predecoding"


def slug(s):
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)


def load_checks():
    """model slug -> decoding check result."""
    out = {}
    for f in OUT.glob("decoding_check__*.json"):
        try:
            d = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        out[slug(d["model"])] = d
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--apply", action="store_true")
    a = p.parse_args()

    checks = load_checks()
    if not checks:
        print("no decoding_check__*.json under out/. Run "
              "scripts/10_check_decoding.py for each model first; without "
              "it there is no evidence either way and nothing is moved.")
        return 1

    stale, keep, unknown = [], [], []
    for d in sorted(OUT.glob("*__*")):
        if not (d / "acts_meta.json").exists():
            continue
        mslug, _, dataset = d.name.rpartition("__")
        try:
            meta = json.loads((d / "acts_meta.json").read_text())
        except json.JSONDecodeError:
            unknown.append((d, "acts_meta.json is unreadable"))
            continue
        pinned = meta.get("decoding") is not None
        chk = checks.get(mslug)

        if pinned:
            keep.append((d, "extracted with decoding pinned"))
        elif chk is None:
            unknown.append((d, "no decoding check for this model yet"))
        elif chk.get("answers_changed"):
            stale.append((d, f"predates pinning and pinning changes "
                             f"{chk['answers_changed']}/{chk['n']} answers"))
        else:
            keep.append((d, "predates pinning, but pinning changes no "
                            "answers for this model"))

    for d, why in keep:
        print(f"  keep       {d.name}: {why}")
    for d, why in unknown:
        print(f"  unknown    {d.name}: {why}")
    for d, why in stale:
        print(f"  STALE      {d.name}: {why}")

    if not stale:
        print("\nnothing to quarantine")
        return 0

    print(f"\n{len(stale)} cache(s) must be re-extracted.")
    if not a.apply:
        print("this was a report; pass --apply to move them")
        return 0

    QUARANTINE.mkdir(parents=True, exist_ok=True)
    for d, _ in stale:
        dest = QUARANTINE / d.name
        if dest.exists():
            n = 2
            while (QUARANTINE / f"{d.name}.{n}").exists():
                n += 1
            dest = QUARANTINE / f"{d.name}.{n}"
        shutil.move(str(d), str(dest))
        print(f"  moved {d.name} -> {dest.relative_to(OUT)}")
    print(f"\n{len(stale)} moved. Nothing was deleted; they are kept for the "
          f"before-and-after comparison.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
