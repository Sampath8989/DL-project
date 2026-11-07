"""Path resolution that works the same on Windows and Linux.

Nothing in this project stores an absolute path. Everything is resolved
relative to the repository root, which is found by walking up from this
file, so the code runs identically from the project folder, from an IDE,
or from a different drive.

Override the data location with the HALLU_DATA environment variable if
the datasets live somewhere other than <repo>/data/raw.
"""
import os
from pathlib import Path

# src/paths.py -> src -> repo root
ROOT = Path(__file__).resolve().parent.parent

DATA = ROOT / "data"
RAW = Path(os.environ.get("HALLU_DATA", DATA / "raw"))
OUT = ROOT / "out"
CACHE = OUT / "acts"
LOGS = ROOT / "logs"


def ensure_dirs():
    for p in (DATA, RAW, OUT, CACHE, LOGS):
        p.mkdir(parents=True, exist_ok=True)


def find(*patterns, root=None, required=True):
    """Return the first file under `root` matching any glob pattern.

    Searches case-insensitively and recursively, so it does not matter
    whether a download landed as PopQA.tsv, popqa.tsv or
    raw/popqa/test.tsv.
    """
    root = Path(root) if root else RAW
    if not root.exists():
        if required:
            raise FileNotFoundError(
                f"{root} does not exist. Create it and put the downloaded "
                f"dataset files there, or set HALLU_DATA to point at them.")
        return None
    lowered = [p.lower() for p in patterns]
    hits = []
    for f in root.rglob("*"):
        if not f.is_file():
            continue
        name = f.name.lower()
        rel = str(f.relative_to(root)).lower().replace("\\", "/")
        for pat in lowered:
            if Path(name).match(pat) or Path(rel).match(pat):
                hits.append(f)
                break
    if not hits:
        if required:
            raise FileNotFoundError(
                f"No file under {root} matched {patterns}. "
                f"Found instead: {sorted(p.name for p in root.rglob('*') if p.is_file())[:20]}")
        return None
    # prefer the largest match: test splits beat tiny sample files
    return max(hits, key=lambda p: p.stat().st_size)


if __name__ == "__main__":
    ensure_dirs()
    print("repo root :", ROOT)
    print("raw data  :", RAW, "(exists)" if RAW.exists() else "(missing)")
    if RAW.exists():
        files = sorted(p for p in RAW.rglob("*") if p.is_file())
        print(f"{len(files)} file(s) found:")
        for p in files[:40]:
            print(f"   {p.relative_to(RAW)}  {p.stat().st_size/1e6:.2f} MB")
