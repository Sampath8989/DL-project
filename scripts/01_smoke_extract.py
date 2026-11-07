"""Smoke test for stage 1: run the extraction path end to end.

Uses a randomly initialised tiny Llama so the pipeline can be exercised
without a pretrained checkpoint. It checks shapes and cache integrity,
not accuracy: a random model cannot answer anything.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.config import ExtractConfig
from src.extract import run
from src.tiny import build_tiny
from data.seed_questions import as_items, label


def main():
    model, tok = build_tiny(seed=0, hidden=64, layers=4)
    cfg = ExtractConfig(model_name="tiny-random-llama", device="cpu",
                        dtype="float32", max_new_tokens=8, max_items=12,
                        max_traj_tokens=8, out_dir="out/smoke")
    path = run(as_items(), cfg, label, model=model, tok=tok)
    z = np.load(path, allow_pickle=True)
    print("cache written to", path)
    for k in ("h_prompt", "h_answer", "traj", "mask", "conf", "y"):
        print(f"  {k:9s} {tuple(z[k].shape)}  {z[k].dtype}")
    print("  label balance:", dict(n=len(z["y"]), n_pos=int(z["y"].sum())))
    print("  tasks present:", sorted(set(z["task"].tolist())))
    assert z["h_prompt"].ndim == 3 and z["traj"].ndim == 4
    assert np.isfinite(z["h_prompt"]).all()
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
