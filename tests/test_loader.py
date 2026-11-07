"""Self-test for the dataset loader, runnable before the real downloads.

Writes fixture files shaped like PopQA and TriviaQA into a temporary
folder, points HALLU_DATA at it, and checks that the loader finds the
right columns and builds a working set. Nothing it writes can be
mistaken for real data: it lives in a temp directory and is deleted.

    python tests/test_loader.py
"""
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def write_fixtures(d):
    rng = np.random.default_rng(0)
    rows = []
    for i in range(400):
        rows.append(dict(
            id=i, subj=f"Entity{i}", prop="occupation", obj=f"Answer{i}",
            s_wiki_title=f"Entity{i}", s_pop=int(10 ** rng.uniform(1.5, 5.5)),
            question=f"What is the occupation of Entity{i}?",
            possible_answers=json.dumps([f"Answer{i}", f"answer{i} alt"])))
    pd.DataFrame(rows).to_csv(d / "popQA.tsv", sep="\t", index=False)

    with open(d / "triviaqa_validation.jsonl", "w") as f:
        for i in range(300):
            f.write(json.dumps(dict(
                question_id=f"q{i}",
                question=f"Trivia question number {i}?",
                answer=dict(value=f"Gold{i}", aliases=[f"Gold{i}", f"gold {i}"],
                            normalized_aliases=[f"gold{i}"]))) + "\n")


def main():
    tmp = Path(tempfile.mkdtemp(prefix="hallu_fixture_"))
    os.environ["HALLU_DATA"] = str(tmp)
    try:
        write_fixtures(tmp)
        # imported after HALLU_DATA is set, so paths picks it up
        from data.datasets import build_working_set, label

        checks = 0
        for src, n in (("popqa", 120), ("triviaqa", 100), ("contamination", 50)):
            items = build_working_set(src, n=n, seed=0)
            assert len(items) == n, f"{src}: expected {n}, got {len(items)}"
            assert all(it["gold"] for it in items), f"{src}: empty gold answer"
            assert all(it["prompt"].endswith("Answer:") for it in items)
            checks += 1

        pq = build_working_set("popqa", n=200, seed=0, verbose=False)
        assert all(np.isfinite(it["log_pop"]) for it in pq), \
            "popqa should carry popularity"
        assert len({it["pop_bucket"] for it in pq}) > 1, \
            "stratification should span more than one popularity band"

        item = dict(gold=["Paris"])
        assert label(item, "The capital is Paris.") == 0
        assert label(item, "It is Berlin.") == 1
        assert label(dict(gold=["Mars"]), "MARS") == 0, "match is case-insensitive"

        print(f"loader self-test passed ({checks} sources, "
              f"stratification and labelling verified)")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
