"""Names and orientations of the baseline features. No heavy imports.

Separated from `src/baselines.py` so that evaluation, which runs on a
CPU with nothing but numpy and sklearn, does not have to import torch to
find out what a column is called or which way it points.

Orientation
    +1  a larger value leans towards hallucinated
    -1  a larger value leans towards correct

It is declared here, once, before any AUROC is computed. Deciding a
baseline's sign after seeing its score is how a baseline gets quietly
flipped from 0.38 to 0.62.
"""
BASE_ORIENT = {
    "mean_logprob_tf": -1,
    "min_logprob_tf": -1,
    "mean_entropy": +1,
    "max_entropy": +1,
    "p_true": -1,
    "semantic_entropy": +1,
    "self_consistency": +1,     # stored as disagreement with the greedy answer
    "distinct_rate": +1,
}
BASE_NAMES = list(BASE_ORIENT)

# the columns extraction already writes, for completeness of the ladder
EXTRACT_ORIENT = {
    "mean_logprob": -1,
    "min_logprob": -1,
    "n_new_tokens": +1,
    "answer_chars": +1,
    "log_popularity": -1,       # obscure entities are hallucinated more
}

ALL_ORIENT = {**EXTRACT_ORIENT, **BASE_ORIENT}
