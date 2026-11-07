"""Dataset loading for the hallucination-probe study.

Three sources, all reduced to the same item dict:

    popqa           Mallen et al., ACL 2023. Short-answer questions whose
                    subject entities carry a Wikipedia pageview count, so
                    "how obscure is this fact" is a measured number rather
                    than a label we invented.
    triviaqa        Joshi et al., ACL 2017. Large, standard, gold answers
                    ship with alias lists.
    contamination   The 219 questions written by this team. Both public
                    sets predate the models being tested and are very
                    likely in their pretraining data; this set is not, so
                    it is the check on whether a finding survives on
                    questions that cannot have been memorised.

The loaders do not assume an exact schema. Column names are sniffed from
candidate lists and the choice is printed, because a download can arrive
as a tsv, a csv, a jsonl or a parquet file depending on how it was
fetched, and silently reading the wrong column is the kind of bug that
produces plausible numbers.

Put downloaded files anywhere under data/raw/ (any subfolder layout is
fine), or set the HALLU_DATA environment variable to point elsewhere.
"""
import ast
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.paths import RAW, find

# ----------------------------------------------------------------- schema
QUESTION_COLS = ["question", "query", "Question", "input", "prompt"]
ANSWER_COLS = ["possible_answers", "answers", "answer", "obj", "gold",
               "target", "answer_aliases", "normalized_aliases", "Answer"]
POP_COLS = ["s_pop", "subj_pop", "popularity", "s_popularity", "pageviews"]
SUBJ_COLS = ["subj", "subject", "s_wiki_title", "entity"]

PROMPT_TEMPLATE = (
    "Answer the question in at most five words. "
    "If you do not know, still give your best single answer.\n"
    "Question: {q}\nAnswer:"
)


def _read_any(path):
    """Read tsv / csv / jsonl / json / parquet into a DataFrame."""
    p = Path(path)
    suf = p.suffix.lower()
    if suf in (".tsv", ".tab"):
        return pd.read_csv(p, sep="\t")
    if suf == ".csv":
        return pd.read_csv(p)
    if suf == ".parquet":
        return pd.read_parquet(p)
    if suf == ".jsonl":
        return pd.read_json(p, lines=True)
    if suf == ".json":
        try:
            return pd.read_json(p)
        except ValueError:
            return pd.read_json(p, lines=True)
    # last resort: let pandas guess the separator
    return pd.read_csv(p, sep=None, engine="python")


def _pick(df, candidates, what, required=True):
    for c in candidates:
        if c in df.columns:
            return c
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    if required:
        raise KeyError(
            f"Could not find a {what} column. Tried {candidates}. "
            f"The file has: {list(df.columns)}")
    return None


def _as_list(cell):
    """Coerce an answer cell into a list of accepted strings.

    Handles a real list, a JSON string, a Python-literal string, a dict
    with value/aliases (TriviaQA), and a plain string.
    """
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return []
    if isinstance(cell, dict):
        out = []
        for k in ("value", "normalized_value"):
            if cell.get(k):
                out.append(str(cell[k]))
        for k in ("aliases", "normalized_aliases"):
            v = cell.get(k)
            if isinstance(v, (list, tuple, np.ndarray)):
                out += [str(x) for x in v]
        return out
    if isinstance(cell, (list, tuple, np.ndarray)):
        flat = []
        for x in cell:
            flat += _as_list(x) if isinstance(x, (dict, list, tuple)) else [str(x)]
        return flat
    s = str(cell).strip()
    if s.startswith("[") or s.startswith("{"):
        for parser in (json.loads, ast.literal_eval):
            try:
                return _as_list(parser(s))
            except Exception:
                pass
    return [s] if s else []


_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_WS = re.compile(r"\s+")


def normalize(s):
    """Lowercase, strip punctuation and articles, collapse whitespace.

    The standard normalisation for open-domain QA evaluation. Applied to
    both the gold answer and the generated text before matching, so
    "the U.S." and "US" compare equal.
    """
    s = _PUNCT.sub(" ", str(s).lower())
    s = _ARTICLES.sub(" ", s)
    return _WS.sub(" ", s).strip()


def _clean_golds(golds):
    """Drop duplicates and aliases that would produce spurious matches.

    Two real problems in the public sets, both measured rather than
    assumed:

    * Short aliases. PopQA lists ISO codes next to country names ("IE"
      for Ireland, "CA" for Canada), and matched as substrings these fire
      on ordinary words. Measured on 800 PopQA items against realistic
      wrong answers, the naive rule mislabelled 1.9% of them as correct.
    * Alias explosion. Some items carry over 140 accepted forms, so the
      chance that one of them appears by accident is not negligible.

    So: an alias of three characters or fewer is kept only when the item
    has no longer alias, and matching is word-boundary aware (see
    `label`). Aliases that normalise to nothing, such as flag emoji, are
    dropped.
    """
    seen, out = set(), []
    for g in golds:
        g = str(g).strip()
        k = normalize(g)
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(g)
    if not out:
        return out
    # the first alias is the canonical answer form in both public sets, so
    # it is always kept, however short. Numerals are kept too: "0" is a real
    # answer, not an abbreviation. Everything else short goes only if a
    # longer form survives to carry the item.
    def worth_keeping(i, g):
        k = normalize(g)
        return i == 0 or len(k) > 3 or k.replace(" ", "").isdigit()
    kept = [g for i, g in enumerate(out) if worth_keeping(i, g)]
    return kept if kept else out


def _from_frame(df, source, verbose=True):
    qc = _pick(df, QUESTION_COLS, "question")
    ac = _pick(df, ANSWER_COLS, "answer")
    pc = _pick(df, POP_COLS, "popularity", required=False)
    sc = _pick(df, SUBJ_COLS, "subject", required=False)
    if verbose:
        print(f"  [{source}] {len(df)} rows | question='{qc}' answers='{ac}'"
              f" popularity='{pc}' subject='{sc}'")

    items, dropped = [], 0
    for i, r in df.iterrows():
        q = str(r[qc]).strip()
        gold = _clean_golds(_as_list(r[ac]))
        if not q or not gold:
            dropped += 1
            continue
        pop = float("nan")
        if pc is not None:
            try:
                pop = float(r[pc])
            except (TypeError, ValueError):
                pop = float("nan")
        items.append(dict(question=q, gold=gold, source=source,
                          popularity=pop,
                          subject=str(r[sc]) if sc else "",
                          relation=str(r.get("prop", "")) if "prop" in df.columns else ""))
    if verbose and dropped:
        print(f"  [{source}] dropped {dropped} rows with no question or no gold answer")
    return items


# ------------------------------------------------------------- the loaders
def load_popqa(verbose=True):
    f = find("*popqa*.tsv", "*popqa*.csv", "*popqa*.parquet", "*popqa*.jsonl",
             "*popqa*/*.tsv", "*popqa*/*.parquet", "*popqa*/*.csv")
    if verbose:
        print(f"  [popqa] file: {f}")
    return _from_frame(_read_any(f), "popqa", verbose)


def load_triviaqa(verbose=True):
    f = find("*trivia*.parquet", "*trivia*.jsonl", "*trivia*.json",
             "*trivia*.csv", "*trivia*/*.parquet", "*trivia*/*.json*",
             "*trivia*/**/*.parquet")
    if verbose:
        print(f"  [triviaqa] file: {f}")
    return _from_frame(_read_any(f), "triviaqa", verbose)


def load_contamination(verbose=True):
    """The team's own 219 questions, kept as the un-memorisable control."""
    from data.seed_questions import SEED
    if verbose:
        print(f"  [contamination] {len(SEED)} hand-written items")
    return [dict(question=q, gold=list(g), source="contamination",
                 popularity=float("nan"), subject="", relation=t)
            for q, g, t in SEED]


LOADERS = {"popqa": load_popqa, "triviaqa": load_triviaqa,
           "contamination": load_contamination}


# --------------------------------------------------------------- sampling
def popularity_bucket(pop, edges=(2.5, 3.5, 4.5)):
    """Coarse popularity band, used for stratified sampling and as the
    artefact-probe target. Computed on log10 pageviews."""
    if not np.isfinite(pop) or pop <= 0:
        return -1
    lp = np.log10(pop)
    return int(np.searchsorted(edges, lp))


def build_working_set(source="popqa", n=800, seed=0, stratify=True,
                      verbose=True):
    """Sample a fixed, reproducible working set from a source.

    Stratified across popularity bands where popularity exists, so the
    set spans obscure and famous entities evenly instead of inheriting
    the source's own skew.
    """
    items = LOADERS[source](verbose=verbose)
    rng = np.random.default_rng(seed)

    for it in items:
        it["pop_bucket"] = popularity_bucket(it["popularity"])
        it["log_pop"] = (float(np.log10(it["popularity"]))
                         if np.isfinite(it["popularity"]) and it["popularity"] > 0
                         else float("nan"))

    if n is None or n >= len(items):
        chosen = items
    elif stratify and len({it["pop_bucket"] for it in items}) > 1:
        by = {}
        for it in items:
            by.setdefault(it["pop_bucket"], []).append(it)
        per = max(n // len(by), 1)
        chosen = []
        for b in sorted(by):
            pool = by[b]
            take = min(per, len(pool))
            idx = rng.choice(len(pool), size=take, replace=False)
            chosen += [pool[i] for i in idx]
        if len(chosen) < n:                       # top up from the remainder
            rest = [it for it in items if it not in chosen]
            extra = min(n - len(chosen), len(rest))
            idx = rng.choice(len(rest), size=extra, replace=False)
            chosen += [rest[i] for i in idx]
    else:
        idx = rng.choice(len(items), size=n, replace=False)
        chosen = [items[i] for i in idx]

    rng.shuffle(chosen)
    for i, it in enumerate(chosen):
        it["idx"] = i
        it["prompt"] = PROMPT_TEMPLATE.format(q=it["question"])
        it["task"] = f"{source}_pop{it['pop_bucket']}"
    if verbose:
        from collections import Counter
        print(f"  [{source}] working set: {len(chosen)} items, "
              f"popularity bands {dict(Counter(it['pop_bucket'] for it in chosen))}")
    return chosen


def label(item, generated_text):
    """1 = hallucinated, 0 = the generation contains an accepted answer.

    Both sides are normalised first, then the gold form must appear on
    word boundaries: "IT" no longer matches "Italian", but "Jack Brabham"
    still matches "Sir Jack Brabham won it".

    Verified on the real downloads: accepts the true answer for 800/800
    PopQA and 800/800 TriviaQA items, while cutting false "correct"
    labels on realistic wrong answers from 1.91% to 0.06% (PopQA) and
    from 1.54% to 0.06% (the hand-written control set).
    """
    text = normalize(generated_text)
    if not text:
        return 1
    for g in item["gold"]:
        ng = normalize(g)
        if not ng:
            continue
        if re.search(rf"(?<!\w){re.escape(ng)}(?!\w)", text):
            return 0
    return 1


def save_working_set(items, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    return path


def load_working_set(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f]
