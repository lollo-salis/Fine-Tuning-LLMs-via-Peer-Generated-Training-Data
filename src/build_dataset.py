"""Build DPO preference datasets from AutoBench peer-assessment scores.

Input  : a CSV with (at least) the columns ``question``, ``answer``,
         ``average_rank_os`` and ``model_name`` -- one row per (prompt, answer).
         ``average_rank_os`` is the mean 1-5 score given to the answer by the
         open-source judge models only.
Output : parquet files with ``prompt``, ``chosen``, ``rejected`` columns
         (conversational format), split *by prompt* into train / test.

.. note::
   The AutoBench export used in the thesis was shared privately and is **not**
   distributed with this repository, so the resulting preference datasets are
   not available either. This module works on any CSV with the columns above;
   the unit tests use small synthetic data.

Two extraction strategies are supported:

* ``best_worst`` -- one pair per prompt: highest- vs lowest-scored answer.
* ``pairwise``   -- every ordered pair of answers per prompt, filtered by a
                    minimum score gap (``min_delta``), a maximum Jaccard
                    similarity and a cap on pairs per prompt.

Example
-------
    python -m src.build_dataset --input data/raw/autobench_run.csv \\
        --strategy pairwise --min-delta 0.3 --max-pairs-per-prompt 20 \\
        --output data/processed/dpo_pairwise_d03.parquet
"""
from __future__ import annotations

import argparse
import logging
from itertools import combinations
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ["question", "answer", "average_rank_os"]


# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #
def fix_encoding(df: pd.DataFrame, columns=("question", "answer")) -> pd.DataFrame:
    """Repair mojibake (e.g. ``â€™`` -> ``'``) with `ftfy`."""
    import ftfy  # imported lazily: only needed for raw AutoBench exports

    df = df.copy()
    for col in columns:
        df[col] = df[col].astype(str).map(ftfy.fix_text)
    return df


def filter_answer_length(df: pd.DataFrame, min_words: int = 3, max_words: int = 2000,
                         col: str = "answer") -> pd.DataFrame:
    """Drop empty / extremely short answers and abnormally long generations."""
    n_words = df[col].astype(str).str.split().map(len)
    kept = df[(n_words >= min_words) & (n_words <= max_words)]
    logger.info("Length filter: dropped %d / %d rows (outside [%d, %d] words)",
                len(df) - len(kept), len(df), min_words, max_words)
    return kept.copy()


# --------------------------------------------------------------------------- #
# Pair extraction helpers
# --------------------------------------------------------------------------- #
def jaccard_similarity(text_a: str, text_b: str) -> float:
    """Jaccard similarity over the sets of lower-cased whitespace tokens."""
    a, b = set(str(text_a).lower().split()), set(str(text_b).lower().split())
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def format_dpo_example(question, chosen, rejected, conversational_format=True):
    """Return (prompt, chosen, rejected) in the chosen serialisation."""
    if conversational_format:
        return ([{"role": "user", "content": question}],
                [{"role": "assistant", "content": chosen}],
                [{"role": "assistant", "content": rejected}])
    prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
    return prompt, f"{chosen}<|im_end|>", f"{rejected}<|im_end|>"


# --------------------------------------------------------------------------- #
# Main builder
# --------------------------------------------------------------------------- #
def extract_pairs(df: pd.DataFrame, strategy: str = "best_worst", min_delta: float = 0.3,
                  similarity_threshold: float = 0.85, max_pairs_per_prompt: int | None = None,
                  conversational_format: bool = True) -> pd.DataFrame:
    """Turn scored answers into (prompt, chosen, rejected, delta) rows."""
    if strategy not in ("best_worst", "pairwise"):
        raise ValueError("strategy must be 'best_worst' or 'pairwise'")

    rows = []
    skipped = {"too_few_answers": 0, "delta": 0, "similarity": 0}

    for question, group in df.groupby("question"):
        if len(group) < 2:
            skipped["too_few_answers"] += 1
            continue

        if strategy == "best_worst":
            # Sorting on (score, answer) makes ties deterministic.
            ordered = group.sort_values(["average_rank_os", "answer"])
            worst, best = ordered.iloc[0], ordered.iloc[-1]
            delta = float(best["average_rank_os"] - worst["average_rank_os"])
            if delta < min_delta:
                skipped["delta"] += 1
                continue
            if jaccard_similarity(best["answer"], worst["answer"]) > similarity_threshold:
                skipped["similarity"] += 1
                continue
            p, c, r = format_dpo_example(question, best["answer"], worst["answer"],
                                         conversational_format)
            rows.append({"prompt": p, "chosen": c, "rejected": r,
                         "delta": delta, "raw_question": question})
            continue

        # pairwise
        prompt_pairs = []
        records = group[["answer", "average_rank_os"]].to_dict("records")
        for r1, r2 in combinations(records, 2):
            delta = abs(r1["average_rank_os"] - r2["average_rank_os"])
            if delta < min_delta:
                skipped["delta"] += 1
                continue
            chosen, rejected = (r1, r2) if r1["average_rank_os"] > r2["average_rank_os"] else (r2, r1)
            if jaccard_similarity(chosen["answer"], rejected["answer"]) > similarity_threshold:
                skipped["similarity"] += 1
                continue
            p, c, r = format_dpo_example(question, chosen["answer"], rejected["answer"],
                                         conversational_format)
            prompt_pairs.append({"prompt": p, "chosen": c, "rejected": r,
                                 "delta": float(delta), "raw_question": question})
        if max_pairs_per_prompt is not None:  # keep the most informative pairs
            prompt_pairs = sorted(prompt_pairs, key=lambda x: x["delta"], reverse=True)
            prompt_pairs = prompt_pairs[:max_pairs_per_prompt]
        rows.extend(prompt_pairs)

    logger.info("Skipped: %s", skipped)
    return pd.DataFrame(rows)


def split_by_prompt(pairs: pd.DataFrame, test_size: float = 0.1, random_state: int = 42):
    """Split so that every pair of a given prompt lands on the same side (no leakage)."""
    unique_q = pd.Series(pairs["raw_question"].unique())
    train_q = unique_q.sample(frac=1 - test_size, random_state=random_state)
    is_train = pairs["raw_question"].isin(train_q)
    return pairs[is_train].copy(), pairs[~is_train].copy()


def build_dpo_dataset(df: pd.DataFrame, output_parquet_path: str | Path,
                      strategy: str = "best_worst", min_delta: float = 0.3,
                      similarity_threshold: float = 0.85,
                      max_pairs_per_prompt: int | None = None,
                      min_answer_words: int = 3, max_answer_words: int = 2000,
                      conversational_format: bool = True, test_size: float = 0.1,
                      random_state: int = 42):
    """Full pipeline. Writes ``*_with_delta``, ``*_train`` and ``*_test`` parquet files."""
    df = df.dropna(subset=REQUIRED_COLUMNS).copy()
    df = filter_answer_length(df, min_answer_words, max_answer_words)

    per_prompt = df.groupby("question").size()
    logger.info("Unique prompts: %d | mean answers per prompt: %.2f | strategy: %s",
                len(per_prompt), per_prompt.mean(), strategy)

    pairs = extract_pairs(df, strategy, min_delta, similarity_threshold,
                          max_pairs_per_prompt, conversational_format)
    if pairs.empty:
        logger.warning("No pairs generated: lower min_delta or raise similarity_threshold.")
        return pd.DataFrame(), pd.DataFrame()

    logger.info("DPO triples: %d | mean gap %.3f | median gap %.3f",
                len(pairs), pairs["delta"].mean(), pairs["delta"].median())

    out = Path(output_parquet_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pairs.drop(columns=["raw_question"]).to_parquet(
        out.with_name(out.stem + "_with_delta.parquet"), index=False)

    train, test = split_by_prompt(pairs, test_size, random_state)
    drop = ["raw_question", "delta"]  # trainers only need prompt/chosen/rejected
    train.drop(columns=drop).to_parquet(out.with_name(out.stem + "_train.parquet"), index=False)
    test.drop(columns=drop).to_parquet(out.with_name(out.stem + "_test.parquet"), index=False)
    logger.info("Train: %d triples (%d prompts) | Test: %d triples (%d prompts)",
                len(train), train["raw_question"].nunique(),
                len(test), test["raw_question"].nunique())
    return train.drop(columns=drop), test.drop(columns=drop)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="AutoBench CSV export")
    ap.add_argument("--output", required=True, help="Output parquet path (suffixes are added)")
    ap.add_argument("--strategy", choices=["best_worst", "pairwise"], default="best_worst")
    ap.add_argument("--min-delta", type=float, default=0.3)
    ap.add_argument("--similarity-threshold", type=float, default=0.85)
    ap.add_argument("--max-pairs-per-prompt", type=int, default=None)
    ap.add_argument("--min-answer-words", type=int, default=3)
    ap.add_argument("--max-answer-words", type=int, default=2000)
    ap.add_argument("--test-size", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-fix-encoding", action="store_true", help="Skip the ftfy step")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    df = pd.read_csv(args.input, encoding="utf-8")
    if not args.no_fix_encoding:
        df = fix_encoding(df)
    build_dpo_dataset(df, args.output, args.strategy, args.min_delta,
                      args.similarity_threshold, args.max_pairs_per_prompt,
                      args.min_answer_words, args.max_answer_words,
                      test_size=args.test_size, random_state=args.seed)


if __name__ == "__main__":
    main()
