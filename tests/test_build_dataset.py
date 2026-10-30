import pandas as pd
import pytest

from src.build_dataset import (build_dpo_dataset, extract_pairs, filter_answer_length,
                               jaccard_similarity, split_by_prompt)


def toy_df():
    rows = []
    # 4 prompts x 4 answers, distinct wording so Jaccard never triggers
    for q in range(4):
        for k, score in enumerate([1.0, 2.0, 3.5, 4.5]):
            rows.append({"question": f"q{q}", "model_name": f"m{k}",
                         "average_rank_os": score,
                         "answer": " ".join(f"w{q}_{k}_{i}" for i in range(10))})
    return pd.DataFrame(rows)


def test_jaccard():
    assert jaccard_similarity("a b c", "a b c") == 1.0
    assert jaccard_similarity("a b", "c d") == 0.0
    assert jaccard_similarity("", "a") == 0.0


def test_length_filter():
    df = pd.DataFrame({"answer": ["hi", "one two three", "w " * 3000]})
    assert len(filter_answer_length(df)) == 1


def test_best_worst_picks_extremes():
    pairs = extract_pairs(toy_df(), "best_worst", min_delta=0.3)
    assert len(pairs) == 4 and (pairs["delta"] == 3.5).all()
    assert pairs.iloc[0]["chosen"][0]["content"].startswith("w0_3")  # score 4.5
    assert pairs.iloc[0]["rejected"][0]["content"].startswith("w0_0")  # score 1.0


def test_pairwise_threshold_and_cap():
    all_pairs = extract_pairs(toy_df(), "pairwise", min_delta=0.0)
    assert len(all_pairs) == 4 * 6  # C(4,2) per prompt
    strict = extract_pairs(toy_df(), "pairwise", min_delta=1.5)
    assert (strict["delta"] >= 1.5).all() and len(strict) < len(all_pairs)
    capped = extract_pairs(toy_df(), "pairwise", min_delta=0.0, max_pairs_per_prompt=2)
    assert len(capped) == 8
    # cap keeps the largest gaps
    assert sorted(capped[capped.raw_question == "q0"]["delta"]) == [2.5, 3.5]


def test_similarity_filter_removes_near_duplicates():
    df = pd.DataFrame({"question": ["q"] * 2, "model_name": ["a", "b"],
                       "average_rank_os": [1.0, 5.0],
                       "answer": ["same words here again"] * 2})
    assert extract_pairs(df, "best_worst").empty


def test_split_is_by_prompt_and_deterministic():
    pairs = extract_pairs(toy_df(), "pairwise", min_delta=0.0)
    tr1, te1 = split_by_prompt(pairs, 0.25, 42)
    tr2, te2 = split_by_prompt(pairs, 0.25, 42)
    assert set(tr1.raw_question).isdisjoint(set(te1.raw_question))
    assert list(te1.raw_question.unique()) == list(te2.raw_question.unique())
    assert len(tr1) + len(te1) == len(pairs)


def test_end_to_end_writes_files(tmp_path):
    out = tmp_path / "ds.parquet"
    train, test = build_dpo_dataset(toy_df(), out, strategy="pairwise", min_delta=0.5,
                                    test_size=0.25)
    for suffix in ("_with_delta", "_train", "_test"):
        assert (tmp_path / f"ds{suffix}.parquet").exists()
    assert list(train.columns) == ["prompt", "chosen", "rejected"]
    assert len(train) + len(test) > 0
