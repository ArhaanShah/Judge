from __future__ import annotations

import pytest

from src.data.prmbench_adapter import UnknownCategoryError, filter_target_row, map_category


MAPPING = {"released_empirical": "Empirical Soundness"}


def valid_row() -> dict:
    return {
        "idx": 1,
        "classification": "released_empirical",
        "original_question": "What is 2 + 2?",
        "modified_question": "What is 2 + 2?",
        "original_process": "Compute two plus two to obtain four.",
        "modified_process": "Compute two plus two to obtain five.",
        "error_steps": [1],
        "modified_steps": ["Compute two plus two to obtain five."],
    }


def test_unknown_category_fails_closed() -> None:
    with pytest.raises(UnknownCategoryError):
        map_category("new-release-label", MAPPING)


def test_target_hard_filters_and_leakage() -> None:
    counter = lambda text: len(text.split())
    assert filter_target_row(valid_row(), MAPPING, token_counter=counter).eligible
    leaked = valid_row()
    leaked["modified_process"] += " classification: hidden"
    result = filter_target_row(leaked, MAPPING, token_counter=counter)
    assert not result.eligible
    assert "metadata_leakage" in result.reasons


def test_relative_length_filter_is_not_relaxed_by_absolute_limit() -> None:
    row = valid_row()
    row["modified_process"] = "five"
    result = filter_target_row(row, MAPPING, token_counter=lambda text: len(text.split()), absolute_delta_limit=192)
    assert "relative_length_delta" in result.reasons
