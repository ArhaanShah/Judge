from __future__ import annotations

from collections import Counter

from conftest import make_config, make_item
from src.build_conditions import build_item_conditions, target_index


TEMPLATE = "A:\n{candidate_a_text}\nB:\n{candidate_b_text}"


def test_builds_six_matched_conditions_per_context() -> None:
    item = make_item()
    rows = build_item_conditions(item, make_config(), TEMPLATE)
    assert len(rows) == 6
    assert Counter(row.candidate_order for row in rows) == {
        "correct_first": 3,
        "flawed_first": 3,
    }


def test_target_depth_and_permutation_invariants() -> None:
    item = make_item()
    rows = build_item_conditions(item, make_config(), TEMPLATE)
    expected = {"early": 1, "middle": 4, "late": 8}
    source_ids = {section.section_id for section in item.correct_sections}
    position_orders = {}
    for row in rows:
        assert row.target_index == expected[row.internal_position]
        assert row.target_index == target_index(
            item.section_count,
            make_config()["positions"][f"{row.internal_position}_fraction"],
        )
        assert row.section_order[row.target_index] == item.flawed_section_id
        assert set(row.section_order) == source_ids
        differences = [
            clean.section_id
            for clean, flawed in zip(
                row.correct_sections, row.flawed_sections
            )
            if clean.text != flawed.text
        ]
        assert differences == [item.flawed_section_id]
        position_orders.setdefault(
            row.internal_position, row.section_order
        )
        assert position_orders[row.internal_position] == row.section_order


def test_position_variants_preserve_section_multisets() -> None:
    item = make_item()
    rows = [
        row
        for row in build_item_conditions(item, make_config(), TEMPLATE)
        if row.candidate_order == "correct_first"
    ]
    correct_sets = [
        sorted((section.section_id, section.text) for section in row.correct_sections)
        for row in rows
    ]
    flawed_sets = [
        sorted((section.section_id, section.text) for section in row.flawed_sections)
        for row in rows
    ]
    assert correct_sets[0] == correct_sets[1] == correct_sets[2]
    assert flawed_sets[0] == flawed_sets[1] == flawed_sets[2]
