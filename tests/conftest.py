from __future__ import annotations

from src.schemas import BaseItem, Section


def make_item(section_count: int = 10) -> BaseItem:
    sections = tuple(
        Section(f"s{index:02d}", f"Step {index} establishes value {index}.")
        for index in range(1, section_count + 1)
    )
    target = sections[section_count // 2]
    return BaseItem(
        item_id="item_0001",
        domain="arithmetic",
        error_type="calculation",
        section_count=section_count,
        correct_sections=sections,
        flawed_section_id=target.section_id,
        flawed_section_text="This step incorrectly establishes value 999.",
        gold_winner="correct",
        error_explanation="999 is not the required intermediate value.",
        gold_source="direct arithmetic verification",
    )


def make_config(section_count: int = 10) -> dict:
    return {
        "items": {"target_count": 1, "section_count": section_count},
        "positions": {
            "early_fraction": 0.10,
            "middle_fraction": 0.50,
            "late_fraction": 0.90,
            "tolerance_fraction": 0.20,
        },
        "context_lengths": [{"name": "tiny", "target_tokens": 256}],
        "padding": {"enabled": False, "target_tolerance_tokens": 20},
        "judge": {"model": "test-model"},
    }
