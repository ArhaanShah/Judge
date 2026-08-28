from __future__ import annotations

import argparse
from dataclasses import replace
from typing import Any, Iterable

from .prompts import load_prompt_template, render_prompt
from .schemas import BaseItem, CandidateOrder, Condition, Position, Section
from .utils import load_config, read_jsonl, require_mapping, token_count, write_jsonl


POSITIONS: tuple[Position, ...] = ("early", "middle", "late")
ORDERS: tuple[CandidateOrder, ...] = ("correct_first", "flawed_first")
DEFAULT_NEUTRAL_PADDING = (
    "Relevant supporting detail."
)


def target_index(section_count: int, fraction: float) -> int:
    if section_count < 1:
        raise ValueError("section_count must be positive")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("position fraction must be between zero and one")
    return min(section_count - 1, max(0, round((section_count - 1) * fraction)))


def relocate(
    sections: tuple[Section, ...], section_id: str, index: int
) -> tuple[Section, ...]:
    selected = next(section for section in sections if section.section_id == section_id)
    remaining = [section for section in sections if section.section_id != section_id]
    remaining.insert(index, selected)
    return tuple(remaining)


def render_candidate(sections: Iterable[Section]) -> str:
    return "\n\n".join(
        f"Section {number}\n{section.text}"
        for number, section in enumerate(sections, start=1)
    )


def _replace_target(
    sections: tuple[Section, ...], item: BaseItem
) -> tuple[Section, ...]:
    source_text = next(
        section.text
        for section in item.correct_sections
        if section.section_id == item.flawed_section_id
    )
    return tuple(
        replace(
            section,
            text=item.flawed_section_text + section.text[len(source_text) :],
        )
        if section.section_id == item.flawed_section_id
        else section
        for section in sections
    )


def _padded_sections(
    item: BaseItem,
    *,
    target_tokens: int,
    prompt_template: str,
    enabled: bool,
    tolerance: int,
    neutral_padding: str,
    model: str | None,
) -> tuple[Section, ...]:
    sections = item.correct_sections
    if not enabled:
        return sections

    def with_repetitions(repetitions: int) -> tuple[Section, ...]:
        suffix = (" " + neutral_padding) * repetitions
        return tuple(replace(section, text=section.text + suffix) for section in sections)

    def prompt_size(repetitions: int) -> int:
        clean = with_repetitions(repetitions)
        flawed = _replace_target(clean, item)
        return token_count(
            render_prompt(
                prompt_template,
                render_candidate(clean),
                render_candidate(flawed),
            ),
            model,
        )

    natural_size = prompt_size(0)
    if natural_size > target_tokens + tolerance:
        raise ValueError(
            f"{item.item_id} naturally renders to {natural_size} tokens, "
            f"above target {target_tokens} + tolerance {tolerance}"
        )
    if natural_size >= target_tokens - tolerance:
        return sections
    low, high = 0, 1
    while prompt_size(high) < target_tokens:
        high *= 2
    while low + 1 < high:
        midpoint = (low + high) // 2
        if prompt_size(midpoint) < target_tokens:
            low = midpoint
        else:
            high = midpoint
    repetitions = min(
        (low, high), key=lambda value: abs(prompt_size(value) - target_tokens)
    )
    return with_repetitions(repetitions)


def build_item_conditions(
    item: BaseItem,
    config: dict[str, Any],
    prompt_template: str,
) -> list[Condition]:
    positions = require_mapping(config.get("positions"), "positions")
    padding = require_mapping(config.get("padding", {}), "padding")
    judge = require_mapping(config.get("judge", {}), "judge")
    context_lengths = config.get("context_lengths")
    if not isinstance(context_lengths, list) or not context_lengths:
        raise ValueError("context_lengths must be a non-empty list")
    configured_count = int(require_mapping(config.get("items"), "items")["section_count"])
    if item.section_count != configured_count:
        raise ValueError(
            f"{item.item_id} has {item.section_count} sections; expected {configured_count}"
        )

    output: list[Condition] = []
    model = str(judge.get("model", "")) or None
    for context_value in context_lengths:
        context = require_mapping(context_value, "context length")
        context_name = str(context["name"])
        target_tokens = int(context["target_tokens"])
        canonical = _padded_sections(
            item,
            target_tokens=target_tokens,
            prompt_template=prompt_template,
            enabled=bool(padding.get("enabled", False)),
            tolerance=int(padding.get("target_tolerance_tokens", 0)),
            neutral_padding=str(
                padding.get("neutral_text", DEFAULT_NEUTRAL_PADDING)
            ),
            model=model,
        )
        for position in POSITIONS:
            index = target_index(
                len(canonical), float(positions[f"{position}_fraction"])
            )
            correct_sections = relocate(canonical, item.flawed_section_id, index)
            flawed_sections = _replace_target(correct_sections, item)
            correct_text = render_candidate(correct_sections)
            flawed_text = render_candidate(flawed_sections)
            correct_id = f"{item.item_id}_correct_{position}_{context_name}"
            flawed_id = f"{item.item_id}_flawed_{position}_{context_name}"
            target_prefix = render_candidate(correct_sections[:index])
            target_depth = token_count(target_prefix, model) / max(
                1, token_count(correct_text, model)
            )
            for order in ORDERS:
                if order == "correct_first":
                    a_id, b_id = correct_id, flawed_id
                    a_text, b_text, display_winner = correct_text, flawed_text, "A"
                else:
                    a_id, b_id = flawed_id, correct_id
                    a_text, b_text, display_winner = flawed_text, correct_text, "B"
                full_prompt = render_prompt(prompt_template, a_text, b_text)
                output.append(
                    Condition(
                        condition_id=(
                            f"{item.item_id}__{context_name}__{position}__{order}"
                        ),
                        item_id=item.item_id,
                        context_length=context_name,
                        target_tokens=target_tokens,
                        actual_prompt_tokens=token_count(full_prompt, model),
                        internal_position=position,
                        target_index=index,
                        target_depth=target_depth,
                        candidate_order=order,
                        correct_candidate_id=correct_id,
                        flawed_candidate_id=flawed_id,
                        candidate_a_id=a_id,
                        candidate_b_id=b_id,
                        candidate_a_text=a_text,
                        candidate_b_text=b_text,
                        candidate_a_tokens=token_count(a_text, model),
                        candidate_b_tokens=token_count(b_text, model),
                        section_order=tuple(
                            section.section_id for section in correct_sections
                        ),
                        correct_sections=correct_sections,
                        flawed_sections=flawed_sections,
                        gold_display_winner=display_winner,
                    )
                )
    return output


def build_conditions(
    config: dict[str, Any], prompt_template: str
) -> list[Condition]:
    paths = require_mapping(config.get("paths"), "paths")
    items = [
        BaseItem.from_dict(row) for row in read_jsonl(paths["source_items"])
    ]
    ids = [item.item_id for item in items]
    if len(ids) != len(set(ids)):
        raise ValueError("source item IDs must be unique")
    target_count = int(
        require_mapping(config.get("items"), "items")["target_count"]
    )
    if len(items) != target_count:
        raise ValueError(f"expected {target_count} source items, found {len(items)}")
    conditions: list[Condition] = []
    for item in items:
        conditions.extend(build_item_conditions(item, config, prompt_template))
    return conditions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build matched position-swap conditions"
    )
    parser.add_argument("--config", default="configs/pilot.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    paths = require_mapping(config.get("paths"), "paths")
    prompt_template = load_prompt_template(paths["prompt_template"])
    conditions = build_conditions(config, prompt_template)
    write_jsonl(
        paths["conditions"], (condition.to_dict() for condition in conditions)
    )
    print(
        f"Wrote {len(conditions)} deterministic conditions "
        f"to {paths['conditions']}"
    )


if __name__ == "__main__":
    main()
