from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
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


def render_answer_sheet(sections: list[dict[str, str]]) -> str:
    """Serialize PRMBench sections without exposing source annotations."""
    return "\n\n".join(
        f"Problem {number}\nQuestion: {section['question']}\nSolution: {section['solution']}"
        for number, section in enumerate(sections, start=1)
    )


def _exact_midpoint(sections: list[dict[str, str]], index: int, counter) -> float:
    blocks = [f"Problem {number}\nQuestion: {section['question']}\nSolution: {section['solution']}" for number, section in enumerate(sections, start=1)]
    start_text = "\n\n".join(blocks[:index]) + ("\n\n" if index else "")
    end_text = "\n\n".join(blocks[:index + 1])
    return (counter(start_text) + counter(end_text)) / 2 / max(counter("\n\n".join(blocks)), 1)


def choose_target_insertion(
    fillers: list[dict[str, str]],
    target: dict[str, str],
    band: dict[str, Any],
    counter,
) -> tuple[list[dict[str, str]], float]:
    """Choose the insertion whose token midpoint is nearest the frozen target."""
    target_fraction = float(band["target"])
    weights = [counter(render_answer_sheet([section])) + 2 for section in fillers]
    target_weight = counter(render_answer_sheet([target]))
    total_weight = sum(weights) + target_weight
    cumulative = 0
    ranked: list[tuple[float, int]] = []
    for index in range(len(fillers) + 1):
        estimate = (cumulative + target_weight / 2) / max(total_weight, 1)
        ranked.append((abs(estimate - target_fraction), index))
        if index < len(weights):
            cumulative += weights[index]
    exact_choices: list[tuple[float, int, list[dict[str, str]], float]] = []
    for _, index in sorted(ranked):
        sections = fillers[:index] + [target] + fillers[index:]
        midpoint = _exact_midpoint(sections, index, counter)
        if float(band["min"]) <= midpoint <= float(band["max"]):
            exact_choices.append((abs(midpoint - target_fraction), index, sections, midpoint))
            # Adjacent estimates are monotonic; checking several valid points is
            # sufficient to find the exact closest candidate without repeatedly
            # tokenizing every full 16K permutation.
            if len(exact_choices) >= 4:
                break
    if not exact_choices:
        closest_index = min(ranked)[1]
        closest_sections = fillers[:closest_index] + [target] + fillers[closest_index:]
        midpoint = _exact_midpoint(closest_sections, closest_index, counter)
        raise ValueError(f"cannot place target in declared band {band}; closest midpoint={midpoint:.4f}")
    _, _, sections, midpoint = min(exact_choices, key=lambda value: (value[0], value[1]))
    return sections, midpoint


def target_midpoint_at(sections: list[dict[str, str]], index: int, counter) -> float:
    return _exact_midpoint(sections, index, counter)


def _candidate_counter(model_name: str):
    """Load tokenizer artifacts only; never load model weights."""
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("install the 'data' extra for the configured Cohere tokenizer") from exc
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return lambda text: len(tokenizer.encode(text, add_special_tokens=False))


def _source_section(row: dict[str, Any], index: int) -> dict[str, str]:
    from .data.prmbench_adapter import row_id
    return {
        "source_id": row_id(row, index),
        "question": str(row["original_question"]),
        "solution": str(row["original_process"]),
    }


def _fillers_for_lengths(
    packet: dict[str, Any],
    source_rows: list[dict[str, Any]],
    lengths: dict[str, Any],
    counter,
    start: int,
) -> tuple[dict[str, list[dict[str, str]]], int]:
    target = packet["target"]
    clean_target = {"source_id": str(packet["target_id"]), "question": str(target["original_question"]), "solution": str(target["original_process"])}
    queue = [int(value) for value in packet["filler_queue_indices"]]
    fillers: list[dict[str, str]] = []
    filler_questions: set[str] = set()
    output: dict[str, list[dict[str, str]]] = {}
    cursor = start
    for length_name, specification in sorted(lengths.items(), key=lambda value: int(value[1]["target"])):
        lower, upper = int(specification["min"]), int(specification["max"])
        desired = int(specification["target"])
        max_section_tokens = max(64, int(lower * 0.07))
        while True:
            size = counter(render_answer_sheet(fillers + [clean_target]))
            if lower <= size <= upper and size >= desired - 100:
                break
            if size > upper:
                removed = fillers.pop()
                filler_questions.remove(removed["question"])
                continue
            estimate = size
            goal = min(upper - 20, desired)
            while estimate < goal:
                if cursor >= len(queue):
                    raise ValueError("filler queue exhausted before reaching the requested length")
                source_index = queue[cursor]
                cursor += 1
                candidate = _source_section(source_rows[source_index], source_index)
                if candidate["source_id"] == clean_target["source_id"] or candidate["question"] == clean_target["question"] or candidate["question"] in filler_questions:
                    continue
                candidate_tokens = counter(render_answer_sheet([candidate]))
                if candidate_tokens > max_section_tokens or estimate + candidate_tokens + 2 > upper - 10:
                    continue
                fillers.append(candidate)
                filler_questions.add(candidate["question"])
                estimate += candidate_tokens + 2
        output[str(length_name)] = list(fillers)
    return output, cursor


def build_prmbench_conditions(
    packets: list[dict[str, Any]],
    source_rows: list[dict[str, Any]],
    config: dict[str, Any],
    prompt_template: str,
    *,
    counter=None,
) -> list[dict[str, Any]]:
    lengths = require_mapping(config.get("lengths"), "lengths")
    positions = require_mapping(config.get("positions"), "positions")
    counter = counter or _candidate_counter(str(config["tokenizer"]))
    conditions: list[dict[str, Any]] = []
    filler_cursor = 0
    for packet in packets:
        filler_sets, filler_cursor = _fillers_for_lengths(packet, source_rows, lengths, counter, filler_cursor)
        target_row = require_mapping(packet.get("target"), "packet target")
        clean_target = {"source_id": str(packet["target_id"]), "question": str(target_row["original_question"]), "solution": str(target_row["original_process"])}
        flawed_target = {**clean_target, "solution": str(target_row["modified_process"])}
        for length_name, specification in sorted(lengths.items(), key=lambda value: int(value[1]["target"])):
            fillers = filler_sets[str(length_name)]
            for position in ("early", "middle", "late"):
                clean_sections, clean_midpoint = choose_target_insertion(fillers, clean_target, require_mapping(positions[position], position), counter)
                target_index = next(index for index, section in enumerate(clean_sections) if section["source_id"] == str(packet["target_id"]))
                flawed_sections = list(clean_sections)
                flawed_sections[target_index] = flawed_target
                clean_text, flawed_text = render_answer_sheet(clean_sections), render_answer_sheet(flawed_sections)
                clean_tokens, flawed_tokens = counter(clean_text), counter(flawed_text)
                lower, upper = int(specification["min"]), int(specification["max"])
                if not (lower <= clean_tokens <= upper and lower <= flawed_tokens <= upper):
                    raise ValueError(
                        f"{packet['base_packet_id']}/{length_name}/{position} candidate bodies "
                        f"outside [{lower}, {upper}]: clean={clean_tokens}, flawed={flawed_tokens}"
                    )
                flawed_midpoint = target_midpoint_at(flawed_sections, target_index, counter)
                band = require_mapping(positions[position], position)
                if not float(band["min"]) <= flawed_midpoint <= float(band["max"]):
                    raise ValueError(f"flawed target midpoint outside {position} band")
                for order in ("clean_first", "flawed_first"):
                    a_text, b_text = (clean_text, flawed_text) if order == "clean_first" else (flawed_text, clean_text)
                    condition_id = f"{packet['base_packet_id']}__{length_name}__{position}__{order}"
                    conditions.append({
                        "condition_id": condition_id,
                        "base_packet_id": str(packet["base_packet_id"]),
                        "item_id": str(packet["base_packet_id"]),
                        "target_id": str(packet["target_id"]),
                        "target_category": str(packet["target_category"]),
                        "length": str(length_name),
                        "context_length": str(length_name),
                        "internal_position": position,
                        "candidate_order": order,
                        "gold_content_winner": "clean",
                        "candidate_a_identity": "clean" if order == "clean_first" else "flawed",
                        "candidate_b_identity": "flawed" if order == "clean_first" else "clean",
                        "candidate_a_text": a_text,
                        "candidate_b_text": b_text,
                        "clean_candidate_text": clean_text,
                        "flawed_candidate_text": flawed_text,
                        "clean_candidate_tokens": clean_tokens,
                        "flawed_candidate_tokens": flawed_tokens,
                        "candidate_a_tokens": clean_tokens if order == "clean_first" else flawed_tokens,
                        "candidate_b_tokens": flawed_tokens if order == "clean_first" else clean_tokens,
                        "complete_prompt_tokens": counter(render_prompt(prompt_template, a_text, b_text)),
                        "target_index": target_index,
                        "target_midpoint_clean": clean_midpoint,
                        "target_midpoint_flawed": flawed_midpoint,
                        "section_ids": [section["source_id"] for section in clean_sections],
                        "filler_ids": [section["source_id"] for section in fillers],
                    })
    return conditions


def build_prmbench_from_config(config: dict[str, Any], split: str) -> list[dict[str, Any]]:
    paths = require_mapping(config.get("paths"), "paths")
    packets = list(read_jsonl(paths[f"{split}_packets"]))
    source_rows = list(read_jsonl(paths["source_rows"]))
    template = load_prompt_template(paths["prompt_template"])
    return build_prmbench_conditions(packets, source_rows, config, template)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build matched position-swap conditions"
    )
    parser.add_argument("--config", default="configs/pilot.yaml")
    parser.add_argument("--split", choices=("smoke", "main"))
    args = parser.parse_args()
    config = load_config(args.config)
    paths = require_mapping(config.get("paths"), "paths")
    if args.split or "source_rows" in paths:
        split = args.split or "main"
        conditions = build_prmbench_from_config(config, split)
        destination = paths[f"{split}_conditions"]
        write_jsonl(destination, conditions)
        print(f"Wrote {len(conditions)} deterministic {split} conditions to {destination}")
        return
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
