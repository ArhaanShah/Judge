from __future__ import annotations

import argparse
from collections import defaultdict
from typing import Any

from .schemas import BaseItem, Condition
from .utils import (
    load_config,
    normalized_hash,
    read_jsonl,
    require_mapping,
    write_json,
)


def validate_conditions(
    conditions: list[Condition],
    items: dict[str, BaseItem],
    config: dict[str, Any],
) -> dict[str, Any]:
    failures: list[dict[str, str]] = []
    positions = require_mapping(config.get("positions"), "positions")
    tolerance = float(positions.get("tolerance_fraction", 0.06))
    padding = require_mapping(config.get("padding", {}), "padding")
    token_tolerance = int(padding.get("target_tolerance_tokens", 0))
    contexts_value = config.get("context_lengths")
    if not isinstance(contexts_value, list):
        raise ValueError("context_lengths must be a list")
    expected_contexts = {str(entry["name"]) for entry in contexts_value}
    grouped: dict[tuple[str, str, str], list[Condition]] = defaultdict(list)

    def fail(scope: str, message: str) -> None:
        failures.append({"scope": scope, "message": message})

    for condition in conditions:
        scope = condition.condition_id
        item = items.get(condition.item_id)
        if item is None:
            fail(scope, "condition references an unknown item")
            continue
        grouped[
            (condition.item_id, condition.context_length, condition.internal_position)
        ].append(condition)
        correct = condition.correct_sections
        flawed = condition.flawed_sections
        if len(correct) != item.section_count or len(flawed) != item.section_count:
            fail(scope, "section count differs from the source item")
        correct_ids = [section.section_id for section in correct]
        source_ids = {section.section_id for section in item.correct_sections}
        if set(correct_ids) != source_ids:
            fail(scope, "section ID set differs from source item")
        if [section.section_id for section in flawed] != correct_ids:
            fail(scope, "clean and flawed candidates use different permutations")
        differences = [
            clean.section_id
            for clean, bad in zip(correct, flawed)
            if normalized_hash(clean.text) != normalized_hash(bad.text)
        ]
        if differences != [item.flawed_section_id]:
            fail(scope, "candidates do not differ at exactly the flawed section")
        for section in correct:
            source_text = next(
                source.text
                for source in item.correct_sections
                if source.section_id == section.section_id
            )
            if not section.text.startswith(source_text):
                fail(scope, f"wording changed for section {section.section_id}")
        flawed_target = next(
            section
            for section in flawed
            if section.section_id == item.flawed_section_id
        )
        if not flawed_target.text.startswith(item.flawed_section_text):
            fail(scope, "flawed target text differs from the source replacement")
        if condition.section_order[condition.target_index] != item.flawed_section_id:
            fail(scope, "flawed section is not at target_index")
        if bool(padding.get("enabled", False)) and abs(
            condition.actual_prompt_tokens - condition.target_tokens
        ) > token_tolerance:
            fail(scope, "rendered prompt token count is outside target tolerance")
        expected_fraction = float(
            positions[f"{condition.internal_position}_fraction"]
        )
        if abs(condition.target_depth - expected_fraction) > tolerance:
            fail(
                scope,
                "target token depth is outside its configured position band",
            )
        if condition.candidate_order == "correct_first":
            expected_a = condition.correct_candidate_id
            expected_b = condition.flawed_candidate_id
            expected_a_text = condition.candidate_a_text == _rendered(correct)
            expected_b_text = condition.candidate_b_text == _rendered(flawed)
        else:
            expected_a = condition.flawed_candidate_id
            expected_b = condition.correct_candidate_id
            expected_a_text = condition.candidate_a_text == _rendered(flawed)
            expected_b_text = condition.candidate_b_text == _rendered(correct)
        if (condition.candidate_a_id, condition.candidate_b_id) != (
            expected_a,
            expected_b,
        ):
            fail(scope, "candidate swap changed content identity")
        if not expected_a_text or not expected_b_text:
            fail(scope, "rendered candidate text does not match section metadata")
        for text in (condition.candidate_a_text, condition.candidate_b_text):
            headings = [
                line.strip().lower()
                for line in text.splitlines()
                if line.strip().lower().startswith("section ")
            ]
            if any(
                label in heading
                for heading in headings
                for label in ("correct", "flawed", "gold", "error")
            ):
                fail(scope, "user-visible section metadata leaks a gold label")

    for item_id in items:
        for context in expected_contexts:
            for position in ("early", "middle", "late"):
                rows = grouped.get((item_id, context, position), [])
                orders = {row.candidate_order for row in rows}
                if len(rows) != 2 or orders != {"correct_first", "flawed_first"}:
                    fail(
                        f"{item_id}/{context}/{position}",
                        "missing or duplicate candidate order",
                    )
                    continue
                first, second = sorted(
                    rows, key=lambda row: row.candidate_order
                )
                if (
                    first.section_order != second.section_order
                    or first.correct_sections != second.correct_sections
                    or first.flawed_sections != second.flawed_sections
                ):
                    fail(
                        f"{item_id}/{context}/{position}",
                        "candidate swap changed condition content",
                    )

    for item_id in items:
        for context in expected_contexts:
            representatives = [
                grouped[(item_id, context, position)][0]
                for position in ("early", "middle", "late")
                if grouped.get((item_id, context, position))
            ]
            if len(representatives) == 3:
                correct_multisets = [
                    sorted(
                        (
                            section.section_id,
                            normalized_hash(section.text),
                        )
                        for section in row.correct_sections
                    )
                    for row in representatives
                ]
                flawed_multisets = [
                    sorted(
                        (
                            section.section_id,
                            normalized_hash(section.text),
                        )
                        for section in row.flawed_sections
                    )
                    for row in representatives
                ]
                if not (
                    correct_multisets[0]
                    == correct_multisets[1]
                    == correct_multisets[2]
                    and flawed_multisets[0]
                    == flawed_multisets[1]
                    == flawed_multisets[2]
                ):
                    fail(
                        f"{item_id}/{context}",
                        "relocation changed a section-text multiset",
                    )

    return {
        "items_checked": len(items),
        "conditions_checked": len(conditions),
        "failures": failures,
        "passed": not failures,
    }


def _rendered(sections: tuple[Any, ...]) -> str:
    return "\n\n".join(
        f"Section {number}\n{section.text}"
        for number, section in enumerate(sections, start=1)
    )


def validate_from_config(config: dict[str, Any]) -> dict[str, Any]:
    paths = require_mapping(config.get("paths"), "paths")
    items = {
        item.item_id: item
        for item in (
            BaseItem.from_dict(row)
            for row in read_jsonl(paths["source_items"])
        )
    }
    conditions = [
        Condition.from_dict(row) for row in read_jsonl(paths["conditions"])
    ]
    return validate_conditions(conditions, items, config)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate generated counterfactual conditions"
    )
    parser.add_argument("--config", default="configs/pilot.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    report = validate_from_config(config)
    report_path = require_mapping(
        config.get("paths"), "paths"
    )["validation_report"]
    write_json(report_path, report)
    status = "PASS" if report["passed"] else "FAIL"
    print(
        f"VALIDATION: {status} "
        f"({report['conditions_checked']} conditions)"
    )
    for failure in report["failures"]:
        print(f"[FAIL] {failure['scope']}: {failure['message']}")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
