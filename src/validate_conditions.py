from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

from .schemas import BaseItem, Condition
from .data.prmbench_adapter import ALLOWED_CATEGORIES, row_id
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


def validate_prmbench_conditions(
    conditions: list[dict[str, Any]],
    source_rows: list[dict[str, Any]],
    config: dict[str, Any],
    split: str,
) -> dict[str, Any]:
    failures: list[dict[str, str]] = []
    lengths = require_mapping(config.get("lengths"), "lengths")
    positions = require_mapping(config.get("positions"), "positions")
    expected_packets = 40 if split == "main" else 2
    expected_conditions = expected_packets * len(lengths) * 3 * 2
    by_target = {row_id(row, index): row for index, row in enumerate(source_rows)}
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    packet_lengths: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)

    def fail(scope: str, message: str) -> None:
        failures.append({"scope": scope, "message": message})

    ids = [str(row.get("condition_id")) for row in conditions]
    if len(conditions) != expected_conditions:
        fail(split, f"expected {expected_conditions} conditions, found {len(conditions)}")
    if len(ids) != len(set(ids)):
        fail(split, "duplicate condition ID")
    target_ids = {str(row.get("target_id")) for row in conditions}
    packet_ids = {str(row.get("base_packet_id")) for row in conditions}
    if len(target_ids) != expected_packets or len(packet_ids) != expected_packets:
        fail(split, "target IDs or packet IDs are not unique and complete")

    for row in conditions:
        scope = str(row.get("condition_id"))
        key = (str(row.get("base_packet_id")), str(row.get("length")), str(row.get("internal_position")))
        grouped[key].append(row)
        packet_lengths[(key[0], key[1])].append(row)
        if row.get("gold_content_winner") != "clean":
            fail(scope, "gold content winner is not clean")
        if row.get("target_category") not in ALLOWED_CATEGORIES:
            fail(scope, "target category is not an eligible Soundness category")
        order = row.get("candidate_order")
        expected = ("clean", "flawed") if order == "clean_first" else ("flawed", "clean") if order == "flawed_first" else None
        if expected is None or (row.get("candidate_a_identity"), row.get("candidate_b_identity")) != expected:
            fail(scope, "candidate identity mapping is invalid")
        a_expected = row.get("clean_candidate_text") if order == "clean_first" else row.get("flawed_candidate_text")
        b_expected = row.get("flawed_candidate_text") if order == "clean_first" else row.get("clean_candidate_text")
        if row.get("candidate_a_text") != a_expected or row.get("candidate_b_text") != b_expected:
            fail(scope, "AB/BA candidate texts are not exact inversions")
        specification = require_mapping(lengths.get(str(row.get("length"))), "length specification")
        for field in ("clean_candidate_tokens", "flawed_candidate_tokens"):
            value = int(row.get(field, -1))
            if not int(specification["min"]) <= value <= int(specification["max"]):
                fail(scope, f"{field} is outside its declared length window")
        band = require_mapping(positions.get(str(row.get("internal_position"))), "position band")
        for field in ("target_midpoint_clean", "target_midpoint_flawed"):
            value = float(row.get(field, -1))
            if not float(band["min"]) <= value <= float(band["max"]):
                fail(scope, f"{field} is outside its declared band")
        section_ids = list(row.get("section_ids", []))
        target_id = str(row.get("target_id"))
        if section_ids.count(target_id) != 1 or section_ids[int(row.get("target_index", -1))] != target_id:
            fail(scope, "target index or section identity is invalid")
        if [value for value in section_ids if value != target_id] != list(row.get("filler_ids", [])):
            fail(scope, "filler relative order changed")
        source = by_target.get(target_id)
        if source is None:
            # Some releases encode idx numerically; string coercion above handles it.
            fail(scope, "target source row is missing")
        else:
            clean_fragment = f"Question: {source['original_question']}\nSolution: {source['original_process']}"
            flawed_fragment = f"Question: {source['original_question']}\nSolution: {source['modified_process']}"
            if clean_fragment not in str(row.get("clean_candidate_text")):
                fail(scope, "clean target is not the source original process")
            if flawed_fragment not in str(row.get("flawed_candidate_text")):
                fail(scope, "flawed target is not the matching modified process")
            clean_text = str(row.get("clean_candidate_text"))
            flawed_text = str(row.get("flawed_candidate_text"))
            if clean_text.count(clean_fragment) != 1 or flawed_text.count(flawed_fragment) != 1 or clean_text.replace(clean_fragment, "<TARGET>", 1) != flawed_text.replace(flawed_fragment, "<TARGET>", 1):
                fail(scope, "clean and flawed candidates differ outside the target solution")
        visible = str(row.get("candidate_a_text", "")) + str(row.get("candidate_b_text", ""))
        if any(marker in visible.lower() for marker in ("error_steps", "classification:", "reason:")):
            fail(scope, "benchmark annotation leaked into candidate text")

    for key, rows in grouped.items():
        if len(rows) != 2 or {row.get("candidate_order") for row in rows} != {"clean_first", "flawed_first"}:
            fail("/".join(key), "missing or duplicate candidate order")
        elif any(rows[0][field] != rows[1][field] for field in ("clean_candidate_text", "flawed_candidate_text", "section_ids")):
            fail("/".join(key), "candidate swap changed underlying content")
    for key, rows in packet_lengths.items():
        representatives = {str(row["internal_position"]): row for row in rows}
        if set(representatives) != {"early", "middle", "late"}:
            fail("/".join(key), "position conditions incomplete")
        elif len({tuple(row["filler_ids"]) for row in representatives.values()}) != 1:
            fail("/".join(key), "section set differs across positions")
    for packet_id in packet_ids:
        short_rows = packet_lengths.get((packet_id, "4K"), [])
        long_rows = packet_lengths.get((packet_id, "16K"), [])
        if short_rows and long_rows:
            short = list(short_rows[0]["filler_ids"])
            long = list(long_rows[0]["filler_ids"])
            if long[:len(short)] != short:
                fail(packet_id, "4K filler queue is not nested in 16K")
    long_fillers: dict[str, set[str]] = {}
    for packet_id in packet_ids:
        rows = packet_lengths.get((packet_id, "16K"), [])
        if rows:
            long_fillers[packet_id] = set(rows[0]["filler_ids"])
    for left_index, left in enumerate(sorted(long_fillers)):
        for right in sorted(long_fillers)[left_index + 1:]:
            if long_fillers[left] & long_fillers[right]:
                fail(f"{left}/{right}", "filler source rows are reused across packets")
    return {"split": split, "conditions_checked": len(conditions), "packets_checked": len(packet_ids), "failures": failures, "passed": not failures}


def validate_prmbench_from_config(config: dict[str, Any], split: str) -> dict[str, Any]:
    paths = require_mapping(config.get("paths"), "paths")
    report = validate_prmbench_conditions(list(read_jsonl(paths[f"{split}_conditions"])), list(read_jsonl(paths["source_rows"])), config, split)
    import json
    manifest = json.loads(Path(paths["source_manifest"]).read_text(encoding="utf-8"))
    if not manifest.get("revision") or manifest.get("revision") == "unresolved" or not manifest.get("dataset_fingerprint"):
        report["failures"].append({"scope": "dataset", "message": "source revision or fingerprint is missing"})
        report["passed"] = False
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate generated counterfactual conditions"
    )
    parser.add_argument("--config", default="configs/pilot.yaml")
    parser.add_argument("--split", choices=("smoke", "main"))
    args = parser.parse_args()
    config = load_config(args.config)
    paths = require_mapping(config.get("paths"), "paths")
    if args.split or "source_rows" in paths:
        split = args.split or "main"
        report = validate_prmbench_from_config(config, split)
        report_path = paths[f"{split}_validation_report"]
        write_json(report_path, report)
        status = "PASS" if report["passed"] else "FAIL"
        print(f"VALIDATION: {status} ({report['conditions_checked']} conditions)")
        for failure in report["failures"]:
            print(f"[FAIL] {failure['scope']}: {failure['message']}")
        if not report["passed"]:
            raise SystemExit(1)
        return
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
