from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml

from ..utils import write_json, write_jsonl

DATASET_ID = "hitsmy/PRMBench_Preview"
ALLOWED_CATEGORIES = ("Empirical Soundness", "Step Consistency", "Domain Consistency")
LEAKAGE_MARKERS = ("error_steps", "classification", "modified_steps", "reason")


class UnknownCategoryError(ValueError):
    """Raised when a released label is absent from the frozen mapping."""


@dataclass(frozen=True)
class FilterResult:
    eligible: bool
    reasons: tuple[str, ...]
    category: str | None


def _nonempty(value: Any) -> bool:
    return bool(value.strip()) if isinstance(value, str) else bool(value)


def _steps_nonempty(value: Any) -> bool:
    if isinstance(value, (list, tuple)):
        return len(value) > 0 and all(_nonempty(step) for step in value)
    return _nonempty(value)


def load_category_map(path: str | Path) -> dict[str, str]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not raw:
        raise ValueError("category mapping must be a non-empty mapping")
    return {str(key): str(value) for key, value in raw.items()}


def map_category(label: Any, mapping: dict[str, str]) -> str:
    key = str(label)
    if key not in mapping:
        raise UnknownCategoryError(f"unmapped PRMBench classification: {key!r}")
    return mapping[key]


def filter_target_row(row: dict[str, Any], category_map: dict[str, str], *, token_counter, absolute_delta_limit: int = 128) -> FilterResult:
    category = map_category(row.get("classification"), category_map)
    reasons: list[str] = []
    question = str(row.get("original_question") or "")
    modified_question = str(row.get("modified_question") or "")
    original = str(row.get("original_process") or "")
    modified = str(row.get("modified_process") or "")
    if question != modified_question:
        reasons.append("question_changed")
    if not original.strip() or not modified.strip():
        reasons.append("empty_process")
    if original == modified:
        reasons.append("process_unchanged")
    if not _steps_nonempty(row.get("error_steps")):
        reasons.append("empty_error_steps")
    if not _steps_nonempty(row.get("modified_steps")):
        reasons.append("empty_modified_steps")
    if category not in ALLOWED_CATEGORIES:
        reasons.append("ineligible_category")
    if any(marker in text.lower() for marker in LEAKAGE_MARKERS for text in (original, modified)):
        reasons.append("metadata_leakage")
    original_tokens, modified_tokens = token_counter(original), token_counter(modified)
    delta = abs(original_tokens - modified_tokens)
    if delta / max(original_tokens, modified_tokens, 1) > 0.10:
        reasons.append("relative_length_delta")
    if delta > absolute_delta_limit:
        reasons.append("absolute_length_delta")
    if not _looks_serializable(question, original, modified):
        reasons.append("malformed_serialization")
    return FilterResult(not reasons, tuple(reasons), category)


def _looks_serializable(question: str, *processes: str) -> bool:
    values = (question, *processes)
    if any(not value.strip() or "\x00" in value for value in values):
        return False
    forbidden = ("<candidate_a>", "</candidate_a>", "<candidate_b>", "</candidate_b>")
    return not any(tag in text.lower() for text in values for tag in forbidden)


def row_id(row: dict[str, Any], index: int) -> str:
    value = row.get("idx")
    return str(value) if value is not None and str(value).strip() else f"row_{index:06d}"


def select_targets(rows: list[dict[str, Any]], eligible_indices: Iterable[int], categories: dict[int, str], *, seed: int, quotas: dict[str, int]) -> list[int]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index in eligible_indices:
        grouped[categories[index]].append(index)
    rng = random.Random(seed)
    for values in grouped.values():
        rng.shuffle(values)
    selected: list[int] = []
    selected_questions: set[str] = set()
    shortfall = 0
    for category in ALLOWED_CATEGORIES:
        wanted = int(quotas.get(category, 0))
        chosen = []
        for index in grouped[category]:
            question = str(rows[index].get("original_question"))
            if question not in selected_questions:
                chosen.append(index)
                selected_questions.add(question)
            if len(chosen) == wanted:
                break
        selected.extend(chosen)
        grouped[category] = [index for index in grouped[category] if index not in set(chosen)]
        shortfall += wanted - len(chosen)
    if shortfall:
        remainder = [index for category in ALLOWED_CATEGORIES for index in grouped[category]]
        rng.shuffle(remainder)
        for index in remainder:
            question = str(rows[index].get("original_question"))
            if question not in selected_questions:
                selected.append(index)
                selected_questions.add(question)
                shortfall -= 1
            if shortfall == 0:
                break
    if len(selected) != sum(quotas.values()):
        raise ValueError(f"only {len(selected)} eligible targets for {sum(quotas.values())} requested")
    return selected


def prepare_packets(rows: list[dict[str, Any]], category_map: dict[str, str], *, token_counter, seed: int = 20260828, main_quotas: dict[str, int] | None = None, smoke_count: int = 2) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    quotas = main_quotas or {"Empirical Soundness": 14, "Step Consistency": 13, "Domain Consistency": 13}
    eligible: list[int] = []
    categories: dict[int, str] = {}
    exclusions: Counter[str] = Counter()
    for index, row in enumerate(rows):
        result = filter_target_row(row, category_map, token_counter=token_counter)
        categories[index] = result.category or ""
        if result.eligible:
            eligible.append(index)
        else:
            exclusions.update(result.reasons)
    protocol_deviation = None
    if len(eligible) < sum(quotas.values()) + smoke_count:
        eligible.clear()
        categories.clear()
        exclusions.clear()
        for index, row in enumerate(rows):
            result = filter_target_row(row, category_map, token_counter=token_counter, absolute_delta_limit=192)
            categories[index] = result.category or ""
            if result.eligible:
                eligible.append(index)
            else:
                exclusions.update(result.reasons)
        protocol_deviation = "absolute target solution token delta relaxed from 128 to 192"
    main_indices = select_targets(rows, eligible, categories, seed=seed, quotas=quotas)
    remaining = [index for index in eligible if index not in set(main_indices)]
    random.Random(seed + 9).shuffle(remaining)
    smoke_indices = remaining[:smoke_count]
    if len(smoke_indices) != smoke_count:
        raise ValueError("not enough disjoint eligible smoke targets")
    selected = set(main_indices + smoke_indices)
    used_questions = {str(rows[index]["original_question"]) for index in selected}
    filler_indices = [
        index
        for index, row in enumerate(rows)
        if index not in selected
        and _nonempty(row.get("original_question"))
        and _nonempty(row.get("original_process"))
    ]
    random.Random(seed + 1).shuffle(filler_indices)

    def packets(indices: list[int], prefix: str, queue: list[int]) -> list[dict[str, Any]]:
        return [{"base_packet_id": f"{prefix}_{number:03d}", "target_id": row_id(rows[index], index), "target_category": categories[index], "target": rows[index], "filler_queue_indices": queue} for number, index in enumerate(indices, start=1)]

    report = {"classification_counts": dict(Counter(str(row.get("classification")) for row in rows)), "eligible_count": len(eligible), "exclusion_reason_counts": dict(exclusions), "protocol_deviation": protocol_deviation, "selected_main_categories": dict(Counter(categories[index] for index in main_indices)), "selected_smoke_ids": [row_id(rows[index], index) for index in smoke_indices], "selected_main_ids": [row_id(rows[index], index) for index in main_indices]}
    return packets(main_indices, "main", filler_indices), packets(smoke_indices, "smoke", list(reversed(filler_indices))), report


def load_prmbench(revision: str | None = None):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("install the 'data' extra to load PRMBench") from exc
    kwargs: dict[str, Any] = {"split": "train"}
    if revision:
        kwargs["revision"] = revision
    return load_dataset(DATASET_ID, **kwargs)


def dataset_manifest(dataset: Any, revision: str | None, seed: int) -> dict[str, Any]:
    return {"dataset": DATASET_ID, "revision": revision or "unresolved", "license": "apache-2.0", "n_rows": len(dataset), "dataset_fingerprint": getattr(dataset, "_fingerprint", None), "retrieved_at": datetime.now(timezone.utc).isoformat(), "target_categories": list(ALLOWED_CATEGORIES), "sampling_seed": seed}


def inspect_dataset(map_path: Path, revision: str | None, output_dir: Path) -> None:
    from ..utils import token_count
    if revision is None:
        try:
            from huggingface_hub import HfApi
            revision = str(HfApi().dataset_info(DATASET_ID).sha)
        except Exception as exc:
            raise RuntimeError("could not resolve the PRMBench repository revision SHA") from exc
    dataset = load_prmbench(revision)
    rows = [dict(row) for row in dataset]
    counts = dict(Counter(str(row.get("classification")) for row in rows))
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "prmbench_classification_counts.json", counts)
    print(json.dumps(counts, indent=2, sort_keys=True))
    main, smoke, report = prepare_packets(rows, load_category_map(map_path), token_counter=token_count)
    write_json(output_dir / "prmbench_manifest.json", dataset_manifest(dataset, revision, 20260828))
    write_json(output_dir / "prmbench_inspection.json", report)
    write_jsonl(output_dir / "prmbench_rows.jsonl", rows)
    write_jsonl(output_dir / "main_packets.jsonl", main)
    write_jsonl(output_dir / "smoke_packets.jsonl", smoke)
    print(f"Rows: {len(rows)}; eligible: {report['eligible_count']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect and prepare PRMBench source data")
    parser.add_argument("--inspect", action="store_true", required=True)
    parser.add_argument("--category-map", default="config/prmbench_category_map.yaml")
    parser.add_argument("--revision")
    parser.add_argument("--output-dir", default="data/source")
    args = parser.parse_args()
    inspect_dataset(Path(args.category_map), args.revision, Path(args.output_dir))


if __name__ == "__main__":
    main()
