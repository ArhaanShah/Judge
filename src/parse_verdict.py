from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Literal

from .schemas import CandidateOrder, Condition
from .utils import read_jsonl, write_jsonl


DisplayVerdict = Literal["A", "B"]
ContentVerdict = Literal["correct", "flawed"]
VERDICT_PATTERN = re.compile(r"^VERDICT:\s*([AB])\s*$", re.MULTILINE)


def parse_display_verdict(text: str) -> tuple[DisplayVerdict | None, str]:
    matches = list(VERDICT_PATTERN.finditer(text))
    if len(matches) != 1:
        return None, "parse_error"
    match = matches[0]
    if text[match.end() :].strip():
        return None, "parse_error"
    return match.group(1), "ok"


def map_content_verdict(
    candidate_order: CandidateOrder,
    display_verdict: DisplayVerdict,
) -> ContentVerdict:
    if candidate_order == "correct_first":
        return "correct" if display_verdict == "A" else "flawed"
    if candidate_order == "flawed_first":
        return "flawed" if display_verdict == "A" else "correct"
    raise ValueError(f"unknown candidate order: {candidate_order}")


def parse_record(
    condition: Condition, response: dict[str, Any]
) -> dict[str, Any]:
    display_verdict, status = parse_display_verdict(
        str(response.get("raw_response_text", ""))
    )
    if response.get("error_status"):
        display_verdict, status = None, "provider_error"
    content_verdict = (
        map_content_verdict(condition.candidate_order, display_verdict)
        if display_verdict is not None
        else None
    )
    return {
        "condition_id": condition.condition_id,
        "item_id": condition.item_id,
        "context_length": condition.context_length,
        "internal_position": condition.internal_position,
        "candidate_order": condition.candidate_order,
        "display_verdict": display_verdict,
        "content_verdict": content_verdict,
        "is_correct": (
            content_verdict == condition.gold_content_winner
            if content_verdict is not None
            else None
        ),
        "parse_status": status,
        "target_depth": condition.target_depth,
        "candidate_a_tokens": condition.candidate_a_tokens,
        "candidate_b_tokens": condition.candidate_b_tokens,
    }


def parse_run(run_dir: str | Path) -> list[dict[str, Any]]:
    run_dir = Path(run_dir)
    manifest = json.loads(
        (run_dir / "manifest.json").read_text(encoding="utf-8")
    )
    conditions_path = manifest["config"]["paths"]["conditions"]
    conditions = {
        row.condition_id: row
        for row in (
            Condition.from_dict(value)
            for value in read_jsonl(conditions_path)
        )
    }
    responses = list(read_jsonl(run_dir / "responses.jsonl"))
    ids = [str(row["condition_id"]) for row in responses]
    counts = {condition_id: ids.count(condition_id) for condition_id in set(ids)}
    parsed = [
        {
            **parse_record(conditions[str(row["condition_id"])], row),
            "duplicate_condition_id": (
                counts[str(row["condition_id"])] > 1
            ),
        }
        for row in responses
    ]
    write_jsonl(run_dir / "parsed.jsonl", parsed)
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Strictly parse judge verdicts")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--runs-dir", default="data/runs")
    args = parser.parse_args()
    rows = parse_run(Path(args.runs_dir) / args.run_id)
    successes = sum(row["parse_status"] == "ok" for row in rows)
    print(f"Parsed {successes}/{len(rows)} responses successfully")


if __name__ == "__main__":
    main()
