from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

from .parse_verdict import parse_run
from .utils import read_jsonl, write_jsonl


def aggregate_pairs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row["item_id"]),
                str(row["context_length"]),
                str(row["internal_position"]),
            )
        ].append(row)
    output: list[dict[str, Any]] = []
    for (item_id, context, position), pair in sorted(grouped.items()):
        by_order = {str(row["candidate_order"]): row for row in pair}
        first_label = "clean_first" if "clean_first" in by_order else "correct_first"
        complete = len(pair) == 2 and set(by_order) == {first_label, "flawed_first"}
        first = by_order.get(first_label)
        second = by_order.get("flawed_first")
        parse_ok = bool(
            complete
            and first
            and second
            and first["parse_status"] == "ok"
            and second["parse_status"] == "ok"
        )
        first_verdict = first["content_verdict"] if first else None
        second_verdict = second["content_verdict"] if second else None
        swap_consistent = (first_verdict == second_verdict and first_verdict != "tie") if parse_ok else None
        consistent_winner = first_verdict if swap_consistent else None
        output.append(
            {
                "item_id": item_id,
                "context_length": context,
                "internal_position": position,
                f"{first_label}_content_verdict": first_verdict,
                "flawed_first_content_verdict": second_verdict,
                "pair_complete": complete,
                "parse_ok": parse_ok,
                "swap_consistent": swap_consistent,
                "consistent_winner": consistent_winner,
                "consistent_wrong": (
                    consistent_winner == "flawed"
                    if swap_consistent
                    else False if swap_consistent is False else None
                ),
            }
        )
    return output


def aggregate_run(run_dir: str | Path) -> list[dict[str, Any]]:
    run_dir = Path(run_dir)
    parsed_path = run_dir / "parsed.jsonl"
    rows = (
        list(read_jsonl(parsed_path))
        if parsed_path.exists()
        else parse_run(run_dir)
    )
    pairs = aggregate_pairs(rows)
    write_jsonl(run_dir / "pair_level.jsonl", pairs)
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate identity-safe AB/BA pairs")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--runs-dir", default="data/runs")
    args = parser.parse_args()
    pairs = aggregate_run(Path(args.runs_dir) / args.run_id)
    complete = sum(row["pair_complete"] for row in pairs)
    print(f"Wrote {len(pairs)} pairs ({complete} complete)")


if __name__ == "__main__":
    main()
