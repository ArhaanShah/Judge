from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


def _float(value: str) -> float | None:
    stripped = value.strip()
    if not stripped:
        return None
    return float(stripped)


def _subtract(left: float | None, right: float | None) -> float | None:
    return left - right if left is not None and right is not None else None


def _format(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def derive_rows(inputs: list[tuple[str, Path]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for model, path in inputs:
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                raw_accuracy = _float(row.get("raw_accuracy", ""))
                coverage = _float(row.get("coverage", row.get("swap_consistency", "")))
                cer = _float(row.get("consistent_error_rate", ""))
                post_filter_accuracy = None if cer is None else 1.0 - cer
                output.append(
                    {
                        "model": model,
                        "length": row["length"],
                        "position": row["position"],
                        "n_calls": row.get("n_calls"),
                        "n_swap_pairs": row.get("n_swap_pairs"),
                        "raw_accuracy": raw_accuracy,
                        "raw_accuracy_ci_low": _float(row.get("raw_accuracy_ci_low", "")),
                        "raw_accuracy_ci_high": _float(row.get("raw_accuracy_ci_high", "")),
                        "post_filter_accuracy": post_filter_accuracy,
                        "coverage": coverage,
                        "coverage_ci_low": _float(row.get("coverage_ci_low", row.get("swap_consistency_ci_low", ""))),
                        "coverage_ci_high": _float(row.get("coverage_ci_high", row.get("swap_consistency_ci_high", ""))),
                        "consistent_error_rate": cer,
                        "reported_gap": _subtract(post_filter_accuracy, raw_accuracy),
                    }
                )
    return output


def write_outputs(rows: list[dict[str, Any]], output_prefix: Path) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "model",
        "length",
        "position",
        "n_calls",
        "n_swap_pairs",
        "raw_accuracy",
        "raw_accuracy_ci_low",
        "raw_accuracy_ci_high",
        "post_filter_accuracy",
        "coverage",
        "coverage_ci_low",
        "coverage_ci_high",
        "consistent_error_rate",
        "reported_gap",
    ]
    with output_prefix.with_suffix(".csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    markdown = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        markdown.append("| " + " | ".join(_format(row.get(column)) for column in columns) + " |")

    for model in sorted({str(row["model"]) for row in rows}):
        hard = next(
            (
                row
                for row in rows
                if row["model"] == model
                and row["length"] == "16K"
                and row["position"] == "middle"
            ),
            None,
        )
        markdown.append("")
        coverage = None if hard is None else hard["coverage"]
        markdown.append(
            f"**Effective hard-distribution coverage (16K middle, {model}):** "
            f"{'' if coverage is None else f'{100 * coverage:.2f}%'}"
        )
    output_prefix.with_suffix(".md").write_text("\n".join(markdown) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Derive coverage-conditional accuracy tables from summary CSVs."
    )
    parser.add_argument(
        "--input",
        nargs=2,
        action="append",
        metavar=("MODEL", "CSV"),
        required=True,
    )
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args()
    rows = derive_rows([(model, Path(path)) for model, path in args.input])
    write_outputs(rows, Path(args.output_prefix))
    print(f"Wrote {len(rows)} rows to {args.output_prefix}.csv/.md")


if __name__ == "__main__":
    main()
