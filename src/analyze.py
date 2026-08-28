from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from .aggregate_pairs import aggregate_pairs
from .parse_verdict import parse_run
from .utils import read_jsonl, require_mapping, write_json, write_jsonl


POSITIONS = ("early", "middle", "late")


@dataclass(frozen=True)
class PilotMetrics:
    early_accuracy: float | None
    middle_accuracy: float | None
    late_accuracy: float | None
    edge_accuracy: float | None
    delta_acc: float | None
    early_swap_consistency: float | None
    middle_swap_consistency: float | None
    late_swap_consistency: float | None
    edge_swap_consistency: float | None
    delta_sc: float | None
    early_cer: float | None
    middle_cer: float | None
    late_cer: float | None
    edge_cer: float | None
    delta_cer: float | None


def _mean(values: Iterable[bool | float]) -> float | None:
    materialized = [float(value) for value in values]
    return (
        sum(materialized) / len(materialized)
        if materialized
        else None
    )


def _subtract(left: float | None, right: float | None) -> float | None:
    return left - right if left is not None and right is not None else None


def accuracy(rows: Iterable[dict[str, Any]]) -> float | None:
    return _mean(
        bool(row["is_correct"])
        for row in rows
        if row.get("parse_status") == "ok"
    )


def swap_consistency(rows: Iterable[dict[str, Any]]) -> float | None:
    return _mean(
        bool(row["swap_consistent"])
        for row in rows
        if row.get("parse_ok")
    )


def consistent_error_rate(rows: Iterable[dict[str, Any]]) -> float | None:
    consistent = [
        row
        for row in rows
        if row.get("parse_ok") and row.get("swap_consistent") is True
    ]
    return _mean(bool(row["consistent_wrong"]) for row in consistent)


def compute_pilot_metrics(
    parsed: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
    context_length: str,
) -> PilotMetrics:
    call_by_position = {
        position: [
            row
            for row in parsed
            if row["context_length"] == context_length
            and row["internal_position"] == position
        ]
        for position in POSITIONS
    }
    pair_by_position = {
        position: [
            row
            for row in pairs
            if row["context_length"] == context_length
            and row["internal_position"] == position
        ]
        for position in POSITIONS
    }
    accuracies = {
        position: accuracy(call_by_position[position])
        for position in POSITIONS
    }
    consistencies = {
        position: swap_consistency(pair_by_position[position])
        for position in POSITIONS
    }
    cers = {
        position: consistent_error_rate(pair_by_position[position])
        for position in POSITIONS
    }
    edge_calls = call_by_position["early"] + call_by_position["late"]
    edge_pairs = pair_by_position["early"] + pair_by_position["late"]
    edge_accuracy = accuracy(edge_calls)
    edge_sc = swap_consistency(edge_pairs)
    edge_cer = consistent_error_rate(edge_pairs)
    return PilotMetrics(
        early_accuracy=accuracies["early"],
        middle_accuracy=accuracies["middle"],
        late_accuracy=accuracies["late"],
        edge_accuracy=edge_accuracy,
        delta_acc=_subtract(edge_accuracy, accuracies["middle"]),
        early_swap_consistency=consistencies["early"],
        middle_swap_consistency=consistencies["middle"],
        late_swap_consistency=consistencies["late"],
        edge_swap_consistency=edge_sc,
        delta_sc=_subtract(edge_sc, consistencies["middle"]),
        early_cer=cers["early"],
        middle_cer=cers["middle"],
        late_cer=cers["late"],
        edge_cer=edge_cer,
        delta_cer=_subtract(cers["middle"], edge_cer),
    )


def evaluate_pilot(
    metrics: PilotMetrics,
    thresholds: dict[str, Any],
    data_integrity_passed: bool,
) -> dict[str, Any]:
    min_accuracy_gap = float(thresholds["min_accuracy_gap"])
    min_middle_sc = float(thresholds["min_middle_swap_consistency"])
    max_sc_gap = float(thresholds["max_swap_consistency_gap"])
    min_cer_gap = float(thresholds["min_cer_gap"])
    checks = {
        "middle_accuracy_deficit_ge_threshold": (
            metrics.delta_acc is not None
            and metrics.delta_acc >= min_accuracy_gap
        ),
        "middle_swap_consistency_ge_threshold": (
            metrics.middle_swap_consistency is not None
            and metrics.middle_swap_consistency >= min_middle_sc
        ),
        "swap_consistency_drop_le_threshold": (
            metrics.delta_sc is not None
            and metrics.delta_sc <= max_sc_gap
        ),
        "middle_cer_increase_ge_threshold": (
            metrics.delta_cer is not None
            and metrics.delta_cer >= min_cer_gap
        ),
        "data_integrity_passed": data_integrity_passed,
    }
    return {
        "decision": "GO" if all(checks.values()) else "KILL",
        "checks": checks,
        "values": {
            "delta_acc": metrics.delta_acc,
            "middle_swap_consistency": metrics.middle_swap_consistency,
            "delta_sc": metrics.delta_sc,
            "delta_cer": metrics.delta_cer,
        },
        "thresholds": {
            "min_accuracy_gap": min_accuracy_gap,
            "min_middle_swap_consistency": min_middle_sc,
            "max_swap_consistency_gap": max_sc_gap,
            "min_cer_gap": min_cer_gap,
        },
    }


def metric_snapshot(
    parsed: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
    pilot_context: str,
) -> dict[str, float | None]:
    contexts = sorted(
        {str(row["context_length"]) for row in parsed}
        | {str(row["context_length"]) for row in pairs}
    )
    snapshot: dict[str, float | None] = {}
    for context in contexts:
        for position in POSITIONS:
            call_rows = [
                row for row in parsed
                if row["context_length"] == context
                and row["internal_position"] == position
            ]
            pair_rows = [
                row for row in pairs
                if row["context_length"] == context
                and row["internal_position"] == position
            ]
            snapshot[f"accuracy.{context}.{position}"] = accuracy(call_rows)
            snapshot[f"swap_consistency.{context}.{position}"] = (
                swap_consistency(pair_rows)
            )
            snapshot[f"cer.{context}.{position}"] = (
                consistent_error_rate(pair_rows)
            )
            for order in ("correct_first", "flawed_first"):
                snapshot[f"accuracy.{context}.{position}.{order}"] = accuracy(
                    row for row in call_rows
                    if row["candidate_order"] == order
                )
    for name, value in asdict(
        compute_pilot_metrics(parsed, pairs, pilot_context)
    ).items():
        snapshot[f"pilot.{name}"] = value
    return snapshot


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = probability * (len(ordered) - 1)
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def bootstrap_intervals(
    parsed: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
    *,
    pilot_context: str,
    n_resamples: int,
    ci: float,
    seed: int,
) -> dict[str, dict[str, float | int] | None]:
    item_ids = sorted({str(row["item_id"]) for row in parsed})
    if not item_ids or n_resamples < 1:
        return {}
    rng = random.Random(seed)
    samples: dict[str, list[float]] = {}
    parsed_by_item = {
        item_id: [row for row in parsed if row["item_id"] == item_id]
        for item_id in item_ids
    }
    pairs_by_item = {
        item_id: [row for row in pairs if row["item_id"] == item_id]
        for item_id in item_ids
    }
    for _ in range(n_resamples):
        selected = rng.choices(item_ids, k=len(item_ids))
        sampled_calls = [
            row for item_id in selected for row in parsed_by_item[item_id]
        ]
        sampled_pairs = [
            row for item_id in selected for row in pairs_by_item[item_id]
        ]
        for key, value in metric_snapshot(
            sampled_calls, sampled_pairs, pilot_context
        ).items():
            if value is not None:
                samples.setdefault(key, []).append(value)
    alpha = (1.0 - ci) / 2.0
    return {
        key: (
            {
                "low": _percentile(values, alpha),
                "high": _percentile(values, 1.0 - alpha),
                "valid_resamples": len(values),
            }
            if values
            else None
        )
        for key, values in samples.items()
    }


def integrity_checks(
    parsed: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
    config: dict[str, Any],
    validation_report: dict[str, Any],
) -> dict[str, Any]:
    integrity_config = require_mapping(config.get("integrity"), "integrity")
    parse_rate = _mean(
        row.get("parse_status") == "ok" for row in parsed
    ) or 0.0
    expected_calls = (
        int(require_mapping(config.get("items"), "items")["target_count"])
        * len(config["context_lengths"])
        * 3
        * 2
    )
    ids = [str(row["condition_id"]) for row in parsed]
    max_difference = max(
        (
            abs(int(row["candidate_a_tokens"]) - int(row["candidate_b_tokens"]))
            / max(
                int(row["candidate_a_tokens"]),
                int(row["candidate_b_tokens"]),
                1,
            )
            for row in parsed
        ),
        default=1.0,
    )
    depths = {
        position: [
            float(row["target_depth"])
            for row in parsed
            if row["internal_position"] == position
        ]
        for position in POSITIONS
    }
    separated = bool(
        all(depths[position] for position in POSITIONS)
        and max(depths["early"]) < min(depths["middle"])
        and max(depths["middle"]) < min(depths["late"])
    )
    checks = {
        "condition_validation": bool(validation_report.get("passed")),
        "parse_success_rate": (
            parse_rate
            >= float(integrity_config["min_parse_success_rate"])
        ),
        "condition_completeness": (
            len(parsed) == expected_calls
            and len(pairs) == expected_calls // 2
            and all(row.get("pair_complete") for row in pairs)
        ),
        "no_duplicate_calls_counted": (
            len(ids) == len(set(ids))
            and not any(
                row.get("duplicate_condition_id", False) for row in parsed
            )
        ),
        "token_positions_separated": separated,
        "candidate_length_parity": (
            max_difference
            <= float(
                integrity_config[
                    "max_candidate_length_difference_fraction"
                ]
            )
        ),
        "relocation_invariance": bool(validation_report.get("passed")),
        "gold_label_integrity": bool(validation_report.get("passed")),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "values": {
            "parse_success_rate": parse_rate,
            "minimum_parse_success_rate": float(
                integrity_config["min_parse_success_rate"]
            ),
            "expected_calls": expected_calls,
            "observed_calls": len(parsed),
            "observed_pairs": len(pairs),
            "max_candidate_length_difference_fraction": max_difference,
            "target_depth_ranges": {
                position: {
                    "min": min(values) if values else None,
                    "max": max(values) if values else None,
                }
                for position, values in depths.items()
            },
        },
    }


def grouped_summaries(
    parsed: list[dict[str, Any]], pairs: list[dict[str, Any]]
) -> dict[str, Any]:
    contexts = sorted({str(row["context_length"]) for row in parsed})
    cross_tabulation: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    for context in contexts:
        for position in POSITIONS:
            position_calls = [
                row for row in parsed
                if row["context_length"] == context
                and row["internal_position"] == position
            ]
            position_pairs = [
                row for row in pairs
                if row["context_length"] == context
                and row["internal_position"] == position
            ]
            for order in ("correct_first", "flawed_first"):
                rows = [
                    row for row in position_calls
                    if row["candidate_order"] == order
                ]
                cross_tabulation.append(
                    {
                        "context_length": context,
                        "internal_position": position,
                        "candidate_order": order,
                        "n": len(rows),
                        "n_parsed": sum(
                            row["parse_status"] == "ok" for row in rows
                        ),
                        "accuracy": accuracy(rows),
                    }
                )
            pair_rows.append(
                {
                    "context_length": context,
                    "internal_position": position,
                    "n": len(position_pairs),
                    "n_parsed": sum(
                        bool(row.get("parse_ok")) for row in position_pairs
                    ),
                    "swap_consistency": swap_consistency(position_pairs),
                    "cer": consistent_error_rate(position_pairs),
                    "fcr": consistent_error_rate(position_pairs),
                }
            )
    def call_group(
        field: str, values: list[str]
    ) -> list[dict[str, Any]]:
        output = []
        for value in values:
            rows = [row for row in parsed if str(row[field]) == value]
            output.append(
                {
                    field: value,
                    "n": len(rows),
                    "n_parsed": sum(
                        row["parse_status"] == "ok" for row in rows
                    ),
                    "accuracy": accuracy(rows),
                }
            )
        return output

    return {
        "call_level": {
            "by_context_length": call_group("context_length", contexts),
            "by_internal_position": call_group(
                "internal_position", list(POSITIONS)
            ),
            "by_candidate_order": call_group(
                "candidate_order", ["correct_first", "flawed_first"]
            ),
            "cross_tabulation": cross_tabulation,
        },
        "pair_level_by_context_and_position": pair_rows,
    }


def generate_figures(
    run_dir: Path,
    snapshot: dict[str, float | None],
    intervals: dict[str, dict[str, float | int] | None],
    contexts: list[str],
    pilot_context: str,
) -> None:
    import matplotlib.pyplot as plt

    figures = run_dir / "figures"
    figures.mkdir(exist_ok=True)
    x = list(range(len(POSITIONS)))

    def series(metric: str, context: str) -> tuple[list[float], list[list[float]]]:
        values: list[float] = []
        lows: list[float] = []
        highs: list[float] = []
        for position in POSITIONS:
            key = f"{metric}.{context}.{position}"
            value = snapshot.get(key)
            estimate = float("nan") if value is None else value
            interval = intervals.get(key)
            values.append(estimate)
            lows.append(
                0.0 if not interval or value is None
                else max(0.0, value - float(interval["low"]))
            )
            highs.append(
                0.0 if not interval or value is None
                else max(0.0, float(interval["high"]) - value)
            )
        return values, [lows, highs]

    def position_plot(metric: str, ylabel: str, filename: str) -> None:
        fig, axis = plt.subplots(figsize=(7, 4.5))
        for context in contexts:
            values, errors = series(metric, context)
            axis.errorbar(
                x, values, yerr=errors, marker="o", capsize=3, label=context
            )
        axis.set_xticks(x, POSITIONS)
        axis.set_ylim(0, 1)
        axis.set_ylabel(ylabel)
        axis.legend(title="Context")
        fig.tight_layout()
        fig.savefig(figures / filename, dpi=180)
        plt.close(fig)

    position_plot("accuracy", "Accuracy", "01_accuracy_by_position.png")
    position_plot(
        "swap_consistency",
        "Swap consistency",
        "02_swap_consistency_by_position.png",
    )

    fig, axis = plt.subplots(figsize=(7, 4.5))
    for metric, label in (
        ("accuracy", "Accuracy"),
        ("swap_consistency", "Swap consistency"),
    ):
        values, errors = series(metric, pilot_context)
        axis.errorbar(
            x, values, yerr=errors, marker="o", capsize=3, label=label
        )
    axis.set_xticks(x, POSITIONS)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Rate")
    axis.set_title(f"Accuracy-consistency dissociation ({pilot_context})")
    axis.legend()
    fig.tight_layout()
    fig.savefig(figures / "03_accuracy_consistency_dissociation.png", dpi=180)
    plt.close(fig)

    position_plot("cer", "Consistent Error Rate", "04_cer_by_position.png")

    fig, axis = plt.subplots(figsize=(8, 4.5))
    for context in contexts:
        for order, style in (
            ("correct_first", "-"),
            ("flawed_first", "--"),
        ):
            values = [
                snapshot.get(f"accuracy.{context}.{position}.{order}")
                for position in POSITIONS
            ]
            axis.plot(
                x,
                [float("nan") if value is None else value for value in values],
                style,
                marker="o",
                label=f"{context} / {order}",
            )
    axis.set_xticks(x, POSITIONS)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Accuracy")
    axis.legend(fontsize="small")
    fig.tight_layout()
    fig.savefig(figures / "05_candidate_order_diagnostic.png", dpi=180)
    plt.close(fig)


def _format_rate(value: float | None) -> str:
    return "null" if value is None else f"{100 * value:.1f}%"


def print_decision(decision: dict[str, Any]) -> None:
    values = decision["values"]
    thresholds = decision["thresholds"]
    checks = decision["checks"]
    print(f"\nPILOT DECISION: {decision['decision']}\n")
    lines = (
        (
            "middle_accuracy_deficit_ge_threshold",
            f"Edge - middle accuracy gap: {_format_rate(values['delta_acc'])} "
            f">= {_format_rate(thresholds['min_accuracy_gap'])}",
        ),
        (
            "middle_swap_consistency_ge_threshold",
            f"Middle swap consistency: "
            f"{_format_rate(values['middle_swap_consistency'])} "
            f">= {_format_rate(thresholds['min_middle_swap_consistency'])}",
        ),
        (
            "swap_consistency_drop_le_threshold",
            f"Edge - middle consistency gap: "
            f"{_format_rate(values['delta_sc'])} "
            f"<= {_format_rate(thresholds['max_swap_consistency_gap'])}",
        ),
        (
            "middle_cer_increase_ge_threshold",
            f"Middle - edge CER gap: {_format_rate(values['delta_cer'])} "
            f">= {_format_rate(thresholds['min_cer_gap'])}",
        ),
        ("data_integrity_passed", "Data integrity checks"),
    )
    for key, message in lines:
        print(f"[{'PASS' if checks[key] else 'FAIL'}] {message}")


def analyze_run(run_dir: str | Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    manifest = json.loads(
        (run_dir / "manifest.json").read_text(encoding="utf-8")
    )
    config = manifest["config"]
    parsed_with_duplicates = parse_run(run_dir)
    seen: set[str] = set()
    parsed = []
    for row in parsed_with_duplicates:
        condition_id = str(row["condition_id"])
        if condition_id not in seen:
            parsed.append(row)
            seen.add(condition_id)
    if any(
        row.get("duplicate_condition_id", False)
        for row in parsed_with_duplicates
    ):
        for row in parsed:
            row["duplicate_condition_id"] = True
    pairs = aggregate_pairs(parsed)
    write_jsonl(run_dir / "pair_level.jsonl", pairs)
    validation_path = Path(config["paths"]["validation_report"])
    validation_report = json.loads(
        validation_path.read_text(encoding="utf-8")
    )
    integrity = integrity_checks(parsed, pairs, config, validation_report)
    pilot_context = str(config["pilot_context_length"])
    pilot_metrics = compute_pilot_metrics(parsed, pairs, pilot_context)
    snapshot = metric_snapshot(parsed, pairs, pilot_context)
    bootstrap_config = require_mapping(config["bootstrap"], "bootstrap")
    intervals = bootstrap_intervals(
        parsed,
        pairs,
        pilot_context=pilot_context,
        n_resamples=int(bootstrap_config["n_resamples"]),
        ci=float(bootstrap_config["ci"]),
        seed=int(bootstrap_config["seed"]),
    )
    decision = evaluate_pilot(
        pilot_metrics,
        require_mapping(config["pilot_thresholds"], "pilot_thresholds"),
        bool(integrity["passed"]),
    )
    summaries = grouped_summaries(parsed, pairs)
    metrics = {
        "pilot_context_length": pilot_context,
        "pilot": asdict(pilot_metrics),
        "stratified": summaries,
        "point_estimates": snapshot,
        "bootstrap": {
            "n_resamples": int(bootstrap_config["n_resamples"]),
            "ci": float(bootstrap_config["ci"]),
            "seed": int(bootstrap_config["seed"]),
            "intervals": intervals,
        },
        "integrity": integrity,
    }
    write_json(run_dir / "metrics.json", metrics)
    write_json(run_dir / "pilot_decision.json", decision)
    contexts = sorted({str(row["context_length"]) for row in parsed})
    generate_figures(
        run_dir, snapshot, intervals, contexts, pilot_context
    )
    print("DATA INTEGRITY")
    for name, passed in integrity["checks"].items():
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")
    print_decision(decision)
    return {"metrics": metrics, "decision": decision}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze a completed judge run"
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--runs-dir", default="data/runs")
    args = parser.parse_args()
    analyze_run(Path(args.runs_dir) / args.run_id)


if __name__ == "__main__":
    main()
