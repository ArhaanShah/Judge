from __future__ import annotations

import argparse
import csv
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
    early_post_filter_accuracy: float | None = None
    middle_post_filter_accuracy: float | None = None
    late_post_filter_accuracy: float | None = None
    edge_post_filter_accuracy: float | None = None
    early_reported_gap: float | None = None
    middle_reported_gap: float | None = None
    late_reported_gap: float | None = None
    edge_reported_gap: float | None = None
    delta_acc_clean_first: float | None = None
    delta_acc_flawed_first: float | None = None


def _mean(values: Iterable[bool | float]) -> float | None:
    materialized = [float(value) for value in values]
    return (
        sum(materialized) / len(materialized)
        if materialized
        else None
    )


def _subtract(left: float | None, right: float | None) -> float | None:
    return left - right if left is not None and right is not None else None


def _model_name(manifest: dict[str, Any], parsed: list[dict[str, Any]]) -> str:
    return next(
        (
            str(row.get("returned_model"))
            for row in parsed
            if row.get("returned_model")
        ),
        str(manifest.get("model") or "unknown"),
    )


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


def post_filter_accuracy(rows: Iterable[dict[str, Any]]) -> float | None:
    """Accuracy restricted to swap-consistent pairs (1 - CER).

    Returns None when no swap-consistent pairs exist (coverage = 0).
    Among swap-consistent pairs, ``consistent_wrong`` is False iff the
    consistent winner is the ground-truth clean/correct candidate.
    """
    consistent = [
        row
        for row in rows
        if row.get("parse_ok") and row.get("swap_consistent") is True
    ]
    if not consistent:
        return None
    return _mean(row.get("consistent_wrong") is False for row in consistent)


def reported_gap(
    raw_accuracy: float | None,
    post_filter: float | None,
) -> float | None:
    """Difference between post-filter and raw accuracy.

    Returns None when either input is None (e.g., zero coverage).
    """
    if raw_accuracy is None or post_filter is None:
        return None
    return post_filter - raw_accuracy


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
    post_filters = {
        position: post_filter_accuracy(pair_by_position[position])
        for position in POSITIONS
    }
    edge_calls = call_by_position["early"] + call_by_position["late"]
    edge_pairs = pair_by_position["early"] + pair_by_position["late"]
    edge_accuracy = accuracy(edge_calls)
    edge_sc = swap_consistency(edge_pairs)
    edge_cer = consistent_error_rate(edge_pairs)
    edge_post_filter = post_filter_accuracy(edge_pairs)
    order_deltas: dict[str, float | None] = {}
    first_order = "clean_first" if any(row.get("candidate_order") == "clean_first" for row in parsed) else "correct_first"
    for order in (first_order, "flawed_first"):
        position_values = {position: accuracy(row for row in call_by_position[position] if row.get("candidate_order") == order) for position in POSITIONS}
        edge = _mean(value for value in (position_values["early"], position_values["late"]) if value is not None)
        order_deltas[order] = _subtract(edge, position_values["middle"])
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
        early_post_filter_accuracy=post_filters["early"],
        middle_post_filter_accuracy=post_filters["middle"],
        late_post_filter_accuracy=post_filters["late"],
        edge_post_filter_accuracy=edge_post_filter,
        early_reported_gap=reported_gap(accuracies["early"], post_filters["early"]),
        middle_reported_gap=reported_gap(accuracies["middle"], post_filters["middle"]),
        late_reported_gap=reported_gap(accuracies["late"], post_filters["late"]),
        edge_reported_gap=reported_gap(edge_accuracy, edge_post_filter),
        delta_acc_clean_first=order_deltas[first_order],
        delta_acc_flawed_first=order_deltas["flawed_first"],
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
            and abs(metrics.delta_sc) <= max_sc_gap
        ),
        "middle_cer_increase_ge_threshold": (
            metrics.delta_cer is not None
            and metrics.delta_cer >= min_cer_gap
        ),
        "data_integrity_passed": data_integrity_passed,
    }
    if metrics.delta_acc_clean_first is not None or metrics.delta_acc_flawed_first is not None:
        checks["clean_first_nonnegative"] = metrics.delta_acc_clean_first is not None and metrics.delta_acc_clean_first >= 0
        checks["flawed_first_nonnegative"] = metrics.delta_acc_flawed_first is not None and metrics.delta_acc_flawed_first >= 0
    return {
        "decision": "GO" if all(checks.values()) else "KILL",
        "checks": checks,
        "values": {
            "delta_acc": metrics.delta_acc,
            "middle_swap_consistency": metrics.middle_swap_consistency,
            "delta_sc": metrics.delta_sc,
            "delta_cer": metrics.delta_cer,
            "delta_acc_clean_first": metrics.delta_acc_clean_first,
            "delta_acc_flawed_first": metrics.delta_acc_flawed_first,
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
            snapshot[f"coverage.{context}.{position}"] = (
                swap_consistency(pair_rows)
            )
            snapshot[f"cer.{context}.{position}"] = (
                consistent_error_rate(pair_rows)
            )
            snapshot[f"post_filter_accuracy.{context}.{position}"] = (
                post_filter_accuracy(pair_rows)
            )
            cell_raw = accuracy(call_rows)
            cell_post = post_filter_accuracy(pair_rows)
            snapshot[f"reported_gap.{context}.{position}"] = (
                reported_gap(cell_raw, cell_post)
            )
            first_order = "clean_first" if any(row.get("candidate_order") == "clean_first" for row in call_rows) else "correct_first"
            for order in (first_order, "flawed_first"):
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
    prmbench = "lengths" in config
    expected_calls = (40 * len(config["lengths"]) * 3 * 2) if prmbench else (
        int(require_mapping(config.get("items"), "items")["target_count"])
        * len(config["context_lengths"]) * 3 * 2
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
        "single_judge_model": len({str(row.get("returned_model")) for row in parsed if row.get("returned_model")}) == 1,
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
            first_order = "clean_first" if any(row.get("candidate_order") == "clean_first" for row in position_calls) else "correct_first"
            for order in (first_order, "flawed_first"):
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
                    "coverage": swap_consistency(position_pairs),
                    "swap_consistency": swap_consistency(position_pairs),
                    "post_filter_accuracy": post_filter_accuracy(position_pairs),
                    "cer": consistent_error_rate(position_pairs),
                    "fcr": consistent_error_rate(position_pairs),
                    "reported_gap": reported_gap(
                        accuracy(position_calls),
                        post_filter_accuracy(position_pairs),
                    ),
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
                "candidate_order",
                (["clean_first", "flawed_first"] if any(row.get("candidate_order") == "clean_first" for row in parsed) else ["correct_first", "flawed_first"]),
            ),
            "cross_tabulation": cross_tabulation,
        },
        "pair_level_by_context_and_position": pair_rows,
    }


def generate_figures(
    output_dir: Path,
    snapshot: dict[str, float | None],
    intervals: dict[str, dict[str, float | int] | None],
    contexts: list[str],
    pilot_context: str,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
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

    position_plot("cer", "Consistent Error Rate", "03_cer_by_position.png")

    fig, axis = plt.subplots(figsize=(8, 4.5))
    for context in [pilot_context]:
        for order, style in (
            (("clean_first" if any(key.endswith(".clean_first") for key in snapshot) else "correct_first"), "-"),
            ("flawed_first", "--"),
        ):
            values = [
                snapshot.get(f"accuracy.{context}.{position}.{order}")
                for position in POSITIONS
            ]
            lows, highs = [], []
            for position, value in zip(POSITIONS, values):
                interval = intervals.get(f"accuracy.{context}.{position}.{order}") or {}
                lows.append(0 if value is None or interval.get("low") is None else max(0, value - float(interval["low"])))
                highs.append(0 if value is None or interval.get("high") is None else max(0, float(interval["high"]) - value))
            axis.errorbar(x, [float("nan") if value is None else value for value in values], yerr=[lows, highs], linestyle=style, marker="o", capsize=3, label=order)
    axis.set_xticks(x, POSITIONS)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Accuracy")
    axis.legend(fontsize="small")
    fig.tight_layout()
    fig.savefig(figures / "04_accuracy_by_candidate_order.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 4.5))
    for context in contexts:
        for position in POSITIONS:
            acc = snapshot.get(f"accuracy.{context}.{position}")
            sc = snapshot.get(f"swap_consistency.{context}.{position}")
            if acc is not None and sc is not None:
                acc_ci = intervals.get(f"accuracy.{context}.{position}") or {}
                sc_ci = intervals.get(f"swap_consistency.{context}.{position}") or {}
                xerr = [[max(0, sc - float(sc_ci.get("low", sc)))], [max(0, float(sc_ci.get("high", sc)) - sc)]]
                yerr = [[max(0, acc - float(acc_ci.get("low", acc)))], [max(0, float(acc_ci.get("high", acc)) - acc)]]
                axis.errorbar(sc, acc, xerr=xerr, yerr=yerr, fmt="o", capsize=2)
                axis.annotate(f"{context}-{position}", (sc, acc), xytext=(4, 4), textcoords="offset points", fontsize="x-small")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_xlabel("Swap consistency")
    axis.set_ylabel("Raw accuracy")
    fig.tight_layout()
    fig.savefig(figures / "05_accuracy_vs_swap_consistency.png", dpi=180)
    plt.close(fig)


def _format_rate(value: float | None) -> str:
    return "null" if value is None else f"{100 * value:.1f}%"


def print_decision(decision: dict[str, Any]) -> None:
    values = decision["values"]
    thresholds = decision["thresholds"]
    checks = decision["checks"]
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
    for key in ("clean_first_nonnegative", "flawed_first_nonnegative"):
        if key in checks:
            print(f"[{'PASS' if checks[key] else 'FAIL'}] {key}")
    if not checks.get("data_integrity_passed", False):
        print("REASON: INVALID PILOT / INTEGRITY FAILURE")
    print(f"PILOT DECISION: {decision['decision']}")


def write_required_outputs(
    results_dir: Path,
    parsed: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
    snapshot: dict[str, float | None],
    intervals: dict[str, dict[str, float | int] | None],
    decision: dict[str, Any],
    *,
    model: str,
    pilot_context: str,
) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for context in sorted({str(row["context_length"]) for row in parsed}):
        for position in POSITIONS:
            calls = [row for row in parsed if row["context_length"] == context and row["internal_position"] == position]
            pair_rows = [row for row in pairs if row["context_length"] == context and row["internal_position"] == position]
            keys = {
                "raw_accuracy": f"accuracy.{context}.{position}",
                "post_filter_accuracy": f"post_filter_accuracy.{context}.{position}",
                "coverage": f"coverage.{context}.{position}",
                "swap_consistency": f"swap_consistency.{context}.{position}",
                "consistent_error_rate": f"cer.{context}.{position}",
                "reported_gap": f"reported_gap.{context}.{position}",
            }
            output: dict[str, Any] = {
                "model": model,
                "length": context,
                "position": position,
                "n_calls": len(calls),
                "n_swap_pairs": len(pair_rows)
            }
            for column, key in keys.items():
                output[column] = snapshot.get(key)
                interval = intervals.get(key) or {}
                prefix = "cer" if column == "consistent_error_rate" else column
                output[f"{prefix}_ci_low"] = interval.get("low")
                output[f"{prefix}_ci_high"] = interval.get("high")
            rows.append(output)
            
    columns = [
        "model", "length", "position", "n_calls", "n_swap_pairs",
        "raw_accuracy", "raw_accuracy_ci_low", "raw_accuracy_ci_high",
        "post_filter_accuracy", "post_filter_accuracy_ci_low", "post_filter_accuracy_ci_high",
        "coverage", "coverage_ci_low", "coverage_ci_high",
        "swap_consistency", "swap_consistency_ci_low", "swap_consistency_ci_high",
        "consistent_error_rate", "cer_ci_low", "cer_ci_high",
        "reported_gap", "reported_gap_ci_low", "reported_gap_ci_high"
    ]
    with (results_dir / "pilot_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    markdown = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    markdown.extend("| " + " | ".join("" if row.get(column) is None else str(row.get(column)) for column in columns) + " |" for row in rows)

    hard_coverage = snapshot.get(f"coverage.{pilot_context}.middle")
    hard_coverage_str = f"{hard_coverage:.2%}" if hard_coverage is not None else "undefined"
    markdown.append("")
    markdown.append(f"**Effective hard-distribution coverage ({pilot_context} middle, {model}):** {hard_coverage_str}")

    (results_dir / "pilot_summary.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    checks, values, thresholds = decision["checks"], decision["values"], decision["thresholds"]
    criteria = {
        "delta_acc": {"value": values.get("delta_acc"), "threshold": thresholds["min_accuracy_gap"], "pass": checks["middle_accuracy_deficit_ge_threshold"]},
        "sc_middle": {"value": values.get("middle_swap_consistency"), "threshold": thresholds["min_middle_swap_consistency"], "pass": checks["middle_swap_consistency_ge_threshold"]},
        "abs_sc_edge_minus_middle": {"value": abs(values["delta_sc"]) if values.get("delta_sc") is not None else None, "threshold": thresholds["max_swap_consistency_gap"], "pass": checks["swap_consistency_drop_le_threshold"]},
        "delta_cer": {"value": values.get("delta_cer"), "threshold": thresholds["min_cer_gap"], "pass": checks["middle_cer_increase_ge_threshold"]},
        "delta_acc_clean_first": {"value": values.get("delta_acc_clean_first"), "threshold": 0.0, "pass": checks.get("clean_first_nonnegative", False)},
        "delta_acc_flawed_first": {"value": values.get("delta_acc_flawed_first"), "threshold": 0.0, "pass": checks.get("flawed_first_nonnegative", False)},
        "integrity": {"pass": checks["data_integrity_passed"]},
    }
    write_json(results_dir / "go_kill.json", {"decision": decision["decision"], "primary_length": "16K", "criteria": criteria})


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
    split = str(manifest.get("split", "main"))
    validation_key = f"{split}_validation_report" if "source_rows" in config["paths"] else "validation_report"
    validation_path = Path(config["paths"][validation_key])
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
    
    model = _model_name(manifest, parsed)
    gradient = []
    for context in ["4K", "16K"]:
        if context not in {str(row["context_length"]) for row in parsed}:
            continue
        for position in POSITIONS:
            gradient.append({
                "model": model,
                "context_length": context,
                "internal_position": position,
                "coverage": snapshot.get(f"coverage.{context}.{position}"),
                "coverage_ci_low": intervals.get(f"coverage.{context}.{position}", {}).get("low") if intervals.get(f"coverage.{context}.{position}") else None,
                "coverage_ci_high": intervals.get(f"coverage.{context}.{position}", {}).get("high") if intervals.get(f"coverage.{context}.{position}") else None,
                "raw_accuracy": snapshot.get(f"accuracy.{context}.{position}"),
                "post_filter_accuracy": snapshot.get(f"post_filter_accuracy.{context}.{position}"),
                "reported_gap": snapshot.get(f"reported_gap.{context}.{position}"),
            })
            
    def _cov(ctx: str, pos: str) -> float | None:
        return snapshot.get(f"coverage.{ctx}.{pos}")
        
    def _edge_cov(ctx: str) -> float | None:
        return swap_consistency(
            row
            for row in pairs
            if row["context_length"] == ctx
            and row["internal_position"] in ("early", "late")
        )
        
    diagnostics = {}
    if "4K" in {str(row["context_length"]) for row in parsed} and "16K" in {str(row["context_length"]) for row in parsed}:
        diagnostics = {
            "coverage_drop_4k_edge_to_4k_middle": _subtract(_edge_cov("4K"), _cov("4K", "middle")),
            "coverage_drop_16k_edge_to_16k_middle": _subtract(_edge_cov("16K"), _cov("16K", "middle")),
            "coverage_drop_early_4k_to_16k": _subtract(_cov("4K", "early"), _cov("16K", "early")),
            "coverage_drop_middle_4k_to_16k": _subtract(_cov("4K", "middle"), _cov("16K", "middle")),
            "coverage_drop_late_4k_to_16k": _subtract(_cov("4K", "late"), _cov("16K", "late")),
        }
        
        mid_lower = all(
            _cov(ctx, "middle") is not None and _cov(ctx, pos) is not None and _cov(ctx, "middle") < _cov(ctx, pos)
            for ctx in ("4K", "16K") for pos in ("early", "late")
        )
        len_lower = all(
            _cov("16K", pos) is not None and _cov("4K", pos) is not None and _cov("16K", pos) < _cov("4K", pos)
            for pos in POSITIONS
        )
        
        diagnostics["middle_lower_than_edges_within_each_length"] = mid_lower
        diagnostics["coverage_lower_at_16k_than_4k_within_each_position"] = len_lower

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
        "coverage_difficulty_gradient": gradient,
        "effective_hard_distribution_coverage": snapshot.get(f"coverage.{pilot_context}.middle"),
        "coverage_diagnostics": diagnostics,
    }
    if "4K" in {str(row["context_length"]) for row in parsed} and pilot_context == "16K":
        short_metrics = compute_pilot_metrics(parsed, pairs, "4K")
        metrics["mechanism_check"] = {
            "delta_acc_16K_minus_4K": _subtract(pilot_metrics.delta_acc, short_metrics.delta_acc),
            "delta_cer_16K_minus_4K": _subtract(pilot_metrics.delta_cer, short_metrics.delta_cer),
        }
    acc_interval = intervals.get("pilot.delta_acc") or {}
    cer_interval = intervals.get("pilot.delta_cer") or {}
    metrics["strong_pilot_evidence"] = bool(
        acc_interval.get("low") is not None and float(acc_interval["low"]) > 0
        and cer_interval.get("low") is not None and float(cer_interval["low"]) > 0
    )
    write_json(run_dir / "metrics.json", metrics)
    write_json(run_dir / "pilot_decision.json", decision)
    contexts = sorted({str(row["context_length"]) for row in parsed})
    output_dir = Path(config["paths"].get("results_dir", run_dir)) if "source_rows" in config["paths"] else run_dir
    generate_figures(output_dir, snapshot, intervals, contexts, pilot_context)
    if "source_rows" in config["paths"]:
        write_required_outputs(
            output_dir,
            parsed,
            pairs,
            snapshot,
            intervals,
            decision,
            model=model,
            pilot_context=pilot_context,
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
    parser.add_argument("--run-id")
    parser.add_argument("--runs-dir", default="data/runs")
    parser.add_argument("--config")
    parser.add_argument("--split", choices=("smoke", "main"), default="main")
    args = parser.parse_args()
    if args.config:
        from .utils import load_config
        config = load_config(args.config)
        runs_dir = Path(config["paths"]["runs_dir"])
        if args.run_id:
            run_dir = runs_dir / args.run_id
        else:
            candidates = sorted((path for path in runs_dir.iterdir() if path.is_dir() and (path / "manifest.json").exists()), key=lambda path: path.stat().st_mtime)
            if not candidates:
                raise SystemExit("no run directory found; pass --run-id")
            run_dir = candidates[-1]
        analyze_run(run_dir)
    else:
        if not args.run_id:
            raise SystemExit("--run-id is required unless --config is supplied")
        analyze_run(Path(args.runs_dir) / args.run_id)


if __name__ == "__main__":
    main()
