from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

from .utils import read_jsonl, sha256_file, write_json


POSITIONS = ("early", "middle", "late")
PAIR_OUTCOMES = (
    "certified_correct",
    "certified_wrong",
    "stable_tie",
    "mixed_tie",
    "order_disagreement",
)


def _mean(values: Iterable[bool | float]) -> float | None:
    materialized = [float(value) for value in values]
    return sum(materialized) / len(materialized) if materialized else None


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    index = probability * (len(ordered) - 1)
    lower, upper = math.floor(index), math.ceil(index)
    if lower == upper:
        return ordered[lower]
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _binomial_cdf(k: int, n: int, probability: float) -> float:
    return sum(
        math.comb(n, index)
        * probability**index
        * (1.0 - probability) ** (n - index)
        for index in range(k + 1)
    )


def _bisect_probability(
    fn: Callable[[float], float], target: float, *, increasing: bool
) -> float:
    low, high = 0.0, 1.0
    for _ in range(100):
        midpoint = (low + high) / 2.0
        value = fn(midpoint)
        if (value < target) == increasing:
            low = midpoint
        else:
            high = midpoint
    return (low + high) / 2.0


def clopper_pearson_interval(
    successes: int, trials: int, confidence: float = 0.95
) -> tuple[float, float] | None:
    """Two-sided exact binomial interval, computed from finite binomial tails."""
    if trials == 0:
        return None
    if not 0 <= successes <= trials:
        raise ValueError("successes must be between zero and trials")
    alpha_tail = (1.0 - confidence) / 2.0
    lower = 0.0 if successes == 0 else _bisect_probability(
        lambda p: 1.0 - _binomial_cdf(successes - 1, trials, p),
        alpha_tail,
        increasing=True,
    )
    upper = 1.0 if successes == trials else _bisect_probability(
        lambda p: _binomial_cdf(successes, trials, p),
        alpha_tail,
        increasing=False,
    )
    return lower, upper


def classify_pair(clean_first: str, flawed_first: str) -> str:
    verdicts = (clean_first, flawed_first)
    if verdicts == ("clean", "clean"):
        return "certified_correct"
    if verdicts == ("flawed", "flawed"):
        return "certified_wrong"
    if verdicts == ("tie", "tie"):
        return "stable_tie"
    if "tie" in verdicts:
        return "mixed_tie"
    if set(verdicts) == {"clean", "flawed"}:
        return "order_disagreement"
    raise ValueError(f"invalid pair verdicts: {verdicts!r}")


def aggregate_paper_pairs(parsed: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in parsed:
        grouped[(str(row["item_id"]), str(row["context_length"]), str(row["internal_position"]))].append(row)
    output: list[dict[str, Any]] = []
    for (item_id, length, position), rows in sorted(grouped.items()):
        by_order = {str(row["candidate_order"]): row for row in rows}
        complete = len(rows) == 2 and set(by_order) == {"clean_first", "flawed_first"}
        parse_ok = complete and all(row.get("parse_status") == "ok" for row in rows)
        clean_verdict = by_order.get("clean_first", {}).get("content_verdict")
        flawed_verdict = by_order.get("flawed_first", {}).get("content_verdict")
        outcome = classify_pair(str(clean_verdict), str(flawed_verdict)) if parse_ok else None
        output.append({
            "item_id": item_id,
            "context_length": length,
            "internal_position": position,
            "clean_first_content_verdict": clean_verdict,
            "flawed_first_content_verdict": flawed_verdict,
            "pair_complete": complete,
            "parse_ok": bool(parse_ok),
            "pair_outcome": outcome,
            "position_consistent": bool(parse_ok and clean_verdict == flawed_verdict),
            "decisively_certified": outcome in ("certified_correct", "certified_wrong"),
            "certified_wrong": outcome == "certified_wrong",
        })
    return output


def _cell_summary(
    calls: list[dict[str, Any]], pairs: list[dict[str, Any]]
) -> dict[str, Any]:
    valid_calls = [row for row in calls if row.get("parse_status") == "ok"]
    valid_pairs = [row for row in pairs if row.get("parse_ok")]
    counts = Counter(str(row["pair_outcome"]) for row in valid_pairs)
    n_certified = counts["certified_correct"] + counts["certified_wrong"]
    selective_risk = counts["certified_wrong"] / n_certified if n_certified else None
    exact = clopper_pearson_interval(counts["certified_wrong"], n_certified)
    result: dict[str, Any] = {
        "n_pairs": len(valid_pairs),
        "n_calls": len(valid_calls),
        "raw_accuracy": _mean(row.get("is_correct") is True for row in valid_calls),
        "position_consistency": _mean(row["position_consistent"] for row in valid_pairs),
        "certification_coverage": n_certified / len(valid_pairs) if valid_pairs else None,
        "n_certified": n_certified,
        "n_certified_correct": counts["certified_correct"],
        "n_certified_wrong": counts["certified_wrong"],
        "selective_risk": selective_risk,
        "selective_risk_exact_ci": list(exact) if exact else None,
    }
    for outcome in PAIR_OUTCOMES:
        result[outcome] = counts[outcome]
        result[f"{outcome}_fraction"] = counts[outcome] / len(valid_pairs) if valid_pairs else None
    result["rejection_rate"] = 1.0 - result["certification_coverage"] if result["certification_coverage"] is not None else None
    tie_involved = counts["stable_tie"] + counts["mixed_tie"]
    noncertified = tie_involved + counts["order_disagreement"]
    result["tie_involved_fraction"] = tie_involved / len(valid_pairs) if valid_pairs else None
    result["pure_order_disagreement_fraction"] = counts["order_disagreement"] / len(valid_pairs) if valid_pairs else None
    result["tie_share_of_noncertification"] = tie_involved / noncertified if noncertified else None
    return result


def summarize_cells(
    parsed: list[dict[str, Any]], pairs: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    lengths = sorted({str(row["context_length"]) for row in parsed}, key=lambda x: int(re.sub(r"\D", "", x) or 0))
    output: dict[str, dict[str, Any]] = {}
    for length in lengths:
        for position in POSITIONS:
            key = f"{length}.{position}"
            output[key] = _cell_summary(
                [row for row in parsed if row["context_length"] == length and row["internal_position"] == position],
                [row for row in pairs if row["context_length"] == length and row["internal_position"] == position],
            )
    return output


def _rate(
    parsed: list[dict[str, Any]], pairs: list[dict[str, Any]], length: str,
    positions: tuple[str, ...], metric: str, order: str | None = None,
) -> float | None:
    if metric == "raw_accuracy":
        rows = [row for row in parsed if row["context_length"] == length and row["internal_position"] in positions and row.get("parse_status") == "ok" and (order is None or row["candidate_order"] == order)]
        return _mean(row.get("is_correct") is True for row in rows)
    rows = [row for row in pairs if row["context_length"] == length and row["internal_position"] in positions and row.get("parse_ok")]
    if metric == "position_consistency":
        return _mean(row["position_consistent"] for row in rows)
    if metric == "certification_coverage":
        return _mean(row["decisively_certified"] for row in rows)
    raise ValueError(metric)


def compute_interactions(
    parsed: list[dict[str, Any]], pairs: list[dict[str, Any]]
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for metric in ("raw_accuracy", "position_consistency", "certification_coverage"):
        by_length: dict[str, Any] = {}
        for length in ("4K", "16K"):
            edge = _rate(parsed, pairs, length, ("early", "late"), metric)
            middle = _rate(parsed, pairs, length, ("middle",), metric)
            by_length[length] = {
                "edge": edge,
                "middle": middle,
                "middle_penalty": edge - middle if edge is not None and middle is not None else None,
            }
        short, long = by_length["4K"]["middle_penalty"], by_length["16K"]["middle_penalty"]
        output[metric] = {
            "by_length": by_length,
            "interaction": long - short if long is not None and short is not None else None,
        }
    order_results: dict[str, Any] = {}
    for order in ("clean_first", "flawed_first"):
        penalties: dict[str, float | None] = {}
        for length in ("4K", "16K"):
            edge = _rate(parsed, pairs, length, ("early", "late"), "raw_accuracy", order)
            middle = _rate(parsed, pairs, length, ("middle",), "raw_accuracy", order)
            penalties[length] = edge - middle if edge is not None and middle is not None else None
        order_results[order] = {
            "middle_penalty_by_length": penalties,
            "interaction": penalties["16K"] - penalties["4K"] if None not in penalties.values() else None,
        }
    output["candidate_order_robustness"] = order_results
    return output


def packet_bootstrap(
    parsed: list[dict[str, Any]], pairs: list[dict[str, Any]], *,
    n_resamples: int = 10_000, confidence: float = 0.95, seed: int = 20260828,
) -> dict[str, Any]:
    packet_ids = sorted({str(row["item_id"]) for row in parsed})
    parsed_by_packet = {packet: [row for row in parsed if row["item_id"] == packet] for packet in packet_ids}
    pairs_by_packet = {packet: [row for row in pairs if row["item_id"] == packet] for packet in packet_ids}
    samples: dict[str, list[float]] = defaultdict(list)
    rng = random.Random(seed)
    for _ in range(n_resamples):
        selected = rng.choices(packet_ids, k=len(packet_ids))
        sampled_calls = [row for packet in selected for row in parsed_by_packet[packet]]
        sampled_pairs = [row for packet in selected for row in pairs_by_packet[packet]]
        for cell, summary in summarize_cells(sampled_calls, sampled_pairs).items():
            for metric in ("raw_accuracy", "position_consistency", "certification_coverage"):
                value = summary[metric]
                if value is not None:
                    samples[f"cells.{cell}.{metric}"].append(value)
        interactions = compute_interactions(sampled_calls, sampled_pairs)
        for metric in ("raw_accuracy", "position_consistency", "certification_coverage"):
            value = interactions[metric]["interaction"]
            if value is not None:
                samples[f"interactions.{metric}"].append(value)
        for order in ("clean_first", "flawed_first"):
            value = interactions["candidate_order_robustness"][order]["interaction"]
            if value is not None:
                samples[f"order.{order}"].append(value)
    alpha = (1.0 - confidence) / 2.0
    return {
        key: {
            "low": _percentile(values, alpha),
            "high": _percentile(values, 1.0 - alpha),
            "valid_draws": len(values),
        }
        for key, values in sorted(samples.items()) if values
    }


def _finish_reason(response: dict[str, Any]) -> str | None:
    raw = response.get("raw_response")
    if not isinstance(raw, dict):
        return None
    candidates = raw.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        return None
    value = candidates[0].get("finish_reason") if isinstance(candidates[0], dict) else None
    return str(value).upper() if value is not None else None


def audit_gemini_run(
    run_dir: Path, conditions_path: Path, prompt_path: Path, freeze_path: Path
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    frozen = json.loads(freeze_path.read_text(encoding="utf-8"))
    conditions = list(read_jsonl(conditions_path))
    responses = list(read_jsonl(run_dir / "responses.jsonl"))
    parsed = list(read_jsonl(run_dir / "parsed.jsonl"))
    pairs = aggregate_paper_pairs(parsed)
    response_ids = [str(row["condition_id"]) for row in responses]
    returned_models = sorted({str(row.get("model_returned") or row.get("returned_model")) for row in responses if row.get("model_returned") or row.get("returned_model")})
    cell_counts = Counter((row["context_length"], row["internal_position"]) for row in pairs if row["parse_ok"])
    incomplete_finish_reasons = [reason for reason in map(_finish_reason, responses) if reason not in (None, "STOP", "COMPLETE", "FINISH_REASON_UNSPECIFIED")]
    condition_hash = sha256_file(conditions_path)
    prompt_hash = sha256_file(prompt_path)
    checks = {
        "exactly_480_completed_conditions": len(responses) == 480 and not any(row.get("error_status") for row in responses),
        "exactly_240_complete_pairs": len(pairs) == 240 and all(row["pair_complete"] for row in pairs),
        "exactly_40_packets": len({row["item_id"] for row in parsed}) == 40,
        "forty_pairs_per_cell": all(cell_counts[(length, position)] == 40 for length in ("4K", "16K") for position in POSITIONS),
        "parse_success_100_percent": len(parsed) == 480 and all(row.get("parse_status") == "ok" for row in parsed),
        "single_returned_model": len(returned_models) == 1,
        "no_duplicate_condition_ids": len(response_ids) == len(set(response_ids)),
        "no_incomplete_pairs": all(row["pair_complete"] for row in pairs),
        "condition_hash_matches": condition_hash == manifest.get("conditions_hash") == frozen.get("condition_file_sha256"),
        "prompt_hash_matches": prompt_hash == frozen.get("prompt_sha256"),
        "verdict_metadata_present": all(row.get("parsed_winner_A_B_tie") in ("A", "B", "tie") for row in responses),
        "no_incomplete_responses": not incomplete_finish_reasons,
    }
    missing_tokens = sum(row.get("prompt_token_count") is None and row.get("prompt_tokens") is None for row in responses)
    report = {
        "passed": all(checks.values()),
        "checks": checks,
        "run_id": manifest.get("run_id", run_dir.name),
        "git_sha": manifest.get("git_commit"),
        "condition_hash": condition_hash,
        "prompt_hash": prompt_hash,
        "configured_model": manifest.get("model"),
        "returned_model_ids": returned_models,
        "n_packets": len({row["item_id"] for row in parsed}),
        "n_calls": len(responses),
        "n_pairs": len(pairs),
        "parse_rate": sum(row.get("parse_status") == "ok" for row in parsed) / len(parsed) if parsed else 0.0,
        "duplicate_count": len(response_ids) - len(set(response_ids)),
        "incomplete_pair_count": sum(not row["pair_complete"] for row in pairs),
        "missing_prompt_token_metadata": missing_tokens,
        "provider_error_count": sum(bool(row.get("error_status")) for row in responses),
        "incomplete_response_count": len(incomplete_finish_reasons),
    }
    return report, conditions, responses, parsed


def _stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {name: None for name in ("n", "mean", "median", "min", "max", "p05", "p95")}
    return {
        "n": len(values), "mean": sum(values) / len(values),
        "median": _percentile(values, 0.5), "min": min(values), "max": max(values),
        "p05": _percentile(values, 0.05), "p95": _percentile(values, 0.95),
    }


def _midpoint_fraction(text: str, target_index: int, counter: Callable[[str], int]) -> float:
    blocks = text.split("\n\n")
    if not 0 <= target_index < len(blocks):
        raise ValueError(f"target index {target_index} outside {len(blocks)} rendered blocks")
    before = "\n\n".join(blocks[:target_index]) + ("\n\n" if target_index else "")
    through = "\n\n".join(blocks[: target_index + 1])
    return (counter(before) + counter(through)) / 2.0 / max(counter(text), 1)


def native_token_rows(
    conditions: list[dict[str, Any]], responses: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], bool, dict[str, Any]]:
    response_by_id = {str(row["condition_id"]): row for row in responses}
    observations: list[dict[str, Any]] = []
    word_counter = lambda text: len(re.findall(r"\S+", text))
    for condition in conditions:
        response = response_by_id.get(str(condition["condition_id"]), {})
        prompt_tokens = response.get("prompt_token_count", response.get("prompt_tokens"))
        text = str(condition["clean_candidate_text"])
        index = int(condition["target_index"])
        observations.append({
            "length": str(condition["length"]),
            "position": str(condition["internal_position"]),
            "prompt_tokens": float(prompt_tokens) if prompt_tokens is not None else None,
            "character_midpoint_fraction": _midpoint_fraction(text, index, len),
            "whitespace_midpoint_fraction": _midpoint_fraction(text, index, word_counter),
        })
    rows: list[dict[str, Any]] = []
    for length in ("4K", "16K"):
        groups = [("length", "all", [row for row in observations if row["length"] == length])]
        groups.extend(("length_position", position, [row for row in observations if row["length"] == length and row["position"] == position]) for position in POSITIONS)
        for scope, position, group in groups:
            output: dict[str, Any] = {"scope": scope, "length": length, "position": position}
            for field in ("prompt_tokens", "character_midpoint_fraction", "whitespace_midpoint_fraction"):
                summary = _stats([float(row[field]) for row in group if row[field] is not None])
                output.update({f"{field}_{name}": value for name, value in summary.items()})
            rows.append(output)
    separation: dict[str, Any] = {}
    passed = True
    for length in ("4K", "16K"):
        separation[length] = {}
        for field in ("character_midpoint_fraction", "whitespace_midpoint_fraction"):
            values = {position: [float(row[field]) for row in observations if row["length"] == length and row["position"] == position] for position in POSITIONS}
            clean = max(values["early"]) < min(values["middle"]) and max(values["middle"]) < min(values["late"])
            separation[length][field] = {"passed": clean, "ranges": {position: [min(items), max(items)] for position, items in values.items()}}
            passed = passed and clean
    return rows, passed, separation


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _ci_text(value: float | None, interval: dict[str, Any] | None) -> str:
    if value is None:
        return "NA"
    if not interval:
        return f"{value:.1%}"
    return f"{value:.1%} [{float(interval['low']):.1%}, {float(interval['high']):.1%}]"


def _main_table_rows(cells: dict[str, dict[str, Any]], intervals: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for length in ("4K", "16K"):
        for position in POSITIONS:
            key, cell = f"{length}.{position}", cells[f"{length}.{position}"]
            exact = cell["selective_risk_exact_ci"]
            risk = "NA" if exact is None else f"{cell['n_certified_wrong']}/{cell['n_certified']} [{exact[0]:.1%}, {exact[1]:.1%}]"
            rows.append({
                "Length": length, "Position": position.title(),
                "Raw accuracy [95% CI]": _ci_text(cell["raw_accuracy"], intervals.get(f"cells.{key}.raw_accuracy")),
                "Position consistency n/N [95% CI]": f"{sum(cell[o] for o in ('certified_correct','certified_wrong','stable_tie'))}/{cell['n_pairs']} " + _ci_text(cell["position_consistency"], intervals.get(f"cells.{key}.position_consistency")),
                "Decisive certification coverage n/N [95% CI]": f"{cell['n_certified']}/{cell['n_pairs']} " + _ci_text(cell["certification_coverage"], intervals.get(f"cells.{key}.certification_coverage")),
                "Certified errors k/n [exact 95% CI]": risk,
                "Stable tie %": f"{cell['stable_tie_fraction']:.1%}",
                "Mixed tie %": f"{cell['mixed_tie_fraction']:.1%}",
                "Order disagreement %": f"{cell['order_disagreement_fraction']:.1%}",
            })
    return rows


def _write_markdown_table(path: Path, rows: list[dict[str, Any]]) -> None:
    headers = list(rows[0])
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(str(row[name]) for name in headers) + " |" for row in rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _generate_figures(output_dir: Path, cells: dict[str, dict[str, Any]], intervals: dict[str, Any]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.5), sharex=True)
    x = range(3)
    for axis, metric, title in zip(axes, ("raw_accuracy", "certification_coverage"), ("A. Raw accuracy", "B. Decisive certification coverage")):
        for length in ("4K", "16K"):
            values, lows, highs = [], [], []
            for position in POSITIONS:
                key = f"{length}.{position}"
                value = float(cells[key][metric])
                ci = intervals[f"cells.{key}.{metric}"]
                values.append(value); lows.append(value - ci["low"]); highs.append(ci["high"] - value)
            axis.errorbar(list(x), values, yerr=[lows, highs], marker="o", capsize=3, label=length)
        axis.set_title(title); axis.set_xticks(list(x), [p.title() for p in POSITIONS]); axis.set_ylim(0, 1); axis.grid(alpha=0.2)
    axes[0].set_ylabel("Rate")
    axes[1].legend(title="Construction length")
    fig.tight_layout(); fig.savefig(figures / "main_accuracy_coverage.png", dpi=220); plt.close(fig)

    fig, axis = plt.subplots(figsize=(7.4, 3.8))
    labels = [f"{length}\n{position.title()}" for length in ("4K", "16K") for position in POSITIONS]
    bottoms = [0.0] * len(labels)
    colors = ("#3b7ddd", "#d9534f", "#8c8c8c", "#f0ad4e", "#7a5aa6")
    for outcome, color in zip(PAIR_OUTCOMES, colors):
        values = [cells[f"{length}.{position}"][f"{outcome}_fraction"] for length in ("4K", "16K") for position in POSITIONS]
        axis.bar(range(len(labels)), values, bottom=bottoms, label=outcome.replace("_", " "), color=color)
        bottoms = [left + value for left, value in zip(bottoms, values)]
    axis.set_xticks(range(len(labels)), labels); axis.set_ylim(0, 1); axis.set_ylabel("Pair fraction"); axis.legend(fontsize=8, ncols=2)
    fig.tight_layout(); fig.savefig(figures / "pair_outcome_decomposition.png", dpi=220); plt.close(fig)


def _selection_diagnostic(parsed: list[dict[str, Any]], pairs: list[dict[str, Any]]) -> dict[str, Any]:
    certified = {(row["item_id"], row["context_length"], row["internal_position"]): row["decisively_certified"] for row in pairs if row["parse_ok"]}
    groups: dict[str, list[bool]] = {"certified_pairs": [], "rejected_pairs": []}
    for row in parsed:
        key = (row["item_id"], row["context_length"], row["internal_position"])
        if row.get("parse_status") == "ok" and key in certified:
            groups["certified_pairs" if certified[key] else "rejected_pairs"].append(row.get("is_correct") is True)
    return {name: {"n_calls": len(values), "raw_accuracy": _mean(values)} for name, values in groups.items()}


def generate_gemini_paper_analysis(
    run_dir: str | Path = "results/gemini/raw/main_pilot_3",
    output_dir: str | Path = "results/gemini/paper",
) -> dict[str, Any]:
    run_path, output_path = Path(run_dir), Path(output_dir)
    manifest = json.loads((run_path / "manifest.json").read_text(encoding="utf-8"))
    paths = manifest["config"]["paths"]
    output_path.mkdir(parents=True, exist_ok=True)
    audit, conditions, responses, parsed = audit_gemini_run(
        run_path, Path(paths["main_conditions"]), Path(paths["prompt_template"]), Path(paths["freeze_manifest"])
    )
    write_json(output_path / "audit.json", audit)
    if not audit["passed"]:
        raise RuntimeError("Gemini audit failed; see audit.json. Historical artifacts were not modified.")
    pairs = aggregate_paper_pairs(parsed)
    cells = summarize_cells(parsed, pairs)
    intervals = packet_bootstrap(parsed, pairs, n_resamples=10_000, confidence=0.95, seed=20260828)
    interactions = compute_interactions(parsed, pairs)
    for metric in ("raw_accuracy", "position_consistency", "certification_coverage"):
        interactions[metric]["bootstrap_95_ci"] = intervals.get(f"interactions.{metric}")
    for order in ("clean_first", "flawed_first"):
        interactions["candidate_order_robustness"][order]["bootstrap_95_ci"] = intervals.get(f"order.{order}")
    interactions["bootstrap"] = {"method": "packet_cluster_percentile", "n_resamples": 10_000, "confidence": 0.95, "seed": 20260828}
    interactions["selection_diagnostic"] = _selection_diagnostic(parsed, pairs)
    token_rows, positions_ok, separation = native_token_rows(conditions, responses)
    interactions["tokenizer_independent_position_check"] = separation
    if not positions_ok:
        raise RuntimeError("tokenizer-independent target locations are not separated; paper outputs stopped")
    pair_rows = []
    for key, cell in cells.items():
        length, position = key.split(".")
        row = {"length": length, "internal_position": position, **cell}
        for name in ("selective_risk_exact_ci",):
            row[name] = json.dumps(row[name])
        pair_rows.append(row)
    _write_csv(output_path / "pair_outcomes.csv", pair_rows)
    write_json(output_path / "pair_outcomes.json", {"cells": cells, "partition": list(PAIR_OUTCOMES)})
    write_json(output_path / "interactions.json", interactions)
    _write_csv(output_path / "native_token_lengths.csv", token_rows)
    table_rows = _main_table_rows(cells, intervals)
    _write_csv(output_path / "main_table.csv", table_rows)
    _write_markdown_table(output_path / "main_table.md", table_rows)
    _generate_figures(output_path, cells, intervals)
    return {"audit": audit, "cells": cells, "interactions": interactions, "output_dir": str(output_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate frozen Gemini paper reanalysis")
    parser.add_argument("--run-dir", default="results/gemini/raw/main_pilot_3")
    parser.add_argument("--output-dir", default="results/gemini/paper")
    args = parser.parse_args()
    result = generate_gemini_paper_analysis(args.run_dir, args.output_dir)
    print(f"Gemini audit passed; paper artifacts written to {result['output_dir']}")


if __name__ == "__main__":
    main()