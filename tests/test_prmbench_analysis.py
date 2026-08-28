from __future__ import annotations

from pathlib import Path

from src.analyze import analyze_run
from src.utils import write_json, write_jsonl


def test_prmbench_analysis_writes_required_outputs_offline(tmp_path: Path) -> None:
    run_dir = tmp_path / "raw" / "run"
    run_dir.mkdir(parents=True)
    conditions_path = tmp_path / "conditions.jsonl"
    validation_path = tmp_path / "validation.json"
    results_dir = tmp_path / "results"
    conditions = []
    responses = []
    for length in ("4K", "16K"):
        for position in ("early", "middle", "late"):
            for order in ("clean_first", "flawed_first"):
                condition_id = f"p1__{length}__{position}__{order}"
                conditions.append({
                    "condition_id": condition_id,
                    "base_packet_id": "p1",
                    "length": length,
                    "internal_position": position,
                    "candidate_order": order,
                    "target_midpoint_clean": {"early": 0.15, "middle": 0.5, "late": 0.85}[position],
                    "candidate_a_tokens": 100,
                    "candidate_b_tokens": 100,
                })
                winner = "A" if order == "clean_first" else "B"
                responses.append({"condition_id": condition_id, "returned_model": "one-model", "raw_response_text": f'{{"winner":"{winner}","brief_reason":"Clean is correct."}}', "error_status": None})
    write_jsonl(conditions_path, conditions)
    write_jsonl(run_dir / "responses.jsonl", responses)
    write_json(validation_path, {"passed": True})
    config = {
        "paths": {"source_rows": "marker", "main_conditions": str(conditions_path), "main_validation_report": str(validation_path), "results_dir": str(results_dir)},
        "lengths": {"4K": {}, "16K": {}},
        "bootstrap": {"n_resamples": 5, "ci": 0.95, "seed": 1},
        "pilot_context_length": "16K",
        "pilot_thresholds": {"min_accuracy_gap": 0.10, "min_middle_swap_consistency": 0.80, "max_swap_consistency_gap": 0.05, "min_cer_gap": 0.10},
        "integrity": {"min_parse_success_rate": 1.0, "max_candidate_length_difference_fraction": 0.10},
    }
    write_json(run_dir / "manifest.json", {"split": "main", "config": config})
    result = analyze_run(run_dir)
    assert result["decision"]["decision"] == "KILL"  # intentionally incomplete: integrity fails closed
    assert (results_dir / "pilot_summary.csv").exists()
    assert (results_dir / "pilot_summary.md").exists()
    assert (results_dir / "go_kill.json").exists()
    assert len(list((results_dir / "figures").glob("*.png"))) == 5
