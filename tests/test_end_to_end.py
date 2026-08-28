from __future__ import annotations

from pathlib import Path

import yaml

from conftest import make_config, make_item
from src.analyze import analyze_run
from src.build_conditions import build_conditions
from src.prompts import load_prompt_template
from src.run_judge import run
from src.utils import write_jsonl


def test_mock_pipeline_writes_auditable_outputs(tmp_path: Path) -> None:
    source = tmp_path / "base_items.jsonl"
    conditions_path = tmp_path / "conditions.jsonl"
    validation_path = tmp_path / "validation.json"
    prompt_path = tmp_path / "prompt.txt"
    runs_dir = tmp_path / "runs"
    config_path = tmp_path / "pilot.yaml"
    item = make_item()
    write_jsonl(source, [item.to_dict()])
    prompt_path.write_text(
        "Compare the responses.\nA:\n{candidate_a_text}\n"
        "B:\n{candidate_b_text}\nEnd with VERDICT: A or VERDICT: B.",
        encoding="utf-8",
    )
    config = make_config()
    config.update(
        {
            "experiment_name": "pipeline_test",
            "seed": 7,
            "paths": {
                "source_items": str(source),
                "conditions": str(conditions_path),
                "validation_report": str(validation_path),
                "runs_dir": str(runs_dir),
                "prompt_template": str(prompt_path),
            },
            "judge": {
                "provider": "mock",
                "model": "deterministic-oracle-v1",
                "temperature": 0,
                "max_output_tokens": 32,
                "timeout_seconds": 10,
                "max_transport_retries": 1,
            },
            "bootstrap": {"n_resamples": 10, "ci": 0.95, "seed": 8},
            "pilot_context_length": "tiny",
            "pilot_thresholds": {
                "min_accuracy_gap": 0.10,
                "min_middle_swap_consistency": 0.80,
                "max_swap_consistency_gap": 0.05,
                "min_cer_gap": 0.10,
            },
            "integrity": {
                "min_parse_success_rate": 0.99,
                "max_candidate_length_difference_fraction": 1.0,
            },
        }
    )
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    template = load_prompt_template(prompt_path)
    conditions = build_conditions(config, template)
    write_jsonl(
        conditions_path, [condition.to_dict() for condition in conditions]
    )

    run_dir = run(config_path, "mock_001")
    result = analyze_run(run_dir)

    assert (run_dir / "manifest.json").exists()
    assert len((run_dir / "requests.jsonl").read_text().splitlines()) == 6
    assert len((run_dir / "responses.jsonl").read_text().splitlines()) == 6
    assert (run_dir / "parsed.jsonl").exists()
    assert (run_dir / "pair_level.jsonl").exists()
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "pilot_decision.json").exists()
    assert len(list((run_dir / "figures").glob("*.png"))) == 5
    assert result["decision"]["decision"] in {"GO", "KILL"}
