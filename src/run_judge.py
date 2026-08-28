from __future__ import annotations

import argparse
import random
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .prompts import load_prompt_template, render_prompt
from .providers.base import (
    JudgeResponse,
    ProviderError,
    RetriableProviderError,
)
from .providers.factory import make_provider
from .schemas import Condition
from .utils import (
    append_jsonl,
    load_config,
    read_jsonl,
    require_mapping,
    sha256_file,
    write_json,
)
from .validate_conditions import validate_from_config


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {str(row["condition_id"]) for row in read_jsonl(path)}


def _prepare_run(
    run_dir: Path,
    run_id: str,
    config: dict[str, Any],
    config_path: Path,
) -> None:
    manifest_path = run_dir / "manifest.json"
    if run_dir.exists() and not manifest_path.exists():
        raise FileExistsError(
            f"{run_dir} already exists without a manifest; choose another run ID"
        )
    paths = require_mapping(config.get("paths"), "paths")
    judge = require_mapping(config.get("judge"), "judge")
    current_fingerprint = {
        "config_hash": sha256_file(config_path),
        "dataset_hash": sha256_file(paths["source_items"]),
        "conditions_hash": sha256_file(paths["conditions"]),
        "prompt_template_hash": sha256_file(paths["prompt_template"]),
    }
    if manifest_path.exists():
        import json

        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        mismatches = [
            name
            for name, value in current_fingerprint.items()
            if existing.get(name) != value
        ]
        if mismatches:
            raise RuntimeError(
                "refusing to resume with changed inputs: "
                + ", ".join(mismatches)
            )
        return
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "run_id": run_id,
        "experiment_name": config["experiment_name"],
        "created_at": utc_now(),
        "git_commit": git_commit(),
        "random_seed": int(config["seed"]),
        "provider": judge["provider"],
        "model": judge["model"],
        "temperature": judge["temperature"],
        "config_path": str(config_path),
        "config": config,
        **current_fingerprint,
    }
    write_json(manifest_path, manifest)


def run(config_path: str | Path, run_id: str) -> Path:
    config_path = Path(config_path)
    config = load_config(config_path)
    paths = require_mapping(config.get("paths"), "paths")
    report = validate_from_config(config)
    write_json(paths["validation_report"], report)
    if not report["passed"]:
        raise RuntimeError(
            "condition validation failed; no judge requests were sent"
        )

    run_dir = Path(paths["runs_dir"]) / run_id
    _prepare_run(run_dir, run_id, config, config_path)
    requests_path = run_dir / "requests.jsonl"
    responses_path = run_dir / "responses.jsonl"
    requested = _existing_ids(requests_path)
    completed = _existing_ids(responses_path)
    conditions = [
        Condition.from_dict(row) for row in read_jsonl(paths["conditions"])
    ]
    random.Random(int(config["seed"])).shuffle(conditions)
    template = load_prompt_template(paths["prompt_template"])
    judge_config = require_mapping(config.get("judge"), "judge")
    provider = make_provider(judge_config)
    retries = int(judge_config.get("max_transport_retries", 0))

    for number, condition in enumerate(conditions, start=1):
        if condition.condition_id in completed:
            continue
        prompt = render_prompt(
            template,
            condition.candidate_a_text,
            condition.candidate_b_text,
        )
        if condition.condition_id not in requested:
            append_jsonl(
                requests_path,
                {
                    "call_id": condition.condition_id,
                    "condition_id": condition.condition_id,
                    "saved_at": utc_now(),
                    "execution_index": number,
                    "provider": judge_config["provider"],
                    "model": judge_config["model"],
                    "temperature": judge_config["temperature"],
                    "max_output_tokens": judge_config["max_output_tokens"],
                    "prompt": prompt,
                },
            )
            requested.add(condition.condition_id)

        result: JudgeResponse | None = None
        last_error: str | None = None
        attempts = 0
        for attempt in range(retries + 1):
            attempts = attempt
            try:
                result = provider.judge(
                    prompt,
                    model=str(judge_config["model"]),
                    temperature=float(judge_config["temperature"]),
                    max_output_tokens=int(judge_config["max_output_tokens"]),
                )
                break
            except RetriableProviderError as exc:
                last_error = str(exc)
                if attempt < retries:
                    time.sleep(min(2**attempt, 30))
            except ProviderError as exc:
                last_error = str(exc)
                break
        if result is None:
            response_record = {
                "condition_id": condition.condition_id,
                "received_at": utc_now(),
                "provider": judge_config["provider"],
                "requested_model": judge_config["model"],
                "returned_model": None,
                "request_id": None,
                "prompt_tokens": None,
                "completion_tokens": None,
                "raw_response_text": "",
                "latency_seconds": None,
                "retry_count": attempts,
                "error_status": last_error or "provider failure",
            }
        else:
            response_record = {
                "condition_id": condition.condition_id,
                "received_at": utc_now(),
                **result.to_dict(),
                "retry_count": attempts,
            }
        append_jsonl(responses_path, response_record)
        print(
            f"[{number}/{len(conditions)}] "
            f"{condition.condition_id}: "
            f"{'ok' if result is not None else 'error'}"
        )
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pairwise judge calls")
    parser.add_argument("--config", default="configs/pilot.yaml")
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    run_dir = run(args.config, args.run_id)
    print(f"Run artifacts written to {run_dir}")


if __name__ == "__main__":
    main()
