from __future__ import annotations

import argparse
import json
import os
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
    load_env_file,
    load_config,
    read_jsonl,
    require_mapping,
    sha256_file,
    write_json,
)
from .validate_conditions import validate_from_config
from .validate_conditions import validate_prmbench_from_config


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
        if existing.get("provider") != judge["provider"] or existing.get("model") != judge["model"]:
            mismatches.append("provider/model (no mid-run failover allowed)")
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


def provider_preflight(config: dict[str, Any], *, trial_confirmed: bool, available_call_budget: int | None) -> None:
    load_env_file()
    judge = require_mapping(config.get("judge"), "judge")
    budget = require_mapping(config.get("budget"), "budget")
    provider = str(judge["provider"]).lower()
    env_var_map = {
        "cohere": "COHERE_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "google": "GEMINI_API_KEY",
        "openai": "OPENAI_API_KEY",
    }
    env_var = env_var_map.get(provider, f"{provider.upper()}_API_KEY")
    required = int(budget["required_available_calls"])
    key_present = bool(os.environ.get(env_var)) or (provider in ("gemini", "google") and bool(os.environ.get("GOOGLE_API_KEY")))
    print(f"provider: {judge['provider']}")
    print(f"model: {judge['model']}")
    print(f"API key ({env_var}) present: {'yes' if key_present else 'no'}")
    print(f"trial/evaluation mode confirmed by operator: {'yes' if trial_confirmed else 'no'}")
    print(f"monthly call budget entered by operator: {available_call_budget if available_call_budget is not None else 'not entered'}")
    print(f"planned maximum calls: {int(budget['hard_api_call_cap'])}")
    if not key_present:
        raise RuntimeError(f"{env_var} is not present")
    if not trial_confirmed:
        raise RuntimeError("operator must explicitly confirm trial/evaluation mode")
    if available_call_budget is None or available_call_budget < required:
        raise RuntimeError(f"available free call budget must be at least {required}")


def _counter_value(path: Path) -> int:
    if not path.exists():
        return 0
    value = json.loads(path.read_text(encoding="utf-8"))
    return int(value.get("api_calls", 0))


def _increment_counter(path: Path, cap: int) -> int:
    current = _counter_value(path)
    if current >= cap:
        raise RuntimeError(f"hard API-call cap of {cap} reached; request not sent")
    current += 1
    write_json(path, {"api_calls": current, "updated_at": utc_now()})
    return current


def run_prmbench(config_path: str | Path, run_id: str, split: str, *, trial_confirmed: bool, available_call_budget: int | None, limit: int | None = None, condition_id: str | None = None) -> Path:
    config_path = Path(config_path)
    config = load_config(config_path)
    paths = require_mapping(config.get("paths"), "paths")
    report = validate_prmbench_from_config(config, split)
    write_json(paths[f"{split}_validation_report"], report)
    if not report["passed"]:
        raise RuntimeError("condition validation failed; no judge requests were sent")
    if split == "main":
        freeze_path = Path(paths["freeze_manifest"])
        if not freeze_path.exists():
            raise RuntimeError("main pilot is not frozen")
        frozen = json.loads(freeze_path.read_text(encoding="utf-8"))
        if frozen.get("condition_file_sha256") != sha256_file(paths["main_conditions"]):
            raise RuntimeError("main condition file changed after freeze")
        source_manifest = json.loads(Path(paths["source_manifest"]).read_text(encoding="utf-8"))
        frozen_checks = {
            "prompt": frozen.get("prompt_sha256") == sha256_file(paths["prompt_template"]),
            "provider": frozen.get("provider") == config["judge"]["provider"],
            "model": frozen.get("model_id") == config["judge"]["model"],
            "dataset revision": frozen.get("dataset_revision") == source_manifest.get("revision"),
            "dataset fingerprint": frozen.get("dataset_fingerprint") == source_manifest.get("dataset_fingerprint"),
            "Git commit": frozen.get("git_commit") == git_commit(),
        }
        changed = [name for name, passed in frozen_checks.items() if not passed]
        if changed:
            raise RuntimeError("main pilot inputs changed after freeze: " + ", ".join(changed))
    provider_preflight(config, trial_confirmed=trial_confirmed, available_call_budget=available_call_budget)
    judge_config = require_mapping(config.get("judge"), "judge")
    run_dir = Path(paths["runs_dir"]) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "manifest.json"
    conditions_path = Path(paths[f"{split}_conditions"])
    fingerprint = sha256_file(conditions_path)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("conditions_hash") != fingerprint:
            raise RuntimeError("refusing to resume with changed conditions")
        if manifest.get("model") != judge_config["model"] or manifest.get("provider") != judge_config["provider"]:
            raise RuntimeError("no mid-run model failover allowed: resuming a run requires the exact same model and provider")
    else:
        write_json(manifest_path, {"run_id": run_id, "split": split, "created_at": utc_now(), "git_commit": git_commit(), "conditions_hash": fingerprint, "config": config, "provider": judge_config["provider"], "model": judge_config["model"]})
    conditions = list(read_jsonl(conditions_path))
    random.Random(int(config["seed"])).shuffle(conditions)
    if condition_id:
        conditions = [c for c in conditions if str(c.get("condition_id")) == condition_id]
        if not conditions:
            raise ValueError(f"condition ID {condition_id} not found in {conditions_path}")
    if limit is not None and limit > 0:
        conditions = conditions[:limit]
    responses_path = run_dir / "responses.jsonl"
    requests_path = run_dir / "requests.jsonl"
    completed = _existing_ids(responses_path)
    requested = _existing_ids(requests_path)
    provider = make_provider(judge_config)
    template = load_prompt_template(paths["prompt_template"])
    scheduler_config = config.get("scheduler", {})
    retries = int(scheduler_config.get("max_transport_retries", judge_config.get("max_transport_retries", 0)))
    budget_map = config.get("budget")
    cap_val = scheduler_config.get("hard_attempt_cap")
    if cap_val is not None:
        cap = int(cap_val)
    else:
        cap = int(require_mapping(budget_map, "budget")["hard_api_call_cap"])
    counter_path = Path(paths["runs_dir"]) / "api_call_counter.json"
    
    # Calculate minimum interval from scheduler or max_requests_per_minute
    if "minimum_interval_seconds" in scheduler_config:
        minimum_interval = float(scheduler_config["minimum_interval_seconds"])
    else:
        minimum_interval = 60.0 / float(judge_config.get("max_requests_per_minute", 15))
        
    base_backoff_seconds = float(scheduler_config.get("base_backoff_seconds", 1.0))
    
    last_attempt_at = 0.0
    for number, condition in enumerate(conditions, start=1):
        condition_id = str(condition["condition_id"])
        if condition_id in completed:
            continue
        prompt = render_prompt(template, str(condition["candidate_a_text"]), str(condition["candidate_b_text"]))
        if condition_id not in requested:
            append_jsonl(requests_path, {"run_id": run_id, "condition_id": condition_id, "saved_at": utc_now(), "execution_index": number, "provider": judge_config["provider"], "model": judge_config["model"], "prompt": prompt})
            requested.add(condition_id)
        request_timestamp = utc_now()
        result = None
        error_status = None
        attempts = 0
        for attempt in range(retries + 1):
            attempts = attempt
            elapsed = time.monotonic() - last_attempt_at
            if elapsed < minimum_interval:
                time.sleep(minimum_interval - elapsed)
            _increment_counter(counter_path, cap)
            last_attempt_at = time.monotonic()
            try:
                result = provider.judge(prompt, model=str(judge_config["model"]), temperature=float(judge_config.get("temperature", 0.0)), max_output_tokens=int(judge_config.get("max_output_tokens", 2048)))
                break
            except RetriableProviderError as exc:
                error_status = str(exc)
                if attempt < retries:
                    retry_after = getattr(exc, "retry_after", None)
                    time.sleep(float(retry_after) if retry_after else base_backoff_seconds * (2 ** attempt))
            except ProviderError as exc:
                error_status = str(exc)
                if attempt < retries:
                    time.sleep(base_backoff_seconds * (2 ** attempt))
        base = {
            "run_id": run_id,
            "condition_id": condition_id,
            "base_packet_id": condition["base_packet_id"],
            "length": condition["length"],
            "internal_position": condition["internal_position"],
            "candidate_order": condition["candidate_order"],
            "gold_content_winner": "clean",
            "request_timestamp": request_timestamp,
            "retry_count": attempts,
        }
        if result:
            result_data = result.to_dict()
            try:
                raw_text = result.raw_response_text.strip()
                if raw_text.startswith("```json"):
                    raw_text = raw_text[7:]
                elif raw_text.startswith("```"):
                    raw_text = raw_text[3:]
                if raw_text.endswith("```"):
                    raw_text = raw_text[:-3]
                raw_text = raw_text.strip()
                structured = json.loads(raw_text)
            except json.JSONDecodeError:
                structured = {}
            display = structured.get("winner") if isinstance(structured, dict) and structured.get("winner") in ("A", "B", "tie") else None
            if display == "tie":
                content = "tie"
            elif display in ("A", "B"):
                a_identity = condition["candidate_a_identity"]
                b_identity = condition["candidate_b_identity"]
                content = a_identity if display == "A" else b_identity
            else:
                content = None
            record = {
                **base,
                **result_data,
                "model_requested": result.requested_model,
                "model_returned": result.returned_model,
                "prompt_token_count": result.prompt_tokens,
                "output_token_count": result.completion_tokens,
                "latency": result.latency_seconds,
                "parsed_winner_A_B_tie": display,
                "parsed_content_winner_clean_flawed_tie": content,
                "error_status": None,
            }
            append_jsonl(responses_path, record)
            print(f"[{number}/{len(conditions)}] {condition_id}: ok")
        else:
            record = {
                **base,
                "provider": judge_config["provider"],
                "requested_model": judge_config["model"],
                "model_requested": judge_config["model"],
                "returned_model": None,
                "model_returned": None,
                "raw_response_text": "",
                "raw_response": None,
                "brief_reason": None,
                "http_status": None,
                "parsed_winner_A_B_tie": None,
                "parsed_content_winner_clean_flawed_tie": None,
                "error_status": error_status or "provider failure",
            }
            append_jsonl(run_dir / "errors.jsonl", record)
            append_jsonl(responses_path, record)
            print(f"[{number}/{len(conditions)}] {condition_id}: error ({record['error_status']})")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pairwise judge calls")
    parser.add_argument("--config", default="configs/pilot.yaml")
    parser.add_argument("--run-id")
    parser.add_argument("--split", choices=("smoke", "main"))
    parser.add_argument("--provider")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--confirm-trial-evaluation", action="store_true")
    parser.add_argument("--available-call-budget", type=int)
    parser.add_argument("--limit", type=int, help="Limit number of condition calls to execute")
    parser.add_argument("--condition-id", type=str, help="Run a specific condition ID only")
    args = parser.parse_args()
    config = load_config(args.config)
    paths = require_mapping(config.get("paths"), "paths")
    configured_provider = str(require_mapping(config.get("judge"), "judge")["provider"])
    if args.provider and args.provider != configured_provider:
        raise SystemExit("provider override must match the frozen configured provider")
    run_id = args.run_id or f"{args.split or 'run'}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    if args.split or "source_rows" in paths:
        run_dir = run_prmbench(
            args.config,
            run_id,
            args.split or "main",
            trial_confirmed=args.confirm_trial_evaluation,
            available_call_budget=args.available_call_budget,
            limit=args.limit,
            condition_id=args.condition_id,
        )
    else:
        run_dir = run(args.config, run_id)
    print(f"Run artifacts written to {run_dir}")


if __name__ == "__main__":
    main()
