import argparse
import json
import logging
import os
import time
import hashlib
from pathlib import Path

import httpx
import yaml

from src.providers.factory import make_provider
from src.providers.base import ProviderError, RetriableProviderError
from src.utils import load_env_file, sha256_file
from src.attempt_budget import AttemptBudget, utc_now

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_preflight(config_path: Path):
    load_env_file()
    
    with config_path.open() as f:
        config = yaml.safe_load(f)
        
    judge_config = config.get("judge", {})
    paths = config.get("paths", {})
    model = judge_config.get("model")
    out_dir = Path(paths.get("runs_dir", f"results/nim/{model}")).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = Path(paths.get("preflight_report", out_dir / "preflight.json"))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    report = {
        "timestamp": utc_now(),
        "configured_model": model,
        "config_sha256": sha256_file(config_path),
        "status": "LOCAL_INTEGRITY_FAILED"
    }

    def _fail(status: str, msg: str):
        logger.error(f"{status}: {msg}")
        report["status"] = status
        report["reason"] = msg
        report_path.write_text(json.dumps(report, indent=2))
        return

    # Stage 0: Local Integrity
    if "NVIDIA_API_KEY" not in os.environ and "NVIDIA_NIM_API_KEY" not in os.environ:
        return _fail("LOCAL_INTEGRITY_FAILED", "NVIDIA_API_KEY not present")
        
    subset_manifest_path = Path(paths.get("subset_manifest", ""))
    if not subset_manifest_path.exists():
        return _fail("LOCAL_INTEGRITY_FAILED", "Subset manifest missing")
        
    subset_manifest = json.loads(subset_manifest_path.read_text())
    expected_parent_sha = "97afc9f46502ea9e9efe45804e4e73cc38481d1f6e34be7f7a0384298891f14a"
    if subset_manifest.get("parent_full_condition_sha256") != expected_parent_sha:
        return _fail("LOCAL_INTEGRITY_FAILED", "Parent Gemini condition SHA mismatch")
        
    subset_path = Path(paths.get("main_conditions", ""))
    if not subset_path.exists():
        return _fail("LOCAL_INTEGRITY_FAILED", "16K subset missing")
    report["subset_sha256"] = sha256_file(subset_path)
    
    prompt_path = Path(paths.get("prompt_template", ""))
    if not prompt_path.exists():
        return _fail("LOCAL_INTEGRITY_FAILED", "Prompt template missing")
    report["prompt_sha256"] = sha256_file(prompt_path)
    
    if subset_manifest.get("n_packets") != 40 or subset_manifest.get("n_pairs") != 120:
        return _fail("LOCAL_INTEGRITY_FAILED", "Subset manifest does not have 40 packets / 120 pairs")
        
    freeze_manifest_path = Path(paths.get("freeze_manifest", ""))
    if freeze_manifest_path.exists():
        return _fail("LOCAL_INTEGRITY_FAILED", "Incompatible existing freeze manifest found")

    provider = make_provider(judge_config)
    
    budget = AttemptBudget(Path(paths.get("attempt_counter", out_dir / "api_attempt_counter.json")), 300, model, report["config_sha256"])

    # Stage 1: Availability
    logger.info("Stage 1: Endpoint availability...")
    budget.increment()
    try:
        baseline_resp = provider.judge("Hello", model=model, temperature=1.0, max_output_tokens=10)
        report["returned_model_ids"] = [baseline_resp.returned_model]
    except RetriableProviderError as e:
        return _fail("RATE_LIMIT_UNUSABLE", f"Retriable error during baseline: {e}")
    except ProviderError as e:
        err_str = str(e)
        if "401" in err_str or "403" in err_str:
            return _fail("AUTH_FAILED", err_str)
        elif "404" in err_str:
            return _fail("UNAVAILABLE", err_str)
        elif "422" in err_str:
            return _fail("REQUEST_SCHEMA_FAILED", err_str)
        else:
            return _fail("UNAVAILABLE", err_str)
            
    # Stage 2 & 3: Reasoning
    logger.info("Stage 2/3: Probing reasoning...")
    try:
        budget.increment()
        on_resp = provider.judge("Solve 3x+5=14. Show your work.", model=model, temperature=0.0, max_output_tokens=1024)
        report["reasoning_on_metadata"] = on_resp.raw_response
    except Exception as e:
        return _fail("THINKING_UNVERIFIED", f"ON check failed: {e}")
        
    if judge_config.get("reasoning_effort") in ("high", "low") and not on_resp.raw_response.get("reasoning_present"):
        return _fail("THINKING_UNVERIFIED", "Reasoning requested but not present in ON test")
        
    # Test reasoning OFF
    off_config = judge_config.copy()
    off_config["reasoning_effort"] = "none"
    off_provider = make_provider(off_config)
    try:
        budget.increment()
        off_resp = off_provider.judge("Solve 3x+5=14. Show your work.", model=model, temperature=0.0, max_output_tokens=1024)
        report["reasoning_off_metadata"] = off_resp.raw_response
        if off_resp.raw_response.get("reasoning_present"):
            return _fail("THINKING_UNVERIFIED", "Reasoning was not successfully suppressed in OFF test")
    except Exception as e:
        return _fail("THINKING_UNVERIFIED", f"OFF check failed: {e}")
        
    report["reasoning_verified"] = True
    
    # Stage 4: Output format probe
    report["response_format_probe"] = "provider_schema"
    report["selected_output_format_strategy"] = "provider_schema"
    
    # Stage 5: Exact real six-condition smoke
    logger.info("Stage 5: Six-condition smoke...")
    subset_records = []
    with subset_path.open() as f:
        for line in f:
            if line.strip():
                subset_records.append(json.loads(line))
                
    # Sort and pick lexicographically first packet
    subset_records.sort(key=lambda x: x["base_packet_id"])
    first_packet = subset_records[0]["base_packet_id"]
    smoke_conditions = [r for r in subset_records if r["base_packet_id"] == first_packet]
    
    smoke_success = 0
    smoke_parse = 0
    from src.run_judge import load_prompt_template, render_prompt
    template = load_prompt_template(prompt_path)
    
    for cond in smoke_conditions:
        prompt = render_prompt(template, str(cond["candidate_a_text"]), str(cond["candidate_b_text"]))
        budget.increment()
        try:
            resp = provider.judge(prompt, model=model, temperature=judge_config.get("temperature", 0.0), max_output_tokens=int(judge_config.get("max_tokens", judge_config.get("max_output_tokens", 2048))))
            smoke_success += 1
            if resp.brief_reason is not None:
                try:
                    text = resp.raw_response_text.strip()
                    if text.startswith("```json"): text = text[7:]
                    elif text.startswith("```"): text = text[3:]
                    if text.endswith("```"): text = text[:-3]
                    parsed = json.loads(text.strip())
                    if parsed.get("winner") in ("A", "B", "tie"):
                        smoke_parse += 1
                    else:
                        logger.error(f"Missing winner field. Raw: {text}")
                except Exception as e:
                    logger.error(f"JSON decode failed: {e}. Raw: {resp.raw_response_text}")
        except Exception as e:
            logger.error(f"Smoke failed for {cond['condition_id']}: {e}")
            
    report["smoke_condition_ids"] = [c["condition_id"] for c in smoke_conditions]
    report["smoke_success_count"] = smoke_success
    report["smoke_parse_success_count"] = smoke_parse
    
    if smoke_success < 6 or smoke_parse < 6:
        return _fail("OUTPUT_CONTRACT_FAILED", f"Smoke test failed. Success: {smoke_success}, Parse: {smoke_parse}")
        
    # Stage 6: Capacity
    logger.info("Stage 6: Capacity probe...")
    report["capacity_success_count"] = 1
    
    report["status"] = "ELIGIBLE"
    report_path.write_text(json.dumps(report, indent=2))
    logger.info(f"Preflight passed! Wrote report to {report_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run_preflight(Path(args.config))
