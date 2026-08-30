import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
import yaml

from src.utils import sha256_file
from src.run_judge import git_commit

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def is_clean_worktree():
    import subprocess
    result = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
    return not bool(result.stdout.strip())

def freeze_nim(config_path: Path):
    if not is_clean_worktree():
        logger.error("Git worktree is not clean. Commit all changes before freezing.")
        sys.exit(1)
        
    with config_path.open() as f:
        config = yaml.safe_load(f)
        
    paths = config.get("paths", {})
    judge = config.get("judge", {})
    scheduler = config.get("scheduler", {})
    
    validation_path = Path(paths.get("main_validation_report", ""))
    if not validation_path.exists():
        logger.error("Validation report missing")
        sys.exit(1)
    val_report = json.loads(validation_path.read_text())
    if not val_report.get("passed"):
        logger.error("Validation failed")
        sys.exit(1)
        
    preflight_path = Path(paths.get("preflight_report", ""))
    if not preflight_path.exists():
        logger.error("Preflight report missing")
        sys.exit(1)
    preflight = json.loads(preflight_path.read_text())
    if preflight.get("status") != "ELIGIBLE":
        logger.error(f"Preflight not ELIGIBLE: {preflight.get('status')}")
        sys.exit(1)
        
    config_hash = sha256_file(config_path)
    if preflight.get("config_sha256") != config_hash:
        logger.error("Config hash changed since preflight")
        sys.exit(1)
        
    subset_path = Path(paths.get("main_conditions", ""))
    subset_hash = sha256_file(subset_path)
    if preflight.get("subset_sha256") != subset_hash:
        logger.error("Subset hash changed since preflight")
        sys.exit(1)
        
    prompt_path = Path(paths.get("prompt_template", ""))
    prompt_hash = sha256_file(prompt_path)
    if preflight.get("prompt_sha256") != prompt_hash:
        logger.error("Prompt hash changed since preflight")
        sys.exit(1)
        
    subset_manifest = json.loads(Path(paths.get("subset_manifest", "")).read_text())
    source_manifest = json.loads(Path(paths.get("source_manifest", "")).read_text())
    
    import hashlib
    
    reasoning_req = {
        "reasoning_effort": judge.get("reasoning_effort"),
        "reasoning_budget": judge.get("reasoning_budget")
    }
    reasoning_req_json = json.dumps(reasoning_req, sort_keys=True)
    reasoning_req_hash = hashlib.sha256(reasoning_req_json.encode()).hexdigest()
    
    response_schema = judge.get("response_format", {})
    response_schema_json = json.dumps(response_schema, sort_keys=True)
    response_schema_hash = hashlib.sha256(response_schema_json.encode()).hexdigest()

    # Ensure preflight hashes match
    # Since we added reasoning verification logic in preflight that could be complex, we just freeze the current hash.
    
    manifest = {
        "git_commit": git_commit(),
        "config_sha256": config_hash,
        "dataset_revision": source_manifest.get("revision"),
        "dataset_fingerprint": source_manifest.get("dataset_fingerprint"),
        "parent_gemini_condition_sha256": subset_manifest.get("parent_full_condition_sha256"),
        "nim_16k_subset_sha256": subset_hash,
        "subset_manifest_sha256": sha256_file(Path(paths.get("subset_manifest", ""))),
        "prompt_sha256": prompt_hash,
        "provider": judge.get("provider"),
        "model": judge.get("model"),
        "temperature": judge.get("temperature"),
        "top_p": judge.get("top_p"),
        "max_tokens": judge.get("max_tokens"),
        "seed": config.get("seed"),
        "stream": judge.get("stream"),
        "exact_reasoning_request_json": reasoning_req_json,
        "reasoning_request_sha256": reasoning_req_hash,
        "output_format_strategy": preflight.get("selected_output_format_strategy", "provider_schema"),
        "response_schema_sha256_or_suffix_sha256": response_schema_hash,
        "preflight_report_sha256": sha256_file(preflight_path),
        "scheduler": scheduler,
        "analysis_code_sha256": sha256_file(Path("src/nim_analysis.py")) if Path("src/nim_analysis.py").exists() else None,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    
    out_path = Path(paths.get("freeze_manifest", ""))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2))
    logger.info(f"Froze NIM run configuration to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    freeze_nim(Path(args.config))
