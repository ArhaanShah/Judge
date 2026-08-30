import argparse
import json
import logging
import os
import sys
import yaml
from pathlib import Path

from src.providers.factory import make_provider
from src.providers.base import ProviderError, RetriableProviderError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_preflight(config_path: Path):
    with config_path.open() as f:
        config = yaml.safe_load(f)
        
    judge_config = config.get("judge", {})
    model = judge_config.get("model")
    out_dir_name = model.replace("/", "_").replace("-", "_")
    out_dir = Path("results/nim") / out_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "preflight.json"
    
    provider = make_provider(judge_config)
    
    report = {
        "model": model,
        "config": str(config_path),
        "status": "UNAVAILABLE"
    }
    
    # Stage 1: Availability
    logger.info("Stage 1: Endpoint availability...")
    try:
        resp = provider.judge("Hello", model=model, temperature=1.0, max_output_tokens=100)
        if not resp.raw_response_text:
            report["status"] = "UNAVAILABLE"
            report["reason"] = "Empty response"
            _write_report(report_path, report)
            return
    except Exception as e:
        logger.error(f"Availability check failed: {e}")
        report["status"] = "UNAVAILABLE"
        report["reason"] = str(e)
        _write_report(report_path, report)
        return
        
    # Stage 2 & 3: Reasoning
    logger.info("Stage 2 & 3: Probing and verifying reasoning...")
    try:
        on_resp = provider.judge("What is 2+2? Think step by step.", model=model, temperature=1.0, max_output_tokens=1024)
        if not on_resp.raw_response.get("reasoning_present"):
            report["status"] = "THINKING_UNVERIFIED"
            report["reason"] = "Reasoning requested but not present in ON test"
            _write_report(report_path, report)
            return
    except Exception as e:
        logger.error(f"Reasoning check failed: {e}")
        report["status"] = "THINKING_UNVERIFIED"
        report["reason"] = str(e)
        _write_report(report_path, report)
        return

    # For strict verification, ideally we'd test with thinking OFF, but NVIDIA's exact schema is fluid.
    # For Nemotron, we could change config to None for reasoning effort.
    # The preflight must at least confirm reasoning is present with the requested config.

    # Stage 4: Output Contract Smoke
    logger.info("Stage 4: Output contract smoke test...")
    # Trivial smoke test for A/B winner
    prompt = """Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants.
[Question] What is 1+1?
[The Start of Assistant A's Answer] 2 [The End of Assistant A's Answer]
[The Start of Assistant B's Answer] 3 [The End of Assistant B's Answer]

Output a JSON object with:
{"winner": "A | B | tie", "brief_reason": "..."}
"""
    try:
        # Testing multiple conditions (just 2 for brevity in basic smoke)
        for _ in range(2):
            smoke_resp = provider.judge(prompt, model=model, temperature=judge_config.get("temperature", 0.0), max_output_tokens=2048)
            text = smoke_resp.raw_response_text
            # Basic validation
            text_cleaned = text.strip()
            if text_cleaned.startswith("```json"):
                text_cleaned = text_cleaned[7:]
            if text_cleaned.startswith("```"):
                text_cleaned = text_cleaned[3:]
            if text_cleaned.endswith("```"):
                text_cleaned = text_cleaned[:-3]
                
            try:
                parsed = json.loads(text_cleaned)
                if "winner" not in parsed or "brief_reason" not in parsed:
                    raise ValueError("Missing required keys")
            except Exception as parse_err:
                logger.error(f"JSON Output parse failed: {parse_err}. Text: {text}")
                report["status"] = "OUTPUT_CONTRACT_FAILED"
                report["reason"] = "Invalid JSON schema"
                _write_report(report_path, report)
                return
    except Exception as e:
        logger.error(f"Smoke test failed: {e}")
        report["status"] = "RATE_LIMIT_UNUSABLE" if isinstance(e, RetriableProviderError) else "OUTPUT_CONTRACT_FAILED"
        report["reason"] = str(e)
        _write_report(report_path, report)
        return
        
    # Stage 5: Capacity Smoke
    logger.info("Stage 5: Full-length capacity smoke...")
    # Send a moderately large string to verify context length isn't immediately erroring
    long_prompt = prompt + "\n" + (" filler word" * 2000)
    try:
        cap_resp = provider.judge(long_prompt, model=model, temperature=judge_config.get("temperature", 0.0), max_output_tokens=2048)
        if not cap_resp.raw_response_text:
            raise ValueError("Empty final content")
    except Exception as e:
        logger.error(f"Capacity test failed: {e}")
        report["status"] = "FULL_LENGTH_FAILED"
        report["reason"] = str(e)
        _write_report(report_path, report)
        return

    logger.info("Preflight passed!")
    report["status"] = "ELIGIBLE"
    report["reason"] = "All preflight checks passed"
    _write_report(report_path, report)
    
def _write_report(path: Path, report: dict):
    with path.open("w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Report written to {path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    
    # We require NVIDIA_API_KEY
    if "NVIDIA_API_KEY" not in os.environ:
        logger.warning("NVIDIA_API_KEY not set! Preflight will fail.")
        
    run_preflight(Path(args.config))
