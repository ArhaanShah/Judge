from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .utils import load_config, read_jsonl, require_mapping, sha256_file, write_json
from .validate_conditions import validate_prmbench_from_config


def _git(*args: str) -> str:
    result = subprocess.run(["git", *args], capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "Git command failed")
    return result.stdout.strip()


def freeze(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    paths = require_mapping(config.get("paths"), "paths")
    report = validate_prmbench_from_config(config, "main")
    if not report["passed"]:
        raise RuntimeError("main conditions failed validation; refusing to freeze")
    if _git("status", "--porcelain"):
        raise RuntimeError("Git worktree must be clean before freezing the main pilot")
    source_manifest = json.loads(Path(paths["source_manifest"]).read_text(encoding="utf-8"))
    conditions = list(read_jsonl(paths["main_conditions"]))
    packet_representatives: dict[str, dict[str, Any]] = {}
    for row in conditions:
        if row["base_packet_id"] not in packet_representatives and row["length"] == "16K":
            packet_representatives[row["base_packet_id"]] = row
    manifest = {
        "git_commit": _git("rev-parse", "HEAD"),
        "dataset_revision": source_manifest.get("revision"),
        "dataset_fingerprint": source_manifest.get("dataset_fingerprint"),
        "sampling_seed": int(config["seed"]),
        "selected_target_ids": sorted({str(row["target_id"]) for row in conditions}),
        "filler_ids_by_packet": {packet: row["filler_ids"] for packet, row in sorted(packet_representatives.items())},
        "model_id": config["judge"]["model"],
        "provider": config["judge"]["provider"],
        "prompt_sha256": sha256_file(paths["prompt_template"]),
        "condition_file_sha256": sha256_file(paths["main_conditions"]),
        "analysis_code_sha256": sha256_file("src/analyze.py"),
        "go_kill_thresholds": config["pilot_thresholds"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    write_json(paths["freeze_manifest"], manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze validated main-pilot inputs")
    parser.add_argument("--config", default="config/pilot_prmbench_cohere.yaml")
    args = parser.parse_args()
    manifest = freeze(args.config)
    print(f"Frozen {len(manifest['selected_target_ids'])} targets at {manifest['git_commit']}")


if __name__ == "__main__":
    main()
