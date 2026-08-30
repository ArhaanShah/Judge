import argparse
import json
from pathlib import Path
import csv
from typing import Any

from src.utils import read_jsonl
from src.paper_analysis import (
    classify_pair, clopper_pearson_interval,
    _cell_summary, packet_bootstrap
)

def nim_analysis(run_dir: Path):
    manifest = json.loads((run_dir / "manifest.json").read_text())
    config = manifest["config"]
    paths = config.get("paths", {})
    responses = list(read_jsonl(run_dir / "responses.jsonl"))
    
    # 16.1 Completed-run integrity gate
    if len(responses) != 240:
        print(f"Warning: Expected 240 calls, got {len(responses)}")
    
    parsed_responses = []
    pairs_map = {}
    
    # parse success
    n_parse_success = sum(1 for r in responses if r.get("parsed_winner_A_B_tie") is not None)
    if n_parse_success / max(len(responses), 1) < 0.98:
        print("Warning: Parse success < 98%. Treating as exploratory.")
        
    for r in responses:
        cond_id = r["condition_id"]
        parsed = r.get("parsed_winner_A_B_tie") is not None
        
        parsed_responses.append({
            "item_id": r["base_packet_id"],
            "context_length": r["length"],
            "internal_position": r["internal_position"],
            "parse_status": "ok" if parsed else "error",
            "is_correct": r.get("parsed_content_winner_clean_flawed_tie") == "clean",
            "candidate_order": r["candidate_order"]
        })
        
        # build pairs
        pair_key = (r["base_packet_id"], r["internal_position"])
        if pair_key not in pairs_map:
            pairs_map[pair_key] = {}
        pairs_map[pair_key][r["candidate_order"]] = r
        
    pairs = []
    for (pkt, pos), calls in pairs_map.items():
        if len(calls) == 2:
            a1 = calls["clean_first"]
            a2 = calls["flawed_first"]
            parse_ok = (a1.get("parsed_winner_A_B_tie") is not None and a2.get("parsed_winner_A_B_tie") is not None)
            
            outcome = None
            certified = False
            consistent = False
            
            if parse_ok:
                w1 = a1["parsed_content_winner_clean_flawed_tie"]
                w2 = a2["parsed_content_winner_clean_flawed_tie"]
                outcome = classify_pair(w1, w2)
                certified = (outcome in ("certified_correct", "certified_wrong"))
                consistent = (outcome in ("certified_correct", "certified_wrong", "stable_tie"))
                
            pairs.append({
                "item_id": pkt,
                "context_length": a1["length"],
                "internal_position": pos,
                "parse_ok": parse_ok,
                "decisively_certified": certified,
                "position_consistent": consistent,
                "pair_outcome": outcome,
                "pair_complete": True
            })

    # write summary
    out_dir = Path(paths.get("results_dir", run_dir.parent / "paper"))
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Do packet bootstrap
    b_results = packet_bootstrap(parsed_responses, pairs, n_resamples=10000, seed=42)
    
    summary = {}
    for pos in ["early", "middle", "late"]:
        pos_calls = [r for r in parsed_responses if r["internal_position"] == pos]
        pos_pairs = [p for p in pairs if p["internal_position"] == pos]
        s = _cell_summary(pos_calls, pos_pairs)
        summary[pos] = s
        
    with open(out_dir / "position_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
        
    print(f"NIM analysis written to {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    nim_analysis(Path(args.run_dir))
