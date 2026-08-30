import argparse
import json
from pathlib import Path
import csv
import random
from collections import defaultdict
from typing import Any

from src.utils import read_jsonl, write_json
from src.paper_analysis import (
    classify_pair, clopper_pearson_interval,
    _cell_summary, packet_bootstrap, POSITIONS, PAIR_OUTCOMES,
    aggregate_paper_pairs, _ci_text, _write_csv, _write_markdown_table,
    summarize_cells, _percentile
)

def edge_minus_middle(parsed_subset, pairs_subset, metric):
    edge_calls = [p for p in parsed_subset if p["internal_position"] in ("early", "late")]
    middle_calls = [p for p in parsed_subset if p["internal_position"] == "middle"]
    edge_pairs = [p for p in pairs_subset if p["internal_position"] in ("early", "late")]
    middle_pairs = [p for p in pairs_subset if p["internal_position"] == "middle"]

    def _get_val(calls, prs):
        if metric == "raw_accuracy":
            c = [x for x in calls if x["parse_status"] == "ok"]
            return sum(x["is_correct"] for x in c) / len(c) if c else None
        elif metric == "position_consistency":
            c = [p for p in prs if p["parse_ok"]]
            return sum(p["position_consistent"] for p in c) / len(c) if c else None
        elif metric == "certification_coverage":
            c = [p for p in prs if p["parse_ok"]]
            return sum(p["decisively_certified"] for p in c) / len(c) if c else None
        return None
        
    edge_val = _get_val(edge_calls, edge_pairs)
    middle_val = _get_val(middle_calls, middle_pairs)
    if edge_val is not None and middle_val is not None:
        return edge_val - middle_val
    return None

def nim_analysis(run_dir: Path, gemini_run_dir: Path | None = None):
    # Load manifest
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        print("Run manifest not found")
        return
        
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = manifest.get("config", {})
    paths = config.get("paths", {})
    out_dir = Path(paths.get("results_dir", run_dir.parent / "paper"))
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 16.1 Completed-run integrity gate
    responses = list(read_jsonl(run_dir / "responses.jsonl"))
    
    if len(responses) != 240:
        print(f"Warning: Expected 240 calls, got {len(responses)}")
    
    parsed = []
    
    for r in responses:
        parsed_ok = r.get("parsed_winner_A_B_tie") is not None
        parsed.append({
            "item_id": str(r["base_packet_id"]),
            "context_length": "16K", # NIM robustness is 16K only
            "internal_position": str(r["internal_position"]),
            "parse_status": "ok" if parsed_ok else "error",
            "is_correct": r.get("parsed_content_winner_clean_flawed_tie") == "clean",
            "candidate_order": str(r["candidate_order"]),
            "content_verdict": r.get("parsed_content_winner_clean_flawed_tie")
        })

    n_parse_success = sum(1 for p in parsed if p["parse_status"] == "ok")
    overall_parse = n_parse_success / max(len(responses), 1)
    
    pos_parse = {}
    for pos in POSITIONS:
        pos_calls = [p for p in parsed if p["internal_position"] == pos]
        pos_parse[pos] = sum(1 for p in pos_calls if p["parse_status"] == "ok") / max(len(pos_calls), 1)

    print(f"Overall parse success: {overall_parse:.1%}")
    if overall_parse < 0.98 or any(v < 0.95 for v in pos_parse.values()):
        print("Warning: Parse success below threshold (98% overall, 95% per pos). Treating as exploratory.")

    # 16.2 Per-position metrics
    pairs = aggregate_paper_pairs(parsed)
    cells = summarize_cells(parsed, pairs) # 16K only
    
    # output audit.json
    audit = {
        "overall_parse_success": overall_parse,
        "per_position_parse": pos_parse,
        "passed_inclusion_gate": overall_parse >= 0.98 and all(v >= 0.95 for v in pos_parse.values()),
        "n_calls": len(responses),
        "n_pairs": len(pairs)
    }
    write_json(out_dir / "audit.json", audit)

    # pair outcomes
    pair_rows = []
    for key, cell in cells.items():
        length, position = key.split(".")
        if length != "16K": continue
        row = {"length": length, "internal_position": position, **cell}
        for name in ("selective_risk_exact_ci",):
            row[name] = json.dumps(row[name]) if row.get(name) else None
        pair_rows.append(row)
    _write_csv(out_dir / "pair_outcomes.csv", pair_rows)
    write_json(out_dir / "pair_outcomes.json", {"cells": cells, "partition": list(PAIR_OUTCOMES)})
    _write_csv(out_dir / "position_summary.csv", pair_rows)
    
    # 16.3 Primary NIM contrasts
    contrasts = {}
    for metric in ("raw_accuracy", "position_consistency", "certification_coverage"):
        val = edge_minus_middle(parsed, pairs, metric)
        contrasts[metric] = val
    write_json(out_dir / "contrasts.json", contrasts)

    # 16.4 Paired NIM-vs-Gemini analysis
    if gemini_run_dir:
        gemini_parsed = list(read_jsonl(Path(gemini_run_dir) / "parsed.jsonl"))
        gemini_parsed = [p for p in gemini_parsed if p["context_length"] == "16K"]
        gemini_pairs = aggregate_paper_pairs(gemini_parsed)
        
        packet_ids = sorted({str(row["item_id"]) for row in parsed})
        parsed_by_packet = {packet: [row for row in parsed if row["item_id"] == packet] for packet in packet_ids}
        pairs_by_packet = {packet: [row for row in pairs if row["item_id"] == packet] for packet in packet_ids}
        
        g_parsed_by_packet = {packet: [row for row in gemini_parsed if row["item_id"] == packet] for packet in packet_ids}
        g_pairs_by_packet = {packet: [row for row in gemini_pairs if row["item_id"] == packet] for packet in packet_ids}
        
        n_resamples = 10000
        seed = 42
        rng = random.Random(seed)
        
        samples = defaultdict(list)
        
        for _ in range(n_resamples):
            selected = rng.choices(packet_ids, k=len(packet_ids))
            
            nim_p = [r for pkt in selected for r in parsed_by_packet[pkt]]
            nim_prs = [r for pkt in selected for r in pairs_by_packet[pkt]]
            
            gem_p = [r for pkt in selected for r in g_parsed_by_packet[pkt]]
            gem_prs = [r for pkt in selected for r in g_pairs_by_packet[pkt]]
            
            for metric in ("raw_accuracy", "position_consistency", "certification_coverage"):
                nim_diff = edge_minus_middle(nim_p, nim_prs, metric)
                gem_diff = edge_minus_middle(gem_p, gem_prs, metric)
                if nim_diff is not None and gem_diff is not None:
                    nim_penalty = -nim_diff
                    gem_penalty = -gem_diff
                    samples[f"{metric}_middle_penalty_diff"].append(nim_penalty - gem_penalty)
                    
        paired_comp = {}
        for metric, vals in samples.items():
            vals.sort()
            low = _percentile(vals, 0.025)
            high = _percentile(vals, 0.975)
            mean = sum(vals)/len(vals)
            paired_comp[metric] = {"mean": mean, "low": low, "high": high, "n": len(vals)}
            
        write_json(out_dir / "gemini_paired_comparison.json", paired_comp)

    # 16.5 Main table
    intervals = packet_bootstrap(parsed, pairs, n_resamples=10000, seed=42)
    main_table = []
    for pos in POSITIONS:
        key = f"16K.{pos}"
        if key not in cells: continue
        c = cells[key]
        exact = c["selective_risk_exact_ci"]
        risk = "NA" if exact is None else f"{c['n_certified_wrong']}/{c['n_certified']} [{exact[0]:.1%}, {exact[1]:.1%}]"
        
        main_table.append({
            "Length": "16K", "Position": pos.title(),
            "Raw accuracy [95% CI]": _ci_text(c["raw_accuracy"], intervals.get(f"cells.{key}.raw_accuracy")),
            "Position consistency n/N [95% CI]": f"{sum(c[o] for o in ('certified_correct','certified_wrong','stable_tie'))}/{c['n_pairs']} " + _ci_text(c["position_consistency"], intervals.get(f"cells.{key}.position_consistency")),
            "Decisive certification coverage n/N [95% CI]": f"{c['n_certified']}/{c['n_pairs']} " + _ci_text(c["certification_coverage"], intervals.get(f"cells.{key}.certification_coverage")),
            "Certified errors k/n [exact 95% CI]": risk,
            "Stable tie %": f"{c['stable_tie_fraction']:.1%}" if c['stable_tie_fraction'] is not None else "NA",
            "Mixed tie %": f"{c['mixed_tie_fraction']:.1%}" if c['mixed_tie_fraction'] is not None else "NA",
            "Order disagreement %": f"{c['order_disagreement_fraction']:.1%}" if c['order_disagreement_fraction'] is not None else "NA",
        })
    if main_table:
        _write_csv(out_dir / "main_table.csv", main_table)
        _write_markdown_table(out_dir / "main_table.md", main_table)

    print(f"NIM analysis written to {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--gemini-run-dir", default=None, help="Path to Gemini run to compare against")
    args = parser.parse_args()
    nim_analysis(Path(args.run_dir), Path(args.gemini_run_dir) if args.gemini_run_dir else None)
