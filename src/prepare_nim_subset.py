import json
import hashlib
import sys
from pathlib import Path
from datetime import datetime, timezone

def hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parent_path = Path("data/processed/pilot_conditions.jsonl")
    out_path = Path("data/processed/pilot_conditions_16k.jsonl")
    manifest_path = Path("artifacts/nim_16k_subset_manifest.json")
    gemini_manifest_path = Path("artifacts/pilot_freeze_manifest_gemini.json")
    self_path = Path(__file__)
    
    EXPECTED_PARENT_HASH = "97afc9f46502ea9e9efe45804e4e73cc38481d1f6e34be7f7a0384298891f14a"
    
    if not parent_path.exists():
        print(f"Error: {parent_path} not found")
        sys.exit(1)
        
    parent_hash = hash_file(parent_path)
    if parent_hash != EXPECTED_PARENT_HASH:
        print(f"Error: Parent hash {parent_hash} != {EXPECTED_PARENT_HASH}")
        sys.exit(1)
        
    records = []
    with open(parent_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("length") == "16K":
                records.append(rec)
                
    # Verifications
    n_conditions = len(records)
    packets = set(rec["base_packet_id"] for rec in records)
    n_packets = len(packets)
    
    # Check 3 positions per packet and 2 orders per packet-position
    pairs_count = 0
    for pkt in packets:
        pkt_recs = [r for r in records if r["base_packet_id"] == pkt]
        positions = set(r["internal_position"] for r in pkt_recs)
        if len(positions) != 3:
            print(f"Error: Packet {pkt} has {len(positions)} positions, expected 3")
            sys.exit(1)
        for pos in positions:
            pos_recs = [r for r in pkt_recs if r["internal_position"] == pos]
            orders = set(r["candidate_order"] for r in pos_recs)
            if len(orders) != 2:
                print(f"Error: Packet {pkt} pos {pos} has {len(orders)} orders, expected 2")
                sys.exit(1)
            pairs_count += 1
            
    if n_conditions != 240:
        print(f"Error: n_conditions={n_conditions}, expected 240")
        sys.exit(1)
    if n_packets != 40:
        print(f"Error: n_packets={n_packets}, expected 40")
        sys.exit(1)
    if pairs_count != 120:
        print(f"Error: pairs={pairs_count}, expected 120")
        sys.exit(1)
        
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, separators=(',', ':')) + "\n")
            
    subset_hash = hash_file(out_path)
    gemini_freeze_hash = hash_file(gemini_manifest_path) if gemini_manifest_path.exists() else None
    
    manifest = {
        "parent_full_condition_sha256": parent_hash,
        "subset_sha256": subset_hash,
        "n_conditions": n_conditions,
        "n_packets": n_packets,
        "n_pairs": pairs_count,
        "source_gemini_freeze_manifest_sha256": gemini_freeze_hash,
        "generation_code_sha256": hash_file(self_path),
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        
    print(f"Successfully wrote 240 16K conditions to {out_path}")
    print(f"Wrote manifest to {manifest_path}")

if __name__ == "__main__":
    main()
