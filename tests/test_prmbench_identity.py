from __future__ import annotations

from src.aggregate_pairs import aggregate_pairs
from src.parse_verdict import map_prmbench_content_verdict, parse_display_verdict


def test_json_identity_mapping_and_tie() -> None:
    assert map_prmbench_content_verdict("clean_first", "A") == "clean"
    assert map_prmbench_content_verdict("clean_first", "B") == "flawed"
    assert map_prmbench_content_verdict("flawed_first", "A") == "flawed"
    assert map_prmbench_content_verdict("flawed_first", "B") == "clean"
    assert parse_display_verdict('{"winner":"tie","brief_reason":"Neither."}') == ("tie", "ok")


def test_any_tie_is_swap_inconsistent() -> None:
    common = {"item_id": "p1", "context_length": "16K", "internal_position": "middle", "parse_status": "ok"}
    rows = [
        {**common, "candidate_order": "clean_first", "content_verdict": "tie"},
        {**common, "candidate_order": "flawed_first", "content_verdict": "tie"},
    ]
    pair = aggregate_pairs(rows)[0]
    assert pair["parse_ok"] is True
    assert pair["swap_consistent"] is False
