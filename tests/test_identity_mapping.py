from src.aggregate_pairs import aggregate_pairs
from src.parse_verdict import map_content_verdict


def test_all_display_to_content_mappings() -> None:
    assert map_content_verdict("correct_first", "A") == "correct"
    assert map_content_verdict("correct_first", "B") == "flawed"
    assert map_content_verdict("flawed_first", "A") == "flawed"
    assert map_content_verdict("flawed_first", "B") == "correct"


def test_ab_a_and_ba_b_are_content_consistent() -> None:
    common = {
        "item_id": "i1",
        "context_length": "16k",
        "internal_position": "middle",
        "parse_status": "ok",
    }
    pairs = aggregate_pairs(
        [
            {
                **common,
                "candidate_order": "correct_first",
                "content_verdict": "correct",
            },
            {
                **common,
                "candidate_order": "flawed_first",
                "content_verdict": "correct",
            },
        ]
    )
    assert pairs[0]["swap_consistent"] is True
    assert pairs[0]["consistent_winner"] == "correct"
