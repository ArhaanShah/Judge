from src.analyze import (
    compute_pilot_metrics,
    consistent_error_rate,
    swap_consistency,
)


def call(item: str, position: str, correct: bool) -> dict:
    return {
        "item_id": item,
        "context_length": "16k",
        "internal_position": position,
        "parse_status": "ok",
        "is_correct": correct,
    }


def pair(
    item: str, position: str, consistent: bool, wrong: bool
) -> dict:
    return {
        "item_id": item,
        "context_length": "16k",
        "internal_position": position,
        "parse_ok": True,
        "swap_consistent": consistent,
        "consistent_wrong": wrong,
    }


def test_known_metrics_and_pooled_edges() -> None:
    parsed = [
        call("a", "early", True),
        call("b", "early", True),
        call("a", "middle", True),
        call("b", "middle", False),
        call("a", "late", False),
        call("b", "late", False),
        call("c", "late", False),
        call("d", "late", False),
        call("e", "late", False),
        call("f", "late", False),
    ]
    pairs = [
        pair("a", "early", True, False),
        pair("b", "early", True, True),
        pair("a", "middle", True, True),
        pair("b", "middle", False, False),
        pair("a", "late", True, False),
        pair("b", "late", True, True),
    ]
    metrics = compute_pilot_metrics(parsed, pairs, "16k")
    assert metrics.early_accuracy == 1.0
    assert metrics.middle_accuracy == 0.5
    assert metrics.late_accuracy == 0.0
    assert metrics.edge_accuracy == 0.25
    assert metrics.delta_acc == -0.25
    assert metrics.middle_swap_consistency == 0.5
    assert metrics.edge_swap_consistency == 1.0
    assert metrics.middle_cer == 1.0
    assert metrics.edge_cer == 0.5
    assert metrics.delta_cer == 0.5


def test_cer_null_when_no_consistent_pairs() -> None:
    rows = [pair("a", "middle", False, False)]
    assert swap_consistency(rows) == 0.0
    assert consistent_error_rate(rows) is None
