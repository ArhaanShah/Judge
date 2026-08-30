from src.analyze import (
    compute_pilot_metrics,
    consistent_error_rate,
    swap_consistency,
    post_filter_accuracy,
    reported_gap,
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
    item: str, position: str, consistent: bool, wrong: bool | None = None
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
    assert metrics.early_post_filter_accuracy == 0.5
    assert metrics.middle_post_filter_accuracy == 0.0
    assert metrics.late_post_filter_accuracy == 0.5
    assert metrics.edge_post_filter_accuracy == 0.5
    assert metrics.early_reported_gap == -0.5
    assert metrics.middle_reported_gap == -0.5
    assert metrics.late_reported_gap == 0.5
    assert metrics.edge_reported_gap == 0.25


def test_cer_null_when_no_consistent_pairs() -> None:
    rows = [pair("a", "middle", False, False)]
    assert swap_consistency(rows) == 0.0
    assert consistent_error_rate(rows) is None
    assert post_filter_accuracy(rows) is None


def test_post_filter_accuracy_helpers() -> None:
    # 1.0 when all swap-consistent pairs are correct (consistent_wrong is False)
    rows_all_correct = [
        pair("a", "middle", True, False),
        pair("b", "middle", True, False),
    ]
    assert post_filter_accuracy(rows_all_correct) == 1.0
    
    # None when no pairs are swap-consistent
    rows_none_consistent = [
        pair("a", "middle", False, None),
    ]
    assert post_filter_accuracy(rows_none_consistent) is None
    
    # Gap checks
    assert reported_gap(0.25, 1.0) == 0.75
    assert reported_gap(0.25, None) is None
    assert reported_gap(None, 0.5) is None


def test_synthetic_prmbench_16k_middle_fixture() -> None:
    # 4 swap pairs, 8 calls total
    # 1 pair is swap-consistent and correct under both orders: 2 correct calls
    # 3 pairs are swap-inconsistent: each has 1 correct and 1 incorrect call 
    # (since the verdict changes across orders). Total: 3 correct, 3 incorrect.
    # Total calls correct: 2 (from consistent) + 3 (from inconsistent) = 5 calls correct.
    # Total calls: 8. Raw accuracy: 5 / 8 = 0.625.
    
    parsed = [
        # Pair 1: consistent, correct
        call("p1", "middle", True),
        call("p1", "middle", True),
        # Pair 2: inconsistent (1 correct, 1 incorrect)
        call("p2", "middle", True),
        call("p2", "middle", False),
        # Pair 3: inconsistent
        call("p3", "middle", True),
        call("p3", "middle", False),
        # Pair 4: inconsistent
        call("p4", "middle", True),
        call("p4", "middle", False),
    ]
    
    pairs = [
        pair("p1", "middle", True, False),  # consistent_wrong is False -> correct
        pair("p2", "middle", False, None),
        pair("p3", "middle", False, None),
        pair("p4", "middle", False, None),
    ]
    
    metrics = compute_pilot_metrics(parsed, pairs, "16k")
    assert metrics.middle_swap_consistency == 0.25  # coverage = 1/4 = 0.25
    assert metrics.middle_accuracy == 0.625         # raw_accuracy = 5/8
    assert metrics.middle_post_filter_accuracy == 1.0
    assert metrics.middle_reported_gap == 0.375
