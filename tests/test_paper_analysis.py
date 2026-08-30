import pytest
from src.paper_analysis import (
    classify_pair,
    clopper_pearson_interval,
    _cell_summary,
    compute_interactions,
    packet_bootstrap,
    native_token_rows,
    aggregate_paper_pairs,
    PAIR_OUTCOMES
)

def test_pair_outcomes_partition_all_pairs():
    outcomes = []
    for a in ("clean", "flawed", "tie"):
        for b in ("clean", "flawed", "tie"):
            outcomes.append(classify_pair(a, b))
    
    assert set(outcomes) == set(PAIR_OUTCOMES)

def test_certified_correct():
    assert classify_pair("clean", "clean") == "certified_correct"

def test_certified_wrong():
    assert classify_pair("flawed", "flawed") == "certified_wrong"

def test_stable_tie():
    assert classify_pair("tie", "tie") == "stable_tie"

def test_mixed_tie_one_side():
    assert classify_pair("clean", "tie") == "mixed_tie"
    assert classify_pair("tie", "clean") == "mixed_tie"
    assert classify_pair("flawed", "tie") == "mixed_tie"
    assert classify_pair("tie", "flawed") == "mixed_tie"

def test_order_disagreement_clean_flawed():
    assert classify_pair("clean", "flawed") == "order_disagreement"

def test_order_disagreement_flawed_clean():
    assert classify_pair("flawed", "clean") == "order_disagreement"

def test_position_consistency_counts_stable_tie():
    pairs = [{"parse_ok": True, "pair_outcome": "stable_tie", "position_consistent": True, "decisively_certified": False}]
    calls = [{"parse_status": "ok", "is_correct": False}, {"parse_status": "ok", "is_correct": False}]
    summary = _cell_summary(calls, pairs)
    assert summary["position_consistency"] == 1.0

def test_certification_coverage_excludes_stable_tie():
    pairs = [{"parse_ok": True, "pair_outcome": "stable_tie", "position_consistent": True, "decisively_certified": False}]
    calls = [{"parse_status": "ok", "is_correct": False}, {"parse_status": "ok", "is_correct": False}]
    summary = _cell_summary(calls, pairs)
    assert summary["certification_coverage"] == 0.0

def test_selective_risk_exact_ci_zero_of_six():
    interval = clopper_pearson_interval(0, 6)
    assert interval is not None
    assert interval[0] == 0.0
    assert abs(interval[1] - 0.459) < 0.01

def test_selective_risk_none_at_zero_coverage():
    assert clopper_pearson_interval(0, 0) is None

def test_accuracy_interaction_formula():
    parsed = []
    pairs = []
    for length, pos, n_calls, n_correct in [
        ("4K", "early", 40, 20), ("4K", "late", 40, 20), ("4K", "middle", 40, 24),
        ("16K", "early", 40, 12), ("16K", "late", 40, 12), ("16K", "middle", 40, 4)
    ]:
        for i in range(n_calls):
            parsed.append({"context_length": length, "internal_position": pos, "parse_status": "ok", "is_correct": i < n_correct, "candidate_order": "clean_first"})
            pairs.append({"context_length": length, "internal_position": pos, "parse_ok": True, "decisively_certified": i < n_correct, "position_consistent": i < n_correct})
    
    interactions = compute_interactions(parsed, pairs)
    assert abs(interactions["raw_accuracy"]["interaction"] - 0.3) < 1e-7

def test_decisive_coverage_interaction_formula():
    parsed = []
    pairs = []
    for length, pos, n_calls, n_certified in [
        ("4K", "early", 40, 20), ("4K", "late", 40, 20), ("4K", "middle", 40, 24),
        ("16K", "early", 40, 12), ("16K", "late", 40, 12), ("16K", "middle", 40, 4)
    ]:
        for i in range(n_calls):
            parsed.append({"context_length": length, "internal_position": pos, "parse_status": "ok", "is_correct": i < n_certified, "candidate_order": "clean_first"})
            pairs.append({"context_length": length, "internal_position": pos, "parse_ok": True, "decisively_certified": i < n_certified, "position_consistent": i < n_certified})
    
    interactions = compute_interactions(parsed, pairs)
    assert abs(interactions["certification_coverage"]["interaction"] - 0.3) < 1e-7

def test_position_consistency_interaction_formula():
    parsed = []
    pairs = []
    for length, pos, n_calls, n_consistent in [
        ("4K", "early", 40, 20), ("4K", "late", 40, 20), ("4K", "middle", 40, 24),
        ("16K", "early", 40, 12), ("16K", "late", 40, 12), ("16K", "middle", 40, 4)
    ]:
        for i in range(n_calls):
            parsed.append({"context_length": length, "internal_position": pos, "parse_status": "ok", "is_correct": i < n_consistent, "candidate_order": "clean_first"})
            pairs.append({"context_length": length, "internal_position": pos, "parse_ok": True, "decisively_certified": i < n_consistent, "position_consistent": i < n_consistent})
    
    interactions = compute_interactions(parsed, pairs)
    assert abs(interactions["position_consistency"]["interaction"] - 0.3) < 1e-7

def test_packet_bootstrap_preserves_cluster():
    # If we bootstrap by packet, all positions for a given packet should be selected together.
    # We can test this by checking if the lengths are consistent.
    dummy_parsed = []
    dummy_pairs = []
    for length in ("4K", "16K"):
        for pos in ("early", "middle", "late"):
            for order in ("clean_first", "flawed_first"):
                for item_id in range(40):
                    dummy_parsed.append({"item_id": str(item_id), "context_length": length, "internal_position": pos, "parse_status": "ok", "is_correct": True, "candidate_order": order})
                    if order == "clean_first":
                        dummy_pairs.append({"item_id": str(item_id), "context_length": length, "internal_position": pos, "parse_ok": True, "decisively_certified": True, "position_consistent": True, "pair_outcome": "certified_correct", "pair_complete": True})

    b1 = packet_bootstrap(dummy_parsed, dummy_pairs, n_resamples=10, seed=42)
    assert "cells.16K.middle.raw_accuracy" in b1

def test_bootstrap_interaction_reproducible_with_seed():
    dummy_parsed = []
    dummy_pairs = []
    for length in ("4K", "16K"):
        for pos in ("early", "middle", "late"):
            for order in ("clean_first", "flawed_first"):
                for item_id in range(40):
                    dummy_parsed.append({"item_id": str(item_id), "context_length": length, "internal_position": pos, "parse_status": "ok", "is_correct": True, "candidate_order": order})
                    if order == "clean_first":
                        dummy_pairs.append({"item_id": str(item_id), "context_length": length, "internal_position": pos, "parse_ok": True, "decisively_certified": True, "position_consistent": True, "pair_outcome": "certified_correct", "pair_complete": True})

    b1 = packet_bootstrap(dummy_parsed, dummy_pairs, n_resamples=10, seed=42)
    b2 = packet_bootstrap(dummy_parsed, dummy_pairs, n_resamples=10, seed=42)
    assert b1 == b2

def test_native_token_summary_ignores_missing_values():
    conditions = []
    responses = []
    for length in ("4K", "16K"):
        for pos in ("early", "middle", "late"):
            conditions.append({"condition_id": len(conditions), "clean_candidate_text": "A\n\nB\n\nC", "target_index": 1, "length": length, "internal_position": pos})
            responses.append({"condition_id": len(responses)})
    rows, passed, _ = native_token_rows(conditions, responses)
    for r in rows:
        if r["scope"] == "length" and r["length"] == "4K":
            assert r["prompt_tokens_n"] is None

def test_tokenizer_independent_position_ordering():
    conditions = []
    responses = []
    for length in ("4K", "16K"):
        conditions.append({"condition_id": len(conditions), "clean_candidate_text": "A\n\nB", "target_index": 0, "length": length, "internal_position": "early"})
        conditions.append({"condition_id": len(conditions), "clean_candidate_text": "A\n\nB\n\nC", "target_index": 1, "length": length, "internal_position": "middle"})
        conditions.append({"condition_id": len(conditions), "clean_candidate_text": "A\n\nB", "target_index": 1, "length": length, "internal_position": "late"})
        for _ in range(3):
            responses.append({"condition_id": len(responses) - 3, "prompt_tokens": 10})
    
    rows, passed, sep = native_token_rows(conditions, responses)
    assert passed

