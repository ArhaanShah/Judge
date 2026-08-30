import json
import pytest
from pathlib import Path
from src.paper_analysis import clopper_pearson_interval

def test_committed_paper_interactions():
    path = Path("results/gemini/paper/interactions.json")
    assert path.exists(), "interactions.json not found"
    data = json.loads(path.read_text())
    
    # Assert raw_accuracy interaction == 0.1875
    assert abs(data["raw_accuracy"]["interaction"] - 0.1875) < 1e-7
    # Assert certification_coverage interaction == 0.1500
    assert abs(data["certification_coverage"]["interaction"] - 0.1500) < 1e-7
    # Assert position_consistency interaction == -0.05
    assert abs(data["position_consistency"]["interaction"] - (-0.05)) < 1e-7

def test_committed_16k_middle_results():
    path = Path("results/gemini/paper/pair_outcomes.json")
    assert path.exists(), "pair_outcomes.json not found"
    data = json.loads(path.read_text())
    
    cell = data["cells"]["16K.middle"]
    
    assert cell["n_pairs"] == 40
    assert abs(cell["raw_accuracy"] - 0.3125) < 1e-7
    assert abs(cell["position_consistency"] - 0.60) < 1e-7
    assert abs(cell["certification_coverage"] - 0.15) < 1e-7
    
    assert cell["stable_tie"] == 18
    assert cell["mixed_tie"] == 14
    assert cell["order_disagreement"] == 2
    
    assert cell["n_certified"] == 6
    assert cell["n_certified_wrong"] == 0

def test_exact_ci_regression_cases():
    def check_ci(k, n, expected_lower, expected_upper):
        ci = clopper_pearson_interval(k, n)
        if expected_lower is None and expected_upper is None:
            assert ci is None
        else:
            assert ci is not None
            assert abs(ci[0] - expected_lower) < 0.01
            assert abs(ci[1] - expected_upper) < 0.01

    # Regression cases
    # 0/6 -> 0%, 45.9% (using standard Clopper-Pearson 95%)
    check_ci(0, 6, 0.0, 0.459)
    # 0/11 -> 0%, 28.5%
    check_ci(0, 11, 0.0, 0.285)
    # 1/17
    check_ci(1, 17, 0.001, 0.287)
    # 1/24
    check_ci(1, 24, 0.001, 0.211)
    # 1/29
    check_ci(1, 29, 0.001, 0.178)
    # 0/0 -> None
    check_ci(0, 0, None, None)
