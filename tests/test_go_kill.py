from dataclasses import replace

import pytest

from src.analyze import PilotMetrics, evaluate_pilot
from src.utils import load_config


THRESHOLDS = {
    "min_accuracy_gap": 0.10,
    "min_middle_swap_consistency": 0.80,
    "max_swap_consistency_gap": 0.05,
    "min_cer_gap": 0.10,
}

PASSING = PilotMetrics(
    early_accuracy=0.8,
    middle_accuracy=0.6,
    late_accuracy=0.8,
    edge_accuracy=0.8,
    delta_acc=0.10,
    early_swap_consistency=0.9,
    middle_swap_consistency=0.8,
    late_swap_consistency=0.8,
    edge_swap_consistency=0.85,
    delta_sc=0.05,
    early_cer=0.1,
    middle_cer=0.2,
    late_cer=0.1,
    edge_cer=0.1,
    delta_cer=0.10,
)


def decision(metrics: PilotMetrics, integrity: bool = True) -> str:
    return evaluate_pilot(metrics, THRESHOLDS, integrity)["decision"]


def test_all_checks_pass_at_boundaries() -> None:
    assert decision(PASSING) == "GO"


@pytest.mark.parametrize(
    "metrics",
    [
        replace(PASSING, delta_acc=0.099),
        replace(PASSING, middle_swap_consistency=0.799),
        replace(PASSING, delta_sc=0.051),
        replace(PASSING, delta_cer=0.099),
    ],
)
def test_each_metric_can_kill(metrics: PilotMetrics) -> None:
    assert decision(metrics) == "KILL"


def test_integrity_failure_kills() -> None:
    assert decision(PASSING, integrity=False) == "KILL"


def test_preregistered_config_threshold_defaults() -> None:
    assert load_config("configs/pilot.yaml")["pilot_thresholds"] == THRESHOLDS
