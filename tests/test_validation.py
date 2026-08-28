from conftest import make_config, make_item
from src.build_conditions import build_item_conditions
from src.validate_conditions import validate_conditions


def test_generated_conditions_pass_invariant_gate() -> None:
    item = make_item()
    config = make_config()
    rows = build_item_conditions(
        item, config, "A:\n{candidate_a_text}\nB:\n{candidate_b_text}"
    )
    report = validate_conditions(rows, {item.item_id: item}, config)
    assert report["passed"], report["failures"]
