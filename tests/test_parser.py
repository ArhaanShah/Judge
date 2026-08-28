import pytest

from src.parse_verdict import parse_display_verdict


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("VERDICT: A", ("A", "ok")),
        ("VERDICT: B", ("B", "ok")),
        ("Reasoning mentions A and B.\nVERDICT:   A  ", ("A", "ok")),
        ("I select A", (None, "parse_error")),
        ("VERDICT: A\nVERDICT: A", (None, "parse_error")),
        ("VERDICT: A\nVERDICT: B", (None, "parse_error")),
        ("Candidate A is stronger than Candidate B.\nVERDICT: B", ("B", "ok")),
        ("VERDICT: A\nTrailing explanation.", (None, "parse_error")),
    ],
)
def test_strict_parser(text: str, expected: tuple[str | None, str]) -> None:
    assert parse_display_verdict(text) == expected
