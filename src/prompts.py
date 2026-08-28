from __future__ import annotations

from pathlib import Path


def load_prompt_template(path: str | Path) -> str:
    template = Path(path).read_text(encoding="utf-8")
    required = {"{candidate_a_text}", "{candidate_b_text}"}
    missing = [marker for marker in required if marker not in template]
    if missing:
        raise ValueError(f"prompt template is missing placeholders: {missing}")
    return template


def render_prompt(template: str, candidate_a_text: str, candidate_b_text: str) -> str:
    return template.format(
        candidate_a_text=candidate_a_text,
        candidate_b_text=candidate_b_text,
    )
