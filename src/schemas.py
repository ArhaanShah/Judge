from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, cast


class SchemaError(ValueError):
    """Raised when persisted experiment data violates its schema."""


def _required_str(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise SchemaError(f"{key} must be a non-empty string")
    return value


@dataclass(frozen=True)
class Section:
    section_id: str
    text: str

    def __post_init__(self) -> None:
        if not self.section_id.strip() or not self.text.strip():
            raise SchemaError("section_id and text must be non-empty")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Section":
        return cls(_required_str(data, "section_id"), _required_str(data, "text"))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BaseItem:
    item_id: str
    domain: str
    error_type: str
    section_count: int
    correct_sections: tuple[Section, ...]
    flawed_section_id: str
    flawed_section_text: str
    gold_winner: Literal["correct"]
    error_explanation: str
    gold_source: str

    def __post_init__(self) -> None:
        ids = [section.section_id for section in self.correct_sections]
        if self.section_count != len(self.correct_sections):
            raise SchemaError("section_count does not match correct_sections")
        if len(ids) != len(set(ids)):
            raise SchemaError("section IDs must be unique")
        if self.flawed_section_id not in ids:
            raise SchemaError("flawed_section_id must identify one section")
        if self.gold_winner != "correct":
            raise SchemaError("gold_winner must be 'correct' for this design")
        correct_target = next(
            s.text for s in self.correct_sections if s.section_id == self.flawed_section_id
        )
        if correct_target.strip() == self.flawed_section_text.strip():
            raise SchemaError("flawed text must differ from its counterpart")
        for name, value in (
            ("item_id", self.item_id),
            ("domain", self.domain),
            ("error_type", self.error_type),
            ("error_explanation", self.error_explanation),
            ("gold_source", self.gold_source),
        ):
            if not value.strip():
                raise SchemaError(f"{name} must be non-empty")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BaseItem":
        raw_sections = data.get("correct_sections")
        if not isinstance(raw_sections, list):
            raise SchemaError("correct_sections must be a list")
        return cls(
            item_id=_required_str(data, "item_id"),
            domain=_required_str(data, "domain"),
            error_type=_required_str(data, "error_type"),
            section_count=int(data.get("section_count", -1)),
            correct_sections=tuple(Section.from_dict(s) for s in raw_sections),
            flawed_section_id=_required_str(data, "flawed_section_id"),
            flawed_section_text=_required_str(data, "flawed_section_text"),
            gold_winner=data.get("gold_winner"),
            error_explanation=_required_str(data, "error_explanation"),
            gold_source=_required_str(data, "gold_source"),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["correct_sections"] = [s.to_dict() for s in self.correct_sections]
        return data


Position = Literal["early", "middle", "late"]
CandidateOrder = Literal["correct_first", "flawed_first"]


@dataclass(frozen=True)
class Condition:
    condition_id: str
    item_id: str
    context_length: str
    target_tokens: int
    actual_prompt_tokens: int
    internal_position: Position
    target_index: int
    target_depth: float
    candidate_order: CandidateOrder
    correct_candidate_id: str
    flawed_candidate_id: str
    candidate_a_id: str
    candidate_b_id: str
    candidate_a_text: str
    candidate_b_text: str
    candidate_a_tokens: int
    candidate_b_tokens: int
    section_order: tuple[str, ...]
    correct_sections: tuple[Section, ...]
    flawed_sections: tuple[Section, ...]
    gold_content_winner: Literal["correct"] = "correct"
    gold_display_winner: Literal["A", "B"] = "A"

    def __post_init__(self) -> None:
        if self.internal_position not in ("early", "middle", "late"):
            raise SchemaError("invalid internal_position")
        if self.candidate_order not in ("correct_first", "flawed_first"):
            raise SchemaError("invalid candidate_order")
        expected_display = "A" if self.candidate_order == "correct_first" else "B"
        if self.gold_content_winner != "correct" or self.gold_display_winner != expected_display:
            raise SchemaError("gold winner fields do not match candidate_order")
        if not self.section_order or not 0 <= self.target_index < len(self.section_order):
            raise SchemaError("invalid section_order or target_index")
        if not 0.0 <= self.target_depth <= 1.0:
            raise SchemaError("target_depth must be between zero and one")
        correct_ids = tuple(section.section_id for section in self.correct_sections)
        flawed_ids = tuple(section.section_id for section in self.flawed_sections)
        if correct_ids != self.section_order or flawed_ids != self.section_order:
            raise SchemaError("persisted section order does not match candidate sections")
        expected_a = self.correct_candidate_id if self.candidate_order == "correct_first" else self.flawed_candidate_id
        expected_b = self.flawed_candidate_id if self.candidate_order == "correct_first" else self.correct_candidate_id
        if self.candidate_a_id != expected_a or self.candidate_b_id != expected_b:
            raise SchemaError("candidate identity mapping does not match candidate_order")
        for name in (
            "condition_id", "item_id", "context_length", "correct_candidate_id",
            "flawed_candidate_id", "candidate_a_id", "candidate_b_id",
            "candidate_a_text", "candidate_b_text",
        ):
            if not getattr(self, name).strip():
                raise SchemaError(f"{name} must be non-empty")
        if min(self.target_tokens, self.actual_prompt_tokens, self.candidate_a_tokens, self.candidate_b_tokens) < 1:
            raise SchemaError("token counts must be positive")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Condition":
        try:
            raw_correct = data["correct_sections"]
            raw_flawed = data["flawed_sections"]
            raw_order = data["section_order"]
        except KeyError as exc:
            raise SchemaError(f"missing condition field: {exc.args[0]}") from exc
        if not isinstance(raw_correct, list) or not isinstance(raw_flawed, list):
            raise SchemaError("condition sections must be lists")
        if not isinstance(raw_order, list):
            raise SchemaError("section_order must be a list")
        return cls(
            condition_id=_required_str(data, "condition_id"),
            item_id=_required_str(data, "item_id"),
            context_length=_required_str(data, "context_length"),
            target_tokens=int(data["target_tokens"]),
            actual_prompt_tokens=int(data["actual_prompt_tokens"]),
            internal_position=cast(Position, _required_str(data, "internal_position")),
            target_index=int(data["target_index"]),
            target_depth=float(data["target_depth"]),
            candidate_order=cast(CandidateOrder, _required_str(data, "candidate_order")),
            correct_candidate_id=_required_str(data, "correct_candidate_id"),
            flawed_candidate_id=_required_str(data, "flawed_candidate_id"),
            candidate_a_id=_required_str(data, "candidate_a_id"),
            candidate_b_id=_required_str(data, "candidate_b_id"),
            candidate_a_text=_required_str(data, "candidate_a_text"),
            candidate_b_text=_required_str(data, "candidate_b_text"),
            candidate_a_tokens=int(data["candidate_a_tokens"]),
            candidate_b_tokens=int(data["candidate_b_tokens"]),
            section_order=tuple(str(value) for value in raw_order),
            correct_sections=tuple(Section.from_dict(s) for s in raw_correct),
            flawed_sections=tuple(Section.from_dict(s) for s in raw_flawed),
            gold_content_winner=data.get("gold_content_winner"),
            gold_display_winner=data.get("gold_display_winner"),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["section_order"] = list(self.section_order)
        data["correct_sections"] = [section.to_dict() for section in self.correct_sections]
        data["flawed_sections"] = [section.to_dict() for section in self.flawed_sections]
        return data
