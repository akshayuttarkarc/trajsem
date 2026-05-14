from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any


@dataclass
class EvidenceRef:
    source: str
    metric: str
    value: Any
    frame: int | None = None
    time_ns: float | None = None
    description: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SemanticEvent:
    event_type: str
    title: str
    time_window_ns: tuple[float, float]
    entities: dict[str, Any]
    evidence: dict[str, Any]
    interpretation: str
    confidence: float
    severity: str = "moderate"
    frame_window: tuple[int, int] | None = None
    evidence_refs: list[EvidenceRef] = field(default_factory=list)
    verification_status: str = "unverified"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["time_window_ns"] = list(self.time_window_ns)
        if self.frame_window is not None:
            d["frame_window"] = list(self.frame_window)
        d["evidence_refs"] = [r.to_dict() for r in self.evidence_refs]
        return d


@dataclass
class VerificationFinding:
    claim: str
    status: str
    supporting_events: list[str]
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)
