from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(slots=True)
class ReasoningClaim:
    claim_type: str
    text: str
    keyword: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)

