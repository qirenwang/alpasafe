from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(slots=True)
class GateDecision:
    accepted: bool
    primary_reason: str
    risk_probability: float
    minimum_margin: float
    rawc_score: float
    selected_candidate_id: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def gate_action(
    candidate_id: str,
    risk_probability: float,
    minimum_margin: float,
    rawc_score: float,
    risk_threshold: float,
    margin_threshold: float,
    rawc_threshold: float,
    fallback_candidate_id: str = "tau_stop",
) -> GateDecision:
    if risk_probability >= risk_threshold:
        return GateDecision(False, "risk_probability_exceeds_threshold", risk_probability, minimum_margin, rawc_score, fallback_candidate_id)
    if minimum_margin <= margin_threshold:
        return GateDecision(False, "safety_margin_below_threshold", risk_probability, minimum_margin, rawc_score, fallback_candidate_id)
    if rawc_score <= rawc_threshold:
        return GateDecision(False, "rawc_score_below_threshold", risk_probability, minimum_margin, rawc_score, fallback_candidate_id)
    return GateDecision(True, "accepted", risk_probability, minimum_margin, rawc_score, candidate_id)

