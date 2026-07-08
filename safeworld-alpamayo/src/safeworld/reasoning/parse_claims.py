from __future__ import annotations

from safeworld.reasoning.claim_schema import ReasoningClaim

CLAIM_KEYWORDS = {
    "perception_claim": ["pedestrian", "vehicle", "cone", "crosswalk", "construction", "lane"],
    "risk_claim": ["hazard", "risk", "cut in", "blocked", "glare", "uncertain", "crossing"],
    "action_intention_claim": ["should", "prepare", "monitor", "cautious"],
    "action_claim": ["slow", "yield", "stop", "keep lane", "shift", "continue", "proceed"],
    "uncertainty_claim": ["uncertain", "may", "if needed", "only if"],
}


def parse_claims(reasoning_text: str) -> list[ReasoningClaim]:
    text = reasoning_text.lower()
    claims: list[ReasoningClaim] = []
    for claim_type, keywords in CLAIM_KEYWORDS.items():
        for keyword in keywords:
            if keyword in text:
                claims.append(ReasoningClaim(claim_type, reasoning_text, keyword))
    return claims


def has_keyword(claims: list[ReasoningClaim], *keywords: str) -> bool:
    wanted = set(keywords)
    return any(claim.keyword in wanted for claim in claims)


def has_type(claims: list[ReasoningClaim], claim_type: str) -> bool:
    return any(claim.claim_type == claim_type for claim in claims)

