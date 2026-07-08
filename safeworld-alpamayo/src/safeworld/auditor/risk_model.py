from __future__ import annotations

from safeworld.data.schema import WorldTarget
from safeworld.world_model.model_v1 import EngineeredRiskModel


def calibrated_risk(model: EngineeredRiskModel, target: WorldTarget, temperature: float = 1.0) -> float:
    risk = model.predict_target(target)
    temperature = max(float(temperature), 1e-6)
    if temperature == 1.0:
        return risk
    odds = risk / max(1.0 - risk, 1e-8)
    adjusted = odds ** (1.0 / temperature)
    return float(adjusted / (1.0 + adjusted))

