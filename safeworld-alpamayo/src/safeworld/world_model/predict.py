from __future__ import annotations

import json

from safeworld.data.schema import WorldTarget
from safeworld.utils.io import project_path
from safeworld.world_model.model_v1 import EngineeredRiskModel


def load_model(path: str) -> EngineeredRiskModel:
    payload = json.loads(project_path(path).read_text(encoding="utf-8"))
    return EngineeredRiskModel.from_dict(payload["model"])


def predict_risk(model_path: str, target: WorldTarget) -> float:
    return load_model(model_path).predict_target(target)

