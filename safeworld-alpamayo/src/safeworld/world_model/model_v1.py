from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from safeworld.data.schema import WorldTarget
from safeworld.geometry.trajectory import accelerations, as_xy_array, curvature, jerks, progress, speeds

FEATURE_NAMES = [
    "bias",
    "mean_speed",
    "max_speed",
    "max_abs_accel",
    "max_abs_jerk",
    "max_curvature",
    "max_abs_lateral",
    "progress",
    "object_count",
    "tag_pedestrian_crossing",
    "tag_vehicle_cut_in",
    "tag_construction_or_blocked_lane",
    "tag_low_light_or_glare",
    "tag_hard_brake",
    "reason_slow_or_yield",
    "reason_keep_lane",
    "candidate_stop",
    "candidate_slow",
    "candidate_lateral",
    "candidate_speed_up",
]


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def extract_features(target: WorldTarget) -> np.ndarray:
    arr = as_xy_array(target.candidate_trajectory)
    speed = speeds(arr)
    acc = accelerations(arr)
    jerk = jerks(arr)
    curv = curvature(arr)
    metadata = target.metadata
    tags = set(metadata.get("scenario_tags", []))
    reasoning = str(metadata.get("reasoning_text", "")).lower()
    candidate_id = target.candidate_id
    values = {
        "bias": 1.0,
        "mean_speed": float(np.mean(speed)) if speed.size else 0.0,
        "max_speed": float(np.max(speed)) if speed.size else 0.0,
        "max_abs_accel": float(np.max(np.abs(acc))) if acc.size else 0.0,
        "max_abs_jerk": float(np.max(np.abs(jerk))) if jerk.size else 0.0,
        "max_curvature": float(np.max(curv)) if curv.size else 0.0,
        "max_abs_lateral": float(np.max(np.abs(arr[:, 1]))) if len(arr) else 0.0,
        "progress": progress(arr),
        "object_count": float(metadata.get("object_count", 0)),
        "tag_pedestrian_crossing": float("pedestrian_crossing" in tags),
        "tag_vehicle_cut_in": float("vehicle_cut_in" in tags),
        "tag_construction_or_blocked_lane": float("construction_or_blocked_lane" in tags),
        "tag_low_light_or_glare": float("low_light_or_glare" in tags),
        "tag_hard_brake": float("hard_brake" in tags),
        "reason_slow_or_yield": float("slow" in reasoning or "yield" in reasoning),
        "reason_keep_lane": float("keep lane" in reasoning),
        "candidate_stop": float("stop" in candidate_id),
        "candidate_slow": float("slow" in candidate_id or "yield" in candidate_id),
        "candidate_lateral": float("left" in candidate_id or "right" in candidate_id),
        "candidate_speed_up": float("speed_up" in candidate_id),
    }
    return np.asarray([values[name] for name in FEATURE_NAMES], dtype=float)


@dataclass(slots=True)
class EngineeredRiskModel:
    feature_names: list[str]
    weights: list[float]
    mean: list[float]
    scale: list[float]
    threshold: float = 0.5

    def predict_proba_matrix(self, x: np.ndarray) -> np.ndarray:
        mean = np.asarray(self.mean, dtype=float)
        scale = np.asarray(self.scale, dtype=float)
        weights = np.asarray(self.weights, dtype=float)
        z = (x - mean) / scale
        return sigmoid(z @ weights)

    def predict_target(self, target: WorldTarget) -> float:
        x = extract_features(target)[None, :]
        return float(self.predict_proba_matrix(x)[0])

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "EngineeredRiskModel":
        return cls(
            feature_names=list(row["feature_names"]),
            weights=[float(v) for v in row["weights"]],
            mean=[float(v) for v in row["mean"]],
            scale=[float(v) for v in row["scale"]],
            threshold=float(row.get("threshold", 0.5)),
        )


def fit_risk_model(
    targets: list[WorldTarget],
    learning_rate: float = 0.08,
    epochs: int = 250,
    l2: float = 0.001,
) -> tuple[EngineeredRiskModel, dict[str, float]]:
    if not targets:
        raise ValueError("at least one target is required")
    x = np.vstack([extract_features(target) for target in targets])
    y = np.asarray([float(target.outcome_labels.get("unsafe", 0.0)) for target in targets])
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    mean[0] = 0.0
    scale[scale < 1e-6] = 1.0
    scale[0] = 1.0
    z = (x - mean) / scale
    weights = np.zeros(z.shape[1], dtype=float)
    for _ in range(epochs):
        prob = sigmoid(z @ weights)
        grad = z.T @ (prob - y) / len(y) + l2 * weights
        grad[0] -= l2 * weights[0]
        weights -= learning_rate * grad
    prob = sigmoid(z @ weights)
    metrics = {
        "train_loss": float(np.mean((prob - y) ** 2)),
        "positive_rate": float(np.mean(y)),
        "mean_probability": float(np.mean(prob)),
    }
    model = EngineeredRiskModel(
        feature_names=FEATURE_NAMES,
        weights=weights.round(8).tolist(),
        mean=mean.round(8).tolist(),
        scale=scale.round(8).tolist(),
        threshold=0.5,
    )
    return model, metrics

