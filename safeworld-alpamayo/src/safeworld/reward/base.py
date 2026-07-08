"""SafeWorld v2 reward-head interface (method_spec_v2).

``RewardHead`` (G_phi) predicts reward *components* from a predicted future
consequence ``Z_hat_i``. It is a different component from the ``RewardSelector``
(decision #7): the head predicts, the selector chooses.

Predicted reward components live in :class:`PredictedRewardComponents`. They are
deliberately a *different type* from evaluator-measured components
(``EvaluationRewardComponents`` in T23) so that predicted and evaluation rewards
can never be silently mixed (decision #6, #10).

Reward component vocabulary (frozen in T21, semantics finalized after T22):

    NC      - no-collision score
    DAC     - drivable-area compliance score
    TTC     - time-to-collision derived score
    EP      - ego progress score
    Comfort - comfort score (jerk / lateral accel aggregate)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

REWARD_COMPONENT_NAMES = ("NC", "DAC", "TTC", "EP", "Comfort")


@dataclass(slots=True)
class PredictedRewardComponents:
    """m_hat_i: predicted reward components from G_phi(Z_hat_i, tau_i, r_i).

    ``final_reward`` (R_hat_i) is the scalar the :class:`RewardSelector` argmaxes
    over. ``uncertainty`` carries the calibrated predictive uncertainty used for
    long-tail-aware selection.
    """

    sample_id: str
    candidate_id: str
    nc: float
    dac: float
    ttc: float
    ep: float
    comfort: float
    final_reward: float
    uncertainty: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def components(self) -> dict[str, float]:
        return {
            "NC": self.nc,
            "DAC": self.dac,
            "TTC": self.ttc,
            "EP": self.ep,
            "Comfort": self.comfort,
        }


@runtime_checkable
class RewardHead(Protocol):
    """G_phi. Predicts reward components from a predicted future consequence."""

    def predict_components(
        self,
        future_prediction: Any,
        candidate: Any,
    ) -> PredictedRewardComponents:
        """Predict m_hat_i for a single candidate from its Z_hat_i."""
        ...
