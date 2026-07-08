"""SafeWorld v2 consequence world-model interface (method_spec_v2).

This module defines the *scientifically valid* world-model contract introduced
in T21. It contains **no training implementation** and **no fake world model**.

Scientific naming rule (see ``docs/terminology_v2.md``):

- A ``ConsequenceWorldModel`` predicts a candidate-conditioned *future
  structured state/latent sequence* ``Z_hat_i``. Anything that maps a candidate
  directly to a scalar reward without future-state supervision is **not** a
  world model; it must be named ``TrajectoryRewardModel`` or ``Critic``.
- The legacy dry-run model in :mod:`safeworld.world_model.model_v1` is a
  ``legacy_dry_run`` engineered classifier. It is excluded from the v2 main CLI.

Equations (method_spec_v2):

    {(r_i, tau_i)} = Alpamayo(O_t, c_nav; K)
    Z_hat_i        = W_theta(O_history, tau_i, r_i)
    m_hat_i        = G_phi(Z_hat_i, tau_i, r_i)
    i_star         = argmax_i R_hat_i
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

# Frozen proposal generator identity. SafeWorld never generates ego trajectories;
# it only consumes Alpamayo-native candidates (see decision #4, #5).
ALPAMAYO_PROPOSAL_GENERATOR = "nvidia/Alpamayo-1.5-10B"


@dataclass(slots=True)
class ObservationHistory:
    """Past observation context O_history passed to W_theta.

    This is the *conditioning* input, not a label. It must not contain any field
    derived from a candidate's own future outcome (leakage guard, see T23).
    """

    sample_id: str
    scene_id: str
    ego_history_states: list[list[float]]
    history_hz: float
    history_frame: str = "ego_t0"
    agent_history_or_occupancy: Any | None = None
    map_context: dict[str, Any] | None = None
    navigation_text: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class FutureConsequencePrediction:
    """Z_hat_i: a *predicted* candidate-conditioned future structured state seq.

    This is a model prediction, never an evaluator label. It is stored separately
    from any :class:`EvaluationRewardComponents` (decision #6/#7; enforced in T23).
    """

    sample_id: str
    candidate_id: str
    future_states: list[list[float]]
    future_hz: float
    horizon_s: float
    future_agent_states_or_occupancy: Any | None = None
    latent_sequence: Any | None = None
    state_uncertainty: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class ConsequenceWorldModel(Protocol):
    """W_theta. Predicts candidate-conditioned future structured consequences.

    Implementations are trained in T26 (not this phase). A class satisfying this
    Protocol must produce a future state/latent *sequence*; returning only a
    scalar reward does not satisfy the contract.
    """

    def predict_future(
        self,
        observation_history: ObservationHistory,
        candidate: Any,
    ) -> FutureConsequencePrediction:
        """Predict Z_hat_i for a single Alpamayo-native candidate."""
        ...
