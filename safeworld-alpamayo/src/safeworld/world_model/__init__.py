"""Action-conditioned world consequence model.

The v2 (scientifically valid) interface lives in :mod:`safeworld.world_model.base`
(``ConsequenceWorldModel``). The legacy dry-run engineered model in
``model_v1`` is retained as ``legacy_dry_run`` and excluded from the v2 main CLI.
"""

from safeworld.world_model.base import (
    ALPAMAYO_PROPOSAL_GENERATOR,
    ConsequenceWorldModel,
    FutureConsequencePrediction,
    ObservationHistory,
)

__all__ = [
    "ALPAMAYO_PROPOSAL_GENERATOR",
    "ConsequenceWorldModel",
    "FutureConsequencePrediction",
    "ObservationHistory",
]
