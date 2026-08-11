"""T26E-A: official-score training dataset (assembly, schema, loaders).

Derived, read-only join of the frozen T26D K8 future-consequence targets,
candidate provenance, and the cross-version official AlpaSim 196d21a scene
scores. Created under T26E-A authorization (dataset + contract preflight only;
no training). See tasks/T26_world_model_and_reward_training.md and
outputs/reports/T26E_A_official_score_training_dataset_and_contract.md.
"""

from safeworld.t26e.loader import (
    T26EGroupIndex,
    T26EOfficialScoreDataset,
    T26ETestInputDataset,
)
from safeworld.t26e.schema import (
    FORBIDDEN_INPUT_KEYS,
    T26ERecord,
    validate_record_structure,
)

__all__ = [
    "FORBIDDEN_INPUT_KEYS",
    "T26EGroupIndex",
    "T26EOfficialScoreDataset",
    "T26ERecord",
    "T26ETestInputDataset",
    "validate_record_structure",
]
