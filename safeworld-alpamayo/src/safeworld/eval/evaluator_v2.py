"""Independent v2 evaluator contract (T23).

This evaluator maps an *independent* :class:`CandidateRolloutRecord` (produced by
a real backend) to evaluator-native :class:`EvaluationRewardComponents`. It is
deliberately separate from prediction: it never sees model outputs and its result
is never fed to the deployable :class:`~safeworld.selection.RewardSelector`.

SCDS (Safety-Critical Driving Score) is intentionally **not** locked here. T22's
audit must confirm the available metric semantics before any custom composite
score is frozen. Until then this module exposes per-component metrics only and
``SCDS_LOCKED`` stays ``False``; ``final_reward`` is a provisional, documented
aggregate flagged ``diagnostic``.
"""

from __future__ import annotations

from safeworld.data.schema_v2 import (
    CandidateRolloutRecord,
    EvaluationRewardComponents,
    assert_real_label_source,
)

# Do not freeze a custom composite score until T22 confirms metric semantics.
SCDS_LOCKED = False
METRIC_DEFINITION_VERSION = "v2_unfrozen"


def evaluate_record(
    record: CandidateRolloutRecord,
    *,
    allow_proxy: bool = False,
) -> EvaluationRewardComponents:
    """Compute evaluator-native components from an independent rollout record.

    Args:
        record: an independent candidate rollout outcome.
        allow_proxy: only unit tests may set this True. The real CLI must leave it
            False so ``proxy_test_only`` records are rejected (decision #10).
    """
    record.validate()
    if not allow_proxy:
        assert_real_label_source(record.label_source, context="evaluator_v2 real evaluation")

    if not record.rollout_success:
        # A failed rollout yields no admissible component evidence.
        return EvaluationRewardComponents(
            sample_id=record.sample_id,
            candidate_id=record.candidate_id,
            label_source=record.label_source,
            nc=0.0,
            dac=0.0,
            ttc=0.0,
            ep=0.0,
            comfort=0.0,
            final_reward=0.0,
            metric_definition_version=record.metric_definition_version,
            metadata={"rollout_failed": True, "failure_reason": record.failure_reason},
        )

    # Evaluator-native component metrics (semantics provisional until T22 freeze).
    nc = 0.0 if record.collision_events else 1.0
    dac = 0.0 if record.drivable_area_events else 1.0
    ttc = _ttc_component(record.ttc_curve)
    ep = float(record.progress) if record.progress is not None else 0.0
    comfort = _comfort_component(record.comfort_components)

    # Provisional aggregate; selection uses PREDICTED reward, not this value.
    final_reward = nc * dac * (0.5 + 0.5 * ttc)

    return EvaluationRewardComponents(
        sample_id=record.sample_id,
        candidate_id=record.candidate_id,
        label_source=record.label_source,
        nc=nc,
        dac=dac,
        ttc=ttc,
        ep=ep,
        comfort=comfort,
        final_reward=final_reward,
        metric_definition_version=record.metric_definition_version,
        metadata={"scds_locked": SCDS_LOCKED, "aggregate_is_diagnostic": True},
    )


def _ttc_component(ttc_curve: list[float]) -> float:
    """Map a TTC curve to a [0,1] score (higher minimum TTC is safer)."""
    if not ttc_curve:
        return 0.0
    min_ttc = min(ttc_curve)
    # 3s+ minimum TTC saturates to 1.0; clamp to [0,1].
    return max(0.0, min(1.0, min_ttc / 3.0))


def _comfort_component(comfort_components: dict[str, float]) -> float:
    """Aggregate comfort sub-metrics into a [0,1] score (provisional)."""
    if not comfort_components:
        return 1.0
    penalty = sum(abs(v) for v in comfort_components.values()) / len(comfort_components)
    return max(0.0, 1.0 - penalty)
