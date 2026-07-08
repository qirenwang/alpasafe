from __future__ import annotations

import hashlib
import random
from dataclasses import asdict, dataclass, field

from safeworld.data.schema import CandidateEvaluation


@dataclass(slots=True)
class SelectionConfig:
    risk_threshold: float = 0.45
    lambda_progress: float = 0.01
    beta_comfort: float = 0.05
    gamma_uncertainty: float = 0.10
    rawc_threshold: float = 0.65
    fallback_candidate: str = "fallback_stop"
    random_seed: int = 7

    @classmethod
    def from_dict(cls, row: dict[str, object]) -> "SelectionConfig":
        return cls(
            risk_threshold=float(row.get("risk_threshold", 0.45)),
            lambda_progress=float(row.get("lambda_progress", 0.01)),
            beta_comfort=float(row.get("beta_comfort", 0.05)),
            gamma_uncertainty=float(row.get("gamma_uncertainty", 0.10)),
            rawc_threshold=float(row.get("rawc_threshold", 0.65)),
            fallback_candidate=str(row.get("fallback_candidate", "fallback_stop")),
            random_seed=int(row.get("random_seed", 7)),
        )


@dataclass(slots=True)
class SelectionDecision:
    sample_id: str
    k: int
    selected_candidate_id: str
    selected_candidate_source: str
    selection_method: str
    predicted_risk: float | None
    progress_score: float | None
    comfort_cost: float | None
    rawc_score: float | None
    decision_reason: str
    metadata: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def selection_objective(evaluation: CandidateEvaluation, config: SelectionConfig) -> float:
    risk = evaluation.predicted_risk if evaluation.predicted_risk is not None else 1.0
    progress = evaluation.progress_score or 0.0
    comfort = evaluation.comfort_cost or 0.0
    uncertainty = evaluation.uncertainty or 0.0
    return (
        risk
        - config.lambda_progress * progress
        + config.beta_comfort * comfort
        + config.gamma_uncertainty * uncertainty
    )


def _native(evaluations: list[CandidateEvaluation]) -> list[CandidateEvaluation]:
    return [e for e in evaluations if e.candidate_source == "alpamayo_native"]


def _fallbacks(evaluations: list[CandidateEvaluation]) -> list[CandidateEvaluation]:
    return [e for e in evaluations if e.candidate_source == "fallback"]


def _rank_key(evaluation: CandidateEvaluation) -> int:
    return evaluation.candidate_rank if evaluation.candidate_rank is not None else 10**6


def _pick_fallback(
    evaluations: list[CandidateEvaluation], config: SelectionConfig
) -> CandidateEvaluation | None:
    fallbacks = _fallbacks(evaluations)
    if not fallbacks:
        return None
    for candidate in fallbacks:
        if candidate.candidate_id == config.fallback_candidate:
            return candidate
    for candidate in fallbacks:
        if candidate.candidate_id == "fallback_slow":
            return candidate
    return fallbacks[0]


def _decision(
    sample_id: str,
    k: int,
    method: str,
    evaluation: CandidateEvaluation,
    reason: str,
    metadata: dict[str, object] | None = None,
) -> SelectionDecision:
    return SelectionDecision(
        sample_id=sample_id,
        k=k,
        selected_candidate_id=evaluation.candidate_id,
        selected_candidate_source=evaluation.candidate_source,
        selection_method=method,
        predicted_risk=evaluation.predicted_risk,
        progress_score=evaluation.progress_score,
        comfort_cost=evaluation.comfort_cost,
        rawc_score=evaluation.rawc_score,
        decision_reason=reason,
        metadata=metadata or {},
    )


class BaseSelector:
    name = "base"

    def select(
        self,
        sample_id: str,
        k: int,
        evaluations: list[CandidateEvaluation],
        config: SelectionConfig,
    ) -> SelectionDecision:
        raise NotImplementedError

    def _fallback_decision(
        self,
        sample_id: str,
        k: int,
        evaluations: list[CandidateEvaluation],
        config: SelectionConfig,
        reason: str,
    ) -> SelectionDecision:
        fallback = _pick_fallback(evaluations, config)
        if fallback is None:
            raise ValueError(f"no fallback candidate available for sample {sample_id}")
        return _decision(sample_id, k, self.name, fallback, reason)


class Top1Selector(BaseSelector):
    name = "top1"

    def select(self, sample_id, k, evaluations, config):
        native = sorted(_native(evaluations), key=_rank_key)
        if not native:
            return self._fallback_decision(sample_id, k, evaluations, config, "no_native_candidates")
        return _decision(sample_id, k, self.name, native[0], "selected_default_top1_candidate")


class RandomTopKSelector(BaseSelector):
    name = "random_topk"

    def select(self, sample_id, k, evaluations, config):
        native = sorted(_native(evaluations), key=_rank_key)
        if not native:
            return self._fallback_decision(sample_id, k, evaluations, config, "no_native_candidates")
        seed_material = f"{config.random_seed}:{sample_id}:{k}".encode("utf-8")
        seed = int(hashlib.sha256(seed_material).hexdigest()[:8], 16)
        chosen = random.Random(seed).choice(native)
        return _decision(sample_id, k, self.name, chosen, "selected_uniform_random_native_candidate")


class AlpamayoScoreSelector(BaseSelector):
    name = "alpamayo_score"

    def select(self, sample_id, k, evaluations, config):
        native = sorted(_native(evaluations), key=_rank_key)
        if not native:
            return self._fallback_decision(sample_id, k, evaluations, config, "no_native_candidates")
        scored = [e for e in native if e.model_score is not None]
        if scored:
            best = max(scored, key=lambda e: e.model_score)
            return _decision(sample_id, k, self.name, best, "selected_best_alpamayo_model_score")
        return _decision(sample_id, k, self.name, native[0], "no_model_score_available_fell_back_to_rank")


class RuleCheckerSelector(BaseSelector):
    name = "rule_checker"

    @staticmethod
    def _rule_safe(evaluation: CandidateEvaluation) -> bool:
        return not bool(evaluation.metadata.get("rule_unsafe", evaluation.outcome_labels.get("unsafe", 0)))

    def select(self, sample_id, k, evaluations, config):
        native = sorted(_native(evaluations), key=_rank_key)
        if not native:
            return self._fallback_decision(sample_id, k, evaluations, config, "no_native_candidates")
        safe = [e for e in native if self._rule_safe(e)]
        if not safe:
            return self._fallback_decision(
                sample_id, k, evaluations, config, "all_native_candidates_violate_rule_predicates"
            )
        return _decision(
            sample_id, k, self.name, safe[0], "selected_highest_rank_candidate_passing_rule_predicates"
        )


class RAWCSelector(BaseSelector):
    name = "rawc"

    def select(self, sample_id, k, evaluations, config):
        native = sorted(_native(evaluations), key=_rank_key)
        if not native:
            return self._fallback_decision(sample_id, k, evaluations, config, "no_native_candidates")
        scored = [e for e in native if e.rawc_score is not None]
        if not scored:
            return _decision(sample_id, k, self.name, native[0], "no_rawc_score_available_fell_back_to_rank")
        best = max(scored, key=lambda e: e.rawc_score)
        if best.rawc_score <= config.rawc_threshold:
            return self._fallback_decision(
                sample_id, k, evaluations, config, "all_native_candidates_below_rawc_threshold"
            )
        return _decision(sample_id, k, self.name, best, "selected_best_rawc_consistency_score")


class SafeWorldSelector(BaseSelector):
    name = "safeworld"

    def select(self, sample_id, k, evaluations, config):
        native = sorted(_native(evaluations), key=_rank_key)
        if not native:
            return self._fallback_decision(sample_id, k, evaluations, config, "no_native_candidates")
        acceptable = [
            e for e in native if e.predicted_risk is not None and e.predicted_risk < config.risk_threshold
        ]
        if not acceptable:
            return self._fallback_decision(
                sample_id, k, evaluations, config, "all_native_candidates_exceed_predicted_risk_threshold"
            )
        best = min(acceptable, key=lambda e: selection_objective(e, config))
        return _decision(
            sample_id,
            k,
            self.name,
            best,
            "selected_min_consequence_objective_among_acceptable_candidates",
            metadata={"objective": round(selection_objective(best, config), 6)},
        )


class OracleBestInKSelector(BaseSelector):
    name = "oracle_best_in_k"

    @staticmethod
    def _oracle_risk(evaluation: CandidateEvaluation) -> float:
        unsafe = float(evaluation.outcome_labels.get("unsafe", 1.0))
        min_margin = float(evaluation.outcome_labels.get("min_margin", 0.0))
        # Tie-break safe candidates by their worst proxy safety margin.
        return unsafe - min(min_margin, 10.0) / 100.0

    def select(self, sample_id, k, evaluations, config):
        native = sorted(_native(evaluations), key=_rank_key)
        if not native:
            return self._fallback_decision(sample_id, k, evaluations, config, "no_native_candidates")
        proxy = all(e.label_source == "proxy_rule" for e in native)
        label_note = "proxy_oracle_only" if proxy else "oracle"
        safe = [e for e in native if float(e.outcome_labels.get("unsafe", 1.0)) < 0.5]
        if not safe:
            return self._fallback_decision(
                sample_id, k, evaluations, config, f"{label_note}:all_native_candidates_unsafe"
            )
        best = min(safe, key=self._oracle_risk)
        return _decision(sample_id, k, self.name, best, f"{label_note}:selected_best_in_k_by_oracle_label")


def all_selectors() -> list[BaseSelector]:
    return [
        Top1Selector(),
        RandomTopKSelector(),
        AlpamayoScoreSelector(),
        RuleCheckerSelector(),
        RAWCSelector(),
        SafeWorldSelector(),
        OracleBestInKSelector(),
    ]
