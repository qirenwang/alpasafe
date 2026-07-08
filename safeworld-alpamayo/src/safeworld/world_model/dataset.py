from __future__ import annotations

import argparse

import numpy as np

from safeworld.counterfactuals.candidate_set import generate_candidates
from safeworld.data.loaders import load_samples, proposals_by_sample, topk_proposals_by_sample
from safeworld.data.schema import CandidateSet, WorldTarget
from safeworld.geometry.occupancy import path_aligned_occupancy
from safeworld.geometry.safety_predicates import (
    distance_curve,
    evaluate_all,
    ttc_curve,
    unsafe_label,
)
from safeworld.utils.io import load_yaml, write_jsonl


def _labels(results: list[object]) -> dict[str, int | float]:
    by_name = {result.name: result for result in results}
    return {
        "collision": int(by_name["minimum_distance_margin"].margin < -1.0),
        "close_encounter": int(by_name["minimum_distance_margin"].margin < 0.0),
        "offroad": int(not by_name["offroad_margin"].is_safe),
        "rule_violation": int(
            (not by_name["ttc_margin"].is_safe)
            or (not by_name["rss_like_longitudinal_margin"].is_safe)
            or (not by_name["offroad_margin"].is_safe)
        ),
        "hard_brake": int(not by_name["comfort_score"].is_safe),
        "unsafe": unsafe_label(results),
        "min_margin": float(min(result.margin for result in results if np.isfinite(result.margin))),
    }


def build_targets(config_path: str, limit: int | None = None) -> str:
    config = load_yaml(config_path)
    index_path = config.get("inputs", {}).get("index_path", "outputs/index/sample_index_mined.jsonl")
    proposal_path = config.get("inputs", {}).get("proposal_path", "outputs/proposals/alpamayo_sample.jsonl")
    output_path = config.get("outputs", {}).get("target_path", "outputs/world_targets/sample_targets.jsonl")
    samples = load_samples(index_path)
    proposal_map = proposals_by_sample(proposal_path)
    targets: list[WorldTarget] = []
    for sample in samples:
        proposal = proposal_map.get(sample.scene_id)
        if proposal is None:
            continue
        for candidate_id, candidate in generate_candidates(proposal.trajectory).items():
            results = evaluate_all(candidate, sample.object_tracks, sample.map_context)
            distances = distance_curve(candidate, sample.object_tracks)
            ttc = ttc_curve(candidate, sample.object_tracks)
            target = WorldTarget(
                sample_id=sample.scene_id,
                candidate_id=candidate_id,
                candidate_trajectory=candidate,
                future_occupancy=path_aligned_occupancy(candidate),
                future_min_distance=[
                    None if not np.isfinite(value) else round(float(value), 4) for value in distances
                ],
                future_ttc=[None if not np.isfinite(value) else round(float(value), 4) for value in ttc],
                future_safety_margins={result.name: [round(float(result.margin), 4)] for result in results},
                outcome_labels=_labels(results),
                metadata={
                    "scenario_tags": sample.scenario_tags,
                    "split": sample.split,
                    "reasoning_text": proposal.reasoning_text,
                    "object_count": len(sample.object_tracks or []),
                    "predicate_results": [result.to_dict() for result in results],
                },
            )
            targets.append(target)
            if limit is not None and len(targets) >= limit:
                write_jsonl(output_path, [item.to_dict() for item in targets])
                return output_path
    write_jsonl(output_path, [target.to_dict() for target in targets])
    return output_path


def build_topk_targets(config_path: str, k: int | None = None, limit: int | None = None) -> str:
    config = load_yaml(config_path)
    topk_cfg = config.get("topk", {})
    k = int(k or topk_cfg.get("default_k", 10))
    index_path = config.get("inputs", {}).get("index_path", "outputs/index/sample_index_mined.jsonl")
    proposal_template = config.get("inputs", {}).get(
        "topk_proposal_path_template", "outputs/proposals/alpamayo_topk_k{k}.jsonl"
    )
    output_template = config.get("outputs", {}).get(
        "topk_target_path_template", "outputs/world_targets/topk_targets_k{k}.jsonl"
    )
    proposal_path = proposal_template.format(k=k)
    output_path = output_template.format(k=k)
    samples = load_samples(index_path)
    proposal_map = topk_proposals_by_sample(proposal_path)
    if not proposal_map:
        raise RuntimeError(
            f"no top-K proposals found at {proposal_path}; run "
            f"safeworld.alpamayo.run_inference --dry-run --k {k} first"
        )
    targets: list[WorldTarget] = []
    for sample in samples:
        # Honor the limit at sample boundaries so every kept sample has its
        # complete candidate set (native + fallback); a truncated set cannot
        # be used for selection.
        if limit is not None and len(targets) >= limit:
            break
        proposal = proposal_map.get(sample.scene_id)
        if proposal is None:
            continue
        candidate_set = CandidateSet.from_proposal(proposal)
        for candidate in candidate_set.candidates:
            results = evaluate_all(candidate.trajectory, sample.object_tracks, sample.map_context)
            distances = distance_curve(candidate.trajectory, sample.object_tracks)
            ttc = ttc_curve(candidate.trajectory, sample.object_tracks)
            target = WorldTarget(
                sample_id=sample.scene_id,
                candidate_id=candidate.candidate_id,
                candidate_trajectory=candidate.trajectory,
                future_occupancy=path_aligned_occupancy(candidate.trajectory),
                future_min_distance=[
                    None if not np.isfinite(value) else round(float(value), 4) for value in distances
                ],
                future_ttc=[None if not np.isfinite(value) else round(float(value), 4) for value in ttc],
                future_safety_margins={result.name: [round(float(result.margin), 4)] for result in results},
                outcome_labels=_labels(results),
                metadata={
                    "scenario_tags": sample.scenario_tags,
                    "split": sample.split,
                    "candidate_source": candidate.candidate_source,
                    "candidate_rank": candidate.candidate_rank,
                    "model_score": candidate.model_score,
                    "k": proposal.k_returned,
                    "label_source": "proxy_rule",
                    "reasoning_text": candidate.reasoning_trace or "",
                    "object_count": len(sample.object_tracks or []),
                    "predicate_results": [result.to_dict() for result in results],
                },
            )
            targets.append(target)
    write_jsonl(output_path, [target.to_dict() for target in targets])
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/world_model_v1.yaml")
    parser.add_argument("--topk", action="store_true", help="build targets from Alpamayo top-K proposals")
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    if args.topk:
        output_path = build_topk_targets(args.config, args.k, args.limit)
    else:
        output_path = build_targets(args.config, args.limit)
    print(f"wrote world targets: {output_path}")


if __name__ == "__main__":
    main()

