from __future__ import annotations

import argparse

from safeworld.alpamayo.wrapper import AlpamayoWrapper
from safeworld.data.loaders import load_samples
from safeworld.utils.io import load_yaml, write_jsonl, write_markdown
from safeworld.utils.seed import set_seed


def _build_wrapper(config: dict, dry_run: bool) -> AlpamayoWrapper:
    topk_cfg = config.get("topk", {})
    return AlpamayoWrapper(
        model_name=config.get("model", {}).get("name", "nvidia/Alpamayo-1.5-10B"),
        dry_run=dry_run or bool(config.get("run", {}).get("dry_run", True)),
        model_version=str(config.get("model", {}).get("version", "1.5-10B")),
        use_real_alpamayo=bool(topk_cfg.get("use_real_alpamayo", False)),
    )


def run_inference(config_path: str, dry_run: bool, limit: int | None) -> str:
    config = load_yaml(config_path)
    set_seed(int(config.get("run", {}).get("seed", 7)))
    input_path = config.get("inputs", {}).get("index_path", "outputs/index/sample_index_mined.jsonl")
    output_path = config.get("outputs", {}).get("proposal_path", "outputs/proposals/alpamayo_sample.jsonl")
    report_path = config.get("outputs", {}).get("e0_report_path", "outputs/reports/e0_reproduction.md")
    samples = load_samples(input_path)
    if limit is not None:
        samples = samples[:limit]
    wrapper = _build_wrapper(config, dry_run)
    proposals = [wrapper.run(sample) for sample in samples]
    write_jsonl(output_path, [proposal.to_dict() for proposal in proposals])
    latencies = [proposal.inference_latency_ms or 0.0 for proposal in proposals]
    mean_latency = sum(latencies) / max(len(latencies), 1)
    report = "\n".join(
        [
            "# E0 - Reproduction and profiling",
            "",
            "Mode: dry_run Alpamayo proposal wrapper.",
            f"Samples: {len(proposals)}",
            f"Model version: {wrapper.model_name}",
            f"Output proposals: `{output_path}`",
            f"Mean latency ms: {mean_latency:.4f}",
            "",
            "Basic ADE/FDE: not reported in E0 dry-run; ground-truth trajectories are synthetic and used in later safety labels.",
            "",
            "Known limitations: this validates schema, provenance, prompt construction, and proposal storage only.",
            "",
            "Reproduction command:",
            "`PYTHONPATH=src python -m safeworld.alpamayo.run_inference --config configs/inference_alpamayo.yaml --dry-run --limit 12`",
            "",
        ]
    )
    write_markdown(report_path, report)
    return output_path


def run_topk_inference(config_path: str, dry_run: bool, k: int | None, limit: int | None) -> str:
    config = load_yaml(config_path)
    set_seed(int(config.get("run", {}).get("seed", 7)))
    topk_cfg = config.get("topk", {})
    if not topk_cfg.get("enabled", True):
        raise RuntimeError("topk.enabled is false in the config; enable it to run top-K inference")
    k = int(k or topk_cfg.get("default_k", 10))
    k_values = [int(v) for v in topk_cfg.get("k_values", [1, 2, 5, 10])]
    if k not in k_values:
        raise ValueError(f"k={k} is not in configured topk.k_values={k_values}")
    input_path = config.get("inputs", {}).get("index_path", "outputs/index/sample_index_mined.jsonl")
    output_template = config.get("outputs", {}).get(
        "topk_proposal_path_template", "outputs/proposals/alpamayo_topk_k{k}.jsonl"
    )
    report_template = config.get("outputs", {}).get(
        "topk_report_path_template", "outputs/reports/topk_inference_k{k}.md"
    )
    output_path = output_template.format(k=k)
    report_path = report_template.format(k=k)
    samples = load_samples(input_path)
    if limit is not None:
        samples = samples[:limit]
    wrapper = _build_wrapper(config, dry_run or bool(topk_cfg.get("dry_run", True)))
    proposals = [wrapper.run_topk(sample, k) for sample in samples]
    write_jsonl(output_path, [proposal.to_dict() for proposal in proposals])
    latencies = [proposal.latency_ms or 0.0 for proposal in proposals]
    mean_latency = sum(latencies) / max(len(latencies), 1)
    k_returned = sorted({proposal.k_returned for proposal in proposals})
    report = "\n".join(
        [
            f"# Top-K Alpamayo proposal inference (K={k})",
            "",
            "Mode: dry_run. Candidates are synthetic Alpamayo-native-style proposals for "
            "pipeline validation; they are NOT real Alpamayo 1.5 outputs.",
            "",
            f"Samples: {len(proposals)}",
            f"K requested: {k}",
            f"K returned values: {k_returned}",
            f"Model: {wrapper.model_name} (version {wrapper.model_version})",
            f"Output proposals: `{output_path}`",
            f"Mean latency ms: {mean_latency:.4f}",
            "GPU memory MB: not available in dry-run mode.",
            "",
            "Each sample stores K native candidates (alpamayo_candidate_01..K) plus three "
            "separate backup controller candidates (fallback_stop, fallback_slow, "
            "fallback_conservative_yield).",
            "",
            "Reproduction command:",
            f"`PYTHONPATH=src python -m safeworld.alpamayo.run_inference --config configs/inference_alpamayo.yaml --dry-run --k {k} --limit {len(proposals)}`",
            "",
        ]
    )
    write_markdown(report_path, report)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/inference_alpamayo.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--k", type=int, default=None, help="number of native Alpamayo candidates per sample")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    if args.k is not None:
        output_path = run_topk_inference(args.config, args.dry_run, args.k, args.limit)
        print(f"wrote top-K proposals: {output_path}")
    else:
        output_path = run_inference(args.config, args.dry_run, args.limit)
        print(f"wrote proposals: {output_path}")


if __name__ == "__main__":
    main()
