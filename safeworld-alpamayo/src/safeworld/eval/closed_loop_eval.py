from __future__ import annotations

import argparse

from safeworld.utils.io import load_yaml, write_markdown


def run_closed_loop_or_replay(config_path: str) -> str:
    config = load_yaml(config_path)
    report_path = config.get("outputs", {}).get("report_path", "outputs/reports/e5_closed_loop_or_replay_gate.md")
    report = "\n".join(
        [
            "# E5 - Closed-loop or replay gate",
            "",
            "AlpaSim/NuRec closed-loop access was not configured in this dry-run pass.",
            "The implemented gate can be evaluated in offline replay through E1/E4 artifacts.",
            "",
            "Known limitations: collision rate and route completion require simulator or replay oracle access.",
            "",
        ]
    )
    write_markdown(report_path, report)
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/eval_closed_loop.yaml")
    args = parser.parse_args()
    path = run_closed_loop_or_replay(args.config)
    print(f"wrote closed-loop/replay report: {path}")


if __name__ == "__main__":
    main()

