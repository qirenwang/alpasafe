from __future__ import annotations

from safeworld.data.schema import DrivingSample


def build_prompt(sample: DrivingSample) -> str:
    nav = sample.navigation_text or "Follow the route."
    tags = ", ".join(sample.scenario_tags) if sample.scenario_tags else "none"
    return (
        "You are Alpamayo 1.5. Produce chain-of-causation reasoning and a "
        f"6.4 second ego trajectory. Navigation: {nav}. Scenario tags: {tags}."
    )

