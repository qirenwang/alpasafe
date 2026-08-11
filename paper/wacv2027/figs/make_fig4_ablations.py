#!/usr/bin/env python3
"""Causal-ablation figure — CV-style bars with 95% CI whiskers
(deltas vs the arm's NORMAL condition, sequence-reading arm, K8).
Reads the locked FM1 analysis JSON."""
import matplotlib.pyplot as plt

import style
from style import GRAY, GRID, INK, MUTED, RED

ROWS = [("WRONG_SCENE_FULL_L2", "wrong-\nscene"),
        ("ZERO_FULL_L2", "zeroed\nrepr."),
        ("LAST_TOKEN_ONLY_DIAGNOSTIC", "last-token\nonly")]


def draw(ax, stats, ylim, ylabel, yticks):
    ax.axhline(0, color=INK, lw=0.7, zorder=2)
    for x, s in enumerate(stats):
        m, lo, hi = s["mean"], s["ci95_low"], s["ci95_high"]
        excludes = (lo > 0) or (hi < 0)
        ax.bar(x, m, width=0.6, color=RED if excludes else GRAY,
               alpha=1.0 if excludes else 0.45, edgecolor="none", zorder=3)
        ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], fmt="none",
                    ecolor="#3a3a3a", elinewidth=0.7, capsize=1.7,
                    capthick=0.7, zorder=4)
    ax.set_xticks(range(len(stats)))
    ax.set_xticklabels([lab for _, lab in ROWS], fontsize=5.4,
                       linespacing=1.25)
    ax.set_xlim(-0.65, len(stats) - 0.35)
    ax.set_ylim(*ylim)
    ax.set_yticks(yticks)
    ax.set_ylabel(ylabel, fontsize=7.0, labelpad=2)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color(style.BASELINE)
    ax.spines["bottom"].set_color(style.BASELINE)
    ax.grid(axis="y", color=GRID, lw=0.5, zorder=0)
    ax.tick_params(axis="y", colors=MUTED, labelsize=6.0, pad=1.5)
    ax.tick_params(axis="x", length=0, pad=2)


def main():
    style.setup()
    abl = style.load(style.FM1_ANALYSIS)["ablations"]

    fig = plt.figure(figsize=(style.SINGLE_COL, 1.55))
    gs = fig.add_gridspec(1, 2, left=0.15, right=0.995, top=0.86,
                          bottom=0.295, wspace=0.45)
    ax_r = fig.add_subplot(gs[0, 0])
    ax_t = fig.add_subplot(gs[0, 1])
    draw(ax_r, [abl[k]["delta_K8_regret_vs_NORMAL"] for k, _ in ROWS],
         (-0.0075, 0.0075), r"$\Delta$ regret ($\downarrow$ better)",
         [-0.005, 0, 0.005])
    draw(ax_t, [abl[k]["delta_K8_top1_vs_NORMAL"] for k, _ in ROWS],
         (-0.075, 0.075), r"$\Delta$ top-1 ($\uparrow$ better)",
         [-0.05, 0, 0.05])

    fig.text(0.15, 0.955,
             "inference-time ablations, training untouched (sequence-reading arm)",
             fontsize=6.2, color=MUTED, ha="left", va="center")
    fig.text(0.15, 0.045,
             "whiskers = 95% CI;  saturated red = CI-supported degradation, "
             "pale = CI crosses 0",
             fontsize=5.0, color=MUTED, ha="left", va="center")

    style.save(fig, "fig4_ablations")


if __name__ == "__main__":
    main()
