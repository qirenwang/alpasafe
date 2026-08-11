#!/usr/bin/env python3
"""Capture & provenance figure (single column, Sec. 4.2).

The same-prefill capture contract: one predict() call, exactly-one-prefill
asserted, h = H[-1] bit-exact, and the hash chain from capture to the
analysis record.
"""
import style
from style import (BLUE, BLUE_EDGE, BLUE_FILL, FAINT, INK, MUTED,
                   NEUTRAL_EDGE, arrow, box, canvas)


def main():
    style.setup()
    fig, ax = canvas(style.SINGLE_COL, 2.30)

    # container: one predict() call
    box(ax, 1, 36, 98, 60, "", fc="#fbfaf8", ec=FAINT, rounding=2.0)
    ax.text(4, 91.5, "one production predict( ) call — exactly-one-prefill asserted",
            fontsize=6.0, color=MUTED, va="center", style="italic")

    # prompt prefill token strip
    for i in range(14):
        box(ax, 5 + i * 3.75, 78, 3.0, 6.0, "", fc="#e7e5df",
            ec=NEUTRAL_EDGE, lw=0.4, rounding=0.5)
    ax.text(5, 73.2, "prompt prefill · 3,086 tokens", fontsize=6.0,
            color=MUTED, va="center")
    arrow(ax, 60, 81, 70, 81)
    box(ax, 71, 74, 25, 14, "sample $K{=}8$\ncandidates", fs=6.0)

    # hidden states H, last row highlighted
    box(ax, 5, 44, 50, 24,
        "$H\\in\\mathbb{R}^{3086\\times4096}$ — final-layer\nhidden states (post final norm)",
        fc="#eef3fa", ec=NEUTRAL_EDGE, fs=6.0)
    arrow(ax, 30, 72.0, 30, 68.6)
    box(ax, 5, 44, 50, 4.6, "", fc=BLUE_FILL, ec=BLUE_EDGE, lw=0.9,
        rounding=0.8)
    ax.text(58.5, 46.5, "$h = H[-1]$", fontsize=6.8, color=BLUE,
            va="center", fontweight="bold")
    ax.text(58.5, 41.2, "bit-exact, sha-verified · 519/519 groups",
            fontsize=5.6, color=MUTED, va="center")
    arrow(ax, 83.5, 73.6, 76, 50.5, color=FAINT,
          connectionstyle="arc3,rad=-0.12")
    ax.text(86, 62, "same\nforward pass", fontsize=5.6, color=MUTED,
            ha="left", va="center", style="italic")

    # provenance chain
    labels = ["capture\nhashes", "training\ntables", "locked\npredictions",
              "analysis\nrecord"]
    for i, lab in enumerate(labels):
        x = 1 + i * 25.5
        box(ax, x, 9, 22, 16, lab, fs=5.9, fc="#f6f5f2")
        ax.text(x + 1.8, 22.6, "sha", fontsize=4.8, color=BLUE,
                va="center", fontweight="bold")
        if i:
            arrow(ax, x - 3.1, 17, x - 0.4, 17)
    arrow(ax, 28, 43.6, 12, 26, connectionstyle="arc3,rad=0.15")
    ax.text(1, 2.5, "provenance chain — every hop hash-verified before analysis",
            fontsize=5.9, color=MUTED, va="center", style="italic")

    style.save(fig, "fig_capture")


if __name__ == "__main__":
    main()
