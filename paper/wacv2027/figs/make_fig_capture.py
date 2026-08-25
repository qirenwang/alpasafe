#!/usr/bin/env python3
"""Same-prefill capture figure (single column, Sec. 3.2).

One production predict() call, one forward pass, two products: the K=8
sampled candidates and the prefill hidden states H. The auditor reads only
the last row h = H[-1], a free byproduct of generation. (Provenance/hash
details live in the text, not in this figure.)
"""
import fancy
import style
from style import (BLUE, BLUE_EDGE, BLUE_FILL, FAINT, INK, MUTED,
                   NEUTRAL_EDGE, arrow, box, canvas)

W_IN, H_IN = style.SINGLE_COL, 2.10
AR = (W_IN / H_IN) * 1.0  # snowflake aspect correction


def main():
    style.setup()
    fig, ax = canvas(W_IN, H_IN)

    # container: one production call
    box(ax, 1, 2, 98, 96, "", fc="#fbfaf8", ec=FAINT, rounding=2.0)
    ax.text(4, 93.5, "one production predict( ) call", fontsize=6.0,
            color=MUTED, va="center", style="italic")

    # input: prompt prefill token strip (last cell highlighted)
    last_cx = fancy.token_strip(ax, 6, 77, 42, 5.4, n=15)
    ax.text(6, 71.6, "prompt prefill · 3,086 tokens", fontsize=6.0,
            color=MUTED, va="center")

    # frozen VLA
    fancy.vgrad(ax, 62, 69, 30, 19, "#dce9fb", "#9ec5f4", ec="#5598e7",
                lw=0.7, rounding=1.6, shadow=True)
    ax.text(77, 78.5, "frozen driving\nVLA (10B)", ha="center", va="center",
            fontsize=6.4, color=INK, zorder=5)
    fancy.snowflake(ax, 91.6, 87.2, r=1.45, ar=AR)
    arrow(ax, 50.5, 80, 61, 79.5)

    # the split: one forward pass, two products
    ax.text(64, 64.5, "one forward pass · two products", fontsize=5.6,
            color=MUTED, ha="center", va="center", style="italic")

    # product 1 (right): sampled candidates
    fancy.road(ax, 70, 27, 24, 29, n=8)
    ax.text(82, 21.0, "$K{=}8$ sampled\ncandidates", fontsize=6.0, color=INK,
            ha="center", va="top")
    arrow(ax, 81, 68.3, 82, 57.5)

    # product 2 (left): hidden states H with the last row highlighted
    box(ax, 6, 30, 48, 27, "", fc="#eef3fa", ec=NEUTRAL_EDGE)
    ax.text(30, 49.5, "final-layer hidden states", fontsize=6.0, color=INK,
            ha="center", va="center", zorder=5)
    ax.text(30, 44.2, r"$H \in \mathbb{R}^{3086\times4096}$", fontsize=6.4,
            color=INK, ha="center", va="center", zorder=5)
    for k in range(3):
        yy = 40.0 - k * 2.1
        ax.plot([7.5, 52.5], [yy, yy], color="#c9d6e8", lw=0.5, zorder=4)
    box(ax, 6, 30, 48, 4.0, "", fc=BLUE_FILL, ec=BLUE_EDGE, lw=0.9,
        rounding=0.8)
    arrow(ax, 67, 68.5, 36, 58.0, connectionstyle="arc3,rad=0.12")

    # extraction: h = H[-1]
    arrow(ax, 22, 29.4, 22, 19.8, color=BLUE, lw=1.1)
    box(ax, 12, 11.5, 20, 5.2, "", fc=BLUE, ec="white", lw=0.7, rounding=0.9)
    ax.text(35, 14.4, "$h = H[-1]$", fontsize=7.0, color=BLUE,
            va="center", fontweight="bold")
    ax.text(35, 8.6, "the only representation the auditor reads · free",
            fontsize=5.6, color=MUTED, va="center")

    style.save(fig, "fig_capture")


if __name__ == "__main__":
    main()
