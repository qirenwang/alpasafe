#!/usr/bin/env python3
"""Architecture figure v2 (figure*, double column) — AV-paper styling.

Story, left to right: multi-view cameras + driving prompt feed the frozen
driving VLA; ONE forward pass yields both the K=8 sampled trajectories
(drawn on a perspective road) and the prefill hidden states whose last
token h is the only representation consumed. The SafeWorld head (stacked
x8, shared weights) scores each candidate; the bounded residual alone
selects; the chosen trajectory is executed (road echo, selected in blue).
Params/latency/memory annotated from the locked FN0 conformance record.
"""
from matplotlib.patches import Rectangle

import style
from style import (BLUE, BLUE_EDGE, BLUE_FILL, FAINT, INK, MUTED,
                   NEUTRAL_EDGE, WARM_EDGE, arrow, canvas)
import fancy
from fancy import (cam_stack, mini_bars, rbox, road, snowflake,
                   token_strip, transformer_stack, vgrad)

W_IN, H_IN = style.DOUBLE_COL, 3.25
AR = W_IN / H_IN

SCORES = [0.42, 0.60, 0.35, 0.86, 0.55, 0.30, 0.66, 0.48]
BEST = 3


def main():
    style.setup()
    conf = style.load(style.FN0_CONFORMANCE)["measurements"]
    params = conf["trainable_parameters"]["F_LASTTOKEN_L3_MATCHED_CONS"]
    lat = conf["median_K8_forward_latency_ms"]
    mem_mb = conf["peak_incremental_inference_bytes"] / 1e6

    fig, ax = canvas(W_IN, H_IN)

    # ================= A: inputs =================
    cam_stack(ax, 1, 66, 10, 26)
    ax.text(6, 62.0, "multi-view cameras", fontsize=4.9, color=MUTED,
            ha="center", va="center")
    rbox(ax, 1, 40, 10, 14, fc="#f6f5f2", rounding=1.0)
    for k in range(3):
        ax.add_patch(Rectangle((2.2, 43.4 + k * 3.2), 7.6 - k * 2.2, 1.1,
                               fc="#c9c7c1", ec="none", zorder=4))
    ax.text(6, 36.5, "driving prompt", fontsize=4.9, color=MUTED,
            ha="center", va="center")
    arrow(ax, 11.4, 79, 15.6, 74)
    arrow(ax, 11.4, 47, 15.6, 52)

    # ================= B: frozen VLA =================
    vgrad(ax, 16, 14, 20, 79, "#fcfdff", "#e9f2fc", ec=BLUE_EDGE, lw=1.0,
          rounding=2.0, shadow=True)
    ax.text(25.4, 88.5, "frozen driving VLA (10B)", fontsize=6.3, color=BLUE,
            ha="center", va="center", fontweight="bold")
    snowflake(ax, 34.0, 89.0, r=1.5, ar=AR)
    ax.text(25.8, 83.8, "one forward pass — no finetuning", fontsize=4.9,
            color=MUTED, ha="center", va="center", style="italic")
    transformer_stack(ax, 19, 52, 14, 26, n=4)
    ax.text(34.4, 65, "36 layers", fontsize=4.5, color=MUTED, rotation=90,
            ha="center", va="center")
    arrow(ax, 26, 50.8, 26, 42.2)
    last_x = token_strip(ax, 18.2, 34, 15.8, 5, n=18)
    ax.text(25.8, 29.8, "prompt prefill · 3,086 tokens", fontsize=5.0,
            color=MUTED, ha="center", va="center")
    ax.text(25.8, 17.6, "captured once per decision group", fontsize=4.7,
            color=MUTED, ha="center", va="center", style="italic")

    # ---- product 2: the last token h ----
    arrow(ax, last_x, 32.8, 41.5, 25.0, color=BLUE_EDGE,
          connectionstyle="arc3,rad=0.25")
    vgrad(ax, 39.5, 14, 15.5, 11, "#e6f0fc", "#cde2fb", ec=BLUE_EDGE, lw=0.9,
          rounding=1.2)
    ax.text(47.25, 21.6, "$h = H[-1]\\in\\mathbb{R}^{4096}$", fontsize=5.5,
            color=INK, ha="center", va="center")
    ax.text(47.25, 17.2, "free byproduct · sha-verified, bit-exact",
            fontsize=4.4, color=MUTED, ha="center", va="center")

    # ================= C: candidates on the road =================
    ax.text(48.2, 96.5, "$K{=}8$ sampled trajectories", fontsize=5.7,
            color=INK, ha="center", va="center")
    ax.text(48.2, 92.3, "(same forward pass)", fontsize=4.7, color=MUTED,
            ha="center", va="center", style="italic")
    road(ax, 39.5, 40, 17, 48, n=8)
    arrow(ax, 36.4, 74, 39.1, 72)

    # ================= D: SafeWorld head (stacked x8) =================
    for off in (2, 1):
        rbox(ax, 60 + off * 1.1, 14 - off * 2.2, 18, 79, fc="#f3f2ef",
             ec=FAINT, lw=0.6, rounding=2.0)
    rbox(ax, 60, 14, 18, 79, fc="#fbfaf8", ec=NEUTRAL_EDGE, lw=0.9,
         rounding=2.0, shadow=True)
    ax.text(80.0, 96.0, "$\\times 8$", fontsize=6.6, color=MUTED,
            ha="center", va="center", style="italic")
    ax.text(69, 88.5, "SafeWorld head", fontsize=6.2, color=INK, ha="center",
            va="center", fontweight="bold")
    ax.text(69, 84.2, "shared weights · one pass per candidate", fontsize=4.7,
            color=MUTED, ha="center", va="center", style="italic")
    rbox(ax, 62, 70, 14, 9, fc="#f0efec", rounding=1.1)
    ax.text(69, 74.5, "trajectory $c_k$ ($64\\times2$)", fontsize=5.2,
            color=INK, ha="center", va="center", zorder=5)
    vgrad(ax, 62, 55.5, 14, 9, "#fbeae3", "#f5d2c2", ec=WARM_EDGE, lw=0.8,
          rounding=1.1)
    ax.text(69, 60, "candidate encoder", fontsize=5.2, color=INK,
            ha="center", va="center", zorder=5)
    vgrad(ax, 62, 38.5, 14, 12, "#fbeae3", "#f5d2c2", ec=WARM_EDGE, lw=0.8,
          rounding=1.1)
    ax.text(69, 44.5, "cross-attention ($\\times1$)\nK,V $\\leftarrow h$",
            fontsize=5.2, color=INK, ha="center", va="center", zorder=5,
            linespacing=1.25)
    vgrad(ax, 62, 23, 14, 9, "#fbeae3", "#f5d2c2", ec=WARM_EDGE, lw=0.8,
          rounding=1.1)
    ax.text(69, 27.5, "shared trunk", fontsize=5.2, color=INK, ha="center",
            va="center", zorder=5)
    arrow(ax, 56.9, 60, 61.6, 74, connectionstyle="arc3,rad=0.2")
    arrow(ax, 69, 69.6, 69, 65.0)
    arrow(ax, 69, 55.1, 69, 50.9)
    ax.text(70.4, 52.8, "Q", fontsize=4.9, color=MUTED)
    arrow(ax, 55.4, 20, 61.6, 41.5, color=BLUE_EDGE,
          connectionstyle="arc3,rad=0.2")
    arrow(ax, 69, 38.1, 69, 32.4)
    ax.text(69, 17.6, "no cross-candidate attention", fontsize=4.7,
            color=MUTED, ha="center", va="center", style="italic")

    # ================= E: outputs & selection =================
    ax.text(90.2, 96.5,
            "score:  $\\hat y(c_k) = \\hat m_g + S\\cdot\\tanh(\\hat r(c_k)/S)$",
            fontsize=5.5, color=INK, ha="center", va="center")
    rbox(ax, 81, 78, 18.5, 8, fc="white", ec=NEUTRAL_EDGE, ls="--",
         rounding=1.0)
    ax.text(90.25, 82, "aux heads (train-only): future · progress ·\n"
            "collision · offroad — deleted at inference", fontsize=4.3,
            color=MUTED, ha="center", va="center", zorder=5, linespacing=1.3)
    rbox(ax, 81, 62, 8.2, 9, fc="#f0efec", rounding=1.0)
    ax.text(85.1, 66.5, "group mean\n$\\hat m_g$", fontsize=4.8, color=INK,
            ha="center", va="center", zorder=5, linespacing=1.2)
    ax.text(85.1, 59.2, "not used for selection", fontsize=4.2, color=MUTED,
            ha="center", va="center", style="italic")
    vgrad(ax, 91.2, 62, 8.3, 9, "#e6f0fc", "#cde2fb", ec=BLUE_EDGE, lw=0.9,
          rounding=1.0)
    ax.text(95.35, 66.5, "residual\n$\\hat r(c_k)$", fontsize=4.8, color=INK,
            ha="center", va="center", zorder=5, linespacing=1.2)
    arrow(ax, 76.4, 27.5, 80.6, 65.5, connectionstyle="arc3,rad=-0.25")
    arrow(ax, 76.4, 26.0, 90.9, 63.0, connectionstyle="arc3,rad=-0.12")
    arrow(ax, 76.4, 29.0, 80.6, 81.5, color=FAINT,
          connectionstyle="arc3,rad=-0.35")

    bx = mini_bars(ax, 82, 36, 16, 16, SCORES, BEST)
    ax.text(82, 54.4, "per-candidate scores", fontsize=4.7, color=MUTED,
            ha="left", va="center")
    arrow(ax, 95.35, 61.6, 95.35, 53.4, color=BLUE_EDGE)
    ax.text(90, 31.4, "$c^{*} = \\arg\\max_k\\,\\hat r(c_k)$", fontsize=5.4,
            color=BLUE, ha="center", va="center", fontweight="bold")
    arrow(ax, bx, 35.3, bx, 33.6, color=BLUE_EDGE)
    road(ax, 83.5, 2, 13, 26, n=8, selected=BEST)
    ax.text(82.6, 15, "execute\n$c^{*}$", fontsize=5.0, color=BLUE,
            ha="right", va="center", fontweight="bold", linespacing=1.2)

    # footer (from the locked conformance record)
    ax.text(1, 4.5,
            f"{params:,} trainable params  ·  {lat:.1f} ms for all "
            f"$K{{=}}8$ candidates  ·  {mem_mb:.1f} MB peak inference memory",
            fontsize=5.0, color=MUTED, ha="left", va="center")

    style.save(fig, "fig_arch")


if __name__ == "__main__":
    main()
