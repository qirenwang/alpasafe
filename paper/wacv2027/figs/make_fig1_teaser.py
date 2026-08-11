#!/usr/bin/env python3
"""Fig.1 — teaser v2 (AV styling): two routes to online trajectory
evaluation, deliberately parallel lanes. Latency/params for our route come
from the locked FN0 conformance record; the explicit-imagination lane is
its paper (verify in the citation pass)."""
import style
from style import (BLUE, BLUE_EDGE, FAINT, INK, MUTED, WARM_EDGE, arrow,
                   canvas)
from fancy import rbox, road, snowflake, token_strip, vgrad

W_IN, H_IN = style.SINGLE_COL, 2.45
AR = W_IN / H_IN


def lane_flow(ax, y0, gen_text, gen_blue, cap_x=43.5):
    """scene road -> generator -> candidate fan road -> select; returns y-mid."""
    road(ax, 1, y0 - 1.5, 13, 17, n=3)
    ax.text(7.5, y0 - 4.6, "scene", fontsize=4.8, color=MUTED, ha="center",
            va="center")
    ym = y0 + 6.5
    arrow(ax, 14.6, ym, 17.0, ym)
    if gen_blue:
        vgrad(ax, 17.5, y0, 15.5, 13, "#eaf3fd", "#cde2fb", ec=BLUE_EDGE,
              lw=0.9, rounding=1.2)
    else:
        vgrad(ax, 17.5, y0, 15.5, 13, "#fbeae3", "#f5d2c2", ec=WARM_EDGE,
              lw=0.9, rounding=1.2)
    ax.text(25.25, ym, gen_text, fontsize=5.2, color=INK, ha="center",
            va="center", zorder=5, linespacing=1.25)
    arrow(ax, 33.5, ym, 36.0, ym)
    road(ax, 36.5, y0 - 1.5, 14, 17, n=8)
    ax.text(cap_x, y0 - 4.6, "$K$ candidates", fontsize=4.8, color=MUTED,
            ha="center", va="center")
    arrow(ax, 51.0, ym, 83.4, ym)
    rbox(ax, 84, y0, 15, 13, fc="white", rounding=1.2, shadow=True)
    ax.text(91.5, ym, "select &\nexecute", fontsize=5.4, color=INK,
            ha="center", va="center", fontweight="bold", zorder=5,
            linespacing=1.2)
    return ym


def main():
    style.setup()
    conf = style.load(style.FN0_CONFORMANCE)["measurements"]
    lat = conf["median_K8_forward_latency_ms"]
    params_m = conf["trainable_parameters"]["F_LASTTOKEN_L3_MATCHED_CONS"] / 1e6

    fig, ax = canvas(W_IN, H_IN)

    # ---------------- lane (a): explicit imagination ----------------
    ax.text(1, 97.5, "(a)  Explicit imagination (WoTE)", fontsize=7.0,
            color=INK, va="top", fontweight="bold")
    ya = 76
    ym_a = lane_flow(ax, ya, "planner\n(trained)", gen_blue=False, cap_x=46.5)
    # branch: world model imagines each candidate's future
    vgrad(ax, 24, 56, 20, 11, "#fbeae3", "#f5d2c2", ec=WARM_EDGE, lw=0.9,
          rounding=1.2)
    ax.text(34, 61.5, "BEV world model\n(trained)", fontsize=5.0, color=INK,
            ha="center", va="center", zorder=5, linespacing=1.25)
    for k, (dx, lab) in enumerate(((0, "$t{+}2$s"), (7.5, "$t{+}4$s"))):
        rbox(ax, 52 + dx, 55.5, 6.5, 12, fc="#eef1f5", rounding=0.8,
             zorder=3 + k)
        ax.text(55.25 + dx, 53.0, lab, fontsize=4.2, color=MUTED,
                ha="center", va="center")
        ax.plot([53 + dx, 57.5 + dx], [64.2, 64.2], color="#b8c4d4", lw=0.6,
                zorder=4 + k)
        ax.fill([53.6 + dx, 56.9 + dx, 56.1 + dx, 54.4 + dx],
                [56.5, 56.5, 63.9, 63.9], color="#d8dde5", lw=0,
                zorder=4 + k)
    ax.text(59.5, 70.6, "imagined futures", fontsize=4.6, color=MUTED,
            ha="center", va="center")
    arrow(ax, 41.0, ya - 2.0, 34, 67.4, connectionstyle="arc3,rad=0.3")
    arrow(ax, 44.4, 61.5, 51.4, 61.5)
    arrow(ax, 66.5, 61.5, 90, ya - 0.6, connectionstyle="arc3,rad=-0.3")
    ax.text(99, 49.0, "trains a world model · per-candidate"
            " inference-time rollouts",
            fontsize=4.8, color=WARM_EDGE, ha="right", va="center")

    # ---------------- lane (b): amortized readout ----------------
    ax.text(1, 44.0, "(b)  Amortized readout (ours)", fontsize=7.0,
            color=INK, va="top", fontweight="bold")
    yb = 23
    ym_b = lane_flow(ax, yb, "frozen driving\nVLA (10B)", gen_blue=True, cap_x=38.0)
    snowflake(ax, 32.2, yb + 12.2, r=1.25, ar=AR)
    # branch: the free last token -> tiny value head
    last_x = token_strip(ax, 20, 8.5, 13, 3.6, n=10)
    ax.text(26.5, 4.6, "$h$ = last prefill token (free)", fontsize=4.6,
            color=BLUE, ha="center", va="center")
    arrow(ax, 25.25, yb - 2.0, 24, 13.6, connectionstyle="arc3,rad=0.25",
          color=BLUE_EDGE)
    vgrad(ax, 44, 7.5, 20, 11, "#fbeae3", "#f5d2c2", ec=WARM_EDGE, lw=0.9,
          rounding=1.2)
    ax.text(54, 13, f"value head · {params_m:.1f}M\n(the only trained part)",
            fontsize=4.8, color=INK, ha="center", va="center", zorder=5,
            linespacing=1.25)
    arrow(ax, 34.6, 10.3, 43.4, 11.5, color=BLUE_EDGE)
    arrow(ax, 66.5, 13, 90, yb - 0.6, connectionstyle="arc3,rad=-0.3")
    ax.text(99, 1.6, f"zero VLA training · zero rollouts · {lat:.1f} ms measured",
            fontsize=4.8, color=BLUE, ha="right", va="center")

    style.save(fig, "fig1_teaser")


if __name__ == "__main__":
    main()
