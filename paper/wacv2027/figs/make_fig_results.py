#!/usr/bin/env python3
"""Main-results figure — CV-style grouped bars with 95% CI whiskers
(replaces the earlier forest plot; same locked numbers, different geometry).

Reads the locked FM1 analysis JSON; after FN1 finalizes, rerun with
  python3 make_fig_results.py --fn1 <fn1_analysis.json>
to append the N1-N4 bars from the FN1 record.
"""
import argparse

import matplotlib.pyplot as plt

import style
from style import GRID, INK, MUTED, classify

M_ROWS = [("M3", "M3: LT$-$Null"), ("M2", "M2: Full$-$Null"),
          ("M1", "M1: Full$-$LT"), ("M5", "M5: LT$-$Geo"),
          ("M4", "M4: Full$-$Geo")]
N_ROWS = [("N1_CKPT", "N1: Ckpt$-$Geo"), ("N1_SEL", "N1$'$: Sel$-$Geo"),
          ("N2_CKPT", "N2: Ckpt$-$Null"), ("N3", "N3: Sel$-$Ckpt")]


def draw(ax, rows, better, ylim, ylabel, yticks):
    xs = range(len(rows))
    ax.axhline(0, color=INK, lw=0.7, zorder=2)
    for x, (stat,) in zip(xs, [(r[2],) for r in rows]):
        m, lo, hi = stat["mean"], stat["ci95_low"], stat["ci95_high"]
        color, filled = classify(m, lo, hi, better)
        ax.bar(x, m, width=0.62, color=color, alpha=1.0 if filled else 0.45,
               edgecolor="none", zorder=3)
        ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], fmt="none",
                    ecolor="#3a3a3a", elinewidth=0.7, capsize=1.7,
                    capthick=0.7, zorder=4)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([r[1] for r in rows], fontsize=5.2,
                       rotation=30, ha="right", rotation_mode="anchor")
    ax.set_xlim(-0.65, len(rows) - 0.35)
    ax.set_ylim(*ylim)
    ax.set_yticks(yticks)
    ax.set_ylabel(ylabel, fontsize=7.0, labelpad=2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(style.BASELINE)
    ax.spines["bottom"].set_color(style.BASELINE)
    ax.grid(axis="y", color=GRID, lw=0.5, zorder=0)
    ax.tick_params(axis="y", colors=MUTED, labelsize=6.0, pad=1.5)
    ax.tick_params(axis="x", length=0, pad=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fn1", default=None)
    args = ap.parse_args()

    style.setup()
    fm1 = style.load(style.FM1_ANALYSIS)
    fn1 = style.load(args.fn1) if args.fn1 else None

    def rows_for(ep):
        rows = [(cid, lab, fm1["comparisons"][cid][ep]) for cid, lab in M_ROWS]
        if fn1:
            rows += [(cid, lab, fn1["comparisons"][cid][ep])
                     for cid, lab in N_ROWS]
        return rows

    fig = plt.figure(figsize=(style.SINGLE_COL, 2.05))
    gs = fig.add_gridspec(1, 2, left=0.135, right=0.995, top=0.885,
                          bottom=0.265, wspace=0.42)
    ax_r = fig.add_subplot(gs[0, 0])
    ax_t = fig.add_subplot(gs[0, 1])
    draw(ax_r, rows_for("K8_regret"), "lower", (-0.016, 0.016),
         r"$\Delta$ regret ($\downarrow$ better)", [-0.01, 0, 0.01])
    draw(ax_t, rows_for("K8_top1"), "higher", (-0.145, 0.145),
         r"$\Delta$ top-1 ($\uparrow$ better)", [-0.1, 0, 0.1])

    fig.text(0.135, 0.955, "matched-visibility study · 173 untouched scenes"
             + ("" if fn1 else "      |      N1–N4 (powered, $N{=}466$): in flight"),
             fontsize=6.2, color=MUTED, ha="left", va="center")
    fig.text(0.135, 0.035,
             "bars = paired scene deltas, whiskers = 95% bootstrap CI;  "
             "saturated blue/red = CI excludes 0 (favors / against first-named arm), pale = crosses 0",
             fontsize=5.0, color=MUTED, ha="left", va="center")

    style.save(fig, "fig_results")


if __name__ == "__main__":
    main()
