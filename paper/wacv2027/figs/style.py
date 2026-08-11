"""Shared style for all paper figures (WACV two-column, Times text).

Color encoding (Fig.3/4): direction + CI support, a diverging pair with a
neutral midpoint — blue = CI-supported in favor of the first-named arm,
red = CI-supported against, gray = CI crosses zero (deliberately neutral;
"reads gray" is the point). Meaning never rides on hue alone: CI-supported
effects use filled markers, indeterminate ones open markers, so the figures
survive grayscale printing and CVD. Poles validated (CVD dE 9.1, contrast
>=3:1 on white).
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FIGS = Path(__file__).parent
ART = Path("/home/qiren/alpasafe/safeworld-alpamayo/artifacts")
FM1_ANALYSIS = (ART / "safeworld_t26g_fm1_full_l2_matched_selector_experiment"
                / "20260803T132326Z/results/t26g_fm1_analysis.json")
FN0_CONFORMANCE = (ART / "safeworld_t26g_fn0_l3_pathway_consolidation_preregistration"
                   / "20260803T210139Z/results/t26g_fn0_conformance_and_budget.json")
FM1_GPU_GATE = (ART / "safeworld_t26g_fm1_full_l2_matched_selector_experiment"
                / "20260803T132326Z/results/t26g_fm1_gpu_gate.json")

# palette (light surface / print)
BLUE = "#1c5cab"      # CI-supported, favors first-named arm
RED = "#d03b3b"       # CI-supported, against first-named arm
GRAY = "#898781"      # CI crosses zero
INK = "#0b0b0b"
MUTED = "#52514e"
FAINT = "#b8b6ae"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
BLUE_FILL = "#cde2fb"   # frozen / ours accent fill
BLUE_EDGE = "#1c5cab"
WARM_FILL = "#f8e3da"   # trained-component accent fill
WARM_EDGE = "#b2543a"
NEUTRAL_FILL = "#f0efec"
NEUTRAL_EDGE = "#898781"

SINGLE_COL = 3.28   # inches (WACV \linewidth)
DOUBLE_COL = 6.90   # inches (\textwidth)


def setup():
    plt.rcParams.update({
        "font.family": "Nimbus Roman",
        "font.size": 8.0,
        "axes.labelsize": 8.0,
        "axes.titlesize": 8.0,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.0,
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.0,
        "ytick.major.size": 2.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 300,
    })


def save(fig, name):
    """Write vector PDF (for the paper) + PNG preview (for eyeballing)."""
    pdf = FIGS / f"{name}.pdf"
    png = FIGS / f"{name}.png"
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.01)
    fig.savefig(png, bbox_inches="tight", pad_inches=0.01, dpi=220)
    print(f"wrote {pdf} and preview {png}")


def load(path):
    with open(path) as fh:
        return json.load(fh)


def box(ax, x, y, w, h, text, fc=NEUTRAL_FILL, ec=NEUTRAL_EDGE, fs=6.2,
        lw=0.8, tc=INK, rounding=1.4, ls="-", weight="normal"):
    """Rounded box centered text; coords in the ax data system."""
    from matplotlib.patches import FancyBboxPatch
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle=f"round,pad=0,rounding_size={rounding}",
                       fc=fc, ec=ec, lw=lw, ls=ls, zorder=3)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tc, zorder=4, linespacing=1.15,
            fontweight=weight)
    return p


def arrow(ax, x1, y1, x2, y2, color=MUTED, lw=0.9, style_="-|>", ms=5.5,
          connectionstyle=None, zorder=4.5):
    # zorder above box patches (3): arrows must never be occluded by fills
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), zorder=zorder,
                arrowprops=dict(arrowstyle=style_, color=color, lw=lw,
                                shrinkA=1.0, shrinkB=1.0,
                                mutation_scale=ms,
                                connectionstyle=connectionstyle))


def canvas(width, height):
    """Blank 0-100 x 0-100 axis with no decorations."""
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    return fig, ax


def classify(mean, lo, hi, better):
    """(color, filled) for an effect with 95% CI.

    better: 'lower' or 'higher' — the favorable direction for the
    first-named arm of the contrast.
    """
    excludes = (lo > 0) or (hi < 0)
    if not excludes:
        return GRAY, False
    favorable = (mean < 0) if better == "lower" else (mean > 0)
    return (BLUE if favorable else RED), True
