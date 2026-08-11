"""AV-paper visual vocabulary for the schematic figures: gradients, soft
shadows, perspective road with trajectory fans, camera stacks, transformer
slabs, token strips. Built on style.py's palette."""
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch

import style
from style import BLUE, BLUE_EDGE, INK, MUTED, NEUTRAL_EDGE


def rbox(ax, x, y, w, h, fc="white", ec=NEUTRAL_EDGE, lw=0.8, rounding=1.4,
         ls="-", zorder=3, shadow=False):
    if shadow:
        ax.add_patch(FancyBboxPatch(
            (x + 0.45, y - 0.7), w, h,
            boxstyle=f"round,pad=0,rounding_size={rounding}",
            fc="#00000012", ec="none", zorder=zorder - 0.2))
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle=f"round,pad=0,rounding_size={rounding}",
                       fc=fc, ec=ec, lw=lw, ls=ls, zorder=zorder)
    ax.add_patch(p)
    return p


def vgrad(ax, x, y, w, h, c0, c1, ec=NEUTRAL_EDGE, lw=0.8, rounding=1.4,
          zorder=3, shadow=False, horizontal=False):
    """Rounded box with a c0->c1 gradient fill."""
    if shadow:
        ax.add_patch(FancyBboxPatch(
            (x + 0.45, y - 0.7), w, h,
            boxstyle=f"round,pad=0,rounding_size={rounding}",
            fc="#00000012", ec="none", zorder=zorder - 0.2))
    clip = FancyBboxPatch((x, y), w, h,
                          boxstyle=f"round,pad=0,rounding_size={rounding}",
                          fc="none", ec="none", zorder=zorder)
    ax.add_patch(clip)
    cmap = LinearSegmentedColormap.from_list("g", [c0, c1])
    g = np.linspace(0, 1, 128).reshape(1, -1) if horizontal \
        else np.linspace(1, 0, 128).reshape(-1, 1)
    im = ax.imshow(g, extent=(x, x + w, y, y + h), cmap=cmap,
                   origin="lower", aspect="auto", zorder=zorder,
                   interpolation="bilinear")
    im.set_clip_path(clip)
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle=f"round,pad=0,rounding_size={rounding}",
                                fc="none", ec=ec, lw=lw, zorder=zorder + 0.1))


def road(ax, x, y, w, h, n=8, selected=None, zorder=3, edge=True):
    """Perspective road with an ego car and a fan of n trajectories.
    selected=None -> uniform candidate shades; else that index is bold blue."""
    bl, br = x + 0.06 * w, x + 0.94 * w
    tl, tr = x + 0.30 * w, x + 0.70 * w
    yt = y + 0.94 * h
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2",
                                fc="#f2f4f7", ec=NEUTRAL_EDGE if edge else "none",
                                lw=0.8, zorder=zorder - 0.1))
    ax.fill([bl, br, tr, tl], [y + 0.02 * h, y + 0.02 * h, yt, yt],
            color="#e3e2dd", zorder=zorder, lw=0)
    for (a, b) in (((bl, tl)), ((br, tr))):
        ax.plot([a, b], [y + 0.02 * h, yt], color="#c6c4bd", lw=0.8,
                zorder=zorder + 0.1)
    # dashed centre line
    for t0 in np.arange(0.06, 0.9, 0.16):
        ax.plot([x + 0.5 * w, x + 0.5 * w],
                [y + t0 * h, y + (t0 + 0.07) * h],
                color="white", lw=1.0, zorder=zorder + 0.1)
    # ego car
    cx = x + 0.5 * w
    ego_w, ego_h = 0.13 * w, 0.10 * h
    ax.add_patch(FancyBboxPatch((cx - ego_w / 2, y + 0.05 * h), ego_w, ego_h,
                                boxstyle="round,pad=0,rounding_size=0.5",
                                fc="#343a42", ec="none", zorder=zorder + 0.3))
    # trajectory fan
    t = np.linspace(0, 1, 40)
    persp = 1 - 0.42 * t
    y0 = y + 0.05 * h + ego_h
    shades = ["#b9c6d8", "#a8bcd6", "#97b3d5", "#86a9d3", "#86a9d3",
              "#97b3d5", "#a8bcd6", "#b9c6d8"]
    for i, lat in enumerate(np.linspace(-1, 1, n)):
        xi = cx + lat * 0.26 * w * (t ** 1.25) * persp * 2.2
        yi = y0 + t * (yt - y0 - 0.03 * h)
        if selected is not None and i == selected:
            ax.plot(xi, yi, color="white", lw=2.6, zorder=zorder + 0.4)
            ax.plot(xi, yi, color=BLUE, lw=1.6, zorder=zorder + 0.5)
        else:
            col = shades[i % len(shades)] if selected is None else "#c3c9d2"
            ax.plot(xi, yi, color=col, lw=0.9, zorder=zorder + 0.2,
                    solid_capstyle="round")


def cam_stack(ax, x, y, w, h, zorder=3):
    """Three offset camera-frame thumbnails."""
    fw, fh = 0.80 * w, 0.72 * h
    for k, (dx, dy) in enumerate(((0.20 * w, 0.28 * h),
                                  (0.10 * w, 0.14 * h), (0, 0))):
        fx, fy = x + dx, y + dy
        rbox(ax, fx, fy, fw, fh, fc="#eef1f5", ec=NEUTRAL_EDGE, lw=0.7,
             rounding=0.8, zorder=zorder + k)
        if k == 2:
            ax.plot([fx + 0.07 * fw, fx + 0.93 * fw],
                    [fy + 0.62 * fh, fy + 0.62 * fh],
                    color="#b8c4d4", lw=0.8, zorder=zorder + k + 0.2)
            ax.fill([fx + 0.5 * fw - 0.26 * fw, fx + 0.5 * fw + 0.26 * fw,
                     fx + 0.5 * fw + 0.08 * fw, fx + 0.5 * fw - 0.08 * fw],
                    [fy + 0.06 * fh, fy + 0.06 * fh,
                     fy + 0.60 * fh, fy + 0.60 * fh],
                    color="#d8dde5", zorder=zorder + k + 0.1, lw=0)


def transformer_stack(ax, x, y, w, h, n=4, zorder=3):
    slab_h = h / (n + 0.6)
    for i in range(n):
        vgrad(ax, x, y + i * (slab_h * 1.18), w, slab_h,
              "#dce9fb", "#9ec5f4", ec="#5598e7", lw=0.6, rounding=0.9,
              zorder=zorder)


def token_strip(ax, x, y, w, h, n=20, zorder=3):
    """Prefill token cells with a gradient; the LAST one is the star."""
    cmap = LinearSegmentedColormap.from_list("t", ["#e2ebf8", "#9ec5f4"])
    cw = w / (n + 1.8)
    for i in range(n):
        ax.add_patch(FancyBboxPatch(
            (x + i * cw * 1.06, y), cw, h,
            boxstyle="round,pad=0,rounding_size=0.25",
            fc=cmap(i / (n - 1)), ec="#7fa8d9", lw=0.3, zorder=zorder))
    lx = x + n * cw * 1.06 + 0.3
    # glow ring + saturated last token
    ax.add_patch(FancyBboxPatch((lx - 0.45, y - 0.9), cw * 1.8 + 0.9, h + 1.8,
                                boxstyle="round,pad=0,rounding_size=0.6",
                                fc="#cde2fb", ec="none", alpha=0.8,
                                zorder=zorder))
    ax.add_patch(FancyBboxPatch((lx, y - 0.45), cw * 1.8, h + 0.9,
                                boxstyle="round,pad=0,rounding_size=0.45",
                                fc=BLUE, ec="white", lw=0.7, zorder=zorder + 0.1))
    return lx + cw * 0.9  # centre x of the highlighted token


def snowflake(ax, x, y, r=1.6, ar=2.12, zorder=6):
    """Frozen badge; ar = figure width/height ratio so the disc looks round."""
    from matplotlib.patches import Ellipse
    ax.add_patch(Ellipse((x, y), 2 * r, 2 * r * ar, fc="white", ec=BLUE_EDGE,
                         lw=0.8, zorder=zorder))
    ax.text(x, y - 0.15, "❄", fontsize=r * 4.2, color=BLUE,
            family="DejaVu Sans", ha="center", va="center", zorder=zorder + 1)


def mini_bars(ax, x, y, w, h, scores, best, zorder=3):
    """Per-candidate score bars, argmax highlighted."""
    n = len(scores)
    bw = w / (n * 1.45)
    ax.plot([x, x + w], [y, y], color=NEUTRAL_EDGE, lw=0.7, zorder=zorder)
    for i, s in enumerate(scores):
        bx = x + i * (w / n) + (w / n - bw) / 2
        col = BLUE if i == best else "#c3c9d2"
        ax.add_patch(FancyBboxPatch((bx, y), bw, s * h,
                                    boxstyle="round,pad=0,rounding_size=0.3",
                                    fc=col, ec="none", zorder=zorder + 0.1))
    bx = x + best * (w / n) + (w / n) / 2
    ax.plot([bx], [y + scores[best] * h + 1.6], marker="v", ms=3.2,
            color=BLUE, zorder=zorder + 0.3)
    return bx
