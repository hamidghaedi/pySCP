"""Drawing for the ``network``, ``enrichmap``, ``wordcloud`` and ``comparison``
enrichment plots.  Each function draws into one ``Axes`` and registers its
legend handles with a :class:`~scp.layout.LegendColumn`; none creates a figure.

ggplot sizes are millimetres; they are converted with ``.pt = 72.27 / 25.4``
(text and point sizes) and ``.pt * 0.75`` (line widths, because R's ``lwd = 1``
is 1/96 inch).

Ported from SCP (Hao Zhang, https://github.com/zhanghao-njmu/SCP, GPL-3),
``R/SCP-plot.R`` 12838-13392 and 13680-14530.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import cast

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.legend import Legend
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox, TextArea, VPacker
from matplotlib.patches import Circle, Patch, Polygon
from matplotlib.text import Text
from matplotlib.textpath import TextToPath
from matplotlib.ticker import MaxNLocator
from matplotlib.transforms import offset_copy

from ..layout import LegendColumn
from ..theme import halo
from ._enrich_graph import EnrichmapTables, NetworkTables

PT = 72.27 / 25.4          # ggplot2's .pt: points per millimetre
DOT = 0.75 * PT           # ggplot point size (mm) -> drawn circle diameter in points
LWD = PT * 72 / 96         # ggplot linewidth (mm) -> matplotlib points


# --------------------------------------------------------------------------- #
# legend key: a numbered circle (SCP's draw_key_cust)
# --------------------------------------------------------------------------- #
class NumberedKey:
    """Legend handle for a network term: its colour and its number."""

    def __init__(self, number: int, color: str, label: str) -> None:
        self.number, self.color, self._label = number, color, label

    def get_label(self) -> str:
        return self._label


class _NumberedKeyHandler(HandlerBase):
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        w, h = handlebox.width, handlebox.height
        tr = handlebox.get_transform()
        r = 0.75 * h
        circ = Circle((x0 + w / 2, y0 + h / 2), r, facecolor=orig_handle.color,
                      edgecolor="white", linewidth=1.0, transform=tr)
        handlebox.add_artist(circ)
        txt = Text(x0 + w / 2, y0 + h / 2, str(orig_handle.number), ha="center",
                   va="center", color="white", fontweight="bold",
                   fontsize=fontsize * 0.85, transform=tr)
        halo(txt, "black", 0.1)
        handlebox.add_artist(txt)
        return circ


Legend.update_default_handler_map({NumberedKey: _NumberedKeyHandler()})


def _size_breaks(lo: float, hi: float, n: int = 4) -> list[float]:
    """``scales::breaks_extended(n)``-like breaks inside ``[lo, hi]``."""
    if not np.isfinite([lo, hi]).all():
        return []
    if hi <= lo:
        return [lo]
    ticks = MaxNLocator(nbins=n, steps=[1, 2, 2.5, 5, 10]).tick_values(lo, hi)
    return [float(t) for t in ticks if lo - 1e-9 <= t <= hi + 1e-9]


def _area_size(v, lo: float, hi: float, rng: tuple[float, float]) -> np.ndarray:
    """``scale_size(range = rng)``: area-proportional, so diameter ~ sqrt."""
    v = np.asarray(v, dtype=float)
    x = np.full(v.shape, 0.5) if hi <= lo else (v - lo) / (hi - lo)
    return rng[0] + (rng[1] - rng[0]) * np.sqrt(np.clip(x, 0, 1))


def _fmt(v: float) -> str:
    return f"{v:g}"


# --------------------------------------------------------------------------- #
# network
# --------------------------------------------------------------------------- #
def draw_network(
    ax: Axes,
    nt: NetworkTables,
    layout: np.ndarray,
    *,
    labelsize: float = 5.0,
    legend: LegendColumn | None = None,
    legend_title: str = "Term",
) -> None:
    """Edges coloured by term, gene labels on blended fills, numbered term
    nodes, and a "Term" legend of numbered circles (``SCP-plot.R:13128-13158``)."""
    nodes, edges = nt.nodes, nt.edges
    xy = np.asarray(layout, dtype=float)
    segs = [[xy[s], xy[t]] for s, t in zip(edges["source"], edges["target"], strict=True)]
    ax.add_collection(LineCollection(segs, colors=list(edges["color"]), linewidths=0.5 * LWD,
                                     capstyle="round", zorder=1))
    is_gene = (nodes["class"] == "gene").to_numpy()
    for i in np.flatnonzero(is_gene):
        ax.text(xy[i, 0], xy[i, 1], str(nodes["name"].iloc[i]), ha="center", va="center",
                fontsize=3 * PT, color=nodes["label_color"].iloc[i], zorder=2,
                bbox=dict(boxstyle="round,pad=0.25,rounding_size=0.15",
                          facecolor=nodes["color"].iloc[i],
                          edgecolor=nodes["label_color"].iloc[i], linewidth=0.25 * LWD))
    term = np.flatnonzero(~is_gene)
    ax.scatter(xy[term, 0], xy[term, 1], s=(8 * DOT) ** 2, c="black", edgecolors="black",
               linewidths=1, zorder=3)
    ax.scatter(xy[term, 0], xy[term, 1], s=(7 * DOT) ** 2,
               c=list(nodes["color"].iloc[term]), edgecolors="white", linewidths=1, zorder=4)
    for i in term:
        t = ax.text(xy[i, 0], xy[i, 1], str(int(nodes["label"].iloc[i])), ha="center",
                    va="center", fontsize=labelsize * PT * 0.6, fontweight="bold",
                    color="white", zorder=5)
        halo(t, "black", 0.1)
    if len(xy):
        ax.set_xlim(*_padded(xy[:, 0], 0.08))
        ax.set_ylim(*_padded(xy[:, 1], 0.08))
    ax.set_xlabel("")
    ax.set_ylabel("")
    if legend is not None:
        legend.add(legend_title, [NumberedKey(i + 1, nt.term_colors[t], t)
                                  for i, t in enumerate(nt.term_levels)])


def _padded(v: np.ndarray, mult: float) -> tuple[float, float]:
    lo, hi = float(np.min(v)), float(np.max(v))
    span = hi - lo if hi > lo else 1.0
    return lo - mult * span, hi + mult * span


# --------------------------------------------------------------------------- #
# enrichmap
# --------------------------------------------------------------------------- #
def _mvee(points: np.ndarray, tol: float = 1e-7) -> tuple[np.ndarray, np.ndarray]:
    """Khachiyan's minimum-volume enclosing ellipse -- what ggforce's
    ``geom_mark_ellipse`` fits.  Returns (centre, A) with
    ``(x - c)^T A (x - c) <= 1``."""
    p = np.asarray(points, dtype=float)
    n, d = p.shape
    q = np.vstack([p.T, np.ones(n)])
    u = np.full(n, 1.0 / n)
    for _ in range(1000):
        x = q @ np.diag(u) @ q.T
        m = np.einsum("ij,ji->i", q.T, np.linalg.solve(x, q))
        j = int(np.argmax(m))
        step = (m[j] - d - 1.0) / ((d + 1) * (m[j] - 1.0))
        new = (1 - step) * u
        new[j] += step
        if np.linalg.norm(new - u) < tol:
            u = new
            break
        u = new
    c = p.T @ u
    a = np.linalg.inv(p.T @ np.diag(u) @ p - np.outer(c, c)) / d
    return c, a


def _ellipse_polygon(pts_mm: np.ndarray, expand: float, n: int = 120) -> np.ndarray:
    """Expanded enclosing ellipse of points given in millimetres."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    unit = np.column_stack([np.cos(t), np.sin(t)])
    if len(pts_mm) == 1:
        return pts_mm[0] + expand * unit
    centred = pts_mm - pts_mm.mean(axis=0)
    if len(pts_mm) == 2 or np.linalg.matrix_rank(centred, tol=1e-9) < 2:
        # degenerate: an ellipse along the points, minor axis zero before expansion
        c = pts_mm.mean(axis=0)
        u, s, vt = np.linalg.svd(centred, full_matrices=False)
        major = vt[0]
        half = float(np.max(np.abs(centred @ major)))
        minor = np.array([-major[1], major[0]])
        return c + np.outer(unit[:, 0] * (half + expand), major) + np.outer(unit[:, 1] * expand,
                                                                             minor)
    c, a = _mvee(pts_mm)
    w, v = np.linalg.eigh(a)
    radii = 1.0 / np.sqrt(w) + expand
    return c + (unit * radii) @ v.T


def _hull_polygon(pts_mm: np.ndarray, expand: float, n: int = 24) -> np.ndarray:
    """Convex hull grown by ``expand`` (the Minkowski sum with a disc).

    ggforce's ``geom_mark_hull`` uses a *concave* hull (``concaveman``); there
    is no maintained Python port, so the convex hull is drawn instead."""
    from scipy.spatial import ConvexHull

    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    disc = expand * np.column_stack([np.cos(t), np.sin(t)])
    cloud = (pts_mm[:, None, :] + disc[None, :, :]).reshape(-1, 2)
    hull = ConvexHull(cloud)
    return cloud[hull.vertices]


def draw_enrichmap(
    ax: Axes,
    et: EnrichmapTables,
    layout: np.ndarray,
    colors: dict[str, str],
    *,
    mark: str = "ellipse",
    labelsize: float = 5.0,
    expand: tuple[float, float] = (0.5, 0.5),
    panel_mm: tuple[float, float] = (80.0, 80.0),
    legend: LegendColumn | None = None,
    legend_suffix: str = "",
) -> None:
    """Cluster marks with labels, overlap edges scaled by weight, and term nodes
    sized by gene count (``SCP-plot.R:13252-13283``)."""
    nodes, edges = et.nodes, et.edges
    xy = np.asarray(layout, dtype=float)
    if len(xy) == 0:
        return
    xlim = _padded(xy[:, 0], expand[0]) if np.ptp(xy[:, 0]) > 0 else (xy[0, 0] - 1, xy[0, 0] + 1)
    ylim = _padded(xy[:, 1], expand[1]) if np.ptp(xy[:, 1]) > 0 else (xy[0, 1] - 1, xy[0, 1] + 1)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    # millimetres per data unit, for the 3 mm mark expansion and label offsets
    sx = panel_mm[0] / (xlim[1] - xlim[0])
    sy = panel_mm[1] / (ylim[1] - ylim[0])
    to_mm = np.array([sx, sy])
    marks = []
    for c in et.cluster_levels:
        idx = np.flatnonzero((nodes["clusters"] == c).to_numpy())
        pts = xy[idx] * to_mm
        poly = (_hull_polygon(pts, 3.0) if mark == "hull" else _ellipse_polygon(pts, 3.0)) / to_mm
        col = colors[c]
        ax.add_patch(Polygon(poly, closed=True, facecolor=col, alpha=0.1,
                             edgecolor="none", zorder=0))
        ax.add_patch(Polygon(poly, closed=True, facecolor="none", edgecolor=col,
                             linewidth=0.5 * LWD, zorder=0.5))
        marks.append((c, xy[idx].mean(axis=0), poly, col))
    _label_marks(ax, marks, et.mark_labels, labelsize, mid_x=float(np.mean(xlim)))

    w = edges["weight"].to_numpy(dtype=float)
    if len(w):
        lw = 0.3 + (2.7 * (w - w.min()) / (w.max() - w.min()) if w.max() > w.min()
                    else np.full(len(w), 1.65 - 0.3))
        segs = [[xy[s], xy[t]] for s, t in zip(edges["source"], edges["target"], strict=True)]
        ax.add_collection(LineCollection(segs, colors="black", alpha=0.1, linewidths=lw * LWD,
                                         capstyle="round", zorder=1))
    cnt = nodes["Count"].to_numpy(dtype=float)
    lo, hi = float(cnt.min()), float(cnt.max())
    diam = _area_size(cnt, lo, hi, (2.0, 6.0))
    ax.scatter(xy[:, 0], xy[:, 1], s=(diam * DOT) ** 2, c=[colors[c] for c in nodes["clusters"]],
               edgecolors="black", linewidths=0.5, zorder=3)
    ax.set_xlabel("")
    ax.set_ylabel("")

    if legend is not None:
        legend.add(f"Count{legend_suffix}", [
            Line2D([], [], marker="o", linestyle="none", markerfacecolor="#4D4D4D",
                   markeredgecolor="black", markersize=float(_area_size([b], lo, hi, (2, 6))[0]) * DOT,
                   label=_fmt(b)) for b in _size_breaks(lo, hi)])
        if len(w):
            wl, wh = float(w.min()), float(w.max())
            legend.add(f"Intersection{legend_suffix}", [
                Line2D([], [], color="grey",
                       linewidth=(0.3 + (2.7 * (b - wl) / (wh - wl) if wh > wl else 1.35)) * LWD,
                       label=_fmt(b)) for b in _size_breaks(wl, wh)])
        legend.add(f"{et.legend_title}{legend_suffix}", [
            Patch(facecolor=colors[c], edgecolor="black", label=et.legend_labels[c])
            for c in et.cluster_levels])



def _label_marks(ax: Axes, marks, descriptions: dict[str, str], labelsize: float,
                 mid_x: float) -> None:
    """Cluster labels in two columns down the panel edges, each tied to its
    mark by a line.

    ggforce's ``geom_mark_*`` searches the panel for free space to put each
    label; this is the deterministic stand-in: labels are stacked from the top
    in order of cluster height, against the left edge for clusters left of
    centre and the right edge otherwise, so the boxes never overlap.  The boxes are kept out of
    the constrained layout so long descriptions do not shrink the panel.
    """
    renderer = ax.figure.canvas.get_renderer()  # type: ignore[attr-defined]
    ax_h = max(ax.get_window_extent(renderer).height, 1.0)
    top = 0.98
    for c, cen, poly, col in sorted(marks, key=lambda m: -m[1][1]):
        left = bool(cen[0] < mid_x)
        anchor = poly[int(np.argmin(poly[:, 0]) if left else np.argmax(poly[:, 0]))]
        desc = descriptions.get(c) or ""
        box = VPacker(children=[
            TextArea(c, textprops=dict(fontsize=labelsize * 2 * 0.9, fontweight="bold")),
            *([TextArea(desc, textprops=dict(fontsize=labelsize * 2 * 0.7))] if desc else []),
        ], align="left", pad=0, sep=1)
        ab = AnnotationBbox(box, tuple(anchor), xycoords="data",
                            xybox=(0.01 if left else 0.99, top), boxcoords="axes fraction",
                            box_alignment=(0.0 if left else 1.0, 1.0),
                            frameon=True, pad=0.25,
                            bboxprops=dict(facecolor="#F2F2F2", edgecolor="none", alpha=0.9),
                            arrowprops=dict(arrowstyle="-", color=col, linewidth=0.5 * LWD,
                                            shrinkA=0, shrinkB=0),
                            annotation_clip=False, zorder=6)
        ab.set_in_layout(False)
        ax.add_artist(ab)
        # one shared stack, alternating sides as the heights dictate, so two
        # wide boxes can never sit side by side and collide
        h = box.get_window_extent(renderer).height + 4 * 0.25 * labelsize * 2
        top -= h / ax_h + 0.015

# --------------------------------------------------------------------------- #
# wordcloud
# --------------------------------------------------------------------------- #
def _square_spiral(step: float, max_r: float):
    """Candidate offsets on concentric squares, centre outwards (ggwordcloud's
    ``shape = "square"``)."""
    yield np.zeros((1, 2))
    k = step
    while k <= max_r:
        n = max(int(8 * k / step), 8)
        t = np.linspace(0, 8, n, endpoint=False)
        side, f = np.divmod(t, 2)
        f = f - 1  # -1..1 along the side
        x = np.select([side == 0, side == 1, side == 2, side == 3], [np.ones_like(f), -f, -np.ones_like(f), f])
        y = np.select([side == 0, side == 1, side == 2, side == 3], [f, np.ones_like(f), -f, -np.ones_like(f)])
        yield np.column_stack([x, y]) * k
        k += step


def draw_wordcloud(
    ax: Axes,
    words: pd.DataFrame,
    colors: Sequence[str],
    *,
    word_size: tuple[float, float] = (2.0, 8.0),
    margin_px: float = 1.5,
) -> list[str]:
    """Pack the words into the panel, largest first is *not* assumed: like
    ggwordcloud the words are placed in table order (highest score first), each
    on a square spiral from the centre at the first position where its box
    collides with nothing and stays inside the panel.  Words that cannot be
    placed are dropped (``rm_outside = TRUE``).  Returns the words placed."""
    from matplotlib.colors import LinearSegmentedColormap, Normalize

    ax.set_xticks([])
    ax.set_yticks([])
    if words.empty:
        return []
    fig = ax.figure
    renderer = fig.canvas.get_renderer()  # type: ignore[attr-defined]
    box = ax.get_window_extent(renderer)
    W, H = box.width, box.height
    cnt = words["count"].to_numpy(dtype=float)
    sizes = _area_size(cnt, float(cnt.min()), float(cnt.max()), word_size)
    score = words["score"].to_numpy(dtype=float)
    cmap = LinearSegmentedColormap.from_list("wc", list(colors), N=256)
    norm = Normalize(float(np.nanmin(score)), float(np.nanmax(score)))
    placed: list[np.ndarray] = []
    kept: list[str] = []
    for word, size_mm, sc, ang in zip(words["keyword"], sizes, score, words["angle"], strict=True):
        t = ax.text(0.5, 0.5, str(word), transform=ax.transAxes, ha="center", va="center",
                    fontsize=float(size_mm) * PT, rotation=float(ang), color=cmap(norm(sc)))
        bb = t.get_window_extent(renderer)
        w, h = bb.width + 2 * margin_px, bb.height + 2 * margin_px
        ok = None
        rects = np.array(placed) if placed else np.zeros((0, 4))
        for cand in _square_spiral(2.0, max(W, H) / 2):
            # candidate centres in panel pixels, origin at the centre
            x0 = cand[:, 0] - w / 2
            y0 = cand[:, 1] - h / 2
            inside = (x0 >= -W / 2) & (x0 + w <= W / 2) & (y0 >= -H / 2) & (y0 + h <= H / 2)
            if not inside.any():
                continue
            if len(rects):
                hit = ((x0[:, None] < rects[None, :, 2]) & (x0[:, None] + w > rects[None, :, 0])
                       & (y0[:, None] < rects[None, :, 3]) & (y0[:, None] + h > rects[None, :, 1]))
                free = inside & ~hit.any(axis=1)
            else:
                free = inside
            if free.any():
                j = int(np.flatnonzero(free)[0])
                ok = (x0[j], y0[j])
                break
        if ok is None:
            t.remove()
            continue
        placed.append(np.array([ok[0], ok[1], ok[0] + w, ok[1] + h]))
        t.set_position((0.5 + (ok[0] + w / 2) / W, 0.5 + (ok[1] + h / 2) / H))
        kept.append(str(word))
    return kept


# --------------------------------------------------------------------------- #
# comparison dot plot
# --------------------------------------------------------------------------- #
def draw_comparison(
    ax: Axes,
    frame: pd.DataFrame,
    *,
    size_col: str,
    size_max: float,
    fill: Sequence,
    edge: Sequence[str],
    stroke: float = 0.5,
    y_levels: Sequence[str],
    x_levels: Sequence[str],
    theme_size: float = 11.0,
) -> None:
    """``geom_point(shape = 21)`` at (Groups, Description), size by
    ``scale_size_area(max_size = 6)``, with SCP's dashed major grid and
    italic wrapped labels."""
    xi = {g: i for i, g in enumerate(x_levels)}
    yi = {d: i for i, d in enumerate(y_levels)}
    xs = np.array([xi[str(g)] for g in frame["Groups"]], dtype=float)
    ys = np.array([yi[str(d)] for d in frame["Description"]], dtype=float)
    v = frame[size_col].to_numpy(dtype=float)
    diam = size_max * np.sqrt(np.clip(v / size_max_of(v), 0, 1))
    ax.set_axisbelow(True)
    ax.grid(True, color="#CCCCCC", linestyle=(0, (4, 4)), linewidth=0.5)
    ax.scatter(xs, ys, s=(diam * DOT) ** 2, c=fill, edgecolors=list(edge),
               linewidths=stroke * 1.5, zorder=3)
    ax.set_xticks(range(len(x_levels)))
    ax.set_xticklabels(list(x_levels), rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(len(y_levels)))
    labels = ax.set_yticklabels(list(y_levels), fontsize=theme_size * 0.8, linespacing=0.9)
    for lab in labels:
        if "\n" in lab.get_text():
            lab.set_fontstyle("italic")
    ax.set_xlim(-0.6, len(x_levels) - 0.4)
    ax.set_ylim(-0.6, len(y_levels) - 0.4)
    ax.set_xlabel("Groups")
    ax.set_ylabel("Description")


def size_max_of(v: np.ndarray) -> float:
    m = float(np.nanmax(v)) if len(v) else 1.0
    return m if m > 0 else 1.0


def size_area_handles(v: np.ndarray, size_max: float, fmt=_fmt) -> list[Line2D]:
    vmax = size_max_of(v)
    out = []
    for b in _size_breaks(0.0, vmax):
        if b <= 0:
            continue
        d = size_max * math.sqrt(b / vmax)
        out.append(Line2D([], [], marker="o", linestyle="none", markerfacecolor="#4D4D4D",
                          markeredgecolor="black", markersize=d * DOT, label=fmt(b)))
    return out


# --------------------------------------------------------------------------- #
# bar / dot / lollipop (EnrichmentPlot, SCP-plot.R:12900-13047)
# --------------------------------------------------------------------------- #
GRID_GREY80 = "#CCCCCC"
_DASHED = (0, (4, 4))


def text_width_in(texts: Sequence[str], fontsize: float) -> float:
    """Width in inches of the widest line among ``texts`` (italic when the
    text wraps, as the term axes draw it), measured from the font itself so a
    panel can be sized for its labels before anything is drawn."""
    ttp = TextToPath()
    widest = 0.0
    for t in texts:
        prop = FontProperties(size=fontsize, style="italic" if "\n" in t else "normal")
        for line in str(t).split("\n"):
            w, _, _ = ttp.get_text_width_height_descent(line, prop, ismath=False)
            widest = max(widest, float(w))
    return widest / 72.0


def _term_axis(ax: Axes, levels: Sequence[str], fontsize: float) -> None:
    """The discrete term axis after ``coord_flip``: level 1 at the bottom,
    ggplot's 0.6 discrete expansion, and SCP's italic face for every level
    that wraps onto a second line (``SCP-plot.R:12937``)."""
    ax.set_yticks(range(len(levels)))
    labels = ax.set_yticklabels(list(levels), fontsize=fontsize, linespacing=0.9)
    for lab in labels:
        lab.set_fontstyle("italic" if "\n" in lab.get_text() else "normal")
    ax.set_ylim(-0.6, len(levels) - 0.4)
    ax.set_ylabel("")
    ax.set_axisbelow(True)
    ax.grid(True, color=GRID_GREY80, linestyle=_DASHED, linewidth=0.5)


def draw_enrichment_bar(
    ax: Axes,
    frame: pd.DataFrame,
    levels: Sequence[str],
    *,
    colors: Sequence[str],
    metric: str,
    fontsize: float,
) -> None:
    """``geom_bar`` of ``-log10(metric)`` filled by ``color_by``, with the
    ``Count`` written twice at the bar end (a white bold underlay, then
    black, ``hjust = -0.5``, size 3.5) and the value axis fixed at
    ``[0, 1.3 * max]`` with no expansion (``SCP-plot.R:12913-12926``)."""
    y = frame["_y"].to_numpy(dtype=float)
    v = frame["metric"].to_numpy(dtype=float)
    ax.barh(y, v, height=0.9, color=list(colors), edgecolor="black", linewidth=0.5 * LWD,
            zorder=2)
    fs = 3.5 * PT
    if "Count" in frame:
        for yi, vi, c in zip(y, v, frame["Count"], strict=True):
            s = str(int(c))
            # hjust = -0.5: the left edge sits half a label width past the bar
            tr = offset_copy(ax.transData, fig=cast(Figure, ax.figure), x=0.3 * fs * len(s), y=0,
                             units="points")
            ax.text(vi, yi, s, transform=tr, ha="left", va="center", fontsize=fs,
                    color="white", fontweight="bold", zorder=3)
            ax.text(vi, yi, s, transform=tr, ha="left", va="center", fontsize=fs,
                    color="black", zorder=4)
    hi = float(np.nanmax(v)) if len(v) else 1.0
    ax.set_xlim(0.0, 1.3 * hi)
    ax.set_xlabel(f"-log10({metric})")
    _term_axis(ax, levels, fontsize)


def draw_enrichment_points(
    ax: Axes,
    frame: pd.DataFrame,
    levels: Sequence[str],
    *,
    plot_type: str,
    fill: Sequence,
    sizes: np.ndarray,
    fontsize: float,
) -> None:
    """``dot``: ``geom_point(shape = 21)`` at ``GeneRatio``, axis to
    ``1.3 * max`` (``SCP-plot.R:12965-12976``).  ``lollipop``: a black
    2-wide stem under a 1-wide stem in the fill colour, then the point, at
    ``FoldEnrichment``, axis to ``1.2 * max`` (``13018-13032``).  ``sizes``
    are ggplot sizes in mm."""
    y = frame["_y"].to_numpy(dtype=float)
    if plot_type == "dot":
        x = frame["GeneRatio"].to_numpy(dtype=float)
        hi, xlabel = 1.3, "GeneRatio"
    else:
        x = frame["FoldEnrichment"].to_numpy(dtype=float)
        hi, xlabel = 1.2, "Fold Enrichment"
        ax.hlines(y, 0, x, colors="black", linewidth=2 * LWD, zorder=1)
        ax.hlines(y, 0, x, colors=list(fill), linewidth=1 * LWD, zorder=2)
    ax.scatter(x, y, s=(np.asarray(sizes, dtype=float) * DOT) ** 2, c=list(fill),
               edgecolors="black", linewidths=0.5 * LWD, zorder=3)
    top = float(np.nanmax(x)) if len(x) else 1.0
    ax.set_xlim(0.0, hi * top)
    ax.set_xlabel(xlabel)
    if plot_type == "dot":
        ax.tick_params(axis="x", labelrotation=45)
        for lab in ax.get_xticklabels():
            lab.set_horizontalalignment("right")
            lab.set_rotation_mode("anchor")
    _term_axis(ax, levels, fontsize)


def size_range_handles(v: np.ndarray, rng: tuple[float, float] = (3.0, 6.0),
                       fmt=_fmt) -> list[Line2D]:
    """Legend keys of ``scale_size(range = rng, breaks_extended(n = 4))``,
    drawn as SCP overrides them (``fill = "grey30"``, shape 21)."""
    v = np.asarray(v, dtype=float)
    lo, hi = float(np.nanmin(v)), float(np.nanmax(v))
    br = _size_breaks(lo, hi)
    sizes = _area_size(br, lo, hi, rng)
    return [Line2D([], [], marker="o", linestyle="none", markerfacecolor="#4D4D4D",
                   markeredgecolor="black", markersize=float(s) * DOT, label=fmt(b))
            for b, s in zip(br, sizes, strict=True)]


# --------------------------------------------------------------------------- #
# GSEA bar and line (GSEAPlot, SCP-plot.R:13740-14078)
# --------------------------------------------------------------------------- #
def draw_gsea_bar(
    ax: Axes,
    frame: pd.DataFrame,
    *,
    colors: dict[str, str],
    fontsize: float,
) -> None:
    """``geom_col(NES)`` filled by direction with ``alpha`` from the metric,
    the description written *inside* the panel at x = 0 (``hjust`` flipped by
    sign, nudged 0.05), y axis text blanked and x limits ``+-max|NES|``
    (``SCP-plot.R:14052-14076``)."""
    y = frame["_y"].to_numpy(dtype=float)
    nes = frame["NES"].to_numpy(dtype=float)
    ax.axvline(0, color="black", linewidth=0.5 * LWD, zorder=1)
    bars = ax.barh(y, nes, height=0.9, color=[colors[d] for d in frame["Direction"]],
                   edgecolor="black", linewidth=0.5 * LWD, zorder=2)
    for p, a in zip(bars.patches, frame["alpha"], strict=True):
        p.set_alpha(float(a))
    for yi, n, d in zip(y, nes, frame["Description"], strict=True):
        pos = n > 0
        # inside the panel, clipped to it as ggplot's coord clips; not a
        # decoration, so constrained layout must not make room for it
        t = ax.text(-0.05 if pos else 0.05, yi, str(d), ha="right" if pos else "left",
                    va="center", fontsize=fontsize, linespacing=0.9, zorder=3, clip_on=True,
                    fontstyle="italic" if "\n" in str(d) else "normal")
        t.set_in_layout(False)
    m = float(np.nanmax(np.abs(nes))) if len(nes) else 1.0
    # coord_cartesian(xlim = c(-m, m)) keeps ggplot's default 5% expansion
    ax.set_xlim(-1.1 * m, 1.1 * m)
    n_lev = int(frame["_y"].max()) + 1 if len(frame) else 0
    ax.set_ylim(-0.6, n_lev - 0.4)
    ax.set_yticks([])
    ax.set_xlabel("NES")
    ax.set_ylabel("Description")


def _classic(ax: Axes) -> None:
    """``theme_classic`` plus the black border and the dashed grey90
    major/minor grid GSEAPlot gives each of its three sub-panels."""
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color("black")
        ax.spines[side].set_linewidth(1.0)
    ax.set_axisbelow(True)
    ax.minorticks_on()
    ax.grid(True, which="both", color="#E5E5E5", linestyle=_DASHED, linewidth=0.5)
    ax.tick_params(which="minor", length=0)


def _spread(y: np.ndarray, ax: Axes, fontsize: float, ylim: tuple[float, float]) -> np.ndarray:
    """Push label rows apart vertically so no two overlap: a one-dimensional
    stand-in for ggrepel, which SCP uses for the core-gene labels."""
    y = np.asarray(y, dtype=float).copy()
    if len(y) < 2:
        return y
    fig = cast(Figure, ax.figure)
    h_in = ax.get_position().height * fig.get_size_inches()[1]
    gap = (fontsize * 1.25 / 72.0) / max(h_in, 1e-6) * (ylim[1] - ylim[0])
    order = np.argsort(-y, kind="stable")
    for a, b in zip(order[:-1], order[1:], strict=True):
        y[b] = min(y[b], y[a] - gap)
    low = ylim[0] + gap / 2 - y[order[-1]]
    if low > 0:
        y += low
    return y


def draw_gsea_line(
    axes: tuple[Axes, Axes, Axes],
    sets: Sequence[dict],
    *,
    colors: Sequence[str],
    gene_list: np.ndarray,
    line_width: float,
    line_alpha: float = 1.0,
    single: dict | None = None,
    heat_strip: Sequence[tuple[float, float, str]] | None = None,
    zero_cross: float | None = None,
    xlim: tuple[float, float],
    fontsize: float,
) -> None:
    """GSEAPlot's three stacked panels for one result key
    (``SCP-plot.R:13792-13970``).

    ``sets`` holds, per gene set, ``x``, ``running_score``, ``hits`` (the x of
    every hit), ``label`` and ``id``; every set is overlaid in the top panel
    and gets its own band of ticks in the middle one, the first set on top.
    ``single`` (one set only) adds the peak triangle with its dashed
    crosshair, the core-gene labels and the title; ``heat_strip`` is the
    ranking-metric strip under the ticks.
    """
    top, mid, bot = axes
    for ax in axes:
        _classic(ax)
        ax.set_xlim(*xlim)

    # p1: running score over red/blue half planes
    lo = min(0.0, *(float(np.min(s["running_score"])) for s in sets))
    hi = max(0.0, *(float(np.max(s["running_score"])) for s in sets))
    span = (hi - lo) or 1.0
    ylo, yhi = lo - 0.05 * span, hi + 0.05 * span
    top.axhspan(0, yhi, facecolor="#C40003", alpha=0.2, linewidth=0, zorder=0)
    top.axhspan(ylo, 0, facecolor="#1D008F", alpha=0.2, linewidth=0, zorder=0)
    top.axhline(0, color="#666666", linewidth=0.5 * LWD, zorder=1)
    for s, c in zip(sets, colors, strict=True):
        top.plot(s["x"], s["running_score"], color=c, linewidth=line_width * LWD,
                 alpha=line_alpha, label=s["label"], gid=s["id"], zorder=2)
    top.set_ylim(ylo, yhi)
    top.set_ylabel("Enrichment Score")
    top.tick_params(axis="x", which="both", labelbottom=False, length=0)

    if single is not None:
        px, py = single["peak"]
        top.plot([0, px], [py, py], color="black", linestyle="--", linewidth=0.5 * LWD, zorder=3)
        top.plot([px, px], [0, py], color="black", linestyle="--", linewidth=0.5 * LWD, zorder=3)
        neg = single["nes"] < 0
        top.scatter([px], [py], marker="v" if neg else "^", s=(2.5 * DOT) ** 2 * 1.6,
                    c="#5E34F5" if neg else "#F52323", edgecolors="black",
                    linewidths=0.5 * LWD, zorder=5)
        genes = single.get("genes")
        if genes is not None and len(genes):
            gx = genes["x"].to_numpy(dtype=float)
            gy = genes["running_score"].to_numpy(dtype=float)
            top.scatter(gx, gy, s=(1.5 * DOT) ** 2, c="black", zorder=4)
            label_fs = single.get("label_size", 4.0) * PT
            nx = 0.05 * (xlim[1] - xlim[0])
            ny = 0.05 * span
            tx = gx + np.where(gy >= 0, nx, -nx)
            ty = _spread(gy + np.where(gy > 0, -ny, ny), top, label_fs, (ylo, yhi))
            for xi, yi, lx, ly, g in zip(gx, gy, tx, ty, genes["gene"], strict=True):
                t = top.annotate(str(g), (xi, yi), xytext=(lx, ly), fontsize=label_fs,
                                 ha="left" if yi >= 0 else "right", va="center",
                                 zorder=6,
                                 arrowprops=dict(arrowstyle="-", color="#666666",
                                                 linewidth=0.5 * LWD, shrinkA=0, shrinkB=2))
                halo(t, "white", 0.1)
        top.set_title(f"{single['title']}\n", loc="left", fontsize=fontsize * 1.2)
        top.text(0.0, 1.01, single["subtitle"], transform=top.transAxes, ha="left",
                 va="bottom", fontstyle="italic", fontsize=fontsize)

    # p2: one band of hit ticks per set, the first set on top
    n = len(sets)
    for i, (s, c) in enumerate(zip(reversed(list(sets)), reversed(list(colors)), strict=True)):
        lc = mid.vlines(s["hits"], i, i + 1, colors="black" if n == 1 else c,
                        linewidth=0.5 * LWD, alpha=line_alpha, zorder=2)
        lc.set_gid(s["id"])
    if heat_strip:
        polys = [[(a, 0.0), (b, 0.0), (b, 0.3), (a, 0.3)] for a, b, _ in heat_strip]
        mid.add_collection(PolyCollection(polys, facecolors=[c for *_, c in heat_strip],
                                          edgecolors="none", alpha=0.95, zorder=3))
    mid.set_ylim(0, n)
    mid.set_xticks([])
    mid.set_yticks([])
    mid.minorticks_off()
    mid.grid(False, which="both")

    # p3: the ranked metric as stems, the correlated-side labels, zero cross
    gl = np.asarray(gene_list, dtype=float)
    x = np.arange(1, len(gl) + 1, dtype=float)
    bot.vlines(x, 0, gl, colors="#4D4D4D", linewidth=0.5 * LWD, zorder=2)
    fs = 4 * PT
    if np.nanmax(gl) > 0:
        bot.text(0, 1, " Positively correlated", transform=bot.get_xaxis_transform(),
                 ha="left", va="top", color="#C81A1F", fontsize=fs)
    if np.nanmin(gl) < 0:
        bot.text(1, 0, "Negatively correlated ", transform=bot.transAxes, ha="right",
                 va="bottom", color="#3C298C", fontsize=fs)
    if zero_cross is not None:
        bot.axvline(zero_cross, color="black", linestyle="--", linewidth=0.5 * LWD, zorder=3)
        up = abs(float(np.nanmax(gl))) > abs(float(np.nanmin(gl)))
        bot.text(zero_cross, 0, f"Zero cross at {zero_cross:.15g}", ha="center",
                 va="bottom" if up else "top", fontsize=fs, zorder=4)
    bot.set_ylabel("Ranked List Metric")
    bot.set_xlabel("Rank in Ordered Dataset")
