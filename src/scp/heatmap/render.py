"""The GridSpec renderer — ComplexHeatmap's layout engine, natively.

ComplexHeatmap gives you four things nothing in Python does: physical-unit
layout of a body plus arbitrary stacked tracks, split-aware coordinates shared
across every track, multi-panel concatenation, and an automatic legend column.
``seaborn.clustermap`` is a dead end (one body, four fixed slots, no splits), so
this is built directly on :class:`matplotlib.gridspec.GridSpec`.

The whole engine rests on one trick: the figure is created at exactly the size
:meth:`HeatmapSpec.figsize` computes, with no margins, so **every GridSpec ratio
is an inch value and maps 1:1 onto the page**.  That is what makes a 5 mm
annotation lane actually 5 mm, and what lets two heatmaps line up when placed
side by side in a manuscript.

Spec: ``docs/porting_briefs/heatmaps.md`` §11.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from .spec import HeatmapSpec, LegendSpec, PanelSpec, Track

__all__ = ["render"]

MM = 1 / 25.4


def _slices(split: pd.Categorical | None, n: int) -> list[tuple[str, np.ndarray]]:
    """Index blocks along one axis, in declared level order.

    Returns one ``(label, positions)`` block per level, or a single unnamed
    block when there is no split.  Level order is taken from the Categorical
    rather than from the data, so the blocks stay in the order the caller
    declared even when a level is sparse.
    """
    if split is None:
        return [("", np.arange(n))]
    s = pd.Categorical(split)
    return [(str(lv), np.flatnonzero(np.asarray(s) == lv)) for lv in s.categories]


def _blank(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)
    return ax


def _measure_text(fig: Figure, labels, fontsize: float, *, horizontal: bool) -> float:
    """Longest label in inches — ComplexHeatmap's ``max_width`` slot.

    Measured against a real renderer rather than estimated from character
    counts, because the whole point of the size engine is that the reserved
    space is the space the text actually needs.
    """
    if not len(labels):
        return 0.0
    renderer = fig.canvas.get_renderer()
    longest = 0.0
    for lab in labels:
        t = fig.text(0, 0, str(lab), fontsize=fontsize)
        bb = t.get_window_extent(renderer=renderer)
        longest = max(longest, (bb.width if horizontal else bb.height) / fig.dpi)
        t.remove()
    return longest + 0.06  # a little breathing room, as ComplexHeatmap pads too


def _draw_simple_track(ax, values, *, vertical: bool, colors=None, cmap=None, norm=None,
                       na_color="none", border=True):
    """A categorical or continuous lane: a 1xN (or Nx1) pcolormesh."""
    v = np.asarray(values)
    n = len(v)
    if colors is not None:
        rgba = [colors.get(str(x), na_color) for x in v]
        for i, c in enumerate(rgba):
            if c == "none":
                continue
            rect = ((0, i, 1, 1) if vertical else (i, 0, 1, 1))
            ax.add_patch(Rectangle(rect[:2], rect[2], rect[3], facecolor=c, edgecolor="none"))
        ax.set_xlim(0, 1 if vertical else n)
        ax.set_ylim(0, n if vertical else 1)
    else:
        arr = v.reshape(-1, 1) if vertical else v.reshape(1, -1)
        ax.pcolormesh(np.ma.masked_invalid(arr.astype(float)), cmap=cmap, norm=norm)
    if vertical:
        ax.invert_yaxis()
    _blank(ax)
    if border:
        for side in ("top", "right", "bottom", "left"):
            ax.spines[side].set_visible(True)
            ax.spines[side].set_linewidth(0.5)


def _draw_custom_track(ax, track: Track, positions: np.ndarray, *, vertical: bool):
    """``anno_customize`` / ``anno_block(panel_fun = ...)``: the track draws itself.

    The axes spans ``len(positions)`` cells along the track (``[0, n]`` in data
    units, first cell at the top when ``vertical``) and ``[0, 1]`` across it;
    ``track.draw(ax, positions)`` is called, with ``flip=True`` added when the
    lane runs vertically (a flipped heatmap's per-panel annotation).
    """
    n = len(positions)
    if vertical:
        ax.set_xlim(0, 1)
        ax.set_ylim(n, 0)
    else:
        ax.set_xlim(0, n)
        ax.set_ylim(0, 1)
    _blank(ax)
    if track.draw is not None:
        if vertical:
            track.draw(ax, positions, flip=True)
        else:
            track.draw(ax, positions)
    if track.border:
        for side in ("top", "right", "bottom", "left"):
            ax.spines[side].set_visible(True)
            ax.spines[side].set_linewidth(0.5)


def _draw_block_track(ax, labels, positions, *, vertical: bool, colors=None,
                      show_labels: bool = True):
    """One labelled rectangle per split block — ComplexHeatmap's ``anno_block``."""
    total = sum(len(p) for p in positions)
    start = 0
    for lab, pos in zip(labels, positions, strict=True):
        size = len(pos)
        c = (colors or {}).get(str(lab), "#DDDDDD")
        if vertical:
            ax.add_patch(Rectangle((0, start), 1, size, facecolor=c, edgecolor="none"))
            if show_labels:
                ax.text(0.5, start + size / 2, str(lab), ha="center", va="center",
                        rotation=90, fontsize=7)
        else:
            ax.add_patch(Rectangle((start, 0), size, 1, facecolor=c, edgecolor="none"))
            if show_labels:
                ax.text(start + size / 2, 0.5, str(lab), ha="center", va="center", fontsize=7)
        start += size
    ax.set_xlim(0, 1 if vertical else total)
    ax.set_ylim(0, total if vertical else 1)
    if vertical:
        ax.invert_yaxis()
    _blank(ax)


def _repel(targets: np.ndarray, min_gap: float, lo: float, hi: float) -> np.ndarray:
    """Push labels apart along one axis, keeping their order.

    ComplexHeatmap's ``anno_mark`` does this so leader lines never cross: sort
    the targets, walk forward enforcing a minimum spacing, then walk back from
    the far edge so the block stays inside the axis.
    """
    out = np.sort(targets.astype(float))
    for i in range(1, len(out)):
        if out[i] - out[i - 1] < min_gap:
            out[i] = out[i - 1] + min_gap
    overflow = out[-1] - hi if len(out) else 0.0
    if overflow > 0:
        out -= overflow
    for i in range(len(out) - 2, -1, -1):
        if out[i + 1] - out[i] < min_gap:
            out[i] = out[i + 1] - min_gap
    return np.clip(out, lo, hi)


def _draw_mark_track(ax, at, labels, n_rows: int, *, fontsize: float = 7.0):
    """Called-out row labels with leader lines, repelled so they never collide.

    ComplexHeatmap's left-side ``anno_mark``: the labels sit on the outside,
    right-aligned, and each leader line runs from its label to its row at the
    edge facing the body, with a dog-leg that absorbs the repulsion.
    """
    ax.set_xlim(0, 1)
    ax.set_ylim(0, n_rows)
    ax.invert_yaxis()
    _blank(ax)
    if not len(at):
        return
    fig = ax.get_figure()
    width_in = max(ax.get_position().width * fig.get_figwidth(), 1e-6)
    x_text = max(1.0 - 0.25 / width_in, 0.05)  # 0.25 in of the lane is for the leaders
    at = np.asarray(at, dtype=float) + 0.5
    gap = max(n_rows / max(len(at), 1) * 0.18, fontsize / 72 / max(fig.get_figheight(), 1) * n_rows)
    placed = _repel(at, gap, 0.0, float(n_rows))
    span = 1.0 - x_text
    for y_true, y_lab, lab in zip(at, placed, labels, strict=True):
        ax.plot([x_text + 0.08 * span, x_text + 0.4 * span, x_text + 0.8 * span, 1.0],
                [y_lab, y_lab, y_true, y_true], color="black", linewidth=0.5,
                solid_joinstyle="miter", clip_on=False)
        ax.text(x_text, y_lab, str(lab), ha="right", va="center", fontsize=fontsize,
                clip_on=False)


def _layout_words(fig, words, mode: str, max_w: float, pad: float):
    """Greedy line-wrap of colored, sized words into ``max_w`` inches.

    Returns ``(lines, height)``: each line a list of ``(text, color, size,
    x_in, width_in)`` plus its height.  Widths are measured on the real
    renderer, as ComplexHeatmap's ``textbox_grob`` measures grobs.
    """
    renderer = fig.canvas.get_renderer()

    def measure(s, size):
        t = fig.text(0, 0, s, fontsize=size)
        bb = t.get_window_extent(renderer=renderer)
        t.remove()
        return bb.width / fig.dpi, bb.height / fig.dpi

    lines: list[list] = []
    heights: list[float] = []
    inner = max(max_w - 2 * pad, 0.2)

    def push(line, h):
        if line:
            lines.append(line)
            heights.append(h)

    line: list = []
    x = 0.0
    h = 0.0
    for text, col, size in words:
        size = float(size)
        tokens = str(text).split(" ") if mode == "lines" else [str(text)]
        if mode == "lines":  # every entry starts a new line, wrapped inside
            push(line, h)
            line, x, h = [], 0.0, 0.0
        for tok in tokens:
            w, th = measure(tok, size)
            # a 0.5 em word space (wider than the font's, so differently styled
            # words stay apart once hinted at a lower dpi): a measured "x x" minus
            # "xx" comes out near zero, as text extents measure ink, not advance
            gap = 0.5 * size / 72.0 if line else 0.0
            if line and x + gap + w > inner:
                push(line, h)
                line, x, h, gap = [], 0.0, 0.0, 0.0
            line.append((tok, col, size, x + gap, w))
            x += gap + w
            h = max(h, th)
    push(line, h)
    return lines, heights


def _draw_textbox_track(fig, cell, track: Track, row_blocks, row_subgrid, body_top_in: float):
    """``anno_textbox``: one rounded word box per row slice, linked to its slice.

    Boxes are centred on their slice and pushed apart top-down when the text is
    taller than the slice, which is what ``anno_zoom`` does in ComplexHeatmap.
    Everything is drawn in figure inches, so a 2-inch box is 2 inches.
    """
    from matplotlib.patches import FancyBboxPatch, Polygon

    sub = row_subgrid(cell)
    fig_w, fig_h = fig.get_size_inches()
    lane = cell.get_position(fig)
    x0 = lane.x0 * fig_w
    x1 = lane.x1 * fig_w
    link_w = min(0.12, 0.25 * (x1 - x0))
    pad = 0.04
    values = np.asarray(track.values) if track.values is not None else None
    boxes = []
    for ri, (_lab, rows) in enumerate(row_blocks):
        if values is None or not len(rows):
            continue
        group = str(values[rows[0]])
        df = (track.text or {}).get(group)
        if df is None or not len(df):
            continue
        pos = sub[ri, 0].get_position(fig)
        s_top, s_bot = pos.y1 * fig_h, pos.y0 * fig_h
        words = list(zip(df["keyword"], df["col"], df["fontsize"], strict=True))
        lines, heights = _layout_words(fig, words, track.text_mode, x1 - x0 - link_w, pad)
        lh = [hh * 1.25 for hh in heights]
        boxes.append([s_top, s_bot, lines, lh, sum(lh) + 2 * pad])
    # push apart, top to bottom, keeping each box as close to its slice as it can
    ceiling = body_top_in
    for b in boxes:
        centre = (b[0] + b[1]) / 2
        top = min(centre + b[4] / 2, ceiling)
        b.append(top)
        ceiling = top - b[4] - 0.03
    bx = x0 + link_w
    for s_top, s_bot, lines, lh, height, top in boxes:
        fig.patches.append(Polygon(
            [(x0, s_top), (bx, top), (bx, top - height), (x0, s_bot)], closed=True,
            transform=fig.dpi_scale_trans, facecolor="#F5F5F5", edgecolor="#9E9E9E",
            linewidth=0.4, zorder=1))
        fig.patches.append(FancyBboxPatch(
            (bx, top - height), x1 - bx, height,
            boxstyle="round,pad=0,rounding_size=0.04", transform=fig.dpi_scale_trans,
            facecolor="#FAFAFA", edgecolor="black", linewidth=0.5, zorder=2))
        y = top - pad
        for line, hh in zip(lines, lh, strict=True):
            y -= hh
            # runs of same-styled words go out as one string, so the font engine
            # spaces them -- separately placed words drift together once the
            # glyphs are hinted at a lower output dpi
            runs: list[list] = []
            for tok, col, size, xx, _w in line:
                if runs and runs[-1][1] == col and runs[-1][2] == size:
                    runs[-1][0].append(tok)
                else:
                    runs.append([[tok], col, size, xx])
            for toks, col, size, xx in runs:
                fig.text((bx + pad + xx) / fig_w, (y + hh * 0.15) / fig_h, " ".join(toks),
                         color=col, fontsize=size, ha="left", va="bottom", zorder=3)
    if track.title:
        fig.text(x0 / fig_w, (body_top_in + 2.5 * MM) / fig_h, track.title, ha="left",
                 va="bottom", fontsize=8)


def _draw_dendrogram(ax, Z, n: int, orientation: str):
    from scipy.cluster.hierarchy import dendrogram

    if not np.asarray(Z)[:, 2].max() > 0:
        # cluster_within_group flattens the within-group trees to height 0:
        # ComplexHeatmap draws them as flat lines on the baseline
        _blank(ax)
        return
    dendrogram(Z, ax=ax, orientation=orientation, no_labels=True,
               link_color_func=lambda _k: "black")
    if orientation == "left":
        ax.set_ylim(0, 10 * n)
        ax.invert_yaxis()
    else:
        ax.set_xlim(0, 10 * n)
    _blank(ax)


def _draw_layers(ax, panel: PanelSpec, layers, rows: np.ndarray, cols: np.ndarray, dpi_scale: float,
                 *, flip: bool = False):
    """Body overlays, in SCP's fixed order: whiteout, bg, reticle, dot, violin.

    ``rows`` / ``cols`` index the panel's matrix (features x observations)
    whatever the orientation; with ``flip`` the slice is drawn transposed, and
    ``custom`` / ``violin`` callbacks receive ``flip=True`` so they can swap
    their own axes.
    """
    order = {"whiteout": 0, "bg": 1, "reticle": 2, "dot": 3, "violin": 4, "custom": 5}

    def view(m):
        sub = m[np.ix_(rows, cols)]
        return sub.T if flip else sub

    n_y, n_x = (len(cols), len(rows)) if flip else (len(rows), len(cols))
    for layer in sorted(layers, key=lambda x: order.get(x.kind, 9)):
        if layer.kind == "whiteout":
            ax.add_patch(Rectangle((0, 0), n_x, n_y, facecolor="white",
                                   edgecolor="none", zorder=2))
        elif layer.kind == "bg":
            ax.pcolormesh(np.ma.masked_invalid(view(panel.matrix)), cmap=panel.cmap,
                          norm=panel.norm, alpha=layer.alpha, zorder=1)
        elif layer.kind == "reticle":
            # restore_matrix(): a line through the centre of every column and row
            for i in range(n_x):
                ax.axvline(i + 0.5, color=layer.color, linewidth=1.5 * 0.75, zorder=3)
            for j in range(n_y):
                ax.axhline(j + 0.5, color=layer.color, linewidth=1.5 * 0.75, zorder=3)
        elif layer.kind == "dot":
            if layer.size_matrix is None:
                continue
            frac = view(layer.size_matrix)
            val = view(panel.matrix)
            yy, xx = np.mgrid[0:n_y, 0:n_x]
            # grid.points(size = dot_size * perc): the DIAMETER is linear in the
            # fraction; scatter's s is the squared diameter in points
            diam = layer.base_size_in * 72.0 * np.clip(frac, 0, 1)
            sm = ScalarMappable(norm=panel.norm, cmap=panel.cmap)
            ax.scatter(xx.ravel() + 0.5, yy.ravel() + 0.5, s=diam.ravel() ** 2,
                       c=sm.to_rgba(val.ravel()), edgecolor="black", linewidths=0.75,
                       zorder=4)
        elif layer.kind in ("violin", "custom") and layer.draw is not None:
            if flip:
                layer.draw(ax, panel, rows, cols, flip=True)
            else:
                layer.draw(ax, panel, rows, cols)


def _draw_legends(fig, ax, legends: list[LegendSpec]) -> float:
    """Stack every legend down one dedicated column; return the height used, in inches.

    Each legend is placed below the previous one by its *measured* height, so
    nothing overlaps whatever the figure's height; :func:`_legend_height_in`
    runs the same stacking on a probe to size the figure first.
    """
    _blank(ax)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    renderer = fig.canvas.get_renderer()
    pos = ax.get_position()
    ax_h_in = max(pos.height * fig.get_figheight(), 1e-6)
    ax_w_in = max(pos.width * fig.get_figwidth(), 1e-6)
    gap_in = 0.12
    y_in = 0.0  # distance from the top of the column
    for spec in legends:
        y = 1.0 - y_in / ax_h_in
        leg = None
        if spec.kind == "categorical" and spec.mapping:
            handles = [Patch(facecolor=color, edgecolor="none", label=lab)
                       for lab, color in spec.mapping.items()]
            leg = ax.legend(handles=handles, title=spec.title, loc="upper left",
                            bbox_to_anchor=(0, y), frameon=False, fontsize=7,
                            title_fontsize=7.5, handlelength=1.0, handleheight=1.0,
                            borderpad=0, labelspacing=0.35)
        elif spec.kind == "continuous":
            bar_h, bar_w, title_h = 0.75, 0.13, 0.2
            cax = ax.inset_axes([0.06,
                                 y - (title_h + bar_h) / ax_h_in, bar_w / ax_w_in,
                                 bar_h / ax_h_in])
            cb = fig.colorbar(ScalarMappable(norm=spec.norm_or_none(), cmap=spec.cmap), cax=cax)
            cb.outline.set_linewidth(0.5)
            cb.ax.tick_params(labelsize=6.5, length=2)
            cax.set_title(spec.title, fontsize=7.5, loc="left", pad=4)
            y_in += title_h + bar_h + gap_in
        elif spec.kind == "size":
            # Legend(type = "points", size = dot_size * breaks): diameters
            base_pt = (spec.size_in if spec.size_in is not None else 8 * MM) * 72.0
            handles = [
                Line2D([], [], marker="o", linestyle="none", markerfacecolor="#7F7F7F",
                       markeredgecolor="black", markeredgewidth=0.75, markersize=base_pt * b,
                       label=f"{int(round(b * 100))}%")
                for b in spec.breaks
            ]
            # rows as tall and handles as wide as the largest dot, so none overlap
            leg = ax.legend(handles=handles, title=spec.title, loc="upper left",
                            bbox_to_anchor=(0, y), frameon=False, fontsize=7,
                            title_fontsize=7.5, borderpad=0, labelspacing=0.3,
                            handleheight=max(base_pt / 7.0, 0.7),
                            handlelength=max(base_pt / 7.0, 1.0))
        if leg is not None:
            leg._legend_box.align = "left"
            ax.add_artist(leg)
            y_in += leg.get_window_extent(renderer).height / fig.dpi + gap_in
    return y_in


def _legend_height_in(legends: list[LegendSpec], width_in: float, dpi: float) -> float:
    """How tall the legend column needs to be: the stacking above, on a probe."""
    import matplotlib.pyplot as plt

    if not legends:
        return 0.0
    probe = plt.figure(figsize=(max(width_in, 0.5), 40.0), dpi=dpi)
    try:
        ax = probe.add_axes((0, 0, 1, 1))
        return _draw_legends(probe, ax, legends)
    finally:
        plt.close(probe)


def render(spec: HeatmapSpec, *, dpi: float = 110.0) -> Figure:
    """Draw a :class:`HeatmapSpec` and return the figure.

    The layout is one top-level GridSpec whose ratios are inches; splits become
    a nested GridSpec shared by the body and every track running along the same
    axis, which is how slices stay aligned.
    """
    import matplotlib.pyplot as plt

    if spec.flip:
        return _render_flipped(spec, dpi=dpi)
    n_rows_total = spec.panels[0].matrix.shape[0]
    row_blocks = _slices(spec.row_split, n_rows_total)

    # --- measure -----------------------------------------------------------
    probe = plt.figure(figsize=(1, 1), dpi=dpi)
    if spec.show_row_names:
        spec.measured["row_names_w"] = _measure_text(
            probe, spec.panels[0].row_labels, 7.0, horizontal=True)
    if spec.show_column_names:
        spec.measured["column_names_h"] = max(
            _measure_text(probe, p.col_labels, 7.0, horizontal=True) for p in spec.panels) * 0.8
    mark_w = max(
        (_measure_text(probe, t.labels or [], 7.0, horizontal=True) + 0.25
         for t in spec.left_tracks if t.kind == "mark"), default=0.0)
    # top-track names print left of the first panel; make sure that margin exists
    anno_names = [t.name for t in spec.panels[0].top_tracks if t.show_name]
    need = _measure_text(probe, anno_names, 6.5, horizontal=True) if anno_names else 0.0
    have = (mark_w + (spec.row_dend.size_in if spec.row_dend else 0.0)
            + sum(t.size_in for t in spec.left_tracks if t.kind != "mark"))
    mark_w += max(need - have, 0.0)
    plt.close(probe)
    if spec.legends:
        spec.measured.setdefault("legend_w", 1.5)

    fig_w, fig_h = spec.figsize()
    fig_w += mark_w
    # a short heatmap still gets room for its legends: the extra goes below
    extra_h = max(_legend_height_in(spec.legends, spec.measured.get("legend_w", 0.0), dpi)
                  + 0.1 - fig_h, 0.0)
    fig_h += extra_h
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)

    # --- top-level grid, in inches ----------------------------------------
    title_h = spec.measured.get("titles_h", 0.25)
    top_h = max((sum(t.size_in for t in p.top_tracks) for p in spec.panels), default=0.0)
    coldend_h = max((d.size_in for d in spec.col_dends.values()), default=0.0)
    body_h = spec.body_height_in if spec.body_height_in is not None else 1.0
    names_h = spec.measured.get("column_names_h", 0.0) if spec.show_column_names else 0.0
    height_ratios = [title_h, top_h, coldend_h, body_h, names_h + extra_h]

    left_w = sum(t.size_in for t in spec.left_tracks if t.kind != "mark")
    rowdend_w = spec.row_dend.size_in if spec.row_dend else 0.0
    right_w = sum(t.size_in for t in spec.right_tracks)
    names_w = spec.measured.get("row_names_w", 0.0) if spec.show_row_names else 0.0
    legend_w = spec.measured.get("legend_w", 0.0) if spec.legends else 0.0
    bodies = [p.body_size_in if p.body_size_in is not None else 1.0 for p in spec.panels]
    width_ratios = [rowdend_w, mark_w, left_w] + bodies + [right_w, names_w, legend_w]

    gs = fig.add_gridspec(
        len(height_ratios), len(width_ratios),
        height_ratios=[max(r, 1e-6) for r in height_ratios],
        width_ratios=[max(r, 1e-6) for r in width_ratios],
        wspace=0, hspace=0,
    )
    BODY_ROW = 3
    FIRST_BODY_COL = 3

    def row_subgrid(cell):
        """Split the row axis into blocks, sized by how many rows each holds."""
        return GridSpecFromSubplotSpec(
            len(row_blocks), 1, subplot_spec=cell,
            height_ratios=[max(len(p), 1) for _, p in row_blocks],
            hspace=spec.row_gap_in / max(body_h, 1e-6) * len(row_blocks),
        )

    # --- bodies ------------------------------------------------------------
    for pi, panel in enumerate(spec.panels):
        col = FIRST_BODY_COL + pi
        col_blocks = _slices(panel.col_split, panel.matrix.shape[1])
        outer = GridSpecFromSubplotSpec(
            len(row_blocks), len(col_blocks), subplot_spec=gs[BODY_ROW, col],
            height_ratios=[max(len(p), 1) for _, p in row_blocks],
            width_ratios=[max(len(p), 1) for _, p in col_blocks],
            hspace=spec.row_gap_in / max(body_h, 1e-6) * len(row_blocks),
            wspace=spec.col_gap_in / max(bodies[pi], 1e-6) * len(col_blocks),
        )
        for ri, (_, rows) in enumerate(row_blocks):
            for ci, (_, cols) in enumerate(col_blocks):
                ax = fig.add_subplot(outer[ri, ci])
                sub = panel.matrix[np.ix_(rows, cols)]
                ax.pcolormesh(np.ma.masked_invalid(sub), cmap=panel.cmap, norm=panel.norm,
                              edgecolors="none", rasterized=sub.size > 20000)
                _draw_layers(ax, panel, panel.layers, rows, cols, dpi)
                ax.set_xlim(0, len(cols))
                ax.set_ylim(0, len(rows))
                ax.invert_yaxis()  # row 0 on top, as ComplexHeatmap does
                _blank(ax)
                for side in ("top", "right", "bottom", "left"):
                    ax.spines[side].set_visible(True)
                    ax.spines[side].set_linewidth(0.5)

                if spec.show_column_names and ri == len(row_blocks) - 1:
                    nax = fig.add_subplot(gs[BODY_ROW + 1, col])
                    _blank(nax)
                    nax.set_xlim(0, panel.matrix.shape[1])
                    nax.set_ylim(0, 1)
                    for x, lab in enumerate(panel.col_labels):
                        nax.text(x + 0.5, 0.92, str(lab), rotation=90, ha="center",
                                 va="top", fontsize=7)
        if panel.title:
            tax = fig.add_subplot(gs[0, col])
            _blank(tax)
            tax.text(0.5, 0.25, panel.title, ha="center", va="bottom", fontsize=8.5)

        # Top tracks stack: give each its own row of gs[1, col], then subdivide
        # that row by the column blocks so the lanes stay aligned with the body.
        # Drawing them all into the one cell makes each overwrite the last.
        if panel.top_tracks:
            lanes = GridSpecFromSubplotSpec(
                len(panel.top_tracks), 1, subplot_spec=gs[1, col],
                height_ratios=[max(t.size_in, 1e-6) for t in panel.top_tracks],
                hspace=0.25)
        for ti, track in enumerate(panel.top_tracks):
            tcell = GridSpecFromSubplotSpec(
                1, len(col_blocks), subplot_spec=lanes[ti, 0],
                width_ratios=[max(len(p), 1) for _, p in col_blocks],
                wspace=spec.col_gap_in / max(bodies[pi], 1e-6) * len(col_blocks))
            for ci, (_, cols) in enumerate(col_blocks):
                tax = fig.add_subplot(tcell[0, ci])
                if track.kind == "custom":
                    _draw_custom_track(tax, track, cols, vertical=False)
                else:
                    _draw_simple_track(tax, np.asarray(track.values)[cols], vertical=False,
                                       colors=track.colors, cmap=track.cmap, norm=track.norm,
                                       na_color=track.na_color, border=track.border)
                # annotation names sit left of the first panel only, as in
                # ComplexHeatmap (annotation_name_side = "left")
                if track.show_name and pi == 0 and ci == 0:
                    tax.annotate(track.name, xy=(0.0, 0.5), xycoords="axes fraction",
                                 xytext=(-3, 0), textcoords="offset points", ha="right",
                                 va="center", fontsize=6.5)

    # --- side tracks, sharing the row subdivision --------------------------
    # Each side column is split into one lane per track, sized in inches, so
    # stacked tracks sit next to each other instead of drawing over each other.
    body_top_in = (fig_h - sum(height_ratios[:BODY_ROW]))

    def side_tracks(tracks, cell):
        if not tracks:
            return
        lanes = GridSpecFromSubplotSpec(
            1, len(tracks), subplot_spec=cell,
            width_ratios=[max(t.size_in, 1e-6) for t in tracks], wspace=0)
        for ti, track in enumerate(tracks):
            if track.kind == "textbox":
                _draw_textbox_track(fig, lanes[0, ti], track, row_blocks, row_subgrid,
                                    body_top_in)
                continue
            if track.kind == "custom" and track.draw is None:
                continue  # anno_empty: a spacer
            sub = row_subgrid(lanes[0, ti])
            for ri, (lab, rows) in enumerate(row_blocks):
                ax = fig.add_subplot(sub[ri, 0])
                if track.kind == "block":
                    _draw_block_track(ax, [lab], [rows], vertical=True, colors=track.colors,
                                      show_labels=track.show_labels)
                elif track.kind == "custom" and track.draw is not None:
                    track.draw(ax, rows)
                else:
                    _draw_simple_track(ax, np.asarray(track.values)[rows], vertical=True,
                                       colors=track.colors, cmap=track.cmap, norm=track.norm,
                                       na_color=track.na_color, border=track.border)
                if ri == 0 and track.show_name and track.kind == "simple":
                    ax.text(0.5, 1.0, f" {track.name}", transform=ax.transAxes, rotation=90,
                            ha="center", va="bottom", fontsize=6.5)

    for track in spec.left_tracks:
        if track.kind == "mark":
            ax = fig.add_subplot(gs[BODY_ROW, 1])
            _draw_mark_track(ax, track.at or [], track.labels or [], n_rows_total)
    side_tracks([t for t in spec.left_tracks if t.kind != "mark"], gs[BODY_ROW, 2])
    side_tracks(list(spec.right_tracks), gs[BODY_ROW, FIRST_BODY_COL + len(spec.panels)])

    # --- row names ---------------------------------------------------------
    if spec.show_row_names:
        sub = row_subgrid(gs[BODY_ROW, FIRST_BODY_COL + len(spec.panels) + 1])
        for ri, (_, rows) in enumerate(row_blocks):
            ax = fig.add_subplot(sub[ri, 0])
            _blank(ax)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, len(rows))
            ax.invert_yaxis()
            for y, idx in enumerate(rows):
                ax.text(0.06, y + 0.5, str(spec.panels[0].row_labels[idx]), ha="left",
                        va="center", fontsize=7)

    # --- dendrograms -------------------------------------------------------
    # With a split, every slice gets its own tree in its own axes, so branches
    # line up with the gaps -- the tree of the slices themselves is not drawn,
    # as ComplexHeatmap does not draw it either.
    rd = spec.row_dend
    if rd is not None:
        if rd.slice_linkages is not None and len(rd.slice_linkages) == len(row_blocks):
            sub = row_subgrid(gs[BODY_ROW, 0])
            for ri, (_, rows) in enumerate(row_blocks):
                Z = rd.slice_linkages[ri]
                ax = fig.add_subplot(sub[ri, 0])
                if Z is None:
                    _blank(ax)
                    continue
                _draw_dendrogram(ax, Z, len(rows), "left")
        elif rd.linkage is not None:
            ax = fig.add_subplot(gs[BODY_ROW, 0])
            _draw_dendrogram(ax, rd.linkage, n_rows_total, "left")
    for pi, panel in enumerate(spec.panels):
        d = spec.col_dends.get(panel.name)
        if d is None:
            continue
        col_blocks = _slices(panel.col_split, panel.matrix.shape[1])
        if d.slice_linkages is not None and len(d.slice_linkages) == len(col_blocks):
            dsub = GridSpecFromSubplotSpec(
                1, len(col_blocks), subplot_spec=gs[2, FIRST_BODY_COL + pi],
                width_ratios=[max(len(p), 1) for _, p in col_blocks],
                wspace=spec.col_gap_in / max(bodies[pi], 1e-6) * len(col_blocks))
            for ci, (_, cols) in enumerate(col_blocks):
                Z = d.slice_linkages[ci]
                ax = fig.add_subplot(dsub[0, ci])
                if Z is None:
                    _blank(ax)
                    continue
                _draw_dendrogram(ax, Z, len(cols), "top")
        elif d.linkage is not None:
            ax = fig.add_subplot(gs[2, FIRST_BODY_COL + pi])
            _draw_dendrogram(ax, d.linkage, panel.matrix.shape[1], "top")

    # --- legends -----------------------------------------------------------
    if spec.legends:
        lax = fig.add_subplot(gs[:, len(width_ratios) - 1])
        _draw_legends(fig, lax, spec.legends)

    return fig


def _draw_mark_track_top(ax, at, labels, n_cols: int, *, fontsize: float = 7.0):
    """``anno_mark(side = "top")``: the flipped heatmap's called-out features.

    The labels stand upright above the body, repelled along x, and each leader
    line drops from its label to its column at the lane's bottom edge.
    """
    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, 1)
    _blank(ax)
    if not len(at):
        return
    fig = ax.get_figure()
    height_in = max(ax.get_position().height * fig.get_figheight(), 1e-6)
    y_text = min(0.25 / height_in, 0.95)  # the bottom 0.25 in of the lane is for the leaders
    at = np.asarray(at, dtype=float) + 0.5
    gap = max(n_cols / max(len(at), 1) * 0.18,
              fontsize / 72 / max(fig.get_figwidth(), 1) * n_cols)
    placed = _repel(at, gap, 0.0, float(n_cols))
    for x_true, x_lab, lab in zip(at, placed, labels, strict=True):
        ax.plot([x_lab, x_lab, x_true, x_true],
                [y_text * 0.92, y_text * 0.6, y_text * 0.2, 0.0], color="black",
                linewidth=0.5, solid_joinstyle="miter", clip_on=False)
        ax.text(x_lab, y_text, str(lab), ha="center", va="bottom", rotation=90,
                fontsize=fontsize, clip_on=False)


def _render_flipped(spec: HeatmapSpec, *, dpi: float) -> Figure:
    """``flip = TRUE``: every panel transposed and the panels stacked vertically.

    What ComplexHeatmap draws for SCP's flipped heatmaps (``ht_list %v%
    Heatmap(t(mat))``): observations run down each panel and the features
    across, shared by every panel and split by the feature split; the panel's
    top annotations become left lanes (named on top of the first panel), its
    column dendrogram a row dendrogram and its title a row title on the left;
    the left annotations (feature marks, split blocks) move above the first
    panel and the right ones (feature annotations, lineage membership) below
    the last.  Feature names, when shown, sit under the bottom lanes; panel
    column names (groups, cells) right of each body.
    """
    import matplotlib.pyplot as plt

    panels = spec.panels
    n_feat = panels[0].matrix.shape[0]
    feat_blocks = _slices(spec.row_split, n_feat)
    feat_labels = panels[0].row_labels

    # --- measure -----------------------------------------------------------
    probe = plt.figure(figsize=(1, 1), dpi=dpi)
    feat_names_h = (_measure_text(probe, feat_labels, 7.0, horizontal=True)
                    if spec.show_row_names else 0.0)
    obs_names_w = (max(_measure_text(probe, p.col_labels, 7.0, horizontal=True) for p in panels)
                   if spec.show_column_names else 0.0)
    titles = [p.title for p in panels if p.title]
    title_w = _measure_text(probe, titles, 8.5, horizontal=True) if titles else 0.0
    marks = [t for t in spec.left_tracks if t.kind == "mark"]
    mark_h = max((_measure_text(probe, t.labels or [], 7.0, horizontal=True) + 0.25
                  for t in marks), default=0.0)
    lane_names = [t.name for t in panels[0].top_tracks if t.show_name]
    lane_names_h = (_measure_text(probe, lane_names, 6.5, horizontal=True)
                    if lane_names else 0.0)
    bottom_names = [t.name for t in spec.right_tracks if t.show_name and t.kind == "simple"]
    bottom_names_w = (_measure_text(probe, bottom_names, 6.5, horizontal=True)
                      if bottom_names else 0.0)
    plt.close(probe)

    top_lanes = [t for t in spec.left_tracks if t.kind != "mark"]
    bottom_lanes = [t for t in spec.right_tracks if t.kind != "textbox"]
    lanes_w = max((sum(t.size_in for t in p.top_tracks) for p in panels), default=0.0)
    pdend_w = max((d.size_in for d in spec.col_dends.values()), default=0.0)
    fdend_h = spec.row_dend.size_in if spec.row_dend else 0.0
    toplanes_h = sum(t.size_in for t in top_lanes)
    bottom_h = sum(t.size_in for t in bottom_lanes)
    body_w = spec.body_height_in if spec.body_height_in is not None else 1.0
    bodies = [p.body_size_in if p.body_size_in is not None else 1.0 for p in panels]
    legend_w = spec.measured.setdefault("legend_w", 1.5) if spec.legends else 0.0
    # the lane names stand above the first panel, beside the top annotations
    mark_h += max(lane_names_h - (mark_h + fdend_h + toplanes_h), 0.0)
    left_w = max(title_w + 0.08, bottom_names_w + 0.06)

    width_ratios = [left_w, pdend_w, lanes_w, body_w, obs_names_w, legend_w]
    TITLE, PDEND, LANES, BODY, NAMES = 0, 1, 2, 3, 4
    height_ratios = [mark_h, fdend_h, toplanes_h]
    body_rows = []
    for pi, b in enumerate(bodies):
        if pi:
            height_ratios.append(spec.panel_gap_in)
        body_rows.append(len(height_ratios))
        height_ratios.append(b)
    BOTTOM = len(height_ratios)
    height_ratios += [bottom_h, feat_names_h]
    # room for the legend column, added below the feature names
    height_ratios[-1] += max(_legend_height_in(spec.legends, legend_w, dpi) + 0.1
                             - sum(height_ratios), 0.0)

    fig_w, fig_h = sum(width_ratios), sum(height_ratios)
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    gs = fig.add_gridspec(
        len(height_ratios), len(width_ratios),
        height_ratios=[max(r, 1e-6) for r in height_ratios],
        width_ratios=[max(r, 1e-6) for r in width_ratios], wspace=0, hspace=0)

    def feat_subgrid(cell, nrows: int = 1, height_ratios=None, hspace=0.0):
        """Split the feature axis (now x) into its blocks."""
        return GridSpecFromSubplotSpec(
            nrows, len(feat_blocks), subplot_spec=cell,
            height_ratios=height_ratios,
            width_ratios=[max(len(p), 1) for _, p in feat_blocks],
            wspace=spec.row_gap_in / max(body_w, 1e-6) * len(feat_blocks), hspace=hspace)

    def obs_subgrid(cell, blocks, height: float, ncols: int = 1, width_ratios=None):
        return GridSpecFromSubplotSpec(
            len(blocks), ncols, subplot_spec=cell, width_ratios=width_ratios,
            height_ratios=[max(len(p), 1) for _, p in blocks],
            hspace=spec.col_gap_in / max(height, 1e-6) * len(blocks), wspace=0)

    def border(ax):
        for side in ("top", "right", "bottom", "left"):
            ax.spines[side].set_visible(True)
            ax.spines[side].set_linewidth(0.5)

    # --- panels --------------------------------------------------------------
    for pi, panel in enumerate(panels):
        r = body_rows[pi]
        obs_blocks = _slices(panel.col_split, panel.matrix.shape[1])
        outer = GridSpecFromSubplotSpec(
            len(obs_blocks), len(feat_blocks), subplot_spec=gs[r, BODY],
            height_ratios=[max(len(p), 1) for _, p in obs_blocks],
            width_ratios=[max(len(p), 1) for _, p in feat_blocks],
            hspace=spec.col_gap_in / max(bodies[pi], 1e-6) * len(obs_blocks),
            wspace=spec.row_gap_in / max(body_w, 1e-6) * len(feat_blocks))
        for oi, (_, obs) in enumerate(obs_blocks):
            for fi, (_, fpos) in enumerate(feat_blocks):
                ax = fig.add_subplot(outer[oi, fi])
                sub = panel.matrix[np.ix_(fpos, obs)].T
                ax.pcolormesh(np.ma.masked_invalid(sub), cmap=cast(Any, panel.cmap),
                              norm=cast(Any, panel.norm), edgecolors="none",
                              rasterized=sub.size > 20000)
                _draw_layers(ax, panel, panel.layers, fpos, obs, dpi, flip=True)
                ax.set_xlim(0, len(fpos))
                ax.set_ylim(len(obs), 0)  # first observation on top
                _blank(ax)
                border(ax)
        if panel.title:
            tax = fig.add_subplot(gs[r, TITLE])
            _blank(tax)
            tax.text(0.97, 0.5, panel.title, ha="right", va="center", fontsize=8.5,
                     transform=tax.transAxes)
        if spec.show_column_names:
            sub = obs_subgrid(gs[r, NAMES], obs_blocks, bodies[pi])
            for oi, (_, obs) in enumerate(obs_blocks):
                nax = fig.add_subplot(sub[oi, 0])
                _blank(nax)
                nax.set_xlim(0, 1)
                nax.set_ylim(len(obs), 0)
                for y, j in enumerate(obs):
                    nax.text(0.06, y + 0.5, str(panel.col_labels[j]), ha="left", va="center",
                             fontsize=7)
        # the panel's top annotations, as left lanes
        if panel.top_tracks:
            lanes = GridSpecFromSubplotSpec(
                1, len(panel.top_tracks), subplot_spec=gs[r, LANES],
                width_ratios=[max(t.size_in, 1e-6) for t in panel.top_tracks], wspace=0.25)
            for ti, track in enumerate(panel.top_tracks):
                sub = obs_subgrid(lanes[0, ti], obs_blocks, bodies[pi])
                for oi, (_, obs) in enumerate(obs_blocks):
                    tax = fig.add_subplot(sub[oi, 0])
                    if track.kind == "custom":
                        _draw_custom_track(tax, track, obs, vertical=True)
                    else:
                        _draw_simple_track(tax, np.asarray(track.values)[obs], vertical=True,
                                           colors=track.colors, cmap=track.cmap,
                                           norm=track.norm, na_color=track.na_color,
                                           border=track.border)
                    # annotation_name_side = "top" when flipped, first panel only
                    if track.show_name and pi == 0 and oi == 0:
                        tax.annotate(track.name, xy=(0.5, 1.0), xycoords="axes fraction",
                                     xytext=(0, 3), textcoords="offset points", rotation=90,
                                     ha="center", va="bottom", fontsize=6.5)
        # the panel's column dendrogram, now on the left
        d = spec.col_dends.get(panel.name)
        if d is not None:
            if d.slice_linkages is not None and len(d.slice_linkages) == len(obs_blocks):
                sub = obs_subgrid(gs[r, PDEND], obs_blocks, bodies[pi])
                for oi, (_, obs) in enumerate(obs_blocks):
                    ax = fig.add_subplot(sub[oi, 0])
                    Z = d.slice_linkages[oi]
                    if Z is None:
                        _blank(ax)
                        continue
                    _draw_dendrogram(ax, Z, len(obs), "left")
            elif d.linkage is not None:
                _draw_dendrogram(fig.add_subplot(gs[r, PDEND]), d.linkage,
                                 panel.matrix.shape[1], "left")

    # --- above the first panel: marks, feature dendrogram, split lanes -------
    for track in marks:
        ax = fig.add_subplot(gs[0, BODY])
        _draw_mark_track_top(ax, track.at or [], track.labels or [], n_feat)
    rd = spec.row_dend
    if rd is not None:
        if rd.slice_linkages is not None and len(rd.slice_linkages) == len(feat_blocks):
            sub = feat_subgrid(gs[1, BODY])
            for fi, (_, fpos) in enumerate(feat_blocks):
                ax = fig.add_subplot(sub[0, fi])
                Z = rd.slice_linkages[fi]
                if Z is None:
                    _blank(ax)
                    continue
                _draw_dendrogram(ax, Z, len(fpos), "top")
        elif rd.linkage is not None:
            _draw_dendrogram(fig.add_subplot(gs[1, BODY]), rd.linkage, n_feat, "top")

    def feature_lanes(tracks, cell, *, names_left: bool):
        if not tracks:
            return
        sub = feat_subgrid(cell, nrows=len(tracks),
                           height_ratios=[max(t.size_in, 1e-6) for t in tracks])
        for ti, track in enumerate(tracks):
            for fi, (lab, fpos) in enumerate(feat_blocks):
                ax = fig.add_subplot(sub[ti, fi])
                if track.kind == "block":
                    _draw_block_track(ax, [lab], [fpos], vertical=False, colors=track.colors,
                                      show_labels=track.show_labels)
                elif track.kind == "custom":
                    _draw_custom_track(ax, track, fpos, vertical=False)
                else:
                    _draw_simple_track(ax, np.asarray(track.values)[fpos], vertical=False,
                                       colors=track.colors, cmap=track.cmap, norm=track.norm,
                                       na_color=track.na_color, border=track.border)
                if (names_left and fi == 0 and track.show_name and track.kind == "simple"):
                    # annotation_name_side = "left" for the flipped feature annotations
                    ax.annotate(track.name, xy=(0.0, 0.5), xycoords="axes fraction",
                                xytext=(-3, 0), textcoords="offset points", ha="right",
                                va="center", fontsize=6.5)

    feature_lanes(top_lanes, gs[2, BODY], names_left=False)
    feature_lanes(bottom_lanes, gs[BOTTOM, BODY], names_left=True)

    if spec.show_row_names:
        sub = feat_subgrid(gs[BOTTOM + 1, BODY])
        for fi, (_, fpos) in enumerate(feat_blocks):
            ax = fig.add_subplot(sub[0, fi])
            _blank(ax)
            ax.set_xlim(0, len(fpos))
            ax.set_ylim(0, 1)
            for x, idx in enumerate(fpos):
                ax.text(x + 0.5, 0.97, str(feat_labels[idx]), rotation=90, ha="center",
                        va="top", fontsize=7)

    if spec.legends:
        lax = fig.add_subplot(gs[:, len(width_ratios) - 1])
        _draw_legends(fig, lax, spec.legends)
    return fig


def _norm_or_none(self):
    from matplotlib.colors import Normalize
    if self.vmin is None or self.vmax is None:
        return None
    return Normalize(self.vmin, self.vmax)


LegendSpec.norm_or_none = _norm_or_none  # type: ignore[attr-defined]
