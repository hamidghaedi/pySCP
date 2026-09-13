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

import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from .spec import HeatmapSpec, LegendSpec, PanelSpec

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


def _draw_block_track(ax, labels, positions, *, vertical: bool, colors=None):
    """One labelled rectangle per split block — ComplexHeatmap's ``anno_block``."""
    total = sum(len(p) for p in positions)
    start = 0
    for lab, pos in zip(labels, positions, strict=True):
        size = len(pos)
        c = (colors or {}).get(str(lab), "#DDDDDD")
        if vertical:
            ax.add_patch(Rectangle((0, start), 1, size, facecolor=c, edgecolor="none"))
            ax.text(0.5, start + size / 2, str(lab), ha="center", va="center",
                    rotation=90, fontsize=7)
        else:
            ax.add_patch(Rectangle((start, 0), size, 1, facecolor=c, edgecolor="none"))
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
    """Called-out row labels with leader lines, repelled so they never collide."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, n_rows)
    ax.invert_yaxis()
    _blank(ax)
    if not len(at):
        return
    at = np.asarray(at, dtype=float) + 0.5
    gap = max(n_rows / max(len(at), 1) * 0.18, fontsize / 72 / max(ax.get_figure().get_figheight(), 1) * n_rows)
    placed = _repel(at, gap, 0.0, float(n_rows))
    for y_true, y_lab, lab in zip(at, placed, labels, strict=True):
        ax.plot([0.0, 0.35, 0.6], [y_true, y_true, y_lab], color="black", linewidth=0.5,
                solid_joinstyle="miter", clip_on=False)
        ax.text(0.68, y_lab, str(lab), ha="left", va="center", fontsize=fontsize, clip_on=False)


def _draw_layers(ax, panel: PanelSpec, layers, rows: np.ndarray, cols: np.ndarray, dpi_scale: float):
    """Body overlays, in SCP's fixed order: whiteout, bg, reticle, dot."""
    order = {"whiteout": 0, "bg": 1, "reticle": 2, "dot": 3, "violin": 4, "custom": 5}
    for layer in sorted(layers, key=lambda x: order.get(x.kind, 9)):
        if layer.kind == "whiteout":
            ax.add_patch(Rectangle((0, 0), len(cols), len(rows), facecolor="white",
                                   edgecolor="none", zorder=2))
        elif layer.kind == "bg":
            sub = panel.matrix[np.ix_(rows, cols)]
            ax.pcolormesh(np.ma.masked_invalid(sub), cmap=panel.cmap, norm=panel.norm,
                          alpha=layer.alpha, zorder=1)
        elif layer.kind == "reticle":
            for i in range(len(cols) + 1):
                ax.axvline(i, color=layer.color, linewidth=0.3, zorder=3)
            for j in range(len(rows) + 1):
                ax.axhline(j, color=layer.color, linewidth=0.3, zorder=3)
        elif layer.kind == "dot":
            if layer.size_matrix is None:
                continue
            frac = layer.size_matrix[np.ix_(rows, cols)]
            val = panel.matrix[np.ix_(rows, cols)]
            yy, xx = np.mgrid[0:len(rows), 0:len(cols)]
            # area in points^2 for a disc whose diameter is base_size_in at 100%
            area = (layer.base_size_in * 72.0) ** 2 * np.clip(frac, 0, 1)
            sm = ScalarMappable(norm=panel.norm, cmap=panel.cmap)
            ax.scatter(xx.ravel() + 0.5, yy.ravel() + 0.5, s=area.ravel(),
                       c=sm.to_rgba(val.ravel()), edgecolor="none", zorder=4)
        elif layer.kind == "custom" and layer.draw is not None:
            layer.draw(ax, panel, rows, cols)


def _draw_legends(fig, ax, legends: list[LegendSpec]):
    """Stack every legend down one dedicated column."""
    _blank(ax)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    y = 1.0
    for spec in legends:
        if spec.kind == "categorical" and spec.mapping:
            handles = [Patch(facecolor=color, edgecolor="none", label=lab)
                       for lab, color in spec.mapping.items()]
            leg = ax.legend(handles=handles, title=spec.title, loc="upper left",
                            bbox_to_anchor=(0, y), frameon=False, fontsize=7,
                            title_fontsize=7.5, handlelength=1.0, handleheight=1.0,
                            borderpad=0, labelspacing=0.35)
            leg._legend_box.align = "left"
            ax.add_artist(leg)
            y -= 0.06 + 0.035 * len(handles)
        elif spec.kind == "continuous":
            cax = ax.inset_axes([0.06, max(y - 0.17, 0.0), 0.22, 0.15])
            cb = fig.colorbar(ScalarMappable(norm=spec.norm_or_none(), cmap=spec.cmap), cax=cax)
            cb.outline.set_linewidth(0.5)
            cb.ax.tick_params(labelsize=6.5, length=2)
            cax.set_title(spec.title, fontsize=7.5, loc="left", pad=4)
            y -= 0.25
        elif spec.kind == "size":
            handles = [
                Line2D([], [], marker="o", linestyle="none", markerfacecolor="#7F7F7F",
                       markeredgecolor="none", markersize=3 + 7 * b, label=f"{int(b * 100)}%")
                for b in spec.breaks
            ]
            leg = ax.legend(handles=handles, title=spec.title, loc="upper left",
                            bbox_to_anchor=(0, y), frameon=False, fontsize=7,
                            title_fontsize=7.5, borderpad=0, labelspacing=0.5)
            leg._legend_box.align = "left"
            ax.add_artist(leg)
            y -= 0.08 + 0.045 * len(handles)


def render(spec: HeatmapSpec, *, dpi: float = 110.0) -> Figure:
    """Draw a :class:`HeatmapSpec` and return the figure.

    The layout is one top-level GridSpec whose ratios are inches; splits become
    a nested GridSpec shared by the body and every track running along the same
    axis, which is how slices stay aligned.
    """
    import matplotlib.pyplot as plt

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
    plt.close(probe)
    if spec.legends:
        spec.measured.setdefault("legend_w", 1.5)

    fig_w, fig_h = spec.figsize()
    fig_w += mark_w
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)

    # --- top-level grid, in inches ----------------------------------------
    title_h = spec.measured.get("titles_h", 0.25)
    top_h = max((sum(t.size_in for t in p.top_tracks) for p in spec.panels), default=0.0)
    coldend_h = max((d.size_in for d in spec.col_dends.values()), default=0.0)
    body_h = spec.body_height_in if spec.body_height_in is not None else 1.0
    names_h = spec.measured.get("column_names_h", 0.0) if spec.show_column_names else 0.0
    height_ratios = [title_h, top_h, coldend_h, body_h, names_h]

    left_w = sum(t.size_in for t in spec.left_tracks if t.kind != "mark")
    rowdend_w = spec.row_dend.size_in if spec.row_dend else 0.0
    right_w = sum(t.size_in for t in spec.right_tracks)
    names_w = spec.measured.get("row_names_w", 0.0) if spec.show_row_names else 0.0
    legend_w = spec.measured.get("legend_w", 0.0) if spec.legends else 0.0
    bodies = [p.body_size_in if p.body_size_in is not None else 1.0 for p in spec.panels]
    width_ratios = [mark_w, rowdend_w, left_w] + bodies + [right_w, names_w, legend_w]

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
                _draw_simple_track(tax, np.asarray(track.values)[cols], vertical=False,
                                   colors=track.colors, cmap=track.cmap, norm=track.norm,
                                   na_color=track.na_color, border=track.border)
                if track.show_name and ci == len(col_blocks) - 1:
                    tax.text(1.02, 0.5, track.name, transform=tax.transAxes, ha="left",
                             va="center", fontsize=6.5)

    # --- left tracks, sharing the row subdivision --------------------------
    for track in spec.left_tracks:
        if track.kind == "mark":
            ax = fig.add_subplot(gs[BODY_ROW, 0])
            _draw_mark_track(ax, track.at or [], track.labels or [], n_rows_total)
            continue
        sub = row_subgrid(gs[BODY_ROW, 2])
        for ri, (lab, rows) in enumerate(row_blocks):
            ax = fig.add_subplot(sub[ri, 0])
            if track.kind == "block":
                _draw_block_track(ax, [lab], [rows], vertical=True, colors=track.colors)
            else:
                _draw_simple_track(ax, np.asarray(track.values)[rows], vertical=True,
                                   colors=track.colors, cmap=track.cmap, norm=track.norm,
                                   na_color=track.na_color, border=track.border)

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
    if spec.row_dend is not None:
        from scipy.cluster.hierarchy import dendrogram
        ax = fig.add_subplot(gs[BODY_ROW, 1])
        dendrogram(spec.row_dend.linkage, ax=ax, orientation="left", no_labels=True,
                   link_color_func=lambda _k: "black")
        ax.invert_yaxis()
        _blank(ax)
    for pi, panel in enumerate(spec.panels):
        d = spec.col_dends.get(panel.name)
        if d is None:
            continue
        from scipy.cluster.hierarchy import dendrogram
        ax = fig.add_subplot(gs[2, FIRST_BODY_COL + pi])
        dendrogram(d.linkage, ax=ax, orientation="top", no_labels=True,
                   link_color_func=lambda _k: "black")
        _blank(ax)

    # --- legends -----------------------------------------------------------
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
