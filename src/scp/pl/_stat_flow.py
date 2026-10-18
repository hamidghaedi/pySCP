"""Flow and set views of :func:`scp.pl.cell_stat_plot`: sankey, chord, venn, upset.

These four ``plot_type``\\ s share nothing with the bar/pie family except the
input: several categorical ``obs`` columns (``stat_by``) read row-wise, one
panel per ``split_by`` level, ``group_by`` ignored. R reaches for three
different packages (SCP's vendored ``ggsankey``, ``circlize``,
``ggVennDiagram`` and ``ggupset``); here each is a pure layout function that returns plain
tables, plus a renderer that draws those tables into a given ``Axes``. The
layout functions are what the parity tests pin against R.

Ported from SCP (Hao Zhang, GPL-3), https://github.com/zhanghao-njmu/SCP,
``SCP-plot.R`` ``StatPlot`` (L4564), branches at L4981 (venn), L5044 (upset),
L5072 (sankey) and L5115 (chord). Spec: ``docs/porting_briefs/stat_plots.md`` §2.3.
"""

from __future__ import annotations

import base64
import math
import warnings
from collections.abc import Mapping, Sequence
from itertools import combinations
from typing import cast

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from matplotlib.patches import Patch, PathPatch, Polygon, Rectangle, Wedge
from matplotlib.path import Path
from matplotlib.textpath import TextPath

from ..colors import blendcolors
from ..fetch import as_ordered_categorical
from ..layout import LegendColumn, PanelGrid
from ..palettes import discrete_palette
from ..theme import LegendPosition, Theme, halo, theme_blank

__all__ = [
    "sankey_layout",
    "chord_layout",
    "venn_regions",
    "upset_layout",
    "resolve_stat_level",
    "flow_plot",
    "upset_plot",
]

_SANKEY_WIDTH = 0.1  # geom_sankey(width = .1)
_SANKEY_SMOOTH = 8  # geom_sankey(smooth = 8)


# --------------------------------------------------------------------------- #
# shared input handling
# --------------------------------------------------------------------------- #
def _r_string(s: pd.Series) -> pd.Series:
    """Booleans spelt the way R prints them, so ``stat_level="TRUE"`` matches."""
    if pd.api.types.is_bool_dtype(s.dtype) or (
        s.dtype == object and s.dropna().map(type).isin([bool, np.bool_]).all() and s.notna().any()
    ):
        return s.map({True: "TRUE", False: "FALSE"})
    return s


def _categorical(s: pd.Series, *, show_na: bool) -> pd.Series:
    """First-appearance factor of one ``stat_by`` column, NA as a real level."""
    return as_ordered_categorical(_r_string(s), show_na=show_na)


def _union_levels(data: pd.DataFrame, columns: Sequence[str]) -> list[str]:
    """``unique(unlist(lapply(dat_all[, stat.by], levels)))`` -- column order first."""
    out: list[str] = []
    for c in columns:
        for lv in _categorical(data[c], show_na=False).cat.categories:
            if str(lv) not in out:
                out.append(str(lv))
    return out


def _flow_colors(data, columns, palette, palcolor, na_color) -> dict[str, str]:
    """Node colors shared across columns, as SCP-plot.R:5073 / 5116 build them.

    ``palette_scp(c(<union of levels>, NA), NA_keep = TRUE)``: one color per
    distinct level name over every ``stat_by`` column, then ``"NA"``. A level
    name appearing in two columns therefore gets one color in both.
    """
    colors = discrete_palette(_union_levels(data, columns), palette=palette, palcolor=palcolor)
    colors["NA"] = na_color
    return colors


def resolve_stat_level(data: pd.DataFrame, stat_cols: Sequence[str], stat_level) -> dict[str, list[str]]:
    """``stat_level`` -> the levels that count as set membership, per column.

    R (SCP-plot.R:4649-4667): ``NULL`` takes each column's first level (for a
    logical column, which has no levels, ``sort(unique(x))[1]``, i.e.
    ``FALSE``); a single value is broadcast; an unnamed vector of the same
    length as ``stat_by`` is matched positionally; a named list by name.
    Booleans compare as ``"TRUE"``/``"FALSE"``, R's spelling.
    """
    stat_cols = list(stat_cols)
    if stat_level is None:
        out = {}
        for c in stat_cols:
            s = _r_string(data[c])
            if isinstance(data[c].dtype, pd.CategoricalDtype):
                first = str(data[c].cat.categories[0])
            elif set(s.dropna().unique()) <= {"TRUE", "FALSE"}:
                first = sorted(s.dropna().unique())[0]
            else:
                first = str(as_ordered_categorical(s).cat.categories[0])
            out[c] = [first]
        warnings.warn(
            "`stat_level` not given; using "
            + ", ".join(f"{k}={v[0]}" for k, v in out.items()),
            stacklevel=4,
        )
        return out

    def _norm(v) -> list[str]:
        vals = [v] if isinstance(v, (str, bool, np.bool_)) or not isinstance(v, Sequence) else list(v)
        return ["TRUE" if x is True else "FALSE" if x is False else str(x) for x in vals]

    if isinstance(stat_level, Mapping):
        missing = [c for c in stat_cols if c not in stat_level]
        if missing:
            raise ValueError(f"`stat_level` has no entry for {missing}")
        return {c: _norm(stat_level[c]) for c in stat_cols}
    if isinstance(stat_level, (str, bool, np.bool_)) or not isinstance(stat_level, Sequence):
        return {c: _norm(stat_level) for c in stat_cols}
    levels = list(stat_level)
    if len(levels) == 1:
        return {c: _norm(levels[0]) for c in stat_cols}
    if len(levels) != len(stat_cols):
        raise ValueError("`stat_level` must be of length 1 or the same length as `stat_by`")
    return {c: _norm(v) for c, v in zip(stat_cols, levels, strict=True)}


def _membership(data: pd.DataFrame, stat_cols, levels: dict[str, list[str]]) -> pd.DataFrame:
    """``meta.data[[i]] <- meta.data[[i]] %in% stat_level[[i]]``; NA is never in."""
    return pd.DataFrame(
        {c: _r_string(data[c]).astype("string").isin(levels[c]).fillna(False).to_numpy(bool)
         for c in stat_cols},
        index=data.index,
    )


# --------------------------------------------------------------------------- #
# sankey
# --------------------------------------------------------------------------- #
def sankey_layout(
    data: pd.DataFrame,
    columns: Sequence[str],
    node_order: Sequence[str],
    *,
    width: float = _SANKEY_WIDTH,
    space: float | None = None,
    flow_order: str = "node",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Node rectangles and flow ribbons of a ``geom_sankey(type = "sankey")``.

    ``data`` holds one row per cell and the ``columns`` as strings (missing
    values already recoded to ``"NA"``). ``node_order`` lists node names
    top to bottom -- SCP passes ``names(colors)``, so the first level of the
    first column sits at the top.

    The arithmetic is ggsankey's ``StatSankeyNode``/``StatSankeyFlow``
    (``SCP/R/ggsankey.R``): within column ``i`` (drawn at ``x = i + 1``) the
    nodes are stacked bottom-up with ``ymax = cumsum(freq) + (k - 1) * space``
    and then centred on zero; ``space`` defaults to the largest
    ``total / n_nodes / 4`` over columns. Between adjacent columns each
    observed (source, target) pair is one flow whose width is its cell count;
    incoming flows are stacked from the top of the target in source order.

    ``flow_order`` sets the order of the *outgoing* flows inside a source node.
    ``"node"`` stacks them from the top in target order, so ribbons leaving a
    node do not cross. ``"r"`` reproduces SCP exactly: it turns ``node`` into a
    factor but leaves ``next_node`` a character vector, so dplyr sorts the
    outgoing flows *alphabetically* (C locale) and ribbons cross whenever the
    level order is not alphabetical. That is an SCP bug, kept here only so the
    parity test can confirm the rest of the arithmetic to the last digit.

    Returns ``(nodes, flows)``. ``nodes``: ``column, node, x, xmin, xmax, ymin,
    ymax, freq``. ``flows``: ``source_column, source, target_column, target,
    value, x_start, x_end, start_ymin, start_ymax, end_ymin, end_ymax``.
    """
    if flow_order not in ("node", "r"):
        raise ValueError("flow_order must be 'node' or 'r'")
    columns = list(columns)
    rank = {str(n): i for i, n in enumerate(node_order)}

    counts = []
    for c in columns:
        vc = data[c].astype(str).value_counts()
        unknown = [n for n in vc.index if n not in rank]
        if unknown:
            raise ValueError(f"nodes {unknown} of {c!r} are missing from `node_order`")
        counts.append(vc.reindex(sorted(vc.index, key=rank.__getitem__)))

    if space is None:
        space = max(float(vc.sum()) / len(vc) / 4 for vc in counts if len(vc))

    node_rows = []
    pos: dict[tuple[int, str], tuple[float, float]] = {}
    for i, (c, vc) in enumerate(zip(columns, counts, strict=True)):
        bottom_up = list(vc.index)[::-1]
        freq = vc[bottom_up].to_numpy(float)
        ymax = np.cumsum(freq) + np.arange(len(freq)) * space
        ymin = ymax - freq
        shift = ymax.max() / 2 if len(ymax) else 0.0
        for name, lo, hi, f in zip(bottom_up, ymin - shift, ymax - shift, freq, strict=True):
            pos[(i, name)] = (lo, hi)
            node_rows.append((c, name, i + 1.0, i + 1 - width / 2, i + 1 + width / 2, lo, hi, f))
    nodes = pd.DataFrame(node_rows, columns=["column", "node", "x", "xmin", "xmax",
                                             "ymin", "ymax", "freq"])
    # ggplot draws rows in the stat's order: by column, then bottom-up
    nodes = nodes.reset_index(drop=True)

    flow_rows = []
    for i in range(len(columns) - 1):
        a, b = columns[i], columns[i + 1]
        pair = data.groupby([data[a].astype(str), data[b].astype(str)], sort=False).size()
        pair = pair[pair > 0]
        out_cum: dict[str, float] = {}
        in_cum: dict[str, float] = {}
        starts: dict[tuple[str, str], tuple[float, float]] = {}
        for src in counts[i].index:  # top to bottom
            targets = [t for (s, t) in pair.index if s == src]
            if flow_order == "r":
                ascending = sorted(targets)
            else:
                ascending = sorted(targets, key=lambda t: -rank[t])
            top = pos[(i, src)][1]
            for t in reversed(ascending):  # stacked from the top
                v = float(pair[(src, t)])
                hi = top - out_cum.get(src, 0.0)
                starts[(src, t)] = (hi - v, hi)
                out_cum[src] = out_cum.get(src, 0.0) + v
        for src in counts[i].index:
            for t in counts[i + 1].index:
                if (src, t) not in starts:
                    continue
                v = float(pair[(src, t)])
                hi = pos[(i + 1, t)][1] - in_cum.get(t, 0.0)
                in_cum[t] = in_cum.get(t, 0.0) + v
                s_lo, s_hi = starts[(src, t)]
                flow_rows.append((a, src, b, t, v, i + 1 + width / 2, i + 2 - width / 2,
                                  s_lo, s_hi, hi - v, hi))
    flows = pd.DataFrame(flow_rows, columns=["source_column", "source", "target_column",
                                             "target", "value", "x_start", "x_end",
                                             "start_ymin", "start_ymax", "end_ymin", "end_ymax"])
    return nodes, flows


def _sigmoid(x0: float, x1: float, y0: float, y1: float, smooth: float, n: int = 100):
    """ggsankey's ``sigmoid()``: a logistic over ``[-smooth, smooth]``."""
    t = np.linspace(-smooth, smooth, n)
    s = np.exp(t) / (np.exp(t) + 1)
    return (t + smooth) / (2 * smooth) * (x1 - x0) + x0, s * (y1 - y0) + y0


def _draw_sankey(ax: Axes, nodes, flows, colors, columns, *, alpha, theme: Theme) -> None:
    """Flows first (layer 1 of geom_sankey), nodes on top (layer 2)."""
    for f in flows.itertuples(index=False):
        x_up, y_up = _sigmoid(f.x_start, f.x_end, f.start_ymax, f.end_ymax, _SANKEY_SMOOTH)
        x_dn, y_dn = _sigmoid(f.x_end, f.x_start, f.end_ymin, f.start_ymin, _SANKEY_SMOOTH)
        ax.add_patch(Polygon(np.column_stack([np.r_[x_up, x_dn], np.r_[y_up, y_dn]]),
                             closed=True, facecolor=to_rgba(colors[f.source], alpha),
                             edgecolor="black", linewidth=0.5, zorder=1))
    for nd in nodes.itertuples(index=False):
        ax.add_patch(Rectangle((nd.xmin, nd.ymin), nd.xmax - nd.xmin, nd.ymax - nd.ymin,
                               facecolor=colors[nd.node], edgecolor="black", linewidth=0.5,
                               zorder=2))
    k = len(columns)
    # scale_x_discrete(expand = c(0, 0.2)); y keeps ggplot's default 5% padding
    ax.set_xlim(1 - 0.2, k + 0.2)
    if len(nodes):
        lo, hi = float(nodes.ymin.min()), float(nodes.ymax.max())
        pad = 0.05 * (hi - lo) if hi > lo else 1.0
        ax.set_ylim(lo - pad, hi + pad)
    # theme_void() + theme(axis.text.x = element_text()): the column names only
    ax.set_xticks(np.arange(1, k + 1))
    ax.set_xticklabels(columns, fontsize=theme.size("axis.text"))
    ax.tick_params(axis="x", length=0, labelbottom=True)
    ax.set_yticks([])


# --------------------------------------------------------------------------- #
# chord
# --------------------------------------------------------------------------- #
def chord_layout(
    table: pd.DataFrame,
    *,
    small_gap: float = 1.0,
    big_gap: float = 10.0,
    reduce: float = 1e-5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sector angles and link positions of ``circlize::chordDiagram(<matrix>)``.

    ``table`` is the contingency table SCP passes (rows: levels of the first
    ``stat_by`` column, columns: the second), zero rows and columns included.
    This follows ``chordDiagramFromMatrix`` -> ``chordDiagramFromDataFrame``
    with SCP's arguments and circlize's defaults (circlize 0.4.18):

    * rows and columns whose totals vanish are dropped (iteratively), as are
      sectors below ``reduce`` of the grand total;
    * sectors are ``union(rownames, colnames)`` -- **a name present on both
      sides becomes one sector** holding both its row and its column links,
      and a diagonal entry is then a link from the sector to itself;
    * a sector's width is its total (row sum plus column sum, a self link
      counted at both ends), laid out clockwise from the start angle with a
      1-degree gap after each sector;
    * when the two sides share no name, circlize's two-group layout applies:
      a 10-degree gap after the last row sector and after the last column
      sector, and the start angle chosen so the row block is centred on the
      vertical axis, i.e. ``start = -(180 - d1) / 2`` with ``d1`` the row
      block's extent;
    * inside a sector the row links come first, ordered so that the link to
      the sector furthest round the circle (clockwise from this one) starts
      at the sector's beginning; column links follow, ordered the same way
      by their row sector. This is ``link.sort = "default"``.

    Returns ``(sectors, links)``. ``sectors``: ``name, start_degree,
    end_degree, xmax`` with degrees in circlize's convention (counter-clockwise
    from 3 o'clock, shifted into ``[0, 720)`` the way circlize stores them,
    start > end). ``links``: ``rn, cn, value, x1, x2`` where ``x1``/``x2`` are
    the *end* positions of the link along its row/column sector, in counts
    (the link occupies ``[x - value, x]``); zero-valued pairs are kept, as
    circlize keeps them, and drawn by nobody.
    """
    m = table.to_numpy(dtype=float).copy()
    m[np.isnan(m)] = 0.0
    rows = [str(r) for r in table.index]
    cols = [str(c) for c in table.columns]

    def _xlim(rows, cols, m):
        nn = list(dict.fromkeys(rows + cols))
        x = dict.fromkeys(nn, 0.0)
        for r, v in zip(rows, np.abs(m).sum(axis=1), strict=True):
            x[r] += v
        for c, v in zip(cols, np.abs(m).sum(axis=0), strict=True):
            x[c] += v
        for s in set(rows) & set(cols):
            x[s] -= abs(m[rows.index(s), cols.index(s)])
        return x

    x = _xlim(rows, cols, m)
    total = sum(x.values())
    keep = {n for n, v in x.items() if total and v / total >= reduce}
    ri = [i for i, r in enumerate(rows) if r in keep]
    ci = [j for j, c in enumerate(cols) if c in keep]
    while True:
        sub = m[np.ix_(ri, ci)]
        r_zero = [ri[k] for k, v in enumerate(sub.sum(axis=1)) if v == 0]
        c_zero = [ci[k] for k, v in enumerate(sub.sum(axis=0)) if v == 0]
        if not r_zero and not c_zero:
            break
        ri = [i for i in ri if i not in r_zero]
        ci = [j for j in ci if j not in c_zero]
    m = m[np.ix_(ri, ci)]
    rows = [rows[i] for i in ri]
    cols = [cols[j] for j in ci]

    # mat2df: column-major, zero entries included
    df = pd.DataFrame(
        [(rows[i], cols[j], m[i, j]) for j in range(len(cols)) for i in range(len(rows))],
        columns=["rn", "cn", "value"],
    )
    cate = list(dict.fromkeys(list(df.rn) + list(df.cn)))

    # chordDiagramFromDataFrame's reduce loop (self.link = 2: a diagonal entry
    # counts at both of its ends)
    def _xsum(df, cate):
        xs = dict.fromkeys(cate, 0.0)
        for rn, cn, v in df[["rn", "cn", "value"]].itertuples(index=False):
            if rn == cn:
                xs[rn] += 2 * abs(v)
            else:
                xs[rn] += abs(v)
                xs[cn] += abs(v)
        return xs

    red = reduce
    while True:
        xs = _xsum(df, cate)
        tot = sum(xs.values())
        keep_names = [n for n in cate if tot and xs[n] / tot >= red]
        mask = df.rn.isin(keep_names) & df.cn.isin(keep_names)
        cate = [n for n in cate if n in keep_names]
        red = 1e-10
        if mask.all():
            break
        df = df[mask].reset_index(drop=True)
    xs = _xsum(df, cate)

    # gaps and start angle (circos.par defaults: gap.after = 1, start = 0)
    gaps = [1.0] * len(cate)
    start = 0.0
    if not set(df.rn) & set(df.cn) and len(cate):
        ind1 = [k for k, n in enumerate(cate) if n in set(df.rn)]
        ind2 = [k for k, n in enumerate(cate) if n in set(df.cn)]
        if max(ind1) < min(ind2) or max(ind2) < min(ind1):
            n1, n2 = df.rn.nunique(), df.cn.nunique()
            s1 = s2 = float(df.value.abs().sum())
            if cate[0] in set(df.cn):
                n1, n2 = n2, n1
            d1 = (360 - small_gap * (n1 + n2 - 2) - big_gap * 2) * (s1 / (s1 + s2)) \
                + small_gap * (n1 - 1)
            start = 0.0 - (180 - d1) / 2
            gaps = [small_gap] * (n1 - 1) + [big_gap] + [small_gap] * (n2 - 1) + [big_gap]

    unit = (360 - sum(gaps)) / sum(xs[n] for n in cate)
    sec_rows = []
    prev_end = start
    for k, n in enumerate(cate):
        s = start if k == 0 else prev_end - gaps[k - 1]
        e = s - xs[n] * unit
        sec_rows.append((n, s + 360, e + 360, xs[n]))
        prev_end = e
    sectors = pd.DataFrame(sec_rows, columns=["name", "start_degree", "end_degree", "xmax"])

    # link.sort = "default"
    pos_in = {n: k for k, n in enumerate(cate)}

    def _rotated_rank(anchor: str, other: str) -> int:
        return (pos_in[other] - pos_in[anchor]) % len(cate)

    x1 = np.zeros(len(df))
    x2 = np.zeros(len(df))
    sum_1: dict[str, float] = {}
    for nm in dict.fromkeys(df.rn):
        idx = np.flatnonzero(df.rn.to_numpy() == nm)
        order = sorted(idx, key=lambda i: _rotated_rank(nm, df.cn.iat[i]), reverse=True)
        x1[order] = np.cumsum(np.abs(df.value.to_numpy()[order]))
        sum_1[nm] = float(np.abs(df.value.to_numpy()[idx]).sum())
    for nm in dict.fromkeys(df.cn):
        idx = np.flatnonzero(df.cn.to_numpy() == nm)
        order = sorted(idx, key=lambda i: _rotated_rank(nm, df.rn.iat[i]), reverse=True)
        x2[order] = np.cumsum(np.abs(df.value.to_numpy()[order])) + sum_1.get(nm, 0.0)
    links = df.assign(x1=x1, x2=x2)
    return sectors, links


def _chord_theta(sectors: pd.DataFrame, name: str, x: float) -> float:
    s = sectors.set_index("name").loc[name]
    return float(s.start_degree - x / s.xmax * (s.start_degree - s.end_degree))


def _nice_step(degree_span: float, data_span: float, by_degree: float = 10.0) -> float:
    """circlize's default ``major.by``: ~10 degrees of data, one significant digit."""
    if degree_span <= 0 or data_span <= 0:
        return 0.0
    step = by_degree / degree_span * data_span
    digits = math.floor(math.log10(abs(step)))
    return float(round(step, -digits)) or step


def _arc(theta_from: float, theta_to: float, r: float, n_per_degree: float = 1.0):
    n = max(2, int(abs(theta_to - theta_from) * n_per_degree) + 1)
    t = np.deg2rad(np.linspace(theta_from, theta_to, n))
    return np.column_stack([r * np.cos(t), r * np.sin(t)])


def _text_width_pt(text: str, fontsize: float) -> float:
    """Advance width of ``text`` in points, from the glyph outlines -- no renderer."""
    if not text:
        return 0.0
    return float(TextPath((0, 0), text, size=fontsize).get_extents().width)


def _draw_chord(ax: Axes, sectors, links, colors, *, theme: Theme,
                side_in: float = 3.9) -> None:
    """Grid ring, axis ticks, sector names and the links, circlize-style.

    Radii follow the default ``annotationTrack = c("name", "grid", "axis")``:
    the grid ring sits just inside the name track and the links end just
    inside the grid. Links are filled with their row sector's color at
    ``transparency = 0.2`` and outlined in black (``link.border = 1``).
    Each link edge is a quadratic Bezier whose control point is the centre --
    what ``getQuadraticPoints(h.ratio = 0.5, w = 1)`` reduces to.

    The axis is ``circos.axis("top", labels.cex = 0.5)``: a major tick every
    ~10 degrees of data, four minor ticks between, and labels facing inside
    with ``niceFacing``.  ``labels.pos.adjust = TRUE`` is reproduced -- the
    first and last label of a sector are pushed inward until they no longer
    hang past its edge, which is what keeps one sector's last label off the
    next one's "0".  Two departures, both because circlize sizes text for a
    7-inch device and this panel is about half that (``side_in`` inches):

    * labels are thinned to every k-th major tick, the smallest k at which
      neighbouring labels clear each other (the ticks themselves all stay);
    * a sector name that would run into its neighbour's is turned radial
      (``facing = "clockwise"``, ``niceFacing``) instead of following the
      circle; the panel's measured decorations make room for it.
    """
    r_link, r_in, r_out = 0.79, 0.80, 0.86
    r_lab, r_name = r_out + 0.03, r_out + 0.12
    tick_fs, name_fs = theme.size("text") * 0.5, theme.size("text") * 0.8
    sec = sectors.set_index("name")
    if len(sectors):
        unit = float((sectors.start_degree - sectors.end_degree).sum() / sectors.xmax.sum())
        step = _nice_step(unit * float(sectors.xmax.max()), float(sectors.xmax.max()))
    else:
        unit, step = 0.0, 0.0

    # names: tangential where they fit beside their neighbours, else radial
    names = list(sec.sort_values("start_degree", ascending=False).index)
    width_pt = {n: _text_width_pt(str(n), name_fs) for n in names}
    lim = 1.25
    ppu = side_in * 72.0 / (2 * lim)  # points per canvas unit
    mids = {n: (sec.loc[n].start_degree + sec.loc[n].end_degree) / 2 for n in names}
    pad_pt = 2.0
    along = {n: np.degrees((width_pt[n] + pad_pt) / ppu / r_name) / 2 for n in names}
    # a radial name still takes its line height out of the arc
    across = np.degrees((name_fs * 1.2 + pad_pt) / ppu / r_name) / 2
    radial: set = set()

    def half(n) -> float:
        return across if n in radial else along[n]

    while len(names) > 1:
        worst, worst_over = None, 0.0
        for i, a in enumerate(names):
            b = names[(i + 1) % len(names)]
            if a == b or (a in radial and b in radial):
                continue
            gap = (mids[a] - mids[b]) % 360  # clockwise layout: a before b
            over = half(a) + half(b) - gap
            if over > worst_over:
                tangential = [n for n in (a, b) if n not in radial]
                worst, worst_over = max(tangential, key=along.__getitem__), over
        if worst is None:
            break
        radial.add(worst)

    for name, s in sec.iterrows():
        ax.add_patch(Wedge((0, 0), r_out, s.end_degree, s.start_degree, width=r_out - r_in,
                           facecolor=colors.get(name, "#BEBEBE"), edgecolor="none", zorder=2))
        if step > 0 and s.xmax > 0:
            deg_per_x = (s.start_degree - s.end_degree) / s.xmax
            major = np.arange(0.0, s.xmax + step * 1e-9, step)
            minor = [m0 + (m1 - m0) * k / 5 for m0, m1 in
                     zip(major, [*major[1:], major[-1] + step], strict=True)
                     for k in range(1, 5)]
            for tv, ln in [(m, 0.02) for m in major] + [(m, 0.01) for m in minor
                                                        if m <= s.xmax]:
                th = np.deg2rad(_chord_theta(sectors, name, tv))
                ax.plot([r_out * np.cos(th), (r_out + ln) * np.cos(th)],
                        [r_out * np.sin(th), (r_out + ln) * np.sin(th)],
                        color="black", linewidth=0.5, zorder=2)
            labels = [f"{tv:g}" for tv in major]
            widths = [_text_width_pt(t, tick_fs) for t in labels]
            spacing = np.radians(step * deg_per_x) * r_lab * ppu
            k = max(1, math.ceil((max(widths) + 2.0) / spacing)) if spacing > 0 else 1
            keep = list(range(0, len(major), k))
            for j, i in enumerate(keep):
                x = float(major[i])
                half_x = widths[i] / 2 / ppu / r_lab / np.radians(deg_per_x)
                if j == 0:                       # labels.pos.adjust
                    x += max(half_x - x, 0.0)
                if j == len(keep) - 1 and len(keep) > 1:
                    x -= max(half_x - (s.xmax - x), 0.0)
                _tangent_text(ax, _chord_theta(sectors, name, x), r_lab, labels[i],
                              fontsize=tick_fs)
            ax.add_patch(PathPatch(Path(_arc(s.end_degree, s.start_degree, r_out)),
                                   fill=False, edgecolor="black", linewidth=0.5, zorder=2))
        if name in radial:
            # starts just beyond the tick labels
            _radial_text(ax, mids[name], r_lab + (tick_fs * 1.2 + 2.0) / ppu, str(name),
                         fontsize=name_fs)
        else:
            _tangent_text(ax, mids[name], r_name, str(name), fontsize=name_fs)

    total = float(links.value.abs().sum()) or 1.0
    for ln in links.itertuples(index=False):
        if abs(ln.value) / total < 1e-6:
            continue
        path = _link_path(_chord_theta(sectors, ln.rn, ln.x1 - abs(ln.value)),
                          _chord_theta(sectors, ln.rn, ln.x1),
                          _chord_theta(sectors, ln.cn, ln.x2 - abs(ln.value)),
                          _chord_theta(sectors, ln.cn, ln.x2), r_link)
        ax.add_patch(PathPatch(path,
                               facecolor=to_rgba(colors.get(ln.rn, "#BEBEBE"), 0.8),
                               edgecolor="black", linewidth=0.75, zorder=1))
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal", adjustable="box")


def _link_path(t11: float, t12: float, t21: float, t22: float, r: float) -> Path:
    """One link's outline, following ``circos.link``'s two-arc branch.

    ``t11``/``t12`` are the angles of the first end's start/end, ``t21``/``t22``
    of the second's. When the two ends touch or overlap (a self link whose
    row and column parts are adjacent, say) circlize draws one arc spanning
    both and a single Bezier between its far ends; otherwise a ribbon: arc,
    Bezier across, arc, Bezier back, both curves bent through the centre.
    """

    def dd2(a: float, b: float) -> float:  # circlize:::degreeDiff2
        return (b - a) % 360

    def _loop(start: float, span: float) -> Path:
        arc = _arc(start, start + span, r)
        verts = np.vstack([arc, [[0.0, 0.0]], arc[:1], arc[:1]])
        codes = [Path.MOVETO] + [Path.LINETO] * (len(arc) - 1) + [Path.CURVE3, Path.CURVE3,
                                                                  Path.CLOSEPOLY]
        return Path(verts, codes)

    if dd2(t12, t21) <= dd2(t22, t21) + dd2(t12, t11) and dd2(t12, t21) >= dd2(t22, t11):
        return _loop(t12, dd2(t12, t21))
    if dd2(t22, t11) <= dd2(t12, t11) + dd2(t22, t21) and dd2(t22, t11) >= dd2(t12, t21):
        return _loop(t22, dd2(t22, t11))
    # clockwise layout: an end's start angle is its counter-clockwise edge
    arc_a = _arc(t12, t11, r)
    arc_b = _arc(t22, t21, r)
    verts = np.vstack([arc_a, [[0.0, 0.0]], arc_b, [[0.0, 0.0]], arc_a[:1], arc_a[:1]])
    codes = ([Path.MOVETO] + [Path.LINETO] * (len(arc_a) - 1) + [Path.CURVE3, Path.CURVE3]
             + [Path.LINETO] * (len(arc_b) - 1) + [Path.CURVE3, Path.CURVE3, Path.CLOSEPOLY])
    return Path(verts, codes)


def _tangent_text(ax: Axes, theta_deg: float, r: float, text: str, *, fontsize: float) -> None:
    """``facing = "inside", niceFacing = TRUE``: along the circle, never upside down."""
    th = theta_deg % 360
    upper = 0 <= th < 180
    rot = th - 90 if upper else th + 90
    t = np.deg2rad(th)
    ax.text(r * np.cos(t), r * np.sin(t), text, rotation=rot, rotation_mode="anchor",
            ha="center", va="bottom" if upper else "top", fontsize=fontsize, clip_on=False)


def _radial_text(ax: Axes, theta_deg: float, r: float, text: str, *, fontsize: float) -> None:
    """``facing = "clockwise", niceFacing = TRUE``: along the radius, reading
    outward on the right half and inward-upright on the left."""
    th = theta_deg % 360
    right = th <= 90 or th > 270
    t = np.deg2rad(th)
    ax.text(r * np.cos(t), r * np.sin(t), text, rotation=th if right else th - 180,
            rotation_mode="anchor", ha="left" if right else "right", va="center",
            fontsize=fontsize, clip_on=False)


# --------------------------------------------------------------------------- #
# venn
# --------------------------------------------------------------------------- #
def venn_regions(sets: Mapping[str, set]) -> pd.DataFrame:
    """Exclusive region counts, in ggVennDiagram's region order.

    One row per non-empty combination of sets -- all singletons, then all
    pairs, ... in ``combn`` order -- counting the items in *exactly* those sets
    (``process_region_data(specific = TRUE)``). ``id`` is the 1-based set
    indices joined by ``"/"``, ``name`` the set names joined likewise, ``pct``
    ``round(count / sum(count) * 100, 1)`` as SCP labels it.
    """
    names = list(sets)
    n = len(names)
    rows = []
    for k in range(1, n + 1):
        for idx in combinations(range(n), k):
            inside = set.intersection(*(set(sets[names[i]]) for i in idx))
            others = [set(sets[names[i]]) for i in range(n) if i not in idx]
            outside = set.union(*others) if others else set()
            rows.append(("/".join(str(i + 1) for i in idx),
                         "/".join(names[i] for i in idx), len(inside - outside), idx))
    out = pd.DataFrame(rows, columns=["id", "name", "count", "members"])
    tot = out["count"].sum()
    out["pct"] = (out["count"] / tot * 100).round(1) if tot else 0.0
    return out


def _venn_shape(n: int):
    from ._venn_shapes import SHAPES

    if n not in SHAPES:
        raise ValueError("venn diagrams are available for 1 to 7 sets")
    sh = SHAPES[n]
    xmin, xmax, ymin, ymax = sh["bbox"]
    edges = []
    for enc in sh["edges"]:
        q = np.frombuffer(base64.b64decode(enc), dtype="<u2").astype(float).reshape(-1, 2)
        edges.append(np.column_stack([xmin + q[:, 0] / 65535 * (xmax - xmin),
                                      ymin + q[:, 1] / 65535 * (ymax - ymin)]))
    return edges, np.asarray(sh["labels"], dtype=float)


def _fmt_r(x: float) -> str:
    """R's default print of a rounded number: ``7`` not ``7.0``."""
    return f"{x:.15g}"


def _draw_venn(ax: Axes, regions: pd.DataFrame, set_names, set_sizes, set_colors, *, alpha,
               label_size, theme: Theme, resolution: int = 500) -> None:
    """Regions filled by blended set colors, set outlines on top, both label layers.

    The outlines are ggVennDiagram's (``_venn_shapes``); region polygons are
    not stored, so each region is recovered as the grid cells inside exactly
    its sets and drawn with ``contourf`` (vector output; the boundary falls
    half a grid cell from the true edge, under the outline). Region labels sit
    at the region's area centroid, as SCP's ``sf::st_centroid`` places them.
    """
    edges, label_xy = _venn_shape(len(set_names))
    allpts = np.vstack(edges)
    x0, y0 = allpts.min(axis=0)
    x1, y1 = allpts.max(axis=0)
    span = max(x1 - x0, y1 - y0)
    nx = max(50, int(resolution * (x1 - x0) / span))
    ny = max(50, int(resolution * (y1 - y0) / span))
    gx = np.linspace(x0, x1, nx)
    gy = np.linspace(y0, y1, ny)
    X, Y = np.meshgrid(gx, gy)
    pts = np.column_stack([X.ravel(), Y.ravel()])
    code = np.zeros(len(pts), dtype=np.int64)
    for k, e in enumerate(edges):
        code |= Path(e).contains_points(pts).astype(np.int64) << k
    code = code.reshape(X.shape)

    for reg in regions.itertuples(index=False):
        bits = sum(1 << i for i in reg.members)
        mask = code == bits
        if not mask.any():
            continue
        color = blendcolors([set_colors[set_names[i]] for i in reg.members], mode="blend")
        rr, cc = np.nonzero(mask)
        r0, r1 = max(rr.min() - 1, 0), min(rr.max() + 2, ny)
        c0, c1 = max(cc.min() - 1, 0), min(cc.max() + 2, nx)
        ax.contourf(X[r0:r1, c0:c1], Y[r0:r1, c0:c1], mask[r0:r1, c0:c1].astype(float),
                    levels=[0.5, 1.5], colors=[color], alpha=alpha, antialiased=True,
                    zorder=1)
        cx, cy = X[mask].mean(), Y[mask].mean()
        t = ax.text(cx, cy, f"{int(reg.count)}\n{_fmt_r(reg.pct)}%", ha="center", va="center",
                    fontsize=label_size * 2, color="black", zorder=4)
        halo(t, foreground="white", radius=0.1)

    for k, e in enumerate(edges):
        ax.plot(np.r_[e[:, 0], e[:1, 0]], np.r_[e[:, 1], e[:1, 1]],
                color=set_colors[set_names[k]], linewidth=1.0, zorder=3)
    for k, name in enumerate(set_names):
        t = ax.text(label_xy[k, 0], label_xy[k, 1], f"{name}\n({set_sizes[name]})",
                    ha="center", va="center", fontsize=(label_size + 0.5) * 2,
                    fontweight="bold", color="black", zorder=5)
        halo(t, foreground="white", radius=0.1)

    lo = np.minimum(allpts.min(axis=0), label_xy.min(axis=0))
    hi = np.maximum(allpts.max(axis=0), label_xy.max(axis=0))
    # coord_sf is equal-aspect; square limits on the square panel keep it so
    centre = (lo + hi) / 2
    half = 0.54 * float((hi - lo).max())
    ax.set_xlim(centre[0] - half, centre[0] + half)
    ax.set_ylim(centre[1] - half, centre[1] + half)
    ax.set_aspect("equal", adjustable="box")


# --------------------------------------------------------------------------- #
# the figure
# --------------------------------------------------------------------------- #
def flow_plot(
    data: pd.DataFrame,
    stat_cols: Sequence[str],
    plot_type: str,
    *,
    split_by: str | None,
    stat_level,
    palette: str,
    palcolor,
    alpha: float,
    na_color: str,
    label_size: float,
    panel_size: tuple[float, float],
    legend_position: str,
    nrow: int | None,
    ncol: int | None,
    byrow: bool,
) -> Figure:
    """``StatPlot``'s sankey / chord / venn branch (SCP-plot.R:4960-5141).

    One panel per ``split_by`` level, ``group_by`` already discarded by the
    caller. As in R, ``na_stat`` and ``label`` have no effect here: sankey and
    chord always keep missing values as an ``"NA"`` node/sector, neither draws
    value labels, and venn always labels every region; chord ignores
    ``alpha`` too (circlize is called with a fixed ``transparency = 0.2``),
    while sankey applies it to the ribbons and venn to the region fills.
    Deliberate divergences from SCP, each a fix:

    * sankey ribbons leaving a node are stacked in target order, not
      alphabetically (see :func:`sankey_layout`), so they do not cross;
    * sankey and chord panels are titled with their split level when
      ``split_by`` is given -- R leaves them unlabelled, so a split figure
      cannot be read;
    * chord treats a row label ``NA`` like a column one (R renames only the
      column ``NA``, leaving a missing sector name), and always reads the table
      as a matrix (``chordDiagram`` silently reinterprets a two-column table
      with more than ten rows as an edge list);
    * the chord grid honours ``nrow`` *or* ``ncol`` alone (R computes the
      missing one as ``ceiling(sqrt(<the other>))``, which can drop panels).
    """
    stat_cols = list(stat_cols)
    if split_by is None:
        split_levels: list[str | None] = [None]
    else:
        scat = as_ordered_categorical(data[split_by])
        split_levels = [str(x) for x in scat.cat.categories]
        data = data.assign(**{split_by: np.asarray(scat).astype(str)})
    keys = [str(s) for s in split_levels]

    th = theme_blank(add_coord=False)
    legend = LegendColumn(
        position=cast(LegendPosition, legend_position if plot_type == "sankey" else "none"))

    if plot_type == "chord" and nrow is None and ncol is None:
        nrow = math.ceil(math.sqrt(len(keys)))
        ncol = math.ceil(len(keys) / nrow)
    # chord and venn are drawn equal-aspect on a square panel. circlize sizes
    # its text for a 7-inch device; at the default 2.6-inch panel the sector
    # names would collide, so the chord panel is half as large again.
    side = panel_size[0] * (1.5 if plot_type == "chord" else 1.0)
    size = panel_size if plot_type == "sankey" else (side, side)
    pg = PanelGrid(len(keys), panel_size=size, nrow=nrow, ncol=ncol, byrow=byrow, keys=keys,
                   theme=th, legend=legend)

    if plot_type in ("sankey", "chord"):
        colors = _flow_colors(data, stat_cols, palette, palcolor, na_color)
        cats = {c: _categorical(data[c], show_na=True) for c in stat_cols}
        frame = pd.DataFrame({c: np.asarray(cats[c]).astype(str) for c in stat_cols},
                             index=data.index)
        if split_by is not None:
            frame[split_by] = data[split_by].to_numpy()
    else:
        levels = resolve_stat_level(data, stat_cols, stat_level)
        members = _membership(data, stat_cols, levels)
        set_colors = discrete_palette(stat_cols, palette=palette, palcolor=palcolor)

    for sp in split_levels:
        ax = pg.axes[str(sp)]
        rows = slice(None) if sp is None else (data[split_by].to_numpy() == sp)
        if plot_type == "sankey":
            block = frame.loc[rows]
            nodes, flows = sankey_layout(block, stat_cols, list(colors))
            _draw_sankey(ax, nodes, flows, colors, stat_cols, alpha=alpha, theme=th)
        elif plot_type == "chord":
            block = frame.loc[rows]
            a, b = stat_cols
            r_lv = [str(x) for x in cats[a].cat.categories]
            c_lv = [str(x) for x in cats[b].cat.categories]
            # table(..., useNA = "ifany"): the NA level only where NA occurs
            r_lv = [x for x in r_lv if x != "NA" or (block[a] == "NA").any()]
            c_lv = [x for x in c_lv if x != "NA" or (block[b] == "NA").any()]
            tab = pd.crosstab(block[a], block[b]).reindex(index=r_lv, columns=c_lv,
                                                          fill_value=0)
            sectors, links = chord_layout(tab)
            _draw_chord(ax, sectors, links, colors, theme=th, side_in=side)
        else:
            mem = members.loc[rows]
            sets = {c: set(mem.index[mem[c].to_numpy()]) for c in stat_cols}
            regions = venn_regions(sets)
            _draw_venn(ax, regions, stat_cols, {c: len(s) for c, s in sets.items()},
                       set_colors, alpha=alpha, label_size=label_size, theme=th)
            if sp is not None:
                # labs(x = sp): theme_blank hides axis labels, so set it as text
                ax.text(0.5, -0.02, str(sp), transform=ax.transAxes, ha="center", va="top",
                        fontsize=th.size("axis.title"))
        if sp is not None and plot_type != "venn":
            ax.set_title(str(sp), fontsize=th.size("plot.title"))

    if plot_type == "sankey" and legend_position != "none":
        for c in stat_cols:
            lvls = [str(x) for x in _categorical(data[c], show_na=False).cat.categories]
            legend.add(c, [Patch(facecolor=colors[lv], edgecolor="black", label=lv)
                           for lv in lvls])
    return pg.finish()


# --------------------------------------------------------------------------- #
# upset
# --------------------------------------------------------------------------- #
#: ggupset joins the set names of an intersection with this before tabling it
_UPSET_SEP = "-__-__-_-"
_UPSET_N = 20  # scale_x_upset(n_intersections = 20)


def upset_layout(members: pd.DataFrame, stat_cols: Sequence[str],
                 n_intersections: int = _UPSET_N) -> tuple[pd.DataFrame, list[str]]:
    """Bars and matrix rows of SCP's upset plot, as ``ggupset`` orders them.

    ``members`` is the binarised cell x set table.  Cells in no set are dropped
    (SCP-plot.R:5047).  Each cell's intersection is the *sorted* list of its
    set names; ``scale_x_upset`` tables the intersections (``table()`` sorts
    the names, C collation as in the reference R session) and orders them by
    ``sort(decreasing = TRUE)`` — a stable radix sort, so equal counts keep
    that alphabetical order — keeping the first ``n_intersections``.  The
    matrix lists the ``stat_by`` columns that occur in a shown intersection,
    in ``stat_by`` order from the top.

    Returns ``(bars, sets)``: ``bars`` has one row per bar, left to right, with
    the ``intersection`` (a tuple of set names) and its ``count``.
    """
    stat_cols = list(stat_cols)
    m = members[stat_cols].to_numpy(bool)
    keys = [_UPSET_SEP.join(sorted(c for c, on in zip(stat_cols, row, strict=True) if on))
            for row in m if row.any()]
    counts = pd.Series(keys, dtype=object).value_counts()
    order = sorted(counts.index, key=lambda k: (-int(counts[k]), k))[:n_intersections]
    bars = pd.DataFrame({"intersection": [tuple(k.split(_UPSET_SEP)) for k in order],
                         "count": [int(counts[k]) for k in order]})
    shown = {s for t in bars["intersection"] for s in t}
    return bars, [c for c in stat_cols if c in shown]


def _draw_upset(bar_ax: Axes, mat_ax: Axes, bars: pd.DataFrame, sets: list[str], *,
                label_size: float, theme: Theme, bar_h: float = 0.0) -> None:
    """One upset block: count bars over the combination matrix."""
    from matplotlib.colors import LinearSegmentedColormap, Normalize

    from ..palettes import continuous_palette

    pt = 72.27 / 25.4  # ggplot .pt
    n = len(bars)
    xs = np.arange(n)
    counts = bars["count"].to_numpy(dtype=float)
    # geom_bar(fill = after_stat(count)) on scale_fill_gradientn(material-indigo)
    cmap = LinearSegmentedColormap.from_list(
        "upset", continuous_palette(palette="material-indigo", n=100), N=256)
    lo, hi = (counts.min(), counts.max()) if n else (0.0, 1.0)
    norm = Normalize(lo, hi if hi > lo else lo + 1)
    bar_ax.bar(xs, counts, width=0.5, color=[cmap(norm(c)) for c in counts],
               edgecolor="black", linewidth=0.5, zorder=2)
    top = counts.max() if n else 1.0
    y0, y1 = -0.05 * top, 1.05 * top
    bar_ax.set_ylim(y0, y1)
    # geom_text_repel(force = 0) leaves each count on its bar top but keeps
    # the label box (text plus box.padding = 0.25 lines) inside the panel, so
    # a small count is lifted clear of the bottom edge, with a black segment
    # back to the bar (min.segment.length = 0)
    fs = label_size * pt
    half = (0.6 + 0.25 * 1.2) * fs / 72.0 / bar_h * (y1 - y0) if bar_h > 0 else 0.0
    for x, c in zip(xs, counts, strict=True):
        yl = float(min(max(c, y0 + half), y1 - half))
        t = bar_ax.text(float(x), yl, f"{int(c)}", ha="center", va="center",
                        fontsize=fs, zorder=3)
        halo(t, foreground="white", radius=0.1)
        if yl - c > half * 0.5:
            bar_ax.plot([x, x], [c, yl - half * 0.6], color="black", linewidth=0.5, zorder=3)
    bar_ax.set_xlim(-0.6, n - 0.4)
    bar_ax.set_ylabel("Intersection size")
    bar_ax.set_axisbelow(True)
    bar_ax.grid(True, color="#CCCCCC", linestyle=(0, (4, 4)), linewidth=0.5)
    bar_ax.tick_params(axis="x", bottom=False, labelbottom=False)

    # the matrix: discrete y with the first set on top, striped from the bottom
    rows = list(reversed(sets))  # bottom-up, as make_combination_matrix_plot gets them
    for k in range(len(rows)):
        # index %% 2 == 0 -> "color.one" (white); index is 1-based from the bottom
        if (k + 1) % 2:
            mat_ax.axhspan(k - 0.5, k + 0.5, color="#F7F7F7", zorder=0, linewidth=0)
    ypos = {s: k for k, s in enumerate(rows)}
    for x, inter in zip(xs, bars["intersection"], strict=True):
        on = sorted(ypos[s] for s in inter)
        if len(on) > 1:
            mat_ax.plot([x, x], [on[0], on[-1]], color="black", linewidth=1.2 * pt * 72 / 96,
                        zorder=2, solid_capstyle="butt")
        mat_ax.scatter(np.full(len(rows), x), range(len(rows)), s=(3 * pt) ** 2,
                       c=["black" if s in inter else "#E0E0E0" for s in rows],
                       linewidths=0, zorder=3)
    mat_ax.set_ylim(-0.5, max(len(rows), 1) - 0.5)
    mat_ax.set_yticks(range(len(rows)))
    mat_ax.set_yticklabels(rows, fontsize=theme.size("axis.title"))
    mat_ax.tick_params(axis="both", length=0)
    mat_ax.tick_params(axis="x", labelbottom=False)
    for side in ("top", "right", "bottom", "left"):
        mat_ax.spines[side].set_visible(False)


def upset_plot(
    data: pd.DataFrame,
    stat_cols: Sequence[str],
    *,
    split_by: str | None,
    stat_level,
    label_size: float,
    panel_size: tuple[float, float],
    nrow: int | None,
    ncol: int | None,
    byrow: bool,
) -> Figure:
    """``StatPlot``'s upset branch (SCP-plot.R:5044-5070): one block per
    ``split_by`` level, each a column of intersection-size bars over ggupset's
    combination matrix, drawn into the two halves of one :class:`PanelGrid`
    panel.  Bars, counts and matrix rows follow :func:`upset_layout`; the
    split level labels the block, as R's ``labs(x = sp)`` does.  As in R,
    ``palette``/``palcolor``/``alpha`` have no effect (the bars are coloured by
    count) and there is no legend.  The panel is ``0.6`` as tall as it is wide
    (R's ``aspect.ratio = 0.6``), wide enough for the bars, with the matrix
    below sized to the number of sets.
    """
    from ..theme import apply_theme, theme_scp

    stat_cols = list(stat_cols)
    members = _membership(data, stat_cols, resolve_stat_level(data, stat_cols, stat_level))
    if split_by is None:
        split_levels: list[str | None] = [None]
        split_of = None
    else:
        scat = as_ordered_categorical(data[split_by])
        split_levels = [str(x) for x in scat.cat.categories]
        split_of = np.asarray(scat).astype(str)
    blocks = []
    for sp in split_levels:
        mem = members if sp is None else members.loc[split_of == sp]
        blocks.append((sp, *upset_layout(mem, stat_cols)))

    n_bars = max((len(b) for _sp, b, _s in blocks), default=1)
    n_sets = max((len(s) for _sp, _b, s in blocks), default=1)
    width = max(panel_size[0], 0.28 * n_bars + 0.4)
    bar_h = 0.6 * width
    # ggupset: (text size + extra spacing) per set, plus 10 pt
    mat_h = ((12 + 6) * max(n_sets, 1) + 10) / 72.0
    th = theme_scp(legend_position="none")
    keys = ["" if sp is None else str(sp) for sp in split_levels]
    pg = PanelGrid(len(keys), panel_size=(width, bar_h + mat_h), nrow=nrow, ncol=ncol,
                   byrow=byrow, keys=keys, theme=th, legend=LegendColumn(position="none"))
    for key, (sp, bars, sets) in zip(keys, blocks, strict=True):
        holder = pg.axes[key]
        spec = holder.get_subplotspec()
        assert spec is not None
        holder.remove()
        sub = spec.subgridspec(2, 1, height_ratios=[bar_h, mat_h], hspace=0.0)
        bar_ax = apply_theme(pg.fig.add_subplot(sub[0]), th)
        mat_ax = pg.fig.add_subplot(sub[1], sharex=bar_ax)
        _draw_upset(bar_ax, mat_ax, bars, sets, label_size=label_size, theme=th, bar_h=bar_h)
        if sp is not None:
            mat_ax.set_xlabel(str(sp))
    return pg.finish()
