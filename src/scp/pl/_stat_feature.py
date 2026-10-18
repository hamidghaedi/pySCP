"""``expression_stat_plot`` — the DataFrame engine behind ``feature_stat_plot``.

Port of ``ExpressionStatPlot`` (``SCP-plot.R:3723``) and the stacking /
``plot.by = "feature"`` logic of ``FeatureStatPlot`` (``SCP-plot.R:3524``) —
Hao Zhang, https://github.com/zhanghao-njmu/SCP (GPL-3).

The code is split the way the R is: :func:`_plan` turns the frame into one
:class:`_Panel` per (feature, group column[, level, split]) with everything a
reader sees already decided — x order, colours, stripes, y limits — and
:func:`_draw_panel` puts one panel into an ``Axes``.  The layouts (an ordinary
grid, or the stacked figure) only arrange axes.  Positions follow ggplot2
exactly where a number is visible: ``position_dodge(0.9)`` with the dodge
count taken per x position, ``coord_cartesian``'s 5 % expansion, the discrete
axis's 0.6 padding, and the significance layer's own arithmetic (see
:mod:`._stat_sig` for the tests themselves).

Spec: ``docs/porting_briefs/stat_plots.md`` §1.
"""

from __future__ import annotations

import inspect
import math
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Literal

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedFormatter, FixedLocator, MaxNLocator

from ..fetch import as_ordered_categorical
from ..layout import LegendColumn, PanelGrid
from ..palettes import continuous_palette, discrete_palette
from ..theme import theme_scp
from . import _stat_sig as sig
from ._scales import format_breaks, log_breaks

__all__ = ["expression_stat_plot"]

ExprPlotType = Literal["violin", "box", "bar", "dot", "col"]

#: ggplot2's ``.pt``: points per millimetre.  Sizes in R are millimetres.
_PT = 72.27 / 25.4
_DODGE = 0.9
_ALL = "All.groups"
#: matplotlib 3.10 replaced ``vert=`` by ``orientation=`` on violinplot/boxplot
_HAS_ORIENTATION = "orientation" in inspect.signature(Axes.violinplot).parameters


def _lw(linewidth: float) -> float:
    """ggplot ``linewidth`` (mm-ish, x .pt in 1/96 in) -> matplotlib points."""
    return linewidth * _PT * 72.0 / 96.0


def _r_round1(v: float) -> str:
    """``as.character(round(v, 1))``: no trailing zeros, no negative zero."""
    s = f"{round(v, 1):.1f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


# --------------------------------------------------------------------------- #
# the plan
# --------------------------------------------------------------------------- #
@dataclass
class _Panel:
    key: str
    g: str                         # group column (``All.groups`` when absent)
    f: str                         # value column
    title: str                     # what the facet strip shows
    frame: pd.DataFrame            # value, group, split; rows in group.unique order
    levels: list[str]              # x order
    split_levels: list[str]        # declared split order ("" when no split)
    has_split: bool
    fill_by: str
    color_of: Callable[[str, str], object]
    bg: list[tuple[str, str]]      # (level, colour) in x order
    y_lo: float
    y_hi: float
    xlab: str
    individual: bool
    ncells: int
    median_of: dict = field(default_factory=dict)
    flip: bool = False             # coord_flip: positions reversed, axes transposed


@dataclass
class _Legend:
    entries: list[tuple[str, list[tuple[str, str]]]] = field(default_factory=list)
    colorbar: tuple[str, list[str], tuple[float, float]] | None = None
    count_range: tuple[float, float] | None = None

    def add(self, title: str, items: list[tuple[str, str]]) -> None:
        if all(t != title for t, _ in self.entries):
            self.entries.append((title, items))


def _parse_limit(spec, pool: np.ndarray) -> float:
    """``y.min``/``y.max``: a number, or ``"qNN"`` for the NN-th percentile."""
    if isinstance(spec, str):
        if not spec.startswith("q"):
            raise ValueError(f"y limit {spec!r} must be numeric or 'qNN'")
        return float(np.quantile(pool, float(spec[1:]) / 100.0))
    return float(spec)


#: the log transforms ``y_trans`` accepts, by base
_LOG_BASE = {"log10": 10.0, "log2": 2.0}


def _kept(pool: np.ndarray, log_t: bool) -> np.ndarray:
    """The values a log scale keeps (positive ones); all of them otherwise.
    Never empty, so the caller can always take an extreme."""
    if not log_t:
        return pool
    pos = pool[pool > 0]
    return pos if pos.size else np.array([1.0])


def _log(v, base: float):
    """``log10`` / ``log2`` as the scale transform computes them, with the
    non-finite results (``log(0) = -Inf``, ``log(-1) = NaN``) set to NaN: ggplot
    drops those rows from every stat."""
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.log10(v) if base == 10 else np.log2(v)
    return np.where(np.isfinite(t), t, np.nan)


def _to_scale(P: _Panel, base: float) -> _Panel:
    """``P`` on the transformed scale: values and limits through the log."""
    fr = P.frame.copy()
    fr["value"] = _log(fr["value"].to_numpy(dtype=float), base)
    lo, hi = float(_log(P.y_lo, base)), float(_log(P.y_hi, base))
    finite = fr["value"].to_numpy()
    finite = finite[np.isfinite(finite)]
    if not np.isfinite(lo):
        # a "qNN" or explicit limit at or below zero: R errors here
        # ('from' must be a finite number); fall back to the smallest kept value
        warnings.warn(f"y limit {P.y_lo:g} is not positive on a log scale; using the "
                      "smallest positive value instead", stacklevel=4)
        lo = float(finite.min()) if finite.size else 0.0
    if not np.isfinite(hi):
        hi = float(finite.max()) if finite.size else 0.0
    return replace(P, frame=fr, y_lo=lo, y_hi=hi)


def _plan(
    data: pd.DataFrame,
    stat_by: list[str],
    *,
    group_by: list[str | None],
    split_by: str | None,
    bg_by: str | None,
    fill_by: str,
    plot_type: str,
    cells: Sequence[str] | None,
    keep_empty: bool,
    individual: bool,
    palette: str,
    palcolor,
    bg_palette: str,
    bg_palcolor,
    same_y_lims: bool,
    y_min,
    y_max,
    sort,
    xlab: str | None,
    titles: dict[str, str] | None,
    force: bool,
    legend: _Legend,
    legend_on: bool,
    y_trans: str = "identity",
) -> list[_Panel]:
    """Everything ``ExpressionStatPlot`` decides before it draws.

    Limits stay on the data scale here; :func:`_draw_panel` moves the panel
    onto the transformed scale.  Under a log ``y_trans`` the default limits are
    the extremes of the values the transform keeps (the positive ones) --
    the only ones ggplot's stats ever see.
    """
    titles = titles or {}
    log_t = y_trans in _LOG_BASE
    dat = data
    if cells is not None:
        dat = dat.loc[dat.index.intersection(pd.Index(cells))]
    if dat.shape[0] == 0:
        raise ValueError("No specified cells found.")

    gcols = [g if g is not None else _ALL for g in group_by]
    frame = pd.DataFrame(index=dat.index)
    declared: dict[str, list[str]] = {}
    for col in dict.fromkeys([*gcols, split_by or _ALL, *([bg_by] if bg_by else [])]):
        if col == _ALL:
            frame[col] = ""
            declared[col] = [""]
            continue
        if col not in dat.columns:
            raise KeyError(f"{col} is not in the data.")
        cat = as_ordered_categorical(dat[col])
        declared[col] = [str(c) for c in cat.cat.categories]
        frame[col] = np.asarray(cat.astype(object).where(cat.notna(), None))
        frame[col] = frame[col].map(lambda v: None if v is None else str(v))
    split_col = split_by or _ALL

    big = [c for c, lv in declared.items() if len(lv) > 100]
    if big and not force:
        raise ValueError(f"{big} have more than 100 levels. Pass force=True to proceed anyway.")

    # bg.by must be a coarser partition of every group column (checked on the
    # full data, before the cell subset, as R does on meta.data)
    bg_map: dict[str, dict[str, str]] = {}
    for g in gcols:
        if bg_by is None:
            continue
        full = data[[c for c in {g, bg_by} if c != _ALL]]
        gv = full[g].astype(str) if g != _ALL else pd.Series("", index=full.index)
        spread = full.assign(__g=gv).groupby("__g", observed=True)[bg_by].nunique()
        if (spread > 1).any():
            raise ValueError("'group.by' must be a part of 'bg.by'")
        bg_map[g] = {str(k): str(v) for k, v in
                     full.assign(__g=gv).dropna(subset=[bg_by])
                     .groupby("__g", observed=True)[bg_by].first().items()}

    stat_by = list(dict.fromkeys(stat_by))
    missing = [f for f in stat_by if f not in dat.columns]
    if missing:
        warnings.warn(f"{','.join(map(str, missing))} are not found.", stacklevel=3)
        stat_by = [f for f in stat_by if f not in missing]
    if not stat_by:
        raise ValueError("none of `stat_by` could be resolved")
    values = pd.DataFrame(index=dat.index)
    for f in stat_by:
        v = dat[f]
        if not pd.api.types.is_numeric_dtype(v):
            raise ValueError("'stat.by' must be type of numeric variable.")
        values[f] = pd.to_numeric(v, errors="coerce").astype(float)

    ncells = int(dat.shape[0])
    if same_y_lims:
        # R reads `values` here although the pool is named `valus`, so a "qNN"
        # string errors whenever same.y.lims=TRUE; this uses the pool it meant.
        pool = values.to_numpy().ravel()
        pool = pool[np.isfinite(pool)]
        kept = _kept(pool, log_t)
        y_max = float(kept.max()) if y_max is None else _parse_limit(y_max, pool)
        y_min = float(kept.min()) if y_min is None else _parse_limit(y_min, pool)

    if group_by == [None] and split_by is None:
        legend_on = False

    # colours that do not depend on the panel
    feat_colors = discrete_palette([str(f) for f in stat_by], palette=palette, palcolor=palcolor)
    medians: dict[str, pd.DataFrame] = {}
    if fill_by == "expression":
        ramp = continuous_palette(palette=palette, palcolor=palcolor, n=100)
        meds_all = []
        for g in gcols:
            m = values.assign(__g=frame[g], __s=frame[split_col]).groupby(
                ["__g", "__s"], observed=True, dropna=True).median()
            medians[g] = m
            meds_all.append(m.to_numpy().ravel())
        pool = np.concatenate(meds_all)
        pool = pool[np.isfinite(pool)]
        lims = (float(pool.min()), float(pool.max())) if pool.size else (0.0, 1.0)
        legend.colorbar = ("Median expression", ramp, lims)

    panels: list[_Panel] = []
    indiv_levels: dict[str, set[str]] = {}
    for g in gcols:
        combos: list[tuple[list[str], list[str]]]
        if individual:
            # R's merge(expand.grid(...)) varies the group level fastest
            combos = [([lv], [sp]) for sp in declared[split_col] for lv in declared[g]]
        else:
            combos = [(declared[g], declared[split_col])]
        for f in stat_by:
            for single_group, sp_use in combos:
                sel = frame[g].isin(single_group) & frame[split_col].isin(sp_use)
                sub = pd.DataFrame({
                    "value": values.loc[sel, f].to_numpy(),
                    "group": frame.loc[sel, g].to_numpy(),
                    "split": (frame.loc[sel, g] if split_col == g
                              else frame.loc[sel, split_col]).to_numpy(),
                }, index=frame.index[sel])
                present = [lv for lv in declared[g] if (sub["group"] == lv).any()]
                # R re-levels to the levels present, so its keep_empty never sees
                # an empty level; here it keeps declared-but-absent ones.
                levels = ([lv for lv in declared[g] if lv in single_group]
                          if keep_empty else present)
                pre_sort = list(levels)

                # bg colours are keyed by level name in the pre-sort order
                if bg_by is not None:
                    bg_cols = discrete_palette(declared[bg_by], palette=bg_palette,
                                               palcolor=bg_palcolor)
                    stripe = {lv: bg_cols[bg_map[g][lv]] for lv in pre_sort if lv in bg_map[g]}
                else:
                    pal = list(bg_palcolor) if bg_palcolor is not None else \
                        [("transparent", "#D9D9D9")[i % 2] for i in range(len(pre_sort))]
                    alt = discrete_palette(pre_sort, palcolor=pal) if pre_sort else {}
                    stripe = dict(alt)

                if sort and sub.shape[0] > 0:
                    med = sub.groupby("group", observed=True)["value"].median()
                    med = med.reindex([lv for lv in levels if lv in med.index])
                    decreasing = not (isinstance(sort, str) and sort == "increasing")
                    order = med.sort_values(ascending=not decreasing, kind="stable",
                                            na_position="last")
                    levels = [str(lv) for lv in order.index]

                color_of: Callable[[str, str], object]
                if fill_by == "feature":
                    color_of = _by_name(feat_colors, str(f))
                    if legend_on:
                        legend.add("Features", [(str(x), feat_colors[str(x)]) for x in stat_by])
                elif fill_by == "group":
                    if split_by is not None:
                        colors = discrete_palette(declared[split_col], palette=palette,
                                                  palcolor=palcolor)
                        color_of = _by_split(colors)
                        if legend_on:
                            legend.add(split_by, [(s, colors[s]) for s in declared[split_col]])
                    else:
                        colors = discrete_palette(declared[g], palette=palette,
                                                  palcolor=palcolor)
                        color_of = _by_level(colors)
                        if legend_on and individual:
                            # R gives each one-group panel its own one-entry
                            # legend; the shared column lists every level the
                            # panels show, once (registered after the loop)
                            indiv_levels.setdefault(g, set()).update(levels)
                        elif legend_on:
                            legend.add(g if g != _ALL else "", [(lv, colors[lv]) for lv in levels])
                else:  # expression
                    assert legend.colorbar is not None
                    _, ramp_c, lims = legend.colorbar
                    color_of = _by_median(medians[g][f], ramp_c, lims)

                # rows in group.unique order: split fastest within each group
                xi = {lv: i for i, lv in enumerate(levels)}
                si = {s: i for i, s in enumerate(declared[split_col])}
                sub = sub[sub["group"].isin(levels)]
                order_key = sub["group"].map(xi) * (len(si) + 1) + sub["split"].map(si)
                sub = sub.iloc[np.argsort(order_key.to_numpy(), kind="stable")]

                finite = sub["value"].to_numpy()
                finite = finite[np.isfinite(finite)]
                pool = finite if finite.size else np.array([0.0])
                kept = _kept(pool, log_t)
                hi = float(kept.max()) if y_max is None else _parse_limit(y_max, pool)
                lo = float(kept.min()) if y_min is None else _parse_limit(y_min, pool)

                key = ":".join([str(f), g, ",".join(single_group), ",".join(sp_use)])
                panels.append(_Panel(
                    key=key, g=g, f=str(f), title=titles.get(str(f), str(f)), frame=sub,
                    levels=levels, split_levels=declared[split_col],
                    has_split=split_by is not None, fill_by=fill_by, color_of=color_of,
                    bg=[] if individual else [(lv, stripe.get(lv, "transparent")) for lv in levels],
                    y_lo=lo, y_hi=hi,
                    xlab=xlab if xlab is not None else ("" if g == _ALL else g),
                    individual=individual, ncells=ncells,
                ))
        if g in indiv_levels:
            colors = discrete_palette(declared[g], palette=palette, palcolor=palcolor)
            legend.add(g if g != _ALL else "",
                       [(lv, colors[lv]) for lv in declared[g] if lv in indiv_levels[g]])
    return panels


def _by_name(colors: dict[str, str], name: str) -> Callable[[str, str], object]:
    return lambda lv, sp: colors[name]


def _by_split(colors: dict[str, str]) -> Callable[[str, str], object]:
    return lambda lv, sp: colors[sp]


def _by_level(colors: dict[str, str]) -> Callable[[str, str], object]:
    return lambda lv, sp: colors[lv]


def _by_median(medians: pd.Series, ramp: list[str],
               lims: tuple[float, float]) -> Callable[[str, str], object]:
    """``fill.by = "expression"``: the colour of the (group, split) median."""
    cmap = LinearSegmentedColormap.from_list("median", ramp, N=256)
    norm = Normalize(*lims)

    def color_of(lv: str, sp: str) -> object:
        v = medians.get((lv, sp), np.nan)
        return cmap(norm(v)) if np.isfinite(v) else "#BEBEBE"
    return color_of


# --------------------------------------------------------------------------- #
# drawing one panel
# --------------------------------------------------------------------------- #
def _nrd0(v: np.ndarray) -> float:
    """R's ``bw.nrd0`` — the bandwidth ``geom_violin`` uses by default."""
    sd = float(np.std(v, ddof=1))
    iqr = float(np.subtract(*np.quantile(v, [0.75, 0.25])))
    lo = min(sd, iqr / 1.34)
    if not lo > 0:
        lo = sd if sd > 0 else (abs(float(v[0])) if v[0] != 0 else 1.0)
    return 0.9 * lo * v.size ** -0.2


def _cut_edges(values: np.ndarray, n_bins: int = 14) -> tuple[np.ndarray, np.ndarray]:
    """Edges of ``cut(x, seq(min, max, length.out = 15), include.lowest = TRUE)``
    and each bin's centre.

    The centre is the median of the two bounds *as printed* in the factor
    label (``dig.lab = 3``, widened until labels are distinct), which is how
    R recovers it — so it is the midpoint of the rounded edges.
    """
    lo, hi = float(values.min()), float(values.max())
    edges = np.linspace(lo, hi, n_bins + 1)
    labs = [f"{e:.3g}" for e in edges]
    for dig in range(3, 13):
        labs = [f"{e:.{dig}g}" for e in edges]
        if all(a != b for a, b in zip(labs[:-1], labs[1:], strict=True)):
            break
    nums = np.array([float(s) for s in labs])
    return edges, (nums[:-1] + nums[1:]) / 2


def _bin(v: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Bin index under ``cut(right = TRUE, include.lowest = TRUE)``."""
    return np.clip(np.searchsorted(edges, v, side="left") - 1, 0, edges.size - 2)


def _dodge(n: int, k: int) -> float:
    """``pos_dodge``: offset of the k-th of n groups sharing one x position."""
    return 0.0 if n <= 1 else _DODGE * ((k + 0.5) / n - 0.5)


def _cells(P: _Panel, min_n: int = 1, by_split: bool = True) -> list[tuple[str, str, int, int, np.ndarray]]:
    """(level, split, k, n, finite values) per drawn group, with its dodge slot."""
    fr = P.frame
    out = []
    for lv in P.levels:
        rows = fr[fr["group"] == lv]
        if by_split:
            groups = [(sp, rows.loc[rows["split"] == sp, "value"].to_numpy())
                      for sp in P.split_levels if (rows["split"] == sp).any()]
        else:
            # one group per x; its fill is the first row's, as in ggplot
            first = rows["split"].iloc[0] if rows.shape[0] else ""
            groups = [(first, rows["value"].to_numpy())]
        groups = [(sp, v[np.isfinite(v)]) for sp, v in groups]
        groups = [(sp, v) for sp, v in groups if v.size >= min_n]
        for k, (sp, v) in enumerate(groups):
            out.append((lv, sp, k, len(groups), v))
    return out


def _span(vals: list[float]) -> tuple[float, float]:
    a = np.asarray([v for v in vals if np.isfinite(v)], dtype=float)
    return (float(a.min()), float(a.max())) if a.size else (0.0, 1.0)


class _Ax:
    """An ``Axes`` addressed in (position, value) space.

    Every layer of a panel is drawn through this, so ``flip=True`` — ggplot's
    ``coord_flip`` — is one switch: the position (group) axis becomes the
    vertical one and the value axis the horizontal one, for the geometry, the
    stripes, the reference lines, the significance layer and the limits alike.
    """

    def __init__(self, ax, flip: bool) -> None:
        self.ax = ax
        self.flip = flip

    def xy(self, p, v):
        return (v, p) if self.flip else (p, v)

    @property
    def value_axis(self):
        return self.ax.xaxis if self.flip else self.ax.yaxis

    @property
    def pos_axis(self):
        return self.ax.yaxis if self.flip else self.ax.xaxis

    def plot(self, p, v, **kw):
        return self.ax.plot(*self.xy(p, v), **kw)

    def scatter(self, p, v, **kw):
        return self.ax.scatter(*self.xy(p, v), **kw)

    def text(self, p, v, s, **kw):
        return self.ax.text(*self.xy(p, v), s, **kw)

    def pos_span(self, p0, p1, **kw):
        """A band across the whole value range (``geom_rect(ymin=-Inf, ymax=Inf)``)."""
        return (self.ax.axhspan if self.flip else self.ax.axvspan)(p0, p1, **kw)

    def value_line(self, v, **kw):
        """``geom_hline(yintercept = v)``."""
        return (self.ax.axvline if self.flip else self.ax.axhline)(v, **kw)

    def pos_line(self, p, **kw):
        """``geom_vline(xintercept = p)``."""
        return (self.ax.axhline if self.flip else self.ax.axvline)(p, **kw)

    def bar(self, p, v, width, *, bottom=None, **kw):
        if self.flip:
            return self.ax.barh(p, v, height=width, left=bottom, **kw)
        return self.ax.bar(p, v, width=width, bottom=bottom, **kw)

    def fill_between(self, p, v0, v1, **kw):
        if self.flip:
            return self.ax.fill_betweenx(p, v0, v1, **kw)
        return self.ax.fill_between(p, v0, v1, **kw)

    def value_segments(self, p, v0, v1, **kw):
        """Segments along the value axis at position(s) ``p``."""
        return (self.ax.hlines if self.flip else self.ax.vlines)(p, v0, v1, **kw)

    def pos_segments(self, v, p0, p1, **kw):
        """Segments along the position axis at value(s) ``v``."""
        return (self.ax.vlines if self.flip else self.ax.hlines)(v, p0, p1, **kw)

    def _orientation(self) -> dict:
        if _HAS_ORIENTATION:
            return {"orientation": "horizontal" if self.flip else "vertical"}
        return {"vert": not self.flip}

    def violinplot(self, data, **kw):
        return self.ax.violinplot(data, **self._orientation(), **kw)

    def boxplot(self, data, **kw):
        return self.ax.boxplot(data, **self._orientation(), **kw)

    def set_value_lim(self, lo, hi) -> None:
        (self.ax.set_xlim if self.flip else self.ax.set_ylim)(lo, hi)

    def get_value_lim(self) -> tuple[float, float]:
        return self.ax.get_xlim() if self.flip else self.ax.get_ylim()

    def set_pos_lim(self, lo, hi) -> None:
        (self.ax.set_ylim if self.flip else self.ax.set_xlim)(lo, hi)

    def get_pos_lim(self) -> tuple[float, float]:
        return self.ax.get_ylim() if self.flip else self.ax.get_xlim()

    def set_value_label(self, s: str) -> None:
        (self.ax.set_xlabel if self.flip else self.ax.set_ylabel)(s)

    def set_pos_label(self, s: str) -> None:
        (self.ax.set_ylabel if self.flip else self.ax.set_xlabel)(s)

    def value_grid(self, **kw) -> None:
        self.value_axis.grid(True, **kw)

    def set_pos_ticks(self, ticks, labels) -> None:
        if self.flip:
            self.ax.set_yticks(ticks)
            self.ax.set_yticklabels(labels)
        else:
            self.ax.set_xticks(ticks)
            self.ax.set_xticklabels(labels, rotation=45, ha="right")

    def hide_pos_ticks(self) -> None:
        (self.ax.set_yticks if self.flip else self.ax.set_xticks)([])


def _positions(P: _Panel) -> dict[str, float]:
    """x position of each level: in level order, or reversed under ``flip``
    (R reverses the factor so that the first level ends up on top)."""
    n = len(P.levels)
    return {lv: float(n - 1 - i if P.flip else i) for i, lv in enumerate(P.levels)}


def _col_range(bounds: list[tuple[str, float, float, int]], n: int,
               individual: bool) -> tuple[float, float]:
    """The cell-axis range R trains for ``plot_type = "col"``.

    The bars span ``[0.55, n + 0.45]``; the stripes add their finite edges,
    computed with R's test ``xmin == min(xmax)`` for the open left edge
    (SCP-plot.R:4102), so the left stripe usually stays finite at ``0.5``.
    ``scale_x_discrete`` then pads a continuous range by 0.6 on either side.
    """
    lo, hi = 0.55, n + 0.45
    if not individual and bounds:
        min_of_max = min(b for _lv, _a, b, _i in bounds)
        max_of_max = max(b for _lv, _a, b, _i in bounds)
        for _lv, a, b, _i in bounds:
            if a != min_of_max:
                lo = min(lo, a - 0.5)
            if b != max_of_max:
                hi = max(hi, b + 0.5)
    return lo - 0.6, hi + 0.6


def _draw_panel(ax, P: _Panel, o: dict, *, stacked: bool = False) -> dict:
    """Draw one ``ExpressionStatPlot`` panel into ``ax``; return what the layout needs."""
    plot_type = o["plot_type"]
    base = _LOG_BASE.get(o["y_trans"])
    if base is not None:
        # scale_y_continuous(trans = ...) transforms before any stat runs: the
        # densities, boxes, means and tests all live on the log scale
        P = _to_scale(P, base)
    A = _Ax(ax, P.flip)
    o["_extra"] = {"dots": [], "trend": []}
    xpos = _positions(P)
    n_lv = len(P.levels)
    split_violin = P.fill_by == "group" and P.has_split
    y_lo, y_hi = P.y_lo, P.y_hi
    vals = P.frame["value"].to_numpy()
    finite = vals[np.isfinite(vals)]

    # ---- background stripes: full height, outermost bleeding past the edge
    if not P.individual and plot_type != "col":
        for lv, col in P.bg:
            if col == "transparent":
                continue
            p = xpos[lv]
            x0 = -1e6 if p == 0 else p - 0.5
            x1 = 1e6 if p == n_lv - 1 else p + 0.5
            A.pos_span(x0, x1, color=col, alpha=o["bg_alpha"], zorder=0.6, linewidth=0)

    final: list[float] = []      # the y range ggplot trains after the stats
    raw: list[float] = list(finite)  # ... and before them (what ggsignif measures)
    if plot_type in ("bar", "col"):
        A.value_line(0, linestyle="--", color="black", linewidth=_lw(0.5), zorder=1.5)
        final.append(0.0)
        raw.append(0.0)

    # ---- the geometry
    if plot_type == "violin":
        for lv, sp, k, n, v in _cells(P, min_n=2, by_split=split_violin):
            pos = xpos[lv] + _dodge(n, k)
            w = _DODGE / n
            final.extend([v.min(), v.max()])
            if np.ptp(v) == 0:
                A.plot([pos - w / 2, pos + w / 2], [v[0], v[0]], color="#333333",
                       linewidth=_lw(0.5), zorder=2)
                continue
            parts = A.violinplot([v], positions=[pos], widths=w, showextrema=False,
                                 bw_method=_nrd0(v) / float(np.std(v, ddof=1)),
                                 points=512)
            for body in parts["bodies"]:
                body.set(facecolor=P.color_of(lv, sp), alpha=o["alpha"],
                         edgecolor="#333333", linewidth=_lw(0.5), zorder=2)
    elif plot_type == "box":
        for lv, sp, k, n, v in _cells(P):
            pos = xpos[lv] + _dodge(n, k)
            final.extend([v.min(), v.max()])
            bp = A.boxplot([v], positions=[pos], widths=0.8 / n, patch_artist=True,
                           showfliers=False, manage_ticks=False, zorder=2,
                           medianprops=dict(color="black", linewidth=_lw(0.5) * 2),
                           whiskerprops=dict(color="black", linewidth=_lw(0.5)),
                           capprops=dict(linewidth=0))
            for box in bp["boxes"]:
                box.set(facecolor=P.color_of(lv, sp), edgecolor="black", linewidth=_lw(0.5))
            A.scatter([pos], [np.median(v)], s=(1.5 * _PT) ** 2, facecolor="white",
                      edgecolor="black", linewidths=0.5, zorder=4)
    elif plot_type == "bar":
        for lv, sp, k, n, v in _cells(P):
            pos = xpos[lv] + _dodge(n, k)
            m = float(v.mean())
            sd = float(v.std(ddof=1)) if v.size > 1 else np.nan
            A.bar(pos, m, 0.8 / n, color=P.color_of(lv, sp), edgecolor="black",
                  linewidth=_lw(0.5), zorder=2)
            final.extend([0.0, m])
            if np.isfinite(sd):
                cap = 0.2 / n / 2
                A.value_segments(pos, m - sd, m + sd, color="black", linewidth=_lw(0.5),
                                 zorder=3)
                A.pos_segments([m - sd, m + sd], pos - cap, pos + cap, color="black",
                               linewidth=_lw(0.5), zorder=3)
                final.extend([m - sd, m + sd])
        # R resets the lower limit to what the bars and error bars span
        y_lo = _span(final)[0]
    elif plot_type == "dot":
        if finite.size and np.ptp(finite) > 0:
            edges, mids = _cut_edges(finite)
            counts_all = []
            groups = []
            for lv, sp, k, n, v in _cells(P, by_split=split_violin):
                u, c = np.unique(_bin(v, edges), return_counts=True)
                groups.append((lv, sp, k, n, u, c))
                counts_all.extend(c.tolist())
            cmax = max(counts_all) if counts_all else 1
            for lv, sp, k, n, u, c in groups:
                pos = xpos[lv] + _dodge(n, k)
                diam = 6.0 * np.sqrt(c / cmax) * _PT
                A.scatter(np.full(u.size, pos), mids[u], s=diam**2, facecolor=P.color_of(lv, sp),
                          edgecolor="black", linewidths=0.5, alpha=o["alpha"], zorder=3)
                final.extend(mids[u].tolist())
                o["_extra"]["dots"].extend(zip([pos] * u.size, mids[u].tolist(), c.tolist(),
                                               strict=True))
            o["_count_range"] = (min(counts_all), cmax) if counts_all else None
    elif plot_type == "col":
        fr = P.frame
        n = fr.shape[0]
        # rows stay in level order; under flip R numbers them from the top
        idx = np.arange(n, 0, -1, dtype=float) if P.flip else np.arange(1, n + 1, dtype=float)
        colors = [P.color_of(g, s) for g, s in zip(fr["group"], fr["split"], strict=True)]
        A.bar(idx, np.nan_to_num(fr["value"].to_numpy()), 0.9, color=colors,
              linewidth=0, zorder=2)
        final.extend(finite.tolist())
        bounds = []
        for i, lv in enumerate(P.levels):
            where = idx[(fr["group"] == lv).to_numpy()]
            if where.size == 0:
                continue
            bounds.append((lv, where.min(), where.max(), i))
        if not P.individual and bounds:
            # R tests `xmin == min(xmax)` for the left edge, a typo for min(xmin)
            # that leaves a gutter on the left; both edges bleed here.
            lowest = min(a for _lv, a, _b, _i in bounds)
            highest = max(b for _lv, _a, b, _i in bounds)
            for _lv, a, b, i in bounds:
                col = P.bg[i][1] if i < len(P.bg) else "transparent"
                if col == "transparent":
                    continue
                x0 = -1e9 if a == lowest else a - 0.5
                x1 = 1e9 if b == highest else b + 0.5
                A.pos_span(x0, x1, color=col, alpha=o["bg_alpha"], zorder=0.6, linewidth=0)
            if len(P.levels) > 1:
                # a dashed border below every group but the lowest-numbered one
                for _lv, a, _b, _i in bounds:
                    if a != lowest:
                        A.pos_line(a - 0.5, linestyle="--", color="black", alpha=0.5,
                                   linewidth=_lw(0.5), zorder=1.5)
        A.set_pos_lim(*_col_range(bounds, n, P.individual))
        A.hide_pos_ticks()

    # ---- significance (never for col; validated by the caller)
    # SCP reads ggplot's trained y range back after each layer it adds
    # (layer_scales) and moves the ceiling: these lines reproduce that range.
    texts: list[tuple] = []
    brackets: list[tuple] = []
    sig_rows: list[tuple] = []
    add_line = o["add_line"]
    if add_line is not None and base is not None:
        add_line = float(_log(float(add_line), base))
        add_line = None if not np.isfinite(add_line) else add_line
    if plot_type != "col":
        if o["comparisons"] is True:
            rows = _sig_split(P, o)
            _emit(rows, o, xpos, y_hi, texts, ha="center", va_off=0.0)
            sig_rows.extend(rows)
            final = final + [y_hi]
            y_hi = _span(final)[1]
        elif o["comparisons"]:
            # ggsignif measures the y range *before* the stats; at the time SCP
            # reads it back that excludes add_line, but the final render does not.
            r_mid = float(np.ptp(_span(raw)))
            r_full = float(np.ptp(_span(raw + ([float(add_line)] if add_line is not None
                                               else []))))
            for i, (a, b) in enumerate(o["comparisons"]):
                p = sig.two_sample_p(_pooled(P, a), _pooled(P, b), o["pairwise_method"])
                lab = sig.signif_strict(p) if o["sig_label"] == "p.signif" else sig.format_2g(p)
                top_mid = y_hi + r_mid * 0.05 + r_mid * 0.1 * i
                final = final + [top_mid, top_mid - r_mid * 0.03]
                ypos = y_hi + r_full * 0.05 + r_full * 0.1 * i
                brackets.append((xpos[a], xpos[b], ypos, ypos - r_full * 0.03, lab, p))
                sig_rows.append(((a, b), p, lab))
            lo_f, hi_f = _span(final)
            y_hi = lo_f + (hi_f - lo_f) * 1.15
        if o["multiplegroup_comparisons"]:
            rows = _sig_multi(P, o)
            _emit(rows, o, xpos, y_hi, texts, ha="left", va_off=0.2)
            sig_rows.extend(rows)
            final = final + [y_hi]
            lo_f, hi_f = _span(final)
            y_hi = lo_f + (hi_f - lo_f) * 1.15

    # ---- overlays, drawn over the comparison layers as in R
    if plot_type != "col":
        _overlays(A, P, o, xpos, split_violin)
    if add_line is not None:
        A.value_line(add_line, color=o["line_color"], linewidth=_lw(o["line_size"]), zorder=6)

    # ---- limits: coord_cartesian(ylim) plus its default 5 % expansion
    lo, hi = y_lo, y_hi
    if not (np.isfinite(lo) and np.isfinite(hi)):
        lo, hi = 0.0, 1.0
    # (on the transformed scale when y_trans is a log: the expansion is 5 % of
    # the log range, as ggplot expands the transformed limits)
    # a zero-width range is expanded as if it were one unit wide (ggplot2 4)
    d = 0.05 * ((hi - lo) or 1.0)
    A.set_value_lim(lo - d, hi + d)
    span_d = A.get_value_lim()[1] - A.get_value_lim()[0]

    for x1, x2, ytop, ytip, lab, _p in brackets:
        lw = _lw(0.3)
        A.plot([x1, x1, x2, x2], [ytip, ytop, ytop, ytip], color="black", linewidth=lw,
               zorder=7, clip_on=False)
        if P.flip:
            # beside the bracket, not over it (see expression_stat_plot)
            A.text((x1 + x2) / 2, ytop + 0.01 * span_d, lab, ha="left", va="center",
                   fontsize=o["sig_labelsize"] * _PT, zorder=7)
        else:
            A.text((x1 + x2) / 2, ytop + 0.01 * span_d, lab, ha="center", va="bottom",
                   fontsize=o["sig_labelsize"] * _PT, zorder=7)
    for x, y, lab, ha, va_off in texts:
        # geom_text keeps its screen-space justification under coord_flip
        ax.annotate(lab, A.xy(x, y), xytext=(0, -va_off * o["sig_labelsize"] * _PT),
                    textcoords="offset points", ha=ha, va="top",
                    fontsize=o["sig_labelsize"] * _PT, zorder=7, annotation_clip=False)
    o["record"][P.key] = {
        "y_use": (y_lo, y_hi), "ylim": A.get_value_lim(), "pos_lim": A.get_pos_lim(),
        "sig": sig_rows,
        "texts": [(x, y, lab) for x, y, lab, _h, _v in texts],
        "brackets": [(x1, x2, yt, yp, lab) for x1, x2, yt, yp, lab, _p in brackets],
        **o.pop("_extra"),
    }

    if stacked:
        A.value_axis.set_major_locator(FixedLocator([y_lo, y_hi]))
        # labels = round(c(y_min_use, y_max_use), 1), on the data scale
        tick_lo, tick_hi = (y_lo, y_hi) if base is None else (base ** y_lo, base ** y_hi)
        if P.flip:
            ax.set_xticklabels([_r_round1(tick_lo), _r_round1(tick_hi)], rotation=90)
            labs = ax.get_xticklabels()
            labs[0].set_ha("left")
            labs[-1].set_ha("right")
        else:
            ax.set_yticklabels([_r_round1(tick_lo), _r_round1(tick_hi)])
            labs = ax.get_yticklabels()
            labs[0].set_va("bottom")
            labs[-1].set_va("top")
    elif base is not None:
        # scales::log_breaks(n = y.nbreaks) over the expanded range, labelled
        # by format() on the data scale ("0.5" "1.0" "3.0" "5.0")
        v0, v1 = A.get_value_lim()
        breaks = [b for b in log_breaks(base ** v0, base ** v1, o["y_nbreaks"], base)
                  if base ** v0 <= b <= base ** v1]
        A.value_axis.set_major_locator(FixedLocator([float(_log(b, base)) for b in breaks]))
        A.value_axis.set_major_formatter(FixedFormatter(format_breaks(breaks)))
        ax.set_axisbelow(True)
        A.value_grid(color="grey", linestyle=(0, (4, 4)), linewidth=_lw(0.5) * 0.6)
        if P.flip:
            ax.tick_params(axis="x", labelrotation=45)
            for t in ax.get_xticklabels():
                t.set_ha("right")
    else:
        A.value_axis.set_major_locator(MaxNLocator(o["y_nbreaks"]))
        ax.set_axisbelow(True)
        A.value_grid(color="grey", linestyle=(0, (4, 4)), linewidth=_lw(0.5) * 0.6)
        if P.flip:
            # axis.text.x = element_text(angle = 45, hjust = 1): the value axis now
            ax.tick_params(axis="x", labelrotation=45)
            for t in ax.get_xticklabels():
                t.set_ha("right")

    if plot_type != "col":
        order = sorted(P.levels, key=xpos.__getitem__)
        A.set_pos_ticks([xpos[lv] for lv in order], order)
        A.set_pos_lim(-0.6, n_lv - 0.4)
    return {"y_lo": y_lo, "y_hi": y_hi, "brackets": brackets, "texts": texts}


def _pooled(P: _Panel, level: str) -> np.ndarray:
    if level not in P.levels:
        raise ValueError(f"comparison level {level!r} is not on the x axis {P.levels}")
    v = P.frame.loc[P.frame["group"] == level, "value"].to_numpy()
    return v[np.isfinite(v)]


def _sig_split(P: _Panel, o: dict) -> list[tuple[str, float]]:
    """``comparisons = TRUE``: split levels compared within each x group.

    Mirrors the R exactly: only groups in which at least two split levels have
    two or more cells are tested, and the multi-group method is used as soon as
    *any* group has three such levels.  ggpubr's ``compare_means`` then runs
    per x position, and its p-values are formatted as one vector.
    """
    fr = P.frame
    tab = fr.groupby(["group", "split"], observed=True).size()
    ok_counts = {lv: int((tab.loc[lv] >= 2).sum()) if lv in tab.index.get_level_values(0) else 0
                 for lv in P.levels}
    group_use = [lv for lv in P.levels if ok_counts[lv] >= 2]
    method = o["multiple_method"] if any(c >= 3 for c in ok_counts.values()) \
        else o["pairwise_method"]
    method = sig.resolve_method(method)
    rows = []
    for lv in group_use:
        sub = fr[fr["group"] == lv]
        groups = [sub.loc[sub["split"] == sp, "value"].to_numpy()
                  for sp in P.split_levels if (sub["split"] == sp).any()]
        if method in sig.PAIRWISE_METHODS:
            for _i, _j, p in sig.pairwise_p(groups, method):
                rows.append((lv, p))
        else:
            rows.append((lv, sig.multigroup_p(groups, method)))
    return rows


def _sig_multi(P: _Panel, o: dict) -> list[tuple[str, float]]:
    """``multiplegroup_comparisons``: one omnibus test per panel, or per x group
    when the panel is split (ggpubr switches on whether ``group != x``)."""
    fr = P.frame
    method = sig.resolve_method(o["multiple_method"])
    per_x = fr.groupby("group", observed=True)["split"].nunique()
    rows: list[tuple[str, float]] = []
    if P.has_split and (per_x > 1).any():
        for lv in P.levels:
            sub = fr[fr["group"] == lv]
            groups = [sub.loc[sub["split"] == sp, "value"].to_numpy()
                      for sp in P.split_levels if (sub["split"] == sp).any()]
            if len(groups) < 2:
                continue  # R stops here ("all observations are in the same group")
            if method in sig.PAIRWISE_METHODS:
                rows.extend((lv, p) for _i, _j, p in sig.pairwise_p(groups, method))
            else:
                rows.append((lv, sig.multigroup_p(groups, method)))
    elif P.levels:
        groups = [fr.loc[fr["group"] == lv, "value"].to_numpy() for lv in P.levels]
        # label.x.npc = "left": the first x position, which flip puts at the bottom
        first = P.levels[-1] if P.flip else P.levels[0]
        if method in sig.PAIRWISE_METHODS:
            rows = [(first, p) for _i, _j, p in sig.pairwise_p(groups, method)]
        else:
            rows = [(first, sig.multigroup_p(groups, method))]
    return rows


def _emit(rows, o, xpos, y, texts, *, ha, va_off) -> None:
    if not rows:
        return
    ps = [p for _lv, p in rows]
    if o["sig_label"] == "p.signif":
        labs = sig.symnum_signif(ps)
    else:
        labs = [f"p = {s}" for s in sig.format_pval(ps)]
    for (lv, _p), lab in zip(rows, labs, strict=True):
        texts.append((xpos[lv], y, lab, ha, va_off))


def _overlays(A: _Ax, P: _Panel, o: dict, xpos: dict, split_violin: bool) -> None:
    plot_type = o["plot_type"]
    if o["add_point"]:
        # position_jitterdodge(seed = 11): R hardcodes the seed, so does this.
        rng = np.random.default_rng(11)
        cells = _cells(P)
        ndodge = max((n for *_, n, _v in cells), default=1)
        jw = o["jitter_width"] / (ndodge + 2)
        size = o["pt_size"] if o["pt_size"] is not None else min(3000 / P.ncells, 0.5)
        for lv, _sp, k, n, v in cells:
            pos = xpos[lv] + _dodge(n, k)
            jx = pos + rng.uniform(-jw, jw, v.size)
            jy = v + rng.uniform(-o["jitter_height"], o["jitter_height"], v.size)
            A.scatter(jx, jy, s=(size * _PT) ** 2, color=o["pt_color"], alpha=o["pt_alpha"],
                      linewidths=0, zorder=3.5)
    if o["add_box"] and plot_type != "box":
        for lv, _sp, k, n, v in _cells(P):
            pos = xpos[lv] + _dodge(n, k)
            bc = o["box_color"]
            A.boxplot([v], positions=[pos], widths=o["box_width"] / n, patch_artist=True,
                      showfliers=False, manage_ticks=False, zorder=4,
                      boxprops=dict(facecolor=bc, edgecolor=bc),
                      whiskerprops=dict(color=bc), capprops=dict(linewidth=0),
                      medianprops=dict(color=bc))
            A.scatter([pos], [np.median(v)], s=(o["box_ptsize"] * _PT) ** 2,
                      facecolor="white", edgecolor="black", linewidths=0.5, zorder=4.5)
    if o["add_trend"] and plot_type in ("violin", "box", "bar"):
        fun = np.mean if plot_type == "bar" else np.median
        pts = [(lv, xpos[lv] + _dodge(n, k), float(fun(v))) for lv, _sp, k, n, v in _cells(P)]
        o["_extra"]["trend"].extend((x, y) for _g, x, y in pts)
        lw = _lw(o["trend_linewidth"])
        if P.has_split:
            # R joins the dodged split summaries *within* each x group
            for lv in P.levels:
                xy = sorted((x, y) for g, x, y in pts if g == lv)
                if len(xy) > 1:
                    A.plot(*zip(*xy, strict=True), color=o["trend_color"], linewidth=lw, zorder=5)
        elif len(pts) > 1:
            A.plot([x for _g, x, _y in pts], [y for *_, y in pts], color=o["trend_color"],
                   linewidth=lw, zorder=5)
        A.scatter([x for _g, x, _y in pts], [y for *_, y in pts],
                  s=(o["trend_ptsize"] * _PT) ** 2, facecolor="white", edgecolor="black",
                  linewidths=0.5, zorder=5.5)
    if o["add_stat"] != "none":
        fun = np.mean if o["add_stat"] == "mean" else np.median
        for lv, _sp, k, n, v in _cells(P):
            A.scatter([xpos[lv] + _dodge(n, k)], [float(fun(v))], marker="v",
                      s=(o["stat_size"] * _PT * 1.5) ** 2, color=o["stat_color"], zorder=6)


# --------------------------------------------------------------------------- #
# layouts
# --------------------------------------------------------------------------- #
def _finish_legend(pg: PanelGrid, legend: _Legend, o: dict) -> None:
    if o["legend_position"] == "none":
        return
    for title, items in legend.entries:
        pg.legend.add(title, [Patch(facecolor=c, edgecolor="black", label=lab)
                              for lab, c in items])
    if o.get("_count_range"):
        lo, hi = o["_count_range"]
        ticks = [t for t in MaxNLocator(4, integer=True).tick_values(lo, hi) if lo <= t <= hi]
        pg.legend.add("Count", [Line2D([], [], marker="o", linestyle="",
                                       markersize=6.0 * math.sqrt(t / hi) * _PT,
                                       markerfacecolor="#4D4D4D", markeredgecolor="black",
                                       label=f"{t:g}") for t in ticks if t > 0])
    if legend.colorbar is not None:
        title, ramp, lims = legend.colorbar
        cmap = LinearSegmentedColormap.from_list("median", ramp, N=256)
        cb = pg.colorbar(ScalarMappable(norm=Normalize(*lims), cmap=cmap))
        cb.set_label(f"{title}:", fontsize=pg.theme.size("legend.title"))
        cb.outline.set_edgecolor("black")


def _title(ax, P: _Panel, o: dict, th) -> None:
    t = o["title"] or P.title
    ax.set_title(f"{o['subtitle']}\n{t}" if o["subtitle"] else t, fontsize=th.size("plot.title"))


def _layout_grid(panels: list[_Panel], legend: _Legend, o: dict) -> Figure:
    flip = o["flip"]
    aspect = o["aspect_ratio"]
    if flip and aspect:
        aspect = 1 / aspect  # R inverts aspect.ratio under coord_flip
    w, h = o["panel_size"]
    th = theme_scp(aspect_ratio=aspect, legend_position=o["legend_position"])
    pg = PanelGrid(len(panels), panel_size=(h, w) if flip else (w, h), nrow=o["nrow"],
                   ncol=o["ncol"], byrow=o["byrow"], keys=[p.key for p in panels], theme=th,
                   legend=LegendColumn(position=o["legend_position"]))
    for P in panels:
        ax = pg.axes[P.key]
        _draw_panel(ax, P, o, stacked=o["stack"])
        A = _Ax(ax, P.flip)
        A.set_pos_label(P.xlab)
        A.set_value_label(o["ylab"])
        if o["stack"] and not flip:
            # a stacked panel keeps its strip on the right, as facet_grid(f ~ .)
            ax.text(1.02, 0.5, P.title, transform=ax.transAxes, ha="left", va="center",
                    fontsize=th.size("strip.text"))
        else:
            _title(ax, P, o, th)
    _finish_legend(pg, legend, o)
    return pg.finish()


def _layout_stack(panels: list[_Panel], legend: _Legend, o: dict) -> Figure:
    """The stacked-violin figure: one column per group column, one row per
    feature, shared x, per-row y limits with exactly two ticks, one y label and
    one legend.  R builds it by binding gtables; this is a single GridSpec."""
    th = theme_scp(aspect_ratio=None, legend_position=o["legend_position"])
    cols = list(dict.fromkeys(p.g for p in panels))
    by_col = {g: [p for p in panels if p.g == g] for g in cols}
    nrow = max(len(v) for v in by_col.values())
    ordered: list[_Panel | None] = []
    for g in cols:
        col = by_col[g]
        ordered.extend(col + [None] * (nrow - len(col)))
    keys = [p.key if p is not None else f"__empty{i}" for i, p in enumerate(ordered)]
    w, h = o["panel_size"]
    pg = PanelGrid(len(keys), panel_size=(w, h * 0.45), nrow=nrow, ncol=len(cols),
                   byrow=False, keys=keys, theme=th, hspace=0.0,
                   legend=LegendColumn(position=o["legend_position"]))
    for ci, g in enumerate(cols):
        col_axes = [pg.axes[keys[ci * nrow + r]] for r in range(nrow)]
        top = col_axes[0]
        for r, ax in enumerate(col_axes):
            P = ordered[ci * nrow + r]
            if P is None:
                ax.set_visible(False)
                continue
            if r > 0:
                ax.sharex(top)
            _draw_panel(ax, P, o, stacked=True)
            ax.text(1.02, 0.5, P.title, transform=ax.transAxes, ha="left", va="center",
                    fontsize=th.size("strip.text"))
            ax.tick_params(axis="x", length=0)
            last = r == len(by_col[g]) - 1
            if not last:
                ax.tick_params(axis="x", labelbottom=False)
                ax.set_xlabel("")
            else:
                ax.set_xlabel(P.xlab)
            ax.set_ylabel("")
        if o["title"] or o["subtitle"]:
            t = o["title"] or ""
            top.set_title(f"{o['subtitle']}\n{t}" if o["subtitle"] else t,
                          fontsize=th.size("plot.title"))
        if len(cols) > 1:
            mid = col_axes[(len(by_col[g]) - 1) // 2]
            mid.set_ylabel(o["ylab"] or "Expression level")
    if len(cols) == 1:
        pg.fig.supylabel(o["ylab"] or "Expression level", fontsize=th.size("axis.title"))
    _finish_legend(pg, legend, o)
    return pg.finish()


def _layout_stack_flip(panels: list[_Panel], legend: _Legend, o: dict) -> Figure:
    """The stacked figure under ``flip``: R binds the panels side by side
    (``cbind``) instead of on top of each other — one row per group column,
    one column per feature, the groups on a shared y axis shown once at the
    left, per-column value limits with exactly two ticks, the feature strips on
    top with their text turned 90 degrees, and one value label at the bottom
    (SCP-plot.R:3635)."""
    th = theme_scp(aspect_ratio=None, legend_position=o["legend_position"])
    rows = list(dict.fromkeys(p.g for p in panels))
    by_row = {g: [p for p in panels if p.g == g] for g in rows}
    ncol = max(len(v) for v in by_row.values())
    ordered: list[_Panel | None] = []
    for g in rows:
        ordered.extend(by_row[g] + [None] * (ncol - len(by_row[g])))
    keys = [p.key if p is not None else f"__empty{i}" for i, p in enumerate(ordered)]
    w, h = o["panel_size"]
    # R binds the panels with plot.margin = c(0, -0.5, 0, 0) mm over ggplot's
    # 5.5 pt default, leaving 5.5 pt - 0.5 mm between neighbours: the room
    # that keeps one column's top label off the next column's bottom one
    gap_in = (5.5 - 0.5 * 72.27 / 25.4) / 72.27
    pg = PanelGrid(len(keys), panel_size=(h * 0.45, w), nrow=len(rows), ncol=ncol,
                   byrow=True, keys=keys, theme=th, wspace=gap_in,
                   legend=LegendColumn(position=o["legend_position"]))
    for ri, g in enumerate(rows):
        row_axes = [pg.axes[keys[ri * ncol + c]] for c in range(ncol)]
        first = row_axes[0]
        for c, ax in enumerate(row_axes):
            P = ordered[ri * ncol + c]
            if P is None:
                ax.set_visible(False)
                continue
            if c > 0:
                ax.sharey(first)
            _draw_panel(ax, P, o, stacked=True)
            ax.text(0.5, 1.02, P.title, transform=ax.transAxes, rotation=90, ha="center",
                    va="bottom", fontsize=th.size("strip.text"))
            ax.tick_params(axis="y", length=0)
            if c > 0:
                ax.tick_params(axis="y", labelleft=False)
                ax.set_ylabel("")
            else:
                ax.set_ylabel(P.xlab)
            ax.set_xlabel("")
        if len(rows) > 1:
            mid = row_axes[(len(by_row[g]) - 1) // 2]
            mid.set_xlabel(o["ylab"] or "Expression level")
    if len(rows) == 1:
        pg.fig.supxlabel(o["ylab"] or "Expression level", fontsize=th.size("axis.title"))
    if o["title"] or o["subtitle"]:
        t = o["title"] or ""
        pg.fig.suptitle(f"{o['subtitle']}\n{t}" if o["subtitle"] else t,
                        fontsize=th.size("plot.title"))
    _finish_legend(pg, legend, o)
    return pg.finish()


# --------------------------------------------------------------------------- #
# the public engine
# --------------------------------------------------------------------------- #
def expression_stat_plot(
    data: pd.DataFrame,
    stat_by: str | Sequence[str],
    *,
    group_by: str | Sequence[str] | None = None,
    split_by: str | None = None,
    bg_by: str | None = None,
    plot_by: Literal["group", "feature"] = "group",
    fill_by: Literal["group", "feature", "expression"] = "group",
    plot_type: ExprPlotType = "violin",
    cells: Sequence[str] | None = None,
    # color
    palette: str = "Paired",
    palcolor: Sequence[str] | dict[str, str] | None = None,
    alpha: float = 1.0,
    bg_palette: str = "Paired",
    bg_palcolor: Sequence[str] | dict[str, str] | None = None,
    bg_alpha: float = 0.2,
    # overlays
    add_box: bool = False,
    add_point: bool = False,
    add_trend: bool = False,
    add_stat: Literal["none", "mean", "median"] = "none",
    add_line: float | None = None,
    jitter_width: float = 0.4,
    jitter_height: float = 0.1,
    # axes
    stack: bool = False,
    flip: bool = False,
    sort: bool | Literal["increasing", "decreasing"] = False,
    same_y_lims: bool = False,
    y_min: float | str | None = None,
    y_max: float | str | None = None,
    y_trans: str = "identity",
    y_nbreaks: int = 5,
    keep_empty: bool = False,
    individual: bool = False,
    # statistics
    comparisons: bool | Sequence[tuple[str, str]] | None = None,
    ref_group: str | None = None,
    pairwise_method: str = "wilcox",
    multiplegroup_comparisons: bool = False,
    multiple_method: str = "kruskal",
    sig_label: Literal["p.signif", "p.format"] = "p.signif",
    sig_labelsize: float = 3.5,
    # layout
    panel_size: tuple[float, float] = (2.6, 2.2),
    aspect_ratio: float | None = None,
    title: str | None = None,
    subtitle: str | None = None,
    xlab: str | None = None,
    ylab: str = "Expression level",
    legend_position: str = "right",
    nrow: int | None = None,
    ncol: int | None = None,
    byrow: bool = True,
    force: bool = False,
    seed: int = 11,
) -> Figure:
    """Per-cell distributions of numeric columns of a DataFrame, by group.

    R original: ``ExpressionStatPlot`` (``SCP-plot.R:3723``), together with the
    ``plot.by = "feature"`` reshape and the ``stack`` assembly that R keeps in
    ``FeatureStatPlot`` (3524) — Hao Zhang, https://github.com/zhanghao-njmu/SCP.
    :func:`scp.pl.feature_stat_plot` is the AnnData wrapper over this.

    ``data`` holds one row per cell: the numeric ``stat_by`` columns and the
    categorical ``group_by`` / ``split_by`` / ``bg_by`` columns.  Level order is
    the column's own (a Categorical is trusted; anything else gets first
    appearance order), never sorted.

    Statistics (ggpubr's ``stat_compare_means``, reproduced in
    :mod:`scp.pl._stat_sig` to R's numbers):

    * ``comparisons=True`` (needs ``split_by``) compares split levels within
      each x group, testing only groups where two or more split levels have at
      least two cells, and switching to ``multiple_method`` as soon as any group
      has three such levels;
    * ``comparisons=[("A", "B"), ...]`` draws ggsignif brackets between x
      groups (splits pooled), stacked by ``step_increase = 0.1`` of the y range
      from ``0.05`` above the panel maximum, tips ``0.03`` long;
    * ``multiplegroup_comparisons=True`` adds one omnibus label at the top left
      — or one per x group when the panel is split, which is what ggpubr does
      when its group and x aesthetics differ.

    Divergences from R, all deliberate:

    * ``ref_group`` is dead in R — ggpubr's bracket path ignores ``ref.group``,
      and SCP passes it nowhere else.  Here ``ref_group`` without
      ``comparisons`` means "every other level against this one"; with
      explicit ``comparisons`` it is ignored with a warning, as in R.
    * split ``multiplegroup_comparisons``: when one x group has a single split
      level, R's ``kruskal.test`` stops and ggpubr drops the whole layer, so
      no group is labelled; here that group is skipped and the rest labelled.
    * ``same_y_lims=True`` with a ``"qNN"`` limit errors in R (``values`` is
      read where the pool is named ``valus``); it works here.
    * ``keep_empty`` is inert in R (absent levels are dropped before the scale
      sees them); here it keeps declared-but-absent levels on the axis.
    * the ``col`` stripes bleed past both panel edges; R's left-edge test
      compares ``xmin`` with ``min(xmax)`` and leaves a gutter.  The cell
      axis's *range* is still R's, typo included: the bars plus the finite
      stripe edges, padded by ``scale_x_discrete``'s 0.6.
    * under ``stack``, each row is ``0.45 * panel_size[1]`` tall and the title
      is drawn once over the stack.
    * ``individual=True`` panels share one legend listing every level they
      show; R gives each one-group panel its own one-entry legend.
    * ``y_trans="log10"`` / ``"log2"`` on data with zeros: R stops (the raw
      minimum is the lower coord limit, ``log(0)`` is ``-Inf`` and the break
      computation errors).  Here the values the transform makes non-finite
      are dropped with ggplot's warning -- every layer, test and limit then
      lives on the log scale, as ggplot's stats do -- and the default limits
      are the kept values' extremes, which is what SCP draws when given that
      ``y.min``.  Breaks are ``scales::log_breaks``, labelled by ``format()``.
    * ``seed`` is accepted for signature parity; jitter uses seed 11 as R's
      ``position_jitterdodge(seed = 11)`` does, so it cannot be changed there
      either.  Jitter positions are not R's — only the RNG seed is shared.

    ``flip=True`` is ``coord_flip``: the values run along x and the groups up
    y, with the factor reversed so the first level is on top, ``aspect_ratio``
    inverted (and ``panel_size`` turned with it), ``col`` cells numbered from
    the top, and every layer — stripes, reference lines, bars and error bars,
    violins, boxes, dots, trend, jitter, brackets and p-values — transposed
    together.  Text keeps its screen-space justification, as ``geom_text``
    does.  Under ``stack`` the panels sit side by side (R's ``cbind``), one row
    per group column, the group labels once at the left, the feature strips on
    top turned 90 degrees, two value ticks per column, and the value label once
    at the bottom; a title becomes the figure's title.  Two divergences:

    * R bug fixed: SCP reverses the factor *after* building the group key it
      hands ggpubr, so an unsplit ``multiplegroup_comparisons`` is tested per
      x position, ``kruskal.test`` fails and R draws no label.  Here the label
      is drawn, at the first (bottom) position.
    * ggsignif positions a flipped bracket's label from mixed coordinates,
      above the bracket's upper end; here it sits beside the bracket's middle.
    """
    if plot_type not in ("violin", "box", "bar", "dot", "col"):
        raise ValueError(f"unknown plot_type {plot_type!r}")
    if fill_by not in ("group", "feature", "expression"):
        raise ValueError(f"unknown fill_by {fill_by!r}")
    if y_trans not in ("identity", "log2", "log10"):
        raise ValueError("`y_trans` must be 'identity', 'log2' or 'log10'")
    if sig_label not in ("p.signif", "p.format"):
        raise ValueError("`sig_label` must be 'p.signif' or 'p.format'")

    feats = [stat_by] if isinstance(stat_by, str) else list(stat_by)
    groups: list[str | None] = ([group_by] if isinstance(group_by, str)
                                else (list(group_by) if group_by else [None]))

    if plot_type == "col":
        if add_box or add_point or add_trend or add_stat != "none":
            warnings.warn("Cannot add other layers when plot_type is 'col'", stacklevel=2)
            add_box = add_point = add_trend = False
            add_stat = "none"
        if multiplegroup_comparisons or comparisons or ref_group is not None:
            warnings.warn("Cannot add comparison when plot_type is 'col'", stacklevel=2)
            multiplegroup_comparisons, comparisons, ref_group = False, None, None
    if comparisons is True and split_by is None:
        raise ValueError("'split.by' must provided when comparisons=TRUE")
    if stack and sort:
        warnings.warn("Set sort to FALSE when stack is TRUE", stacklevel=2)
        sort = False
    if add_line is not None and not isinstance(add_line, (int, float)):
        raise ValueError("`add_line` must be numeric")

    comps: bool | list[tuple[str, str]] | None
    if comparisons is True:
        comps = True
    elif comparisons:
        comps = [tuple(map(str, c)) for c in comparisons]  # type: ignore[misc]
        if any(len(c) != 2 for c in comps):
            raise ValueError("'comparisons' must be a list in which all elements "
                             "must be vectors of length 2")
        if ref_group is not None:
            warnings.warn("`ref_group` is ignored when `comparisons` are given, as in "
                          "ggpubr's bracket layer", stacklevel=2)
    else:
        comps = None
    for m in (pairwise_method,):
        if sig.resolve_method(m) not in sig.PAIRWISE_METHODS:
            raise ValueError(f"`pairwise_method` must be a two-sample test, got {m!r}")
    sig.resolve_method(multiple_method)

    o: dict[str, Any] = dict(
        plot_type=plot_type, alpha=alpha, bg_alpha=bg_alpha, add_box=add_box,
        add_point=add_point, add_trend=add_trend, add_stat=add_stat, add_line=add_line,
        jitter_width=jitter_width, jitter_height=jitter_height, stack=stack,
        y_trans=y_trans, y_nbreaks=y_nbreaks, comparisons=comps,
        pairwise_method=pairwise_method, multiplegroup_comparisons=multiplegroup_comparisons,
        multiple_method=multiple_method, sig_label=sig_label, sig_labelsize=sig_labelsize,
        panel_size=panel_size, aspect_ratio=aspect_ratio, title=title, subtitle=subtitle,
        ylab=ylab, legend_position=legend_position, nrow=nrow, ncol=ncol, byrow=byrow,
        box_color="black", box_width=0.1, box_ptsize=2.0, pt_color="#4D4D4D", pt_size=None,
        pt_alpha=1.0, trend_color="black", trend_linewidth=1.0, trend_ptsize=2.0,
        stat_color="black", stat_size=1.0, line_color="red", line_size=1.0,
        flip=flip, record={},
    )
    legend = _Legend()
    common: dict[str, Any] = dict(fill_by=fill_by, plot_type=plot_type, keep_empty=keep_empty,
                  individual=individual, palette=palette, palcolor=palcolor,
                  bg_palette=bg_palette, bg_palcolor=bg_palcolor, same_y_lims=same_y_lims,
                  y_min=y_min, y_max=y_max, sort=sort, force=force, legend=legend,
                  legend_on=legend_position != "none", y_trans=y_trans)
    if y_trans in _LOG_BASE:
        # ggplot's two warnings, once per call rather than once per panel
        cols = [f for f in dict.fromkeys(feats) if f in data.columns]
        vals = data[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        n_bad = int((np.isfinite(vals) & (vals <= 0)).sum())
        if n_bad:
            warnings.warn(f"{y_trans} transformation introduced infinite values; removed "
                          f"{n_bad} non-positive values from every layer", stacklevel=2)

    if plot_by == "feature":
        panels = _plan_by_feature(data, feats, groups, split_by, bg_by, cells, xlab, common)
    elif plot_by == "group":
        panels = _plan(data, feats, group_by=groups, split_by=split_by, bg_by=bg_by,
                       cells=cells, xlab=xlab, titles=None, **common)
    else:
        raise ValueError("`plot_by` must be 'group' or 'feature'")

    if ref_group is not None and comps is None:
        lv_all = panels[0].levels if panels else []
        if ref_group not in lv_all:
            raise ValueError(f"`ref_group` {ref_group!r} is not on the x axis {lv_all}")
        o["comparisons"] = [(ref_group, lv) for lv in lv_all if lv != ref_group]

    for P in panels:
        P.flip = flip

    n_feats = len(feats)
    if stack and n_feats > 1 and not individual:
        fig = (_layout_stack_flip if flip else _layout_stack)(panels, legend, o)
    else:
        fig = _layout_grid(panels, legend, o)
    # what was computed per panel (y limits, p-values, labels): kept for tests
    # and for anyone who wants the numbers behind the figure
    fig._scp_stat_record = o["record"]  # type: ignore[attr-defined]
    return fig


def _plan_by_feature(data, feats, groups, split_by, bg_by, cells, xlab, common) -> list[_Panel]:
    """``plot.by = "feature"``: x is the feature, one panel per cell group.

    R melts cells x features into long form and calls ``ExpressionStatPlot``
    once per group level with ``group.by = "Features"``.  The panels come in
    the order the levels first *appear* in the cells (R loops over
    ``unique()`` of the column), not in declared level order.
    """
    if len(groups) > 1:
        raise ValueError("The 'group.by' must have a length of 1 when 'plot.by' is set to 'feature'")
    if bg_by is not None:
        warnings.warn("'bg.by' is invalid when plot.by is set to 'feature'", stacklevel=3)
    g = groups[0]
    dat = data
    if cells is not None:
        dat = dat.loc[dat.index.intersection(pd.Index(cells))]
    feats = [f for f in dict.fromkeys(feats) if f in dat.columns]
    if not feats:
        raise ValueError("none of `stat_by` could be resolved")
    gv = (pd.Series("All groups", index=dat.index) if g is None
          else as_ordered_categorical(dat[g]).astype(object))
    panels: list[_Panel] = []
    for lv in pd.unique(gv.dropna()):
        rows = gv.index[(gv == lv).to_numpy()]
        long = {
            "Features": pd.Categorical(np.repeat(feats, len(rows)), categories=feats),
            "__value__": np.concatenate([pd.to_numeric(dat.loc[rows, f], errors="coerce")
                                         .to_numpy(dtype=float) for f in feats]),
        }
        idx = [f"{c}-{f}" for f in feats for c in rows]
        frame = pd.DataFrame(long, index=idx)
        if split_by is not None:
            sp = as_ordered_categorical(dat[split_by])
            frame[split_by] = pd.Categorical(np.tile(sp.loc[rows].to_numpy(), len(feats)),
                                             categories=sp.cat.categories)
        panels.extend(_plan(frame, ["__value__"], group_by=["Features"], split_by=split_by,
                            bg_by=None, cells=None, xlab=xlab,
                            titles={"__value__": str(lv)}, **common))
        for p in panels[-1:]:
            p.key = p.key.replace("__value__", str(lv), 1)
    return panels
