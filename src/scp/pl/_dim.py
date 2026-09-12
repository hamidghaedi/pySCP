"""Dimensional-reduction plots.

``cell_dim_plot`` is implemented as the **reference pattern** every other plot
function should follow: resolve -> build a per-panel frame -> allocate a
PanelGrid -> draw into each Axes -> register legend handles.  Read it before
porting anything else.

The remaining functions here carry their full, final signatures but raise
``NotImplementedError``.  Do not change a signature without updating
``docs/04_api_mapping.md`` — the mapping table is the contract.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_hex, to_rgb
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from ..colors import adjcolors, blendcolors
from ..fetch import as_ordered_categorical, default_reduction, fetch_data, reduction_key
from ..layout import LegendColumn, PanelGrid
from ..palettes import NA_COLOR_DEFAULT, continuous_palette, discrete_palette
from ..theme import Theme, halo, theme_scp

__all__ = ["cell_dim_plot", "feature_dim_plot", "cell_dim_plot_3d", "feature_dim_plot_3d"]

BlendMode = Literal["blend", "average", "screen", "multiply"]


def _auto_pt_size(n: int) -> float:
    """SCP: ``min(3000 / n_cells, 0.5)`` in ggplot mm; scaled to matplotlib pt^2."""
    mm = min(3000.0 / max(n, 1), 0.5)
    return max((mm * 2.845) ** 2, 0.5)  # mm -> pt, then area


def cell_dim_plot(
    adata,
    group_by: str | Sequence[str],
    *,
    reduction: str | None = None,
    dims: tuple[int, int] = (1, 2),
    split_by: str | None = None,
    cells: Sequence[str] | None = None,
    # aesthetics
    palette: str = "Paired",
    palcolor: Sequence[str] | dict[str, str] | None = None,
    bg_color: str = NA_COLOR_DEFAULT,
    pt_size: float | None = None,
    pt_alpha: float = 1.0,
    show_na: bool = False,
    show_stat: bool | None = None,
    raster: bool | None = None,
    raster_dpi: tuple[int, int] = (512, 512),
    # labels
    label: bool = False,
    label_size: float = 4.0,
    label_fg: str = "white",
    label_bg: str = "black",
    label_bg_r: float = 0.1,
    label_insitu: bool = False,
    label_repel: bool = False,
    label_repulsion: float = 20.0,
    # highlight
    cells_highlight: Sequence[str] | bool | None = None,
    cols_highlight: str = "black",
    sizes_highlight: float = 1.0,
    stroke_highlight: float = 0.5,
    # overlays (see docs/porting_briefs/dim_plots.md §1.6)
    add_density: bool = False,
    density_filled: bool = False,
    add_mark: bool = False,
    mark_type: Literal["hull", "ellipse", "rect", "circle"] = "hull",
    graph: str | None = None,
    lineages: Sequence[str] | None = None,
    paga: bool | dict | None = None,
    velocity: str | None = None,
    stat_by: str | None = None,
    # theme / combination
    theme: Theme | None = None,
    aspect_ratio: float | None = 1.0,
    panel_size: tuple[float, float] = (2.6, 2.6),
    title: str | None = None,
    subtitle: str | None = None,
    xlab: str | None = None,
    ylab: str | None = None,
    legend_position: Literal["none", "left", "right", "bottom", "top"] = "right",
    nrow: int | None = None,
    ncol: int | None = None,
    byrow: bool = True,
    force: bool = False,
    seed: int = 11,
    ax: Axes | None = None,
) -> Figure:
    """Color cells on a 2-D embedding by one or more categorical ``obs`` columns.

    The R original is ``CellDimPlot``.  Behaviours that are easy to lose and
    must be preserved:

    * every split panel contains **all** cells — those outside the current
      split are drawn in ``bg_color``, not dropped (contrast with
      :func:`feature_dim_plot`, which genuinely partitions);
    * axis limits come from the *global* embedding range, so panels are
      directly comparable;
    * draw order is background cells first, then a **seeded shuffle** of the
      rest, so no group systematically overplots another;
    * labels sit at the per-group *marginal* median of x and y (median of x and
      median of y computed independently — not a medoid, not a density peak);
    * with ``label_insitu=False`` the plot shows ``1, 2, 3 ...`` and the legend
      reads ``"1: Ductal"``.

    Parameters are documented in ``docs/04_api_mapping.md``; overlay arguments
    (``graph``, ``lineages``, ``paga``, ``velocity``, ``stat_by``,
    ``add_density``, ``add_mark``) are declared here but raise until the
    corresponding milestone lands.
    """
    rng = np.random.default_rng(seed)
    groups = [group_by] if isinstance(group_by, str) else list(group_by)

    for unsupported, name in (
        (add_density, "add_density"),
        (add_mark, "add_mark"),
        (graph, "graph"),
        (lineages, "lineages"),
        (paga, "paga"),
        (velocity, "velocity"),
        (stat_by, "stat_by"),
    ):
        if unsupported:
            raise NotImplementedError(
                f"`{name}` is specified in docs/porting_briefs/dim_plots.md but not yet "
                "implemented; see docs/07_milestones.md."
            )

    reduction = reduction or default_reduction(adata)
    key = reduction_key(reduction)
    emb = np.asarray(adata.obsm[reduction])
    xi, yi = dims[0] - 1, dims[1] - 1

    obs = adata.obs
    frame = pd.DataFrame(
        {"x": emb[:, xi], "y": emb[:, yi]}, index=pd.Index(adata.obs_names)
    )
    # Capture the level order HERE, while the Categorical still exists.
    # Storing `.to_numpy()` flattens it to an object array, and re-deriving the
    # levels from those values later yields first-appearance-in-data order
    # rather than the column's declared order -- which silently shifts every
    # palette index and legend position.
    group_levels: dict[str, list[str]] = {}
    for g in groups:
        cat = as_ordered_categorical(obs[g], show_na=show_na)
        group_levels[g] = [str(lv) for lv in cat.cat.categories]
        frame[g] = cat.to_numpy()
    split_levels: list[str | None]
    if split_by is None:
        split_levels = [None]
        frame["__split__"] = ""
    else:
        sp = as_ordered_categorical(obs[split_by])
        frame["__split__"] = sp.to_numpy()
        split_levels = list(sp.cat.categories)

    # Global limits, computed before any `cells` subsetting -- panels stay aligned.
    xlim = (float(np.nanmin(emb[:, xi])), float(np.nanmax(emb[:, xi])))
    ylim = (float(np.nanmin(emb[:, yi])), float(np.nanmax(emb[:, yi])))

    if cells is not None:
        frame = frame.loc[frame.index.intersection(pd.Index(cells))]

    n_cells = len(frame)
    size = pt_size if pt_size is not None else _auto_pt_size(n_cells)
    if raster is None:
        raster = n_cells > 100_000
    if show_stat is None:
        show_stat = not (theme.blank if theme else False)

    for g in groups:
        nlev = len(pd.unique(frame[g].dropna()))
        if nlev > 100 and not force:
            raise ValueError(
                f"`{g}` has {nlev} levels, which will produce an unreadable legend. "
                "Pass force=True to proceed anyway."
            )

    th = theme or theme_scp(aspect_ratio=aspect_ratio, legend_position=legend_position)
    keys = [f"{s or ''}:{g}" for g in groups for s in split_levels]
    legend = LegendColumn(position=legend_position)
    pg = PanelGrid(
        len(keys), panel_size=panel_size, nrow=nrow, ncol=ncol, byrow=byrow,
        keys=keys, theme=th, legend=legend,
    )

    seen_legend: set[str] = set()
    for g in groups:
        levels = group_levels[g]
        colors = discrete_palette(levels, palette=palette, palcolor=palcolor)

        for s in split_levels:
            panel = pg.axes[f"{s or ''}:{g}"]
            dat = frame.copy()
            if s is not None:
                dat.loc[dat["__split__"] != s, g] = np.nan  # mask, do not drop

            is_bg = dat[g].isna().to_numpy()
            order = np.concatenate([np.flatnonzero(is_bg), rng.permutation(np.flatnonzero(~is_bg))])
            dat = dat.iloc[order]
            cvec = np.where(
                dat[g].isna().to_numpy(),
                bg_color,
                [colors.get(str(v), bg_color) for v in dat[g]],
            )
            panel.scatter(
                dat["x"], dat["y"], s=size, c=cvec, alpha=pt_alpha,
                linewidths=0, rasterized=raster,
            )

            if cells_highlight is not None:
                hl = (
                    dat.index[~dat[g].isna()]
                    if cells_highlight is True
                    else dat.index.intersection(pd.Index(cells_highlight))
                )
                sub = dat.loc[hl]
                panel.scatter(sub["x"], sub["y"], s=(sizes_highlight + stroke_highlight) ** 2 * 8,
                              c=cols_highlight, linewidths=0, rasterized=raster)
                panel.scatter(sub["x"], sub["y"], s=sizes_highlight ** 2 * 8,
                              c=[colors.get(str(v), bg_color) for v in sub[g]],
                              linewidths=0, rasterized=raster)

            present = [lv for lv in levels if (dat[g] == lv).any()]
            if label and present:
                centers = dat.dropna(subset=[g]).groupby(g, observed=True)[["x", "y"]].median()
                for i, lv in enumerate(present, start=1):
                    if lv not in centers.index:
                        continue
                    cx, cy = centers.loc[lv, "x"], centers.loc[lv, "y"]
                    txt = str(lv) if label_insitu else str(i)
                    t = panel.text(cx, cy, txt, ha="center", va="center",
                                   fontsize=label_size * 2.5, color=label_fg,
                                   fontweight="bold", zorder=5)
                    halo(t, foreground=label_bg, radius=label_bg_r)

            panel.set_xlim(*xlim)
            panel.set_ylim(*ylim)
            panel.set_xlabel(xlab if xlab is not None else f"{key}{dims[0]}")
            panel.set_ylabel(ylab if ylab is not None else f"{key}{dims[1]}")
            # title and subtitle are separate slots -- ggplot stacks them, and
            # so must we, or show_stat silently eats a user-supplied title.
            if title:
                panel.set_title(title, fontsize=th.size("plot.title"), loc="center")
            if show_stat or subtitle is not None:
                n_shown = int((~dat[g].isna()).sum())
                sub_t = subtitle if subtitle is not None else f"{s or ''} nCells:{n_shown}"
                panel.set_title(sub_t, fontsize=th.size("plot.subtitle"), loc="left")

            if g not in seen_legend and legend_position != "none":
                seen_legend.add(g)
                counts = dat[g].value_counts()
                handles = []
                for i, lv in enumerate(present, start=1):
                    lbl = str(lv)
                    if show_stat:
                        lbl = f"{lbl}({int(counts.get(lv, 0))})"
                    if label and not label_insitu:
                        lbl = f"{i}: {lbl}"
                    handles.append(
                        Line2D([], [], marker="o", linestyle="none", markersize=6,
                               markerfacecolor=colors[str(lv)], markeredgecolor="none", label=lbl)
                    )
                legend.add(g, handles)

    if ax is not None:
        warnings.warn("`ax=` is ignored when more than one panel is produced.", stacklevel=2)
    return pg.finish()


def feature_dim_plot(
    adata,
    features: str | Sequence[str] | dict[str, Sequence[str]],
    *,
    reduction: str | None = None,
    dims: tuple[int, int] = (1, 2),
    split_by: str | None = None,
    cells: Sequence[str] | None = None,
    layer: str | None = None,
    use_raw: bool = False,
    # scaling -- see porting brief §2.4
    palette: str = "Spectral",
    palcolor: Sequence[str] | None = None,
    bg_cutoff: float | None = 0.0,
    bg_color: str = NA_COLOR_DEFAULT,
    keep_scale: Literal["feature", "all"] | None = "feature",
    lower_quantile: float = 0.0,
    upper_quantile: float = 0.99,
    lower_cutoff: float | None = None,
    upper_cutoff: float | None = None,
    # multi-feature blending
    compare_features: bool = False,
    color_blend_mode: BlendMode = "blend",
    calculate_coexp: bool = False,
    # shared with cell_dim_plot
    pt_size: float | None = None,
    pt_alpha: float = 1.0,
    raster: bool | None = None,
    raster_dpi: tuple[int, int] = (512, 512),
    label: bool = False,
    label_size: float = 4.0,
    label_fg: str = "white",
    label_bg: str = "black",
    label_bg_r: float = 0.1,
    label_insitu: bool = False,
    label_repel: bool = False,
    cells_highlight: Sequence[str] | bool | None = None,
    cols_highlight: str = "black",
    sizes_highlight: float = 1.0,
    stroke_highlight: float = 0.5,
    add_density: bool = False,
    density_filled: bool = False,
    graph: str | None = None,
    lineages: Sequence[str] | None = None,
    theme: Theme | None = None,
    aspect_ratio: float | None = 1.0,
    panel_size: tuple[float, float] = (2.6, 2.6),
    show_stat: bool | None = None,
    title: str | None = None,
    subtitle: str | Sequence[str] | None = None,
    xlab: str | None = None,
    ylab: str | None = None,
    legend_position: Literal["none", "left", "right", "bottom", "top"] = "right",
    nrow: int | None = None,
    ncol: int | None = None,
    byrow: bool = True,
    force: bool = False,
    seed: int = 11,
) -> Figure:
    """Color cells on a 2-D embedding by continuous feature values.

    R original: ``FeatureDimPlot``.  Implementation notes live in
    ``docs/porting_briefs/dim_plots.md`` §2.  The three things that define how
    these figures look:

    * **draw order** is ascending value with NaN first, so the highest
      expressing cells land on top;
    * ``bg_cutoff=0`` NaN-masks every value ``<= 0`` *before* plotting, so
      zero-expression cells render in ``bg_color``.  Signed scores need
      ``bg_cutoff=None``;
    * the color limits are **winsorized** at ``upper_quantile=0.99`` — values
      above are saturated, never hidden — and ``keep_scale`` controls whether
      that range is pooled per feature, across everything, or per panel.

    ``compare_features=True`` with more than one feature switches to the
    per-cell color-blending path (``scp.colors.blendcolors``), which produces
    one colorbar per feature rather than a single scale.
    """
    rng = np.random.default_rng(seed)
    for unsupported, name in (
        (add_density, "add_density"), (graph, "graph"), (lineages, "lineages"),
    ):
        if unsupported:
            raise NotImplementedError(
                f"`{name}` is specified in docs/porting_briefs/dim_plots.md but not yet "
                "implemented; see docs/07_milestones.md."
            )

    # A named mapping flattens to a flat list whose names become subtitles.
    subtitles: dict[str, str] = {}
    if isinstance(features, dict):
        flat: list[str] = []
        for group, members in features.items():
            members = [members] if isinstance(members, str) else list(members)
            flat.extend(members)
            subtitles.update({m: group for m in members})
        feats = flat
    else:
        feats = [features] if isinstance(features, str) else list(features)

    # R guards at >50 features here, despite the roxygen text saying 100.
    if len(feats) > 50 and not force:
        raise ValueError(
            f"{len(feats)} features will produce an unreadable figure. "
            "Pass force=True to proceed anyway."
        )

    reduction = reduction or default_reduction(adata)
    key = reduction_key(reduction)
    emb = np.asarray(adata.obsm[reduction])
    xi, yi = dims[0] - 1, dims[1] - 1

    values = fetch_data(adata, feats, layer=layer, use_raw=use_raw)
    resolved = [f for f in feats if f in values.columns]
    if not resolved:
        raise ValueError("none of `features` could be resolved")

    frame = pd.DataFrame(
        {"x": emb[:, xi], "y": emb[:, yi]}, index=pd.Index(adata.obs_names)
    )
    for f in resolved:
        frame[f] = pd.to_numeric(values[f], errors="coerce").to_numpy()

    if split_by is None:
        split_levels: list[str | None] = [None]
        frame["__split__"] = ""
    else:
        sp = as_ordered_categorical(adata.obs[split_by])
        frame["__split__"] = sp.to_numpy()
        split_levels = list(sp.cat.categories)

    # Global limits, from every cell, so split panels stay comparable.
    xlim = (float(np.nanmin(emb[:, xi])), float(np.nanmax(emb[:, xi])))
    ylim = (float(np.nanmin(emb[:, yi])), float(np.nanmax(emb[:, yi])))

    if cells is not None:
        frame = frame.loc[frame.index.intersection(pd.Index(cells))]

    # bg_cutoff is applied globally and destructively BEFORE plotting: every
    # value <= it becomes NaN and renders in bg_color.  The comparison is <=,
    # so exact zeros are background.  Signed scores need bg_cutoff=None.
    if bg_cutoff is not None:
        for f in resolved:
            frame.loc[frame[f] <= bg_cutoff, f] = np.nan

    n_cells = len(frame)
    size = pt_size if pt_size is not None else _auto_pt_size(n_cells)
    if raster is None:
        raster = n_cells > 100_000
    if show_stat is None:
        show_stat = not (theme.blank if theme else False)

    blend_path = bool(compare_features) and len(resolved) > 1
    if calculate_coexp and not blend_path:
        raise NotImplementedError(
            "`calculate_coexp` is specified in docs/porting_briefs/dim_plots.md §2.2 "
            "but not yet implemented."
        )

    th = theme or theme_scp(aspect_ratio=aspect_ratio, legend_position=legend_position)
    panel_feats = ["|".join(resolved)] if blend_path else resolved
    keys = [f"{s or ''}:{f}" for s in split_levels for f in panel_feats]
    legend = LegendColumn(position=legend_position)
    pg = PanelGrid(
        len(keys), panel_size=panel_size,
        nrow=nrow if nrow is not None else (None if ncol else len(split_levels)),
        ncol=ncol if ncol is not None else (None if nrow else len(panel_feats)),
        byrow=byrow, keys=keys, theme=th, legend=legend,
    )

    def _limits(f: str, panel: pd.DataFrame) -> tuple[float, float]:
        """Winsorized colour limits; `keep_scale` chooses the pooling scope."""
        if keep_scale == "all":
            pool = frame[resolved].to_numpy().ravel()
        elif keep_scale == "feature":
            pool = frame[f].to_numpy()
        else:
            pool = panel[f].to_numpy()
        pool = pool[np.isfinite(pool)]
        if pool.size == 0:
            return (0.0, 1.0)
        lo = lower_cutoff if lower_cutoff is not None else float(np.quantile(pool, lower_quantile))
        hi = upper_cutoff if upper_cutoff is not None else float(np.quantile(pool, upper_quantile))
        return (lo, hi + 0.001)  # +0.001 so the range can never be degenerate

    ramp = continuous_palette(palette=palette if not blend_path else "Spectral",
                              palcolor=None if blend_path else palcolor, n=100)
    cmap = LinearSegmentedColormap.from_list("scp", ramp, N=256).with_extremes(bad=bg_color)

    seen_cbar: set[str] = set()
    for s in split_levels:
        # Unlike cell_dim_plot this genuinely partitions -- a split panel holds
        # only its own cells, with no grey "other split" points.
        part = frame if s is None else frame[frame["__split__"] == s]
        for f in panel_feats:
            ax_p = pg.axes[f"{s or ''}:{f}"]

            if blend_path:
                hues = discrete_palette(resolved, palette="Set1", palcolor=palcolor)
                per_feature: list[list] = []
                for feat in resolved:
                    lo, hi = _limits(feat, part)
                    stops = [adjcolors(hues[feat], 0.1), hues[feat]]
                    sub_cmap = LinearSegmentedColormap.from_list(f"b{feat}", stops, N=256)
                    norm = Normalize(vmin=lo, vmax=hi, clip=True)
                    vals = part[feat].to_numpy()
                    per_feature.append(
                        [None if not np.isfinite(v) else to_hex(sub_cmap(norm(v))) for v in vals]
                    )
                blended = [
                    blendcolors([c[i] for c in per_feature], mode=color_blend_mode)
                    for i in range(len(part))
                ]
                # Sort brightest-first so the most co-expressing points land on top;
                # cells with no signal on any feature sort to the bottom.
                lum = np.array([
                    np.nan if c is None else float(sum(to_rgb(c))) for c in blended
                ])
                order = np.concatenate([
                    np.flatnonzero(np.isnan(lum)),
                    np.flatnonzero(~np.isnan(lum))[np.argsort(-lum[~np.isnan(lum)], kind="stable")],
                ])
                cvec = [bg_color if blended[i] is None else blended[i] for i in order]
                d = part.iloc[order]
                ax_p.scatter(d["x"], d["y"], s=size, c=cvec, alpha=pt_alpha,
                             linewidths=0, rasterized=raster)
                for feat in resolved:
                    if feat in seen_cbar or legend_position == "none":
                        continue
                    seen_cbar.add(feat)
                    lo, hi = _limits(feat, part)
                    stops = [adjcolors(hues[feat], 0.1), hues[feat]]
                    sm = ScalarMappable(norm=Normalize(lo, hi),
                                        cmap=LinearSegmentedColormap.from_list(f"c{feat}", stops))
                    cb = pg.fig.colorbar(sm, ax=list(pg.axes.values()), fraction=0.02, pad=0.02)
                    cb.outline.set_edgecolor("black")
                    cb.set_label(feat, fontsize=th.size("legend.title"))
            else:
                lo, hi = _limits(f, part)
                vals = part[f].to_numpy(dtype=float)
                # NaN first, then ascending, so the highest expressers draw last.
                order = np.concatenate([
                    np.flatnonzero(np.isnan(vals)),
                    np.flatnonzero(~np.isnan(vals))[np.argsort(vals[~np.isnan(vals)], kind="stable")],
                ])
                d = part.iloc[order]
                v = np.clip(d[f].to_numpy(dtype=float), lo, hi)  # winsorize, never drop
                # Two layers, as SCP does: background cells first so they can
                # never overplot a cell that actually carries signal.
                bg = np.isnan(v)
                if bg.any():
                    ax_p.scatter(d["x"][bg], d["y"][bg], s=size, c=bg_color,
                                 alpha=pt_alpha, linewidths=0, rasterized=raster)
                if (~bg).any():
                    ax_p.scatter(d["x"][~bg], d["y"][~bg], s=size, c=v[~bg],
                                 cmap=cmap, norm=Normalize(lo, hi), alpha=pt_alpha,
                                 linewidths=0, rasterized=raster)
                cbar_key = f if keep_scale == "feature" else ("__all__" if keep_scale == "all" else f"{s}:{f}")
                if cbar_key not in seen_cbar and legend_position != "none":
                    seen_cbar.add(cbar_key)
                    sm = ScalarMappable(norm=Normalize(lo, hi), cmap=cmap)
                    # ax MUST be a list. Handing constrained layout a bare Axes
                    # here zeroes a height ratio and savefig dies dividing by it.
                    cb = pg.fig.colorbar(sm, ax=[ax_p], fraction=0.046, pad=0.03)
                    cb.outline.set_edgecolor("black")

            if cells_highlight is not None:
                hl = (
                    part.index[part[resolved].notna().any(axis=1)]
                    if cells_highlight is True
                    else part.index.intersection(pd.Index(cells_highlight))
                )
                sub = part.loc[hl]
                ax_p.scatter(sub["x"], sub["y"], s=(sizes_highlight + stroke_highlight) ** 2 * 8,
                             c=cols_highlight, linewidths=0, rasterized=raster)

            ax_p.set_xlim(*xlim)
            ax_p.set_ylim(*ylim)
            ax_p.set_xlabel(xlab if xlab is not None else f"{key}{dims[0]}")
            ax_p.set_ylabel(ylab if ylab is not None else f"{key}{dims[1]}")
            strip = "|".join(resolved) if blend_path else f
            head = title if title else strip
            if show_stat:
                # R reports the POSITIVE cells and their share, not the panel
                # total: cells at or below bg_cutoff are background and excluded.
                cols = resolved if blend_path else [f]
                n_pos = int(part[cols].notna().any(axis=1).sum())
                pct = (100.0 * n_pos / len(part)) if len(part) else 0.0
                stat = f"nPos:{n_pos}, {pct:.1f}%"
                if s is not None:
                    stat = f"{s} | {stat}"
                if not blend_path and f in subtitles:
                    stat = f"{subtitles[f]} | {stat}"
                # Stacked in one title: ggplot puts the subtitle above the strip,
                # and two matplotlib titles at different loc share a line.
                ax_p.set_title(f"{stat}\n{head}", fontsize=th.size("plot.subtitle"))
            else:
                ax_p.set_title(head, fontsize=th.size("plot.title"))

    _ = rng  # seed is accepted for signature parity; this path has no shuffle
    return pg.finish()


def cell_dim_plot_3d(adata, group_by: str, **kwargs):
    """plotly Scatter3d version of :func:`cell_dim_plot`. Milestone 8."""
    raise NotImplementedError("Milestone 8. Spec: docs/porting_briefs/dim_plots.md §3.1.")


def feature_dim_plot_3d(adata, features: str | Sequence[str], **kwargs):
    """plotly Scatter3d version of :func:`feature_dim_plot`. Milestone 8."""
    raise NotImplementedError("Milestone 8. Spec: docs/porting_briefs/dim_plots.md §3.2.")
