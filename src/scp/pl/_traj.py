"""Trajectory, graph and velocity plots.

Most of this family is cheaper in Python than in R, because scanpy and scvelo
already persist what SCP had to reconstruct: ``adata.uns['paga']`` has the same
keys SCP reads, and ``adata.obsm['velocity_<basis>']`` is the velocity field.

Spec: ``docs/porting_briefs/trajectory_enrichment.md``.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure

from ..fetch import as_ordered_categorical, default_reduction, fetch_data, reduction_key
from ..io import read_paga
from ..layout import LegendColumn, PanelGrid
from ..palettes import discrete_palette
from ..theme import halo, theme_scp

__all__ = [
    "graph_plot",
    "paga_plot",
    "lineage_plot",
    "velocity_plot",
    "dynamic_plot",
    "projection_plot",
    "shorten_segments",
    "compute_velocity_on_grid",
]


def shorten_segments(
    x: np.ndarray, y: np.ndarray, xend: np.ndarray, yend: np.ndarray,
    shorten_start: float = 0.0, shorten_end: float = 0.0, offset: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Port of ``segementsDf`` — pure geometry, port it verbatim.

    Shortening pulls segment ends back so arrowheads do not sit under node
    circles.  ``offset`` displaces the whole segment perpendicular to itself,
    which is how the forward and reverse transitions between the same pair of
    nodes avoid overlapping.
    """
    dx, dy = xend - x, yend - y
    dist = np.hypot(dx, dy)
    with np.errstate(invalid="ignore", divide="ignore"):
        px, py = np.where(dist > 0, dx / dist, 0.0), np.where(dist > 0, dy / dist, 0.0)
    x = x + px * shorten_start
    y = y + py * shorten_start
    xend = xend - px * shorten_end
    yend = yend - py * shorten_end
    return x - py * offset, y + px * offset, xend - py * offset, yend + px * offset


def graph_plot(
    node: pd.DataFrame,
    edge: np.ndarray,
    *,
    transition: np.ndarray | None = None,
    node_coord: tuple[str, str] = ("x", "y"),
    node_group: str | None = None,
    node_palette: str = "Paired",
    node_size: float | str = 4.0,
    edge_threshold: float = 0.01,
    use_triangular: Literal["upper", "lower", "both"] = "upper",
    edge_line: Literal["straight", "curved"] = "straight",
    edge_line_curvature: float = 0.3,
    edge_color: str = "#666666",
    edge_size: tuple[float, float] = (0.2, 1.0),
    edge_alpha: float = 0.5,
    edge_shorten: float = 0.0,
    edge_offset: float = 0.0,
    transition_threshold: float = 0.01,
    transition_shorten: float = 0.05,
    transition_offset: float = 0.0,
    label: bool = False,
    label_insitu: bool = False,
    ax: Axes | None = None,
    **kwargs,
) -> Axes:
    """The generic node/edge engine. Milestone 11 — build this first.

    Anything expressible as (node table, edge matrix, optional directed
    transition matrix) routes through here, which is why ``paga_plot`` is
    nearly free once this exists.

    Two behaviours to port exactly:

    * **net transitions.** Only the net directed flow between each pair is
      drawn, as one arrow in the dominant direction::

          t1, t2 = T.copy(), T.copy()
          t1[np.tril_indices_from(t1)] = 0
          t2[np.triu_indices_from(t2)] = 0
          net = t1.T - t2          # net[i,j] = T[j,i] - T[i,j] for i > j
          # negative entries flip (from, to) and take abs

      Drawing every directed edge instead gives a visibly different plot.
    * **edges go dashed when transitions are present** and solid otherwise, so
      connectivity and directed flow read differently.

    Nodes are drawn twice — a black disc at ``node_size * 1.2`` then the
    colored disc — which is what gives them their outline.
    """
    xk, yk = node_coord
    X = node[xk].to_numpy(dtype=float)
    Y = node[yk].to_numpy(dtype=float)
    names = [str(i) for i in node.index]
    E = np.asarray(edge, dtype=float).copy()
    if E.shape[0] != E.shape[1] or E.shape[0] != len(node):
        raise ValueError(f"edge must be square and match node: {E.shape} vs {len(node)}")

    if ax is None:
        _fig, ax = plt.subplots(figsize=(4, 4))
    # global_size converts the fractional shorten/offset into data units
    global_size = float(np.hypot(np.nanmax(np.abs(X)), np.nanmax(np.abs(Y)))) or 1.0

    E[E <= edge_threshold] = np.nan
    if use_triangular == "upper":
        E[np.tril_indices_from(E)] = np.nan
    elif use_triangular == "lower":
        E[np.triu_indices_from(E)] = np.nan

    fi, ti = np.where(np.isfinite(E))
    if len(fi):
        w = E[fi, ti]
        lo, hi = np.nanmin(w), np.nanmax(w)
        span = (hi - lo) or 1.0
        widths = edge_size[0] + (w - lo) / span * (edge_size[1] - edge_size[0])
        x0, y0, x1, y1 = shorten_segments(
            X[fi], Y[fi], X[ti], Y[ti],
            global_size * edge_shorten, global_size * edge_shorten,
            global_size * edge_offset)
        # dashed when transitions are drawn, so connectivity and directed flow
        # read differently
        style = "--" if transition is not None else "-"
        segs = [[(a, b), (c, d)] for a, b, c, d in zip(x0, y0, x1, y1, strict=True)]
        ax.add_collection(LineCollection(segs, linewidths=widths, colors=edge_color,
                                         alpha=edge_alpha, linestyles=style, zorder=1))

    if transition is not None:
        T = np.asarray(transition, dtype=float)
        t1, t2 = T.copy(), T.copy()
        t1[np.tril_indices_from(t1)] = 0
        t2[np.triu_indices_from(t2)] = 0
        # only the NET flow between each pair is drawn, as one arrow in the
        # dominant direction; drawing every directed edge looks quite different
        net = t1.T - t2
        net[np.abs(net) <= transition_threshold] = np.nan
        ai, bi = np.where(np.isfinite(net))
        for a, b in zip(ai, bi, strict=True):
            v = net[a, b]
            src, dst = (a, b) if v > 0 else (b, a)
            x0, y0, x1, y1 = shorten_segments(
                np.array([X[src]]), np.array([Y[src]]),
                np.array([X[dst]]), np.array([Y[dst]]),
                global_size * transition_shorten, global_size * transition_shorten,
                global_size * transition_offset)
            ax.annotate("", xy=(x1[0], y1[0]), xytext=(x0[0], y0[0]),
                        arrowprops=dict(arrowstyle="-|>", color="black",
                                        linewidth=float(edge_size[1]), shrinkA=0, shrinkB=0),
                        zorder=2)

    if node_group is not None:
        cat = as_ordered_categorical(node[node_group])
        levels = [str(x) for x in cat.cat.categories]
        colors = discrete_palette(levels, palette=node_palette)
        cvec = [colors[str(v)] for v in np.asarray(cat).astype(str)]
    else:
        levels, colors = names, discrete_palette(names, palette=node_palette)
        cvec = [colors[n] for n in names]

    sizes = (node[node_size].to_numpy(dtype=float)
             if isinstance(node_size, str) and node_size in node.columns
             else np.full(len(node), float(node_size) if not isinstance(node_size, str) else 4.0))
    area = (sizes / np.nanmax(sizes) * 200.0) if sizes.max() > 20 else sizes ** 2 * 8
    # each node is drawn twice: a black disc, then the coloured one on top
    ax.scatter(X, Y, s=area * 1.2, c="black", zorder=3, linewidths=0)
    ax.scatter(X, Y, s=area, c=cvec, zorder=4, linewidths=0)

    if label:
        for i, n in enumerate(names):
            txt = str(i + 1) if not label_insitu else n
            t = ax.text(X[i], Y[i], txt, ha="center", va="center", fontsize=8,
                        color="white", fontweight="bold", zorder=5)
            halo(t, foreground="black", radius=0.1)

    ax.set_xlabel(kwargs.get("xlab", xk))
    ax.set_ylabel(kwargs.get("ylab", yk))
    if kwargs.get("title"):
        ax.set_title(kwargs["title"])
    ax.set_aspect("equal", adjustable="datalim")
    return ax


def paga_plot(
    adata,
    *,
    reduction: str | None = None,
    dims: tuple[int, int] = (1, 2),
    type: Literal["connectivities", "connectivities_tree"] = "connectivities",
    show_transition: bool = False,
    node_palette: str = "Paired",
    node_size: float | str = 4.0,
    edge_threshold: float = 0.01,
    cells: Sequence[str] | None = None,
    ax: Axes | None = None,
    **kwargs,
) -> Axes:
    """PAGA graph laid over an embedding. Milestone 11.

    Reads ``adata.uns['paga']`` natively — ``connectivities``,
    ``connectivities_tree``, ``groups`` (scanpy) and ``transitions_confidence``
    (scvelo).  Rows and columns are indexed by
    ``adata.obs[groups].cat.categories``.

    Note SCP deliberately **ignores** ``uns['paga']['pos']`` and recomputes
    node positions as the per-group *median* of the chosen embedding, with
    node size available as the group's cell count.  That is what lets the same
    graph be drawn over UMAP, PCA or diffusion coordinates.  Keep that default
    and expose ``use_paga_pos=True`` as an opt-in.

    ``type="connectivities_tree"`` reads both triangles and disables the
    threshold, because the tree matrix is not symmetric-complete.
    """
    conn, groups_key, categories = read_paga(adata, type=type)
    obs_cat = as_ordered_categorical(adata.obs[groups_key])
    levels = [str(x) for x in obs_cat.cat.categories]
    if conn.shape[0] != len(levels):
        raise ValueError(
            f"paga matrix is {conn.shape[0]}x{conn.shape[1]} but "
            f"{groups_key!r} has {len(levels)} levels"
        )

    reduction = reduction or default_reduction(adata)
    key = reduction_key(reduction)
    emb = np.asarray(adata.obsm[reduction])
    xi, yi = dims[0] - 1, dims[1] - 1

    vals = np.asarray(obs_cat).astype(str)
    keep = np.ones(adata.n_obs, dtype=bool)
    if cells is not None:
        keep = np.isin(np.asarray(adata.obs_names), np.asarray(cells))

    # SCP deliberately ignores uns['paga']['pos'] and puts each node at the
    # per-group MEDIAN of the chosen embedding, which is what lets the same
    # graph be drawn over UMAP, PCA or diffusion coordinates.
    rows, sizes, present = [], [], []
    for lv in levels:
        m = keep & (vals == lv)
        if not m.any():
            continue
        rows.append((float(np.median(emb[m, xi])), float(np.median(emb[m, yi]))))
        sizes.append(int(m.sum()))
        present.append(lv)
    idx = [levels.index(lv) for lv in present]
    node = pd.DataFrame(rows, columns=["x", "y"], index=pd.Index(present))
    node["GroupSize"] = sizes
    node[groups_key] = pd.Categorical(present, categories=present)

    E = np.asarray(conn.todense() if hasattr(conn, "todense") else conn, dtype=float)
    E = E[np.ix_(idx, idx)]
    transition = None
    if show_transition:
        tr, _k, _c = read_paga(adata, type="transitions_confidence")
        transition = np.asarray(tr.todense() if hasattr(tr, "todense") else tr,
                                dtype=float)[np.ix_(idx, idx)]

    # the tree matrix is not symmetric-complete, so read both triangles and
    # drop the threshold
    triangular = "both" if type == "connectivities_tree" else "upper"
    thresh = 0.0 if type == "connectivities_tree" else edge_threshold

    kwargs.setdefault("title", "PAGA")
    kwargs.setdefault("xlab", f"{key}{dims[0]}")
    kwargs.setdefault("ylab", f"{key}{dims[1]}")
    return graph_plot(node, E, transition=transition, node_group=groups_key,
                      node_palette=node_palette, node_size="GroupSize",
                      edge_threshold=thresh, use_triangular=triangular, ax=ax, **kwargs)


def lineage_plot(
    adata,
    lineages: Sequence[str],
    *,
    reduction: str | None = None,
    dims: tuple[int, int] = (1, 2),
    trim: tuple[float, float] = (0.01, 0.99),
    span: float = 0.75,
    palette: str = "Dark2",
    linewidth: float = 1.0,
    line_bg: str = "white",
    line_bg_stroke: float = 0.5,
    whiskers: bool = False,
    ax: Axes | None = None,
    **kwargs,
) -> Axes:
    """Principal curves over an embedding, one per pseudotime column. Milestone 12.

    Each embedding axis is LOESS-smoothed **independently** against pseudotime
    (``degree=2``, ``span``), after trimming to the 1st..99th percentile.  Use
    ``skmisc.loess`` for degree-2 parity; ``statsmodels`` lowess is degree-1.

    The curve is drawn twice — a white halo at ``linewidth + line_bg_stroke``
    then the colored path — so it stays readable over dense points.

    ``lineages`` are ``obs`` columns of pseudotime with NaN off-lineage,
    whatever produced them (scFates, Palantir, CellRank, pyslingshot).
    """
    reduction = reduction or default_reduction(adata)
    key = reduction_key(reduction)
    emb = np.asarray(adata.obsm[reduction])
    xi, yi = dims[0] - 1, dims[1] - 1
    if ax is None:
        _fig, ax = plt.subplots(figsize=(4, 4))

    lineages = list(lineages)
    colors = discrete_palette([str(x) for x in lineages], palette=palette)
    ax.scatter(emb[:, xi], emb[:, yi], s=2, c="#D9D9D9", linewidths=0, zorder=1)

    for lin in lineages:
        pt = pd.to_numeric(adata.obs[lin], errors="coerce").to_numpy(dtype=float)
        m = np.isfinite(pt)
        if not m.any():
            continue
        # trim the pseudotime tails before fitting, as SCP does, so the curve is
        # not dragged around by a handful of extreme cells
        lo, hi = np.nanquantile(pt[m], trim[0]), np.nanquantile(pt[m], trim[1])
        m = m & (pt >= lo) & (pt <= hi)
        order = np.argsort(pt[m])
        t = pt[m][order]
        px, py = emb[m, xi][order], emb[m, yi][order]

        try:
            # degree-2 LOESS: statsmodels' lowess is degree 1 and will not match
            from skmisc.loess import loess
            fx = loess(t, px, span=span, degree=2)
            fx.fit()
            fy = loess(t, py, span=span, degree=2)
            fy.fit()
            sx, sy = fx.outputs.fitted_values, fy.outputs.fitted_values
        except ImportError:
            warnings.warn(
                "skmisc is not installed, so the lineage curve falls back to a "
                "rolling mean rather than the degree-2 LOESS SCP uses "
                "(`pip install scp-plot[stats]`).",
                stacklevel=2,
            )
            w = max(len(t) // 20, 3)
            sx = pd.Series(px).rolling(w, center=True, min_periods=1).mean().to_numpy()
            sy = pd.Series(py).rolling(w, center=True, min_periods=1).mean().to_numpy()

        # a white casing under the coloured line keeps it readable over points
        ax.plot(sx, sy, color=line_bg, linewidth=linewidth + 2 * line_bg_stroke, zorder=3)
        ax.plot(sx, sy, color=colors[str(lin)], linewidth=linewidth, zorder=4, label=str(lin))

    ax.set_xlabel(kwargs.get("xlab", f"{key}{dims[0]}"))
    ax.set_ylabel(kwargs.get("ylab", f"{key}{dims[1]}"))
    if kwargs.get("title"):
        ax.set_title(kwargs["title"])
    return ax


def compute_velocity_on_grid(
    X_emb: np.ndarray,
    V_emb: np.ndarray,
    *,
    density: float = 1.0,
    smooth: float = 0.5,
    n_neighbors: int | None = None,
    min_mass: float = 1.0,
    scale: float = 1.0,
    adjust_for_stream: bool = False,
    cutoff_perc: float = 5.0,
):
    """Delegate to scvelo rather than reimplementing.

    SCP's R version is a transliteration of
    ``scvelo.plotting.velocity_embedding_grid.compute_velocity_on_grid`` with
    one deliberate divergence: scvelo's +-1% range padding is commented out, so
    grids differ slightly at the border.  Match scvelo, not SCP.

    Note the ``min_mass`` reparameterisation in stream mode:
    ``min_mass -> 10 ** (min_mass - 6)``, so the default 1 means ``1e-5``.
    """
    try:
        from scvelo.plotting.velocity_embedding_grid import compute_velocity_on_grid as _cvog
        from scvelo.tools.velocity_embedding import velocity_embedding  # noqa: F401
    except ImportError as e:  # pragma: no cover - optional dependency
        raise ImportError("velocity plots need `pip install scp-plot[velocity]`") from e
    # Delegating rather than reimplementing: R's version is a transliteration of
    # this very function, with one deliberate divergence (no +/-1% grid padding).
    # stream mode reparameterises the mass floor: min_mass -> 10 ** (min_mass - 6),
    # so the default 1 means 1e-5.
    mass = 10 ** (min_mass - 6) if adjust_for_stream else min_mass
    return _cvog(X_emb=X_emb, V_emb=V_emb, density=density, smooth=smooth,
                 n_neighbors=n_neighbors, min_mass=mass, autoscale=not adjust_for_stream,
                 adjust_for_stream=adjust_for_stream, cutoff_perc=cutoff_perc)


def velocity_plot(
    adata,
    *,
    basis: str = "umap",
    vkey: str = "velocity",
    dims: tuple[int, int] = (1, 2),
    plot_type: Literal["raw", "grid", "stream"] = "raw",
    group_by: str | None = None,
    n_neighbors: int | None = None,
    density: float = 1.0,
    smooth: float = 0.5,
    scale: float = 1.0,
    min_mass: float = 1.0,
    cutoff_perc: float = 5.0,
    arrow_color: str = "black",
    streamline_palette: str = "RdYlBu",
    ax: Axes | None = None,
    **kwargs,
) -> Axes:
    """RNA velocity as per-cell arrows, a grid field, or streamlines. Milestone 12.

    Reads ``adata.obsm[f"{vkey}_{basis}"]`` — scvelo's own convention.  SCP
    names the Seurat reduction ``"<mode>_<REDUCTION>"`` instead; do not port
    that naming.

    ``raw`` -> ``ax.quiver`` with per-arrow head length proportional to
    magnitude; ``grid`` -> quiver on the smoothed grid; ``stream`` ->
    ``ax.streamplot`` with three stacked passes (white background stroke,
    speed-colored line, arrowheads only).
    """
    basis_key = basis if basis.startswith("X_") else f"X_{basis}"
    vkey_emb = f"{vkey}_{basis}"
    if basis_key not in adata.obsm:
        raise KeyError(f"{basis_key!r} not in adata.obsm")
    if vkey_emb not in adata.obsm:
        raise KeyError(
            f"{vkey_emb!r} not in adata.obsm; run scvelo.tl.velocity_embedding("
            f"basis={basis!r}) first"
        )
    X = np.asarray(adata.obsm[basis_key], dtype=float)
    V = np.asarray(adata.obsm[vkey_emb], dtype=float)
    xi, yi = dims[0] - 1, dims[1] - 1
    X, V = X[:, [xi, yi]], V[:, [xi, yi]]

    if ax is None:
        _fig, ax = plt.subplots(figsize=(4, 4))

    if group_by is not None:
        cat = as_ordered_categorical(adata.obs[group_by])
        levels = [str(x) for x in cat.cat.categories]
        colors = discrete_palette(levels, palette=kwargs.get("palette", "Paired"))
        cvec = [colors[str(v)] for v in np.asarray(cat).astype(str)]
    else:
        cvec = "#D9D9D9"
    ax.scatter(X[:, 0], X[:, 1], s=3, c=cvec, linewidths=0, zorder=1)

    if plot_type == "raw":
        ax.quiver(X[:, 0], X[:, 1], V[:, 0], V[:, 1], color=arrow_color,
                  angles="xy", scale_units="xy", scale=1.0 / max(scale, 1e-9),
                  width=0.0025, zorder=2)
    else:
        gx, gv = compute_velocity_on_grid(X, V, density=density, smooth=smooth,
                                          n_neighbors=n_neighbors, min_mass=min_mass,
                                          cutoff_perc=cutoff_perc,
                                          adjust_for_stream=plot_type == "stream")
        if plot_type == "grid":
            ax.quiver(gx[:, 0], gx[:, 1], gv[:, 0], gv[:, 1], color=arrow_color,
                      angles="xy", scale_units="xy", scale=1.0 / max(scale, 1e-9),
                      width=0.003, zorder=2)
        else:
            side = int(np.sqrt(len(gx)))
            ax.streamplot(gx[:, 0].reshape(side, side)[0, :],
                          gx[:, 1].reshape(side, side)[:, 0],
                          gv[:, 0].reshape(side, side), gv[:, 1].reshape(side, side),
                          color=arrow_color, linewidth=0.6, density=density,
                          arrowsize=0.8, zorder=2)

    ax.set_xlabel(kwargs.get("xlab", f"{basis}_{dims[0]}"))
    ax.set_ylabel(kwargs.get("ylab", f"{basis}_{dims[1]}"))
    if kwargs.get("title"):
        ax.set_title(kwargs["title"])
    return ax


def dynamic_plot(
    adata,
    lineages: Sequence[str],
    features: Sequence[str],
    *,
    group_by: str | None = None,
    layer: str | None = "counts",
    exp_method: str = "log1p",
    compare_lineages: bool = True,
    compare_features: bool = False,
    add_line: bool = True,
    add_interval: bool = True,
    add_point: bool = True,
    add_rug: bool = True,
    x_order: Literal["value", "rank"] = "value",
    **kwargs,
) -> Figure:
    """Fitted expression trends along pseudotime with confidence ribbons.

    R original: ``DynamicPlot``.  Milestone 9.

    .. warning::

       **Exact parity with R is not achievable here and is not attempted.**
       ``mgcv`` selects its smoothing parameter by GCV/REML over a penalised
       spline basis; ``pygam`` grid-searches a different basis and does not
       reproduce that criterion, so two fits of the same data differ by more
       than floating-point noise and the smooth is visibly different.  What is
       stable across both is the *shape*: the trend direction, the location of
       the peak, and the ordering of features by peak time.  Compare those, not
       the fitted values.

       Where ``scFates`` has already fitted and stored trends, reading them is
       strictly better than refitting and sidesteps the problem entirely.
    """
    feats = list(features)
    lins = list(lineages)
    exp = fetch_data(adata, feats, layer=layer)
    feats = [f for f in feats if f in exp.columns]
    if not feats:
        raise ValueError("none of `features` could be resolved")

    keys = [f"{lin}:{f}" for lin in lins for f in feats]
    th = theme_scp()
    pg = PanelGrid(len(keys), panel_size=kwargs.get("panel_size", (2.8, 2.0)),
                   nrow=kwargs.get("nrow"), ncol=kwargs.get("ncol"),
                   keys=keys, theme=th, legend=LegendColumn(position="right"))
    colors = discrete_palette([str(x) for x in lins], palette=kwargs.get("palette", "Dark2"))

    for lin in lins:
        t_raw = pd.to_numeric(adata.obs[lin], errors="coerce").to_numpy(dtype=float)
        for f in feats:
            ax = pg.axes[f"{lin}:{f}"]
            y = pd.to_numeric(exp[f], errors="coerce").to_numpy(dtype=float)
            m = np.isfinite(t_raw) & np.isfinite(y)
            t, yy = t_raw[m], y[m]
            if x_order == "rank":
                t = pd.Series(t).rank().to_numpy()
            order = np.argsort(t)
            t, yy = t[order], yy[order]
            if exp_method == "log1p":
                yy = np.log1p(yy)

            if add_point:
                ax.scatter(t, yy, s=2, color="#BFBFBF", linewidths=0, zorder=1)
            if add_rug:
                ax.plot(t, np.full_like(t, ax.get_ylim()[0]), "|", color="#808080",
                        markersize=2, zorder=1)

            fit, lo, hi = _fit_trend(t, yy)
            if add_interval and lo is not None:
                ax.fill_between(t, lo, hi, color=colors[str(lin)], alpha=0.2, zorder=2)
            if add_line:
                ax.plot(t, fit, color=colors[str(lin)], linewidth=1.5, zorder=3)

            ax.set_xlabel(lin)
            ax.set_ylabel(f"{exp_method}(expression)" if exp_method else "expression")
            ax.set_title(f, fontsize=th.size("plot.title"))

    return pg.finish()


def _fit_trend(t: np.ndarray, y: np.ndarray):
    """A smooth trend plus an interval, via pygam when available.

    See the warning on :func:`dynamic_plot`: this does not reproduce mgcv's
    smoothing-parameter selection and is not meant to.
    """
    try:
        from pygam import LinearGAM, s
    except ImportError:
        warnings.warn(
            "pygam is not installed, so dynamic trends fall back to a rolling "
            "mean with no confidence interval (`pip install scp-plot[gam]`).",
            stacklevel=3,
        )
        w = max(len(t) // 20, 3)
        fit = pd.Series(y).rolling(w, center=True, min_periods=1).mean().to_numpy()
        return fit, None, None
    gam = LinearGAM(s(0)).fit(t, y)
    fit = gam.predict(t)
    ci = gam.confidence_intervals(t, width=0.95)
    return fit, ci[:, 0], ci[:, 1]


def projection_plot(
    adata_query,
    adata_ref,
    *,
    query_group: str | None = None,
    ref_group: str | None = None,
    query_reduction: str = "X_umap",
    ref_reduction: str = "X_umap",
    pt_size: float = 0.8,
    stroke_highlight: float = 0.5,
    ax: Axes | None = None,
    **kwargs,
) -> Axes:
    """Query cells mapped onto a reference embedding. Milestone 13.

    Two scatters on one axes with shared limits; query points get
    ``edgecolors="black", linewidths=stroke_highlight`` so they read as
    outlined discs over the reference cloud.
    """
    if ax is None:
        _fig, ax = plt.subplots(figsize=(4.2, 4.2))
    Xr = np.asarray(adata_ref.obsm[ref_reduction], dtype=float)
    Xq = np.asarray(adata_query.obsm[query_reduction], dtype=float)

    # Shared limits over BOTH objects, or the projection reads as a shift that
    # is really just two different axis ranges.
    allx = np.concatenate([Xr[:, 0], Xq[:, 0]])
    ally = np.concatenate([Xr[:, 1], Xq[:, 1]])

    if ref_group is not None:
        cat = as_ordered_categorical(adata_ref.obs[ref_group])
        levels = [str(x) for x in cat.cat.categories]
        colors = discrete_palette(levels, palette=kwargs.get("palette", "Paired"))
        cref = [colors[str(v)] for v in np.asarray(cat).astype(str)]
    else:
        cref = "#D9D9D9"
    ax.scatter(Xr[:, 0], Xr[:, 1], s=pt_size * 6, c=cref, linewidths=0, zorder=1)

    if query_group is not None:
        qcat = as_ordered_categorical(adata_query.obs[query_group])
        qlev = [str(x) for x in qcat.cat.categories]
        qcolors = discrete_palette(qlev, palette=kwargs.get("query_palette", "Set1"))
        cq = [qcolors[str(v)] for v in np.asarray(qcat).astype(str)]
    else:
        cq = "black"
    # halo then fill, so query cells read on top of a dense reference
    ax.scatter(Xq[:, 0], Xq[:, 1], s=(pt_size + stroke_highlight) ** 2 * 10,
               c="black", linewidths=0, zorder=2)
    ax.scatter(Xq[:, 0], Xq[:, 1], s=pt_size ** 2 * 10, c=cq, linewidths=0, zorder=3)

    ax.set_xlim(float(allx.min()), float(allx.max()))
    ax.set_ylim(float(ally.min()), float(ally.max()))
    ax.set_xlabel(kwargs.get("xlab", "Dim 1"))
    ax.set_ylabel(kwargs.get("ylab", "Dim 2"))
    if kwargs.get("title"):
        ax.set_title(kwargs["title"])
    return ax
