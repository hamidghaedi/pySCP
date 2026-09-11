"""Trajectory, graph and velocity plots.

Most of this family is cheaper in Python than in R, because scanpy and scvelo
already persist what SCP had to reconstruct: ``adata.uns['paga']`` has the same
keys SCP reads, and ``adata.obsm['velocity_<basis>']`` is the velocity field.

Spec: ``docs/porting_briefs/trajectory_enrichment.md``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

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
    raise NotImplementedError("Milestone 11. Spec: docs/porting_briefs/trajectory_enrichment.md §3.")


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
    raise NotImplementedError("Milestone 11. Spec: docs/porting_briefs/trajectory_enrichment.md §2.")


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
    raise NotImplementedError("Milestone 12. Spec: docs/porting_briefs/trajectory_enrichment.md §1.")


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
    raise NotImplementedError("Milestone 12. Prefer scvelo's implementation directly.")


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
    raise NotImplementedError("Milestone 12. Spec: docs/porting_briefs/trajectory_enrichment.md §4.")


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
    """Fitted expression trends along pseudotime with confidence ribbons. Milestone 9."""
    raise NotImplementedError("Milestone 9. Spec: docs/porting_briefs/trajectory_enrichment.md §5.")


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
    raise NotImplementedError("Milestone 13. Spec: docs/porting_briefs/trajectory_enrichment.md §7.")
