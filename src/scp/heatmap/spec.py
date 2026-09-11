"""The declarative heatmap layout model — our replacement for ComplexHeatmap.

ComplexHeatmap gives R four things nothing in Python offers together:

1. physical-unit layout of a body plus arbitrarily many stacked annotation tracks,
2. split-aware coordinate systems shared across body and tracks,
3. horizontal/vertical concatenation of several heatmaps into one figure,
4. an automatic legend column.

``seaborn.clustermap`` is a dead end for this (one body, four fixed slots, no
splits).  The design here is a nested ``GridSpec`` whose ratios are inches:

    Figure (figsize computed by HeatmapSpec.figsize())
    +-- row 0: column titles
    +-- row 1: top annotation tracks           (one sub-row per Track)
    +-- row 2: column dendrogram
    +-- row 3: BODY ROW
    |     +-- col 0: row marks (anno_mark leader lines)
    |     +-- col 1: row dendrogram
    |     +-- col 2: left tracks (split blocks)
    |     +-- col 3..N: panel bodies (one per group_by / lineage)
    |     +-- col N+1: right tracks
    |     +-- col N+2: legend column
    +-- row 4: column names

Splits are a *nested* GridSpec inside the body, with ``height_ratios``
proportional to slice sizes.  Every track that runs along the split axis gets
the same subdivision, which is what keeps slices aligned.

Spec: ``docs/porting_briefs/heatmaps.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal, Sequence

import numpy as np
import pandas as pd

__all__ = ["Track", "Layer", "LegendSpec", "Dendrogram", "PanelSpec", "HeatmapSpec", "HeatmapResult"]

Side = Literal["top", "bottom", "left", "right"]


@dataclass
class LegendSpec:
    title: str
    kind: Literal["categorical", "continuous", "size"]
    #: level -> color, for categorical
    mapping: dict[str, str] | None = None
    #: (cmap, vmin, vmax) for continuous
    cmap: object | None = None
    vmin: float | None = None
    vmax: float | None = None
    #: for the dot-size legend: the five breaks SCP draws (20%..100%)
    breaks: Sequence[float] = (0.2, 0.4, 0.6, 0.8, 1.0)


@dataclass
class Track:
    """One annotation lane, occupying a fixed thickness on one side."""

    name: str
    side: Side
    kind: Literal["simple", "block", "mark", "textbox", "custom"]
    size_in: float = 5 / 25.4  # ComplexHeatmap's default 5mm
    values: np.ndarray | pd.Series | None = None
    #: categorical: level -> color
    colors: dict[str, str] | None = None
    #: continuous: a matplotlib Colormap plus its Normalize
    cmap: object | None = None
    norm: object | None = None
    na_color: str = "none"
    border: bool = True
    show_name: bool = True
    name_side: Literal["left", "top"] = "left"
    #: for kind="mark": positions and labels of the features to call out
    at: Sequence[int] | None = None
    labels: Sequence[str] | None = None
    #: for kind="custom" / "block": draw into the supplied sub-axes yourself.
    #: This replaces SCP's anno_customize, which captures rendered ggplot grobs.
    draw: Callable[..., None] | None = None
    legend: LegendSpec | None = None


@dataclass
class Layer:
    """An overlay drawn on top of the heatmap body — SCP's ``layer_fun``.

    Draw order is fixed and must be preserved: ``whiteout`` (when dots or
    violins are present), ``bg``, ``reticle``, ``dot``, ``violin``.
    """

    kind: Literal["whiteout", "bg", "reticle", "dot", "violin", "custom"]
    #: for kind="dot": fraction of cells above exp_cutoff, same shape as the body
    size_matrix: np.ndarray | None = None
    #: diameter at 100%, in inches (SCP default 8mm)
    base_size_in: float = 8 / 25.4
    alpha: float = 0.5
    color: str = "#7F7F7F"
    draw: Callable[..., None] | None = None
    legend: LegendSpec | None = None


@dataclass
class Dendrogram:
    linkage: np.ndarray
    order: np.ndarray
    #: True when built by cluster_within_group2 (per-slice trees grafted onto a
    #: tree of the slice centroids)
    within_group: bool = False
    size_in: float = 0.4


@dataclass
class PanelSpec:
    """One heatmap body — the equivalent of a single ``ComplexHeatmap::Heatmap``."""

    name: str
    #: features x observations, already through matrix_process
    matrix: np.ndarray
    row_labels: pd.Index
    col_labels: pd.Index
    cmap: object
    norm: object
    col_split: pd.Categorical | None = None
    #: body width in inches (height when flipped); None -> proportional to ncol
    body_size_in: float | None = None
    layers: list[Layer] = field(default_factory=list)
    top_tracks: list[Track] = field(default_factory=list)
    title: str = ""


@dataclass
class HeatmapSpec:
    """The whole figure, declared before anything is drawn."""

    panels: list[PanelSpec]
    row_split: pd.Categorical | None = None
    row_dend: Dendrogram | None = None
    col_dends: dict[str, Dendrogram] = field(default_factory=dict)
    left_tracks: list[Track] = field(default_factory=list)
    right_tracks: list[Track] = field(default_factory=list)
    legends: list[LegendSpec] = field(default_factory=list)
    body_height_in: float | None = None
    row_gap_in: float = 0.04
    col_gap_in: float = 0.04
    panel_gap_in: float = 0.08
    flip: bool = False
    show_row_names: bool = False
    show_column_names: bool = True
    row_names_side: Side | None = None
    column_names_side: Side | None = None
    #: measured lazily by render(); populated here so figsize() is testable
    measured: dict[str, float] = field(default_factory=dict)

    def figsize(self) -> tuple[float, float]:
        """Total figure size in inches — the ``heatmap_rendersize`` equivalent.

        ``width`` is a per-panel quantity that is **summed** along the
        concatenation axis; ``height`` is scalar and shared.  ``flip`` swaps
        the two roles.  Defaults to a 1-inch body, as R does.
        """
        m = self.measured
        bodies = [p.body_size_in if p.body_size_in is not None else 1.0 for p in self.panels]
        w = sum(bodies) + max(len(bodies) - 1, 0) * self.panel_gap_in
        h = self.body_height_in if self.body_height_in is not None else 1.0

        left = sum(t.size_in for t in self.left_tracks)
        right = sum(t.size_in for t in self.right_tracks)
        top = max((sum(t.size_in for t in p.top_tracks) for p in self.panels), default=0.0)

        w += left + right
        w += self.row_dend.size_in if self.row_dend else 0.0
        w += m.get("row_names_w", 0.0)
        w += m.get("legend_w", 1.4) if self.legends else 0.0
        h += top
        h += max((d.size_in for d in self.col_dends.values()), default=0.0)
        h += m.get("column_names_h", 0.0)
        h += m.get("titles_h", 0.25)

        return (h, w) if self.flip else (w, h)


@dataclass
class HeatmapResult:
    """What every ``*_heatmap`` function returns.

    Mirrors the named list the R functions return, plus a ``spec`` escape hatch
    so callers can retune the layout and re-render without recomputing.
    """

    fig: object
    matrices: dict[str, pd.DataFrame]
    feature_split: pd.Series | None = None
    obs_meta: pd.DataFrame | None = None
    var_meta: pd.DataFrame | None = None
    enrichment: pd.DataFrame | None = None
    spec: HeatmapSpec | None = None
