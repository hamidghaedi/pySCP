"""Enrichment and GSEA plots.

These read **tables**, not AnnData internals, which is what makes them
backend-agnostic.  The contract is fixed in ``scp.io``; ``gseapy``,
``decoupler`` and ``GOATOOLS`` output all map onto it with renaming.

Spec: ``docs/porting_briefs/trajectory_enrichment.md`` §8-§9.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

__all__ = ["enrichment_plot", "gsea_plot", "gsea_scores", "adjust_layout"]

EnrichPlotType = Literal["bar", "dot", "lollipop", "network", "enrichmap", "wordcloud", "comparison"]
GSEAPlotType = Literal["line", "bar", "network", "enrichmap", "wordcloud", "comparison"]


def enrichment_plot(
    enrichment: pd.DataFrame,
    *,
    plot_type: EnrichPlotType = "bar",
    split_by: Sequence[str] = ("Database", "Groups"),
    color_by: str = "Database",
    group_use: Sequence[str] | None = None,
    id_use: Sequence[str] | dict[str, Sequence[str]] | None = None,
    pvalue_cutoff: float | None = None,
    padjust_cutoff: float | None = 0.05,
    top_term: int | None = None,
    character_width: int = 50,
    palette: str = "Spectral",
    palcolor: Sequence[str] | None = None,
    # network / enrichmap
    network_layout: str = "fr",
    network_blendmode: str = "blend",
    network_layoutadjust: bool = True,
    enrichmap_cluster: str = "fast_greedy",
    enrichmap_mark: Literal["ellipse", "hull"] = "ellipse",
    enrichmap_nlabel: int = 4,
    # wordcloud
    top_word: int = 100,
    word_type: Literal["term", "feature"] = "term",
    word_size: tuple[float, float] = (2.0, 8.0),
    words_excluded: Sequence[str] | None = None,
    panel_size: tuple[float, float] = (3.2, 2.6),
    nrow: int | None = None,
    ncol: int | None = None,
    seed: int = 11,
) -> Figure:
    """Over-representation results, seven ways. Milestone 14.

    Required columns (see ``scp.io.ENRICHMENT_COLUMNS``): ``ID``,
    ``Description``, ``GeneRatio``, ``BgRatio``, ``pvalue``, ``p.adjust``,
    ``geneID`` (``/``-joined), ``Count``, ``Groups``, ``Database``.

    Derived in-function: ``metric = -log10(p.adjust or pvalue)`` — note
    ``padjust_cutoff`` wins, and ``pvalue_cutoff`` only applies when
    ``padjust_cutoff is None``; and ``FoldEnrichment = GeneRatio / BgRatio``
    after parsing both ``"k/n"`` strings.

    ``top_term`` is applied per (Database x Group), not globally.  Descriptions
    are capitalised and wrapped at ``character_width``, and wrapped labels are
    italicised — a neat "this was wrapped" cue worth keeping.

    Start with ``bar``/``dot``/``lollipop``/``comparison``: they are pure
    tabular plots with no graph dependencies.
    """
    raise NotImplementedError("Milestone 14. Spec: docs/porting_briefs/trajectory_enrichment.md §8.")


def gsea_plot(
    results: dict[str, dict] | pd.DataFrame,
    *,
    plot_type: GSEAPlotType = "line",
    direction: Literal["pos", "neg", "both"] = "pos",
    top_term: int = 6,
    n_coregene: int = 10,
    sample_coregene: bool = False,
    features_label: Sequence[str] | None = None,
    line_width: float = 1.5,
    line_color: str | Sequence[str] = "#6BB82D",
    palette: str = "Spectral",
    padjust_cutoff: float | None = 0.05,
    panel_size: tuple[float, float] = (3.2, 3.0),
    seed: int = 11,
) -> Figure:
    """The classic three-panel GSEA figure, plus six alternatives. Milestone 15.

    ``plot_type="line"`` stacks three sub-panels at relative heights
    ``1.5 : 0.5 : 1``:

    1. the running enrichment score, over red/blue half-plane shading, with a
       peak marker (up-triangle when NES > 0, down when NES < 0) and optional
       core-gene labels;
    2. the hit ticks — one band per gene set, or, for a single set, a
       ranking-metric heat strip below the ticks;
    3. the ranked metric as stems to zero, with a dashed line at the
       zero-crossing rank.

    Input is ``{f"{group}-{db}": {"result": df, "geneList": Series,
    "geneSets": dict, "exponent": float}}``; ``gseapy.prerank`` output maps on
    with renaming (``Term`` -> ``ID``/``Description``, ``FDR q-val`` ->
    ``p.adjust``, ``Lead_genes`` -> ``core_enrichment``).
    """
    raise NotImplementedError("Milestone 15. Spec: docs/porting_briefs/trajectory_enrichment.md §9.")


def gsea_scores(gene_list: pd.Series, gene_set: Sequence[str], exponent: float = 1.0) -> pd.DataFrame:
    """The weighted KS running score (Subramanian et al.).

    ``gene_list`` must be sorted **descending** by the ranking metric::

        hits  = gene_list.index.isin(gene_set)
        Phit  = np.where(hits, np.abs(gene_list) ** exponent, 0)
        Phit  = np.cumsum(Phit / Phit.sum())
        Pmiss = np.cumsum(np.where(hits, 0, 1 / (N - Nh)))
        running = Phit - Pmiss

    If you are already using ``gseapy``, take its per-term ``RES`` array
    instead of recomputing — this exists so the plot can be driven by a
    ranking the user computed themselves.
    """
    gl = gene_list.astype(float)
    n = len(gl)
    hits = np.isin(gl.index.to_numpy(), np.asarray(list(gene_set)))
    nh = int(hits.sum())
    if nh == 0 or n == nh:
        raise ValueError("gene set does not overlap the ranked list (or covers all of it).")
    phit = np.where(hits, np.abs(gl.to_numpy()) ** exponent, 0.0)
    phit = np.cumsum(phit / phit.sum())
    pmiss = np.cumsum(np.where(hits, 0.0, 1.0 / (n - nh)))
    return pd.DataFrame(
        {
            "x": np.arange(1, n + 1),
            "running_score": phit - pmiss,
            "position": hits.astype(int),
            "gene": gl.index.to_numpy(),
            "metric": gl.to_numpy(),
        }
    )


def adjust_layout(
    layout: np.ndarray,
    widths: np.ndarray,
    edges: Sequence[tuple[int, int]],
    *,
    height: float = 2.0,
    scale: float = 60.0,
    iterations: int = 100,
    seed: int = 11,
) -> np.ndarray:
    """Separate overlapping *label boxes* after a force-directed layout. Milestone 14.

    A spring layout treats nodes as points, but gene nodes in the enrichment
    network are rendered as text whose width scales with the symbol length, so
    long labels collide badly.  ``adjustlayout`` fixes that in two passes:

    1. visit vertices in decreasing degree and push each not-yet-adjusted
       neighbour radially outward by the deficit ``(r + nr) - dist``; each
       vertex is pushed at most once, so hubs keep their positions;
    2. ``iterations`` rounds of nearest-neighbour box decollision — for each
       overlapping pair displace by the full overlap in x *or* y, chosen at
       random per collision.

    It is ~40 lines and deterministic apart from the coin flips; transcribe it
    and seed the RNG rather than substituting ``adjustText``, if visual parity
    with R matters.
    """
    raise NotImplementedError("Milestone 14. Spec: docs/porting_briefs/trajectory_enrichment.md §8.")
