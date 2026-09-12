"""Enrichment and GSEA plots.

These read **tables**, not AnnData internals, which is what makes them
backend-agnostic.  The contract is fixed in ``scp.io``; ``gseapy``,
``decoupler`` and ``GOATOOLS`` output all map onto it with renaming.

Spec: ``docs/porting_briefs/trajectory_enrichment.md`` §8-§9.
"""

from __future__ import annotations

import textwrap
from collections.abc import Sequence
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.figure import Figure

from ..layout import LegendColumn, PanelGrid
from ..palettes import continuous_palette
from ..theme import theme_scp

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
    if plot_type in ("network", "enrichmap", "wordcloud"):
        raise NotImplementedError(
            f"plot_type={plot_type!r} is specified in "
            "docs/porting_briefs/trajectory_enrichment.md §8 but not yet implemented; "
            "it needs python-igraph layouts (network/enrichmap) or a keyword-"
            "enrichment backend (wordcloud). See docs/07_milestones.md."
        )
    df = enrichment.copy()
    missing = [c for c in ("Description", "p.adjust") if c not in df.columns]
    if missing:
        raise ValueError(
            f"enrichment table is missing {missing}; expected the columns in "
            "scp.io.ENRICHMENT_COLUMNS (scp.io.enrichment_from_gseapy builds them)"
        )
    if pvalue_cutoff is not None and "pvalue" in df:
        df = df[df["pvalue"] <= pvalue_cutoff]
    if padjust_cutoff is not None:
        df = df[df["p.adjust"] <= padjust_cutoff]
    if group_use is not None and "Groups" in df:
        df = df[df["Groups"].astype(str).isin([str(g) for g in group_use])]
    if id_use is not None and "ID" in df:
        wanted = (list(id_use) if not isinstance(id_use, dict)
                  else [i for v in id_use.values() for i in v])
        df = df[df["ID"].astype(str).isin([str(i) for i in wanted])]
    if df.empty:
        raise ValueError("no terms survive the cutoffs")

    df["_score"] = -np.log10(df["p.adjust"].astype(float).clip(lower=1e-300))
    panels = [c for c in split_by if c in df.columns]
    keys = ([" | ".join(str(k) for k in (kk if isinstance(kk, tuple) else (kk,)))
             for kk, _ in df.groupby(panels, observed=True)] if panels else ["all"])
    groups = (list(df.groupby(panels, observed=True)) if panels
              else [("all", df)])

    if top_term:
        groups = [(k, g.nlargest(top_term, "_score")) for k, g in groups]

    th = theme_scp()
    pg = PanelGrid(len(groups), panel_size=panel_size, nrow=nrow, ncol=ncol,
                   keys=keys, theme=th, legend=LegendColumn(position="right"))

    ramp = continuous_palette(palette=palette, palcolor=palcolor, n=100)
    cmap = LinearSegmentedColormap.from_list("enr", ramp, N=256)

    for key, (_k, g) in zip(keys, groups, strict=True):
        ax = pg.axes[key]
        g = g.sort_values("_score")
        # terms are long; wrap them or the axis eats the panel
        labels = [textwrap.fill(str(d), character_width) for d in g["Description"]]
        y = np.arange(len(g))
        counts = (g["Count"].to_numpy(dtype=float) if "Count" in g
                  else np.full(len(g), 1.0))
        norm = Normalize(float(g["_score"].min()), float(g["_score"].max()))
        cvec = cmap(norm(g["_score"].to_numpy()))

        if plot_type == "bar":
            ax.barh(y, g["_score"], color=cvec, edgecolor="black", linewidth=0.4)
        elif plot_type == "dot":
            ax.scatter(g["_score"], y, s=20 + 80 * counts / max(counts.max(), 1),
                       color=cvec, edgecolor="black", linewidth=0.4, zorder=3)
        elif plot_type == "lollipop":
            ax.hlines(y, 0, g["_score"], color="#B0B0B0", linewidth=1, zorder=1)
            ax.scatter(g["_score"], y, s=20 + 80 * counts / max(counts.max(), 1),
                       color=cvec, edgecolor="black", linewidth=0.4, zorder=3)
        elif plot_type == "comparison":
            # one column per group, colour = significance, size = gene count
            gcol = "Groups" if "Groups" in g else None
            if gcol is None:
                raise ValueError("plot_type='comparison' needs a 'Groups' column")
            glev = [str(x) for x in pd.unique(g[gcol])]
            terms = list(dict.fromkeys(str(d) for d in g["Description"]))
            for _, row in g.iterrows():
                ax.scatter(glev.index(str(row[gcol])), terms.index(str(row["Description"])),
                           s=20 + 80 * (float(row.get("Count", 1)) / max(counts.max(), 1)),
                           color=cmap(norm(row["_score"])), edgecolor="black",
                           linewidth=0.4)
            ax.set_xticks(range(len(glev)))
            ax.set_xticklabels(glev, rotation=45, ha="right")
            ax.set_yticks(range(len(terms)))
            ax.set_yticklabels([textwrap.fill(t, character_width) for t in terms],
                               fontsize=6.5)
            ax.set_title(str(key), fontsize=th.size("plot.title"))
            continue

        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=6.5)
        ax.set_xlabel("-log10(p.adjust)")
        ax.set_title(str(key), fontsize=th.size("plot.title"))
        ax.set_ylim(-0.7, len(g) - 0.3)

    sm = ScalarMappable(cmap=cmap, norm=Normalize(0, 1))
    cb = pg.fig.colorbar(sm, ax=list(pg.axes.values()), fraction=0.02, pad=0.02)
    cb.set_label("-log10(p.adjust), scaled per panel", fontsize=th.size("legend.title"))
    return pg.finish()


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
    if isinstance(results, pd.DataFrame):
        table = results.copy()
        curves: dict[str, pd.DataFrame] = {}
    else:
        table = pd.DataFrame(results.get("table", []))
        curves = results.get("curves", {})
    if table.empty:
        raise ValueError("no GSEA results to plot")
    if padjust_cutoff is not None and "p.adjust" in table:
        table = table[table["p.adjust"] <= padjust_cutoff]
    if direction == "pos":
        table = table[table["NES"] > 0]
    elif direction == "neg":
        table = table[table["NES"] < 0]
    if table.empty:
        raise ValueError("no gene sets survive the cutoffs")
    table = table.reindex(table["NES"].abs().sort_values(ascending=False).index).head(top_term)

    if plot_type in ("comparison", "network", "enrichmap", "wordcloud"):
        raise NotImplementedError(
            f"plot_type={plot_type!r} reuses the enrichment_plot machinery of "
            "docs/porting_briefs/trajectory_enrichment.md §8-§9 and is not yet implemented."
        )
    if plot_type == "bar":
        ax_fig = plt.figure(figsize=(panel_size[0] * 1.4, panel_size[1]))
        ax = ax_fig.add_subplot()
        y = np.arange(len(table))
        ax.barh(y, table["NES"], color=np.where(table["NES"] > 0, "#B2182B", "#2166AC"),
                edgecolor="black", linewidth=0.4)
        ax.set_yticks(y)
        ax.set_yticklabels([textwrap.fill(str(d), 50) for d in table["Description"]],
                           fontsize=6.5)
        ax.set_xlabel("NES")
        ax_fig.set_layout_engine("constrained")
        return ax_fig

    # plot_type == "line": the canonical three-panel GSEA figure
    n = len(table)
    fig = plt.figure(figsize=(panel_size[0], panel_size[1] * n))
    gs = fig.add_gridspec(n * 3, 1,
                          height_ratios=[6, 1.2, 2.5] * n, hspace=0.08)
    colors = ([line_color] * n if isinstance(line_color, str) else list(line_color))
    for i, (_, row) in enumerate(table.iterrows()):
        cur = curves.get(str(row["ID"]))
        if cur is None:
            raise ValueError(
                f"no running-score curve for {row['ID']!r}; pass "
                "{'table': df, 'curves': {id: scp.pl.gsea_scores(...)}}"
            )
        x = cur["x"].to_numpy() if "x" in cur else np.arange(len(cur))
        # gsea_scores emits the port's own column names; accept R's too so a
        # frame lifted straight out of clusterProfiler still plots.
        rs_col = "running_score" if "running_score" in cur else "runningScore"
        rs = cur[rs_col].to_numpy()
        hits = np.flatnonzero(cur["position"].to_numpy() == 1) if "position" in cur else []

        top = fig.add_subplot(gs[i * 3])
        top.plot(x, rs, color=colors[i % len(colors)], linewidth=line_width)
        top.axhline(0, color="grey", linewidth=0.6, linestyle="--")
        top.set_xticks([])
        top.set_ylabel("Running\nenrichment", fontsize=7)
        top.set_title(textwrap.fill(str(row["Description"]), 46), fontsize=8)
        top.text(0.98, 0.95, f"NES = {row['NES']:.2f}\np.adj = {row['p.adjust']:.2g}",
                 transform=top.transAxes, ha="right", va="top", fontsize=6.5)

        mid = fig.add_subplot(gs[i * 3 + 1])
        mid.vlines(x[hits], 0, 1, color="black", linewidth=0.4)
        mid.set_xticks([])
        mid.set_yticks([])
        mid.set_xlim(x.min(), x.max())

        bot = fig.add_subplot(gs[i * 3 + 2])
        metric_col = "metric" if "metric" in cur else "geneList"
        if metric_col in cur:
            gl = cur[metric_col].to_numpy()
            bot.fill_between(x, 0, gl, color="#B0B0B0")
            bot.axhline(0, color="grey", linewidth=0.6)
        bot.set_xlabel("Rank in ordered gene list", fontsize=7)
        bot.set_ylabel("Ranked\nmetric", fontsize=7)
        bot.set_xlim(x.min(), x.max())
    fig.set_layout_engine("constrained")
    return fig


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
    pos = np.array(layout, dtype=float).copy()
    # `widths` are label widths in the same units as `layout`; `scale` converts
    # them, so it divides the widths rather than multiplying the coordinates --
    # scaling only one of the two makes the overlap test meaningless.
    w = np.asarray(widths, dtype=float) / max(scale, 1e-9)
    height = height / max(scale, 1e-9)
    rng = np.random.default_rng(seed)
    for _ in range(iterations):
        moved = False
        for i in range(len(pos)):
            for j in range(i + 1, len(pos)):
                dx = pos[j, 0] - pos[i, 0]
                dy = pos[j, 1] - pos[i, 1]
                # label boxes, not points: overlap is per-axis, so the test is a
                # rectangle intersection rather than a distance
                min_dx = (w[i] + w[j]) / 2
                if abs(dx) < min_dx and abs(dy) < height:
                    push = (min_dx - abs(dx)) / 2 + 1e-6
                    sign = 1.0 if dx >= 0 else -1.0
                    if dx == 0:
                        sign = 1.0 if rng.random() > 0.5 else -1.0
                    pos[i, 0] -= sign * push
                    pos[j, 0] += sign * push
                    moved = True
        if not moved:
            break
    _ = edges  # edges are accepted for signature parity; repulsion is node-only
    return pos
