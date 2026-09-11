"""``scp.pl`` — the plotting namespace.

Names are snake_case ports of the R originals and coexist with ``sc.pl.*``
without shadowing anything::

    import scanpy as sc
    import scp

    scp.pl.cell_dim_plot(adata, "leiden", label=True)
    scp.pl.group_heatmap(adata, features=genes, group_by="leiden", add_dot=True)
"""

from ..palettes import list_palettes, palette_scp  # re-exported: users reach for these here
from ._dim import cell_dim_plot, cell_dim_plot_3d, feature_dim_plot, feature_dim_plot_3d
from ._enrich import adjust_layout, enrichment_plot, gsea_plot, gsea_scores
from ._heatmap import (
    cell_cor_heatmap,
    dynamic_heatmap,
    feature_cor_heatmap,
    feature_heatmap,
    group_heatmap,
)
from ._stat import (
    cell_density_plot,
    cell_stat_plot,
    expression_stat_plot,
    feature_cor_plot,
    feature_stat_plot,
    stat_plot,
    volcano_plot,
)
from ._traj import (
    compute_velocity_on_grid,
    dynamic_plot,
    graph_plot,
    lineage_plot,
    paga_plot,
    projection_plot,
    shorten_segments,
    velocity_plot,
)

__all__ = [
    # palettes
    "list_palettes",
    "palette_scp",
    # dimensional reduction
    "cell_dim_plot",
    "feature_dim_plot",
    "cell_dim_plot_3d",
    "feature_dim_plot_3d",
    # statistics
    "feature_stat_plot",
    "expression_stat_plot",
    "cell_stat_plot",
    "stat_plot",
    "feature_cor_plot",
    "cell_density_plot",
    "volcano_plot",
    # heatmaps
    "group_heatmap",
    "feature_heatmap",
    "dynamic_heatmap",
    "cell_cor_heatmap",
    "feature_cor_heatmap",
    # trajectory / graph
    "graph_plot",
    "paga_plot",
    "lineage_plot",
    "velocity_plot",
    "dynamic_plot",
    "projection_plot",
    "shorten_segments",
    "compute_velocity_on_grid",
    # enrichment
    "enrichment_plot",
    "gsea_plot",
    "gsea_scores",
    "adjust_layout",
]
