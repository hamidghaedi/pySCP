"""scp — SCP's plotting grammar, for scanpy.

A port of the plotting layer of the R package
`SCP <https://github.com/zhanghao-njmu/SCP>`_ (Hao Zhang, GPL-3), reading
AnnData directly instead of Seurat.

::

    import scanpy as sc
    import scp

    adata = sc.datasets.pbmc3k_processed()
    scp.pl.cell_dim_plot(adata, "louvain", label=True, theme=scp.theme_blank())

The public surface is deliberately small:

``scp.pl``
    every plotting function
``scp.theme_scp`` / ``scp.theme_blank``
    the two themes, as objects you pass around
``scp.palettes``
    229 palettes extracted verbatim from the R package
``scp.fetch_data``
    the data-access contract every plot funnels through
``scp.io``
    adapters from scanpy / gseapy / decoupler results to our table contracts
"""

from __future__ import annotations

from . import colors, fetch, io, layout, palettes, pl, theme
from .colors import adjcolors, blendcolors
from .fetch import default_reduction, fetch_data, fetch_var
from .heatmap.matrix import matrix_process
from .layout import PanelGrid, combine
from .palettes import continuous_palette, discrete_palette, list_palettes, palette_scp
from .theme import Theme, apply_theme, theme_blank, theme_scp

__version__ = "0.0.1.dev0"

__all__ = [
    "__version__",
    # namespaces
    "pl",
    "io",
    "palettes",
    "colors",
    "fetch",
    "layout",
    "theme",
    # themes
    "Theme",
    "theme_scp",
    "theme_blank",
    "apply_theme",
    # palettes and color algebra
    "palette_scp",
    "discrete_palette",
    "continuous_palette",
    "list_palettes",
    "blendcolors",
    "adjcolors",
    # data access
    "fetch_data",
    "fetch_var",
    "default_reduction",
    "matrix_process",
    # composition
    "PanelGrid",
    "combine",
]
