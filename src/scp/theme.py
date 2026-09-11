"""The SCP look, expressed as matplotlib rcParams and axes post-processors.

``theme_scp`` and ``theme_blank`` in R are ggplot2 ``theme()`` objects, which
are declarative.  matplotlib has two places the same information can live:
``rcParams`` (applies at artist creation) and per-axes mutation (applies after).
We use both, and the split is deliberate:

* ``theme_scp_rc()`` returns an rcParams dict — use it with
  ``matplotlib.rc_context`` around figure construction so fonts and sizes are
  right from the start.
* ``apply_theme(ax, ...)`` fixes up what rcParams cannot express (the aspect
  ratio, the four-sided black panel border with no axis line, legend placement).

Reference: ``SCP/R/SCP-plot.R`` lines 16--167.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import matplotlib as mpl
from matplotlib.axes import Axes

LegendPosition = Literal["none", "left", "right", "bottom", "top"]

__all__ = ["Theme", "theme_scp", "theme_blank", "theme_scp_rc", "apply_theme", "add_coord_arrows"]

# Base font sizes at base_size = 12, taken verbatim from theme_scp().
_BASE = {
    "text": 12.0,
    "plot.title": 14.0,
    "plot.subtitle": 13.0,
    "axis.title": 13.0,
    "axis.text": 12.0,
    "strip.text": 12.5,
    "legend.title": 12.0,
    "legend.text": 11.0,
}


@dataclass
class Theme:
    """Resolved theme settings for one panel."""

    base_size: float = 12.0
    aspect_ratio: float | None = None
    blank: bool = False
    panel_border: bool = True
    border_color: str = "black"
    border_width: float = 1.0
    background: str = "white"
    legend_position: LegendPosition = "right"
    legend_direction: Literal["vertical", "horizontal"] = "vertical"
    #: only used when blank=True
    add_coord: bool = True
    xlen_npc: float = 0.15
    ylen_npc: float = 0.15
    xlab: str = ""
    ylab: str = ""
    lab_size: float = 12.0
    overrides: dict = field(default_factory=dict)

    @property
    def scale(self) -> float:
        return self.base_size / 12.0

    def size(self, element: str) -> float:
        return _BASE[element] * self.scale


def theme_scp(
    aspect_ratio: float | None = None,
    base_size: float = 12.0,
    *,
    legend_position: LegendPosition = "right",
    legend_direction: Literal["vertical", "horizontal"] = "vertical",
    **overrides,
) -> Theme:
    """The default SCP panel: white background, black 1pt border, no axis line."""
    return Theme(
        base_size=base_size,
        aspect_ratio=aspect_ratio,
        legend_position=legend_position,
        legend_direction=legend_direction,
        overrides=overrides,
    )


def theme_blank(
    *,
    add_coord: bool = True,
    xlen_npc: float = 0.15,
    ylen_npc: float = 0.15,
    xlab: str = "",
    ylab: str = "",
    lab_size: float = 12.0,
    aspect_ratio: float | None = None,
    base_size: float = 12.0,
    **overrides,
) -> Theme:
    """No frame, no ticks — optionally the little corner axis arrows.

    This is the idiom every published UMAP uses.  ``add_coord=True`` draws an
    x-arrow spanning ``xlen_npc`` of the panel width along the bottom edge and
    a y-arrow spanning ``ylen_npc`` of the height up the left edge, each
    labelled outside the panel.
    """
    return Theme(
        base_size=base_size,
        aspect_ratio=aspect_ratio,
        blank=True,
        panel_border=False,
        add_coord=add_coord,
        xlen_npc=xlen_npc,
        ylen_npc=ylen_npc,
        xlab=xlab,
        ylab=ylab,
        lab_size=lab_size,
        overrides=overrides,
    )


def theme_scp_rc(theme: Theme | None = None) -> dict:
    """rcParams equivalent of ``theme_scp``, for use with ``mpl.rc_context``."""
    t = theme or Theme()
    return {
        "figure.facecolor": t.background,
        "savefig.facecolor": t.background,
        "axes.facecolor": t.background,
        "axes.edgecolor": t.border_color,
        "axes.linewidth": t.border_width,
        "axes.grid": False,
        "axes.titlesize": t.size("plot.title"),
        "axes.labelsize": t.size("axis.title"),
        "axes.labelcolor": "black",
        "xtick.labelsize": t.size("axis.text"),
        "ytick.labelsize": t.size("axis.text"),
        "xtick.color": "black",
        "ytick.color": "black",
        "font.size": t.size("text"),
        "legend.fontsize": t.size("legend.text"),
        "legend.title_fontsize": t.size("legend.title"),
        "legend.frameon": False,
        "legend.handletextpad": 0.4,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }


def apply_theme(ax: Axes, theme: Theme | None = None) -> Axes:
    """Post-process one axes into the SCP look."""
    t = theme or Theme()

    if t.blank:
        for side in ("top", "right", "bottom", "left"):
            ax.spines[side].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        if t.add_coord:
            add_coord_arrows(ax, t)
    else:
        # theme_scp: a closed rectangle, no protruding axis line.
        for side in ("top", "right", "bottom", "left"):
            sp = ax.spines[side]
            sp.set_visible(t.panel_border)
            sp.set_color(t.border_color)
            sp.set_linewidth(t.border_width)
        ax.tick_params(direction="out", length=3, width=1, colors="black")

    ax.set_facecolor(t.background)
    if t.aspect_ratio is not None:
        # aspect_ratio in ggplot is panel height / panel width, in *panel*
        # units, which is exactly set_box_aspect.
        ax.set_box_aspect(t.aspect_ratio)
    return ax


def add_coord_arrows(ax: Axes, theme: Theme | None = None) -> None:
    """The ``theme_blank(add_coord=TRUE)`` corner arrows, in axes coordinates."""
    t = theme or Theme(blank=True)
    kw = dict(
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="-|>", color="black", linewidth=2, mutation_scale=12),
        annotation_clip=False,
    )
    ax.annotate("", xy=(t.xlen_npc, 0.0), xytext=(0.0, 0.0), **kw)
    ax.annotate("", xy=(0.0, t.ylen_npc), xytext=(0.0, 0.0), **kw)
    ax.annotate(
        t.xlab,
        xy=(0.0, 0.0),
        xycoords="axes fraction",
        xytext=(0, -t.lab_size * 1.15),
        textcoords="offset points",
        ha="left",
        va="top",
        fontsize=t.lab_size,
        annotation_clip=False,
    )
    ax.annotate(
        t.ylab,
        xy=(0.0, 0.0),
        xycoords="axes fraction",
        xytext=(-t.lab_size * 1.15, 0),
        textcoords="offset points",
        ha="left",
        va="bottom",
        rotation=90,
        fontsize=t.lab_size,
        annotation_clip=False,
    )


def halo(text_artist, foreground: str = "black", radius: float = 0.1) -> None:
    """ggrepel's ``bg.color``/``bg.r`` shadow text, as a path effect.

    ggrepel redraws the string eight times offset by ``bg.r * fontsize`` before
    drawing the real glyphs.  A stroke of ``2 * bg.r * fontsize`` is visually
    equivalent and far cheaper.
    """
    from matplotlib import patheffects

    size = text_artist.get_fontsize()
    text_artist.set_path_effects(
        [patheffects.withStroke(linewidth=2 * radius * size, foreground=foreground)]
    )


def _unused() -> None:  # pragma: no cover
    _ = mpl
