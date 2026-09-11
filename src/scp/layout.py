"""Panel composition — the patchwork / ``panel_fix`` replacement.

SCP builds one ggplot per (split x group) or (split x feature) cell, then
combines them with ``patchwork::wrap_plots`` and, where exact physical sizing
matters, surgically rewrites the gtable with ``panel_fix``.

In matplotlib there is no such thing as "combine finished plots", so the
architecture has to be inverted: a plotting function receives an ``Axes`` and
draws into it, and a :class:`PanelGrid` allocates axes at exact inch sizes up
front.  That inversion is the single most important design decision in the
port — build it before any plot function.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Iterable, Literal, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

from .theme import Theme, apply_theme, theme_scp_rc

__all__ = ["PanelGrid", "grid_shape", "combine", "LegendColumn", "panel_size_for"]


def grid_shape(n: int, nrow: int | None = None, ncol: int | None = None, byrow: bool = True) -> tuple[int, int]:
    """patchwork's near-square default, with explicit overrides."""
    if nrow and ncol:
        return nrow, ncol
    if nrow:
        return nrow, math.ceil(n / nrow)
    if ncol:
        return math.ceil(n / ncol), ncol
    ncol = math.ceil(math.sqrt(n))
    return math.ceil(n / ncol), ncol


def panel_size_for(
    n: int,
    panel_w: float,
    panel_h: float,
    nrow: int,
    ncol: int,
    *,
    margin: float = 0.35,
    legend_w: float = 0.0,
) -> tuple[float, float]:
    """Figure size in inches for a grid of fixed-size panels.

    This is what ``panel_fix`` buys you in R: the *panel* is the thing with a
    known size, and the figure grows to accommodate decorations.  Saving at a
    fixed ``figsize`` and letting the panel float is the mistake that makes
    multi-panel figures inconsistent.
    """
    w = ncol * panel_w + (ncol + 1) * margin + legend_w
    h = nrow * panel_h + (nrow + 1) * margin
    return w, h


@dataclass
class LegendColumn:
    """A right-hand (or bottom) strip that owns every legend in the figure.

    SCP composes multiple independent legends (group colors + lineage colors +
    PAGA + velocity + stat pies) by rendering each from a throwaway plot,
    binding the gtables and gluing the result onto the panel.  Here, artists
    register handles and the column lays them out once.
    """

    position: Literal["right", "left", "top", "bottom", "none"] = "right"
    width_in: float = 1.4
    entries: list[tuple[str, list]] = field(default_factory=list)

    def add(self, title: str, handles: Sequence) -> None:
        self.entries.append((title, list(handles)))

    def draw(self, fig: Figure, theme: Theme | None = None) -> None:
        t = theme or Theme()
        if self.position == "none" or not self.entries:
            return
        y = 0.98
        for title, handles in self.entries:
            leg = fig.legend(
                handles=handles,
                title=f"{title}:" if title else None,
                loc="upper left",
                bbox_to_anchor=(1.0, y),
                frameon=False,
                fontsize=t.size("legend.text"),
                title_fontsize=t.size("legend.title"),
                alignment="left",
            )
            fig.add_artist(leg)
            fig.canvas.draw_idle()
            y -= 0.05 + 0.03 * len(handles)


class PanelGrid:
    """Allocate a grid of equally sized panels at a known physical size.

    Example
    -------
    >>> pg = PanelGrid(n=4, panel_size=(2.5, 2.5), theme=theme_scp(aspect_ratio=1))
    >>> for ax, key in pg:
    ...     ...  # draw
    >>> fig = pg.finish()
    """

    def __init__(
        self,
        n: int,
        *,
        panel_size: tuple[float, float] = (2.5, 2.5),
        nrow: int | None = None,
        ncol: int | None = None,
        byrow: bool = True,
        keys: Sequence[str] | None = None,
        theme: Theme | None = None,
        legend: LegendColumn | None = None,
        margin: float = 0.45,
        wspace: float = 0.25,
        hspace: float = 0.3,
    ) -> None:
        self.theme = theme or Theme()
        self.legend = legend if legend is not None else LegendColumn(position=self.theme.legend_position)
        self.keys = list(keys) if keys is not None else [str(i) for i in range(n)]
        self.nrow, self.ncol = grid_shape(n, nrow, ncol, byrow)
        self.byrow = byrow
        legend_w = self.legend.width_in if self.legend.position in ("left", "right") else 0.0
        figsize = panel_size_for(n, *panel_size, self.nrow, self.ncol, margin=margin, legend_w=legend_w)
        with mpl.rc_context(theme_scp_rc(self.theme)):
            self.fig: Figure = plt.figure(figsize=figsize)
        self.gs: GridSpec = self.fig.add_gridspec(
            self.nrow, self.ncol, wspace=wspace, hspace=hspace
        )
        self.axes: dict[str, Axes] = {}
        for i, key in enumerate(self.keys):
            r, c = (i // self.ncol, i % self.ncol) if byrow else (i % self.nrow, i // self.nrow)
            ax = self.fig.add_subplot(self.gs[r, c])
            apply_theme(ax, self.theme)
            self.axes[key] = ax

    def __iter__(self):
        return iter(self.axes.items())

    def finish(self, *, tight: bool = True) -> Figure:
        self.legend.draw(self.fig, self.theme)
        if tight:
            self.fig.set_layout_engine("constrained")
        return self.fig


def combine(
    draw: Callable[[Axes, str], None],
    keys: Sequence[str],
    *,
    panel_size: tuple[float, float] = (2.5, 2.5),
    nrow: int | None = None,
    ncol: int | None = None,
    byrow: bool = True,
    theme: Theme | None = None,
) -> Figure:
    """Functional shorthand: run ``draw(ax, key)`` over a grid and return the figure."""
    pg = PanelGrid(len(keys), panel_size=panel_size, nrow=nrow, ncol=ncol, byrow=byrow, keys=keys, theme=theme)
    for ax, key in ((a, k) for k, a in pg.axes.items()):
        draw(ax, key)
    return pg.finish()


def _iter_unused(x: Iterable) -> None:  # pragma: no cover
    list(x)
