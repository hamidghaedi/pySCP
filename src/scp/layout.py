"""Panel composition — the patchwork / ``panel_fix`` replacement.

SCP builds one ggplot per (split x group) or (split x feature) cell, then
combines them with ``patchwork::wrap_plots`` and, where exact physical sizing
matters, surgically rewrites the gtable with ``panel_fix``.

In matplotlib there is no such thing as "combine finished plots", so the
architecture has to be inverted: a plotting function receives an ``Axes`` and
draws into it, and a :class:`PanelGrid` allocates axes at exact inch sizes up
front.  That inversion is the single most important design decision in the
port — build it before any plot function.

The layout is ``panel_fix``'s, not matplotlib's: the *panel* has the size the
caller asked for, and the figure is whatever is left over once every
decoration has been measured.  :meth:`PanelGrid.finish` does that in two
deterministic passes -- draw, measure each panel's tick labels, axis titles,
strips and attached colourbars, then place every axes at an exact inch
position and size the figure to fit.  The constrained layout engine is not
used: it keeps the figure size fixed and shrinks or floats the panels, which
is the behaviour ``panel_fix`` exists to prevent.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colorbar import Colorbar
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.layout_engine import LayoutEngine
from matplotlib.legend import Legend
from matplotlib.text import Text
from matplotlib.transforms import Bbox

from .theme import Theme, apply_theme, theme_scp_rc

__all__ = ["PanelGrid", "grid_shape", "combine", "LegendColumn", "panel_size_for"]

#: ggplot2's ``legend.box.spacing`` and patchwork's plot spacing: 2 x half_line,
#: 11 pt.  Used between a panel block and its legends and between legends.
SPACING = 11 / 72
#: ``theme_scp``'s ``legend.key.size`` (10 pt).  A colourbar is one key thick
#: and five keys long, ggplot2's ``guide_colourbar`` default.
KEY = 10 / 72
#: Outer figure margin, inches (ggplot's ``plot.margin`` is 5.5 pt).
MARGIN = 0.1

Side = Literal["right", "left", "top", "bottom"]


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
    """A first estimate of the figure size, in inches, for a grid of panels.

    Only an estimate: :class:`PanelGrid` measures the decorations once they
    are drawn and sizes the figure from those, so the panels -- not the
    figure -- end up with the requested size.  Kept for callers that need a
    rough canvas before anything is drawn.
    """
    w = ncol * panel_w + (ncol + 1) * margin + legend_w
    h = nrow * panel_h + (nrow + 1) * margin
    return w, h


# --------------------------------------------------------------------------- #
# measurement helpers
# --------------------------------------------------------------------------- #
def _renderer(fig: Figure):
    get = getattr(fig.canvas, "get_renderer", None)
    return get() if get is not None else fig._get_renderer()  # type: ignore[attr-defined]


def _inches(bb: Bbox, fig: Figure) -> Bbox:
    return bb.transformed(fig.dpi_scale_trans.inverted())


def _union(boxes: Iterable[Bbox]) -> Bbox | None:
    boxes = [b for b in boxes if b is not None and math.isfinite(b.width) and math.isfinite(b.height)]
    return Bbox.union(boxes) if boxes else None


class _LegendBox:
    """A ``fig.legend`` placed by its upper-left corner, in inches."""

    def __init__(self, legend: Legend) -> None:
        self.legend = legend
        self.w = self.h = 0.0

    def measure(self, fig: Figure, renderer) -> tuple[float, float]:
        bb = _inches(self.legend.get_window_extent(renderer), fig)
        self.w, self.h = bb.width, bb.height
        return self.w, self.h

    def place(self, fig: Figure, x: float, top: float) -> None:
        W, H = fig.get_size_inches()
        # loc="upper left" + borderaxespad=0: the anchor IS the corner
        self.legend.set_bbox_to_anchor((x / W, top / H), transform=fig.transFigure)


class _ColorbarBox:
    """A colourbar of fixed inch thickness and length, plus its measured labels."""

    def __init__(self, cb: Colorbar, orientation: str, thickness: float, length: float) -> None:
        self.cb = cb
        self.cax = cb.ax
        self.vertical = orientation == "vertical"
        self.bar_w, self.bar_h = (thickness, length) if self.vertical else (length, thickness)
        self.pad = (0.0, 0.0, 0.0, 0.0)  # left, right, bottom, top
        self.w = self.h = 0.0

    def measure(self, fig: Figure, renderer) -> tuple[float, float]:
        box = _inches(self.cax.get_window_extent(renderer), fig)
        tb = self.cax.get_tightbbox(renderer)
        tight = box if tb is None else _inches(tb, fig)
        self.pad = (max(box.x0 - tight.x0, 0.0), max(tight.x1 - box.x1, 0.0),
                    max(box.y0 - tight.y0, 0.0), max(tight.y1 - box.y1, 0.0))
        self.w = self.pad[0] + self.bar_w + self.pad[1]
        self.h = self.pad[3] + self.bar_h + self.pad[2]
        return self.w, self.h

    def place(self, fig: Figure, x: float, top: float) -> None:
        W, H = fig.get_size_inches()
        x0 = x + self.pad[0]
        y0 = top - self.pad[3] - self.bar_h
        self.cax.set_position((x0 / W, y0 / H, self.bar_w / W, self.bar_h / H))


_Box = _LegendBox | _ColorbarBox


def _pack(sizes: Sequence[tuple[float, float]], avail: float, gap: float,
          vertical: bool) -> tuple[list[list[int]], list[float], list[float]]:
    """Greedy wrap: stack along the main axis until ``avail`` is used, then
    start another column (``vertical``) or row.  Returns the groups and, per
    group, its extent along the main axis and across it."""
    main = 1 if vertical else 0
    groups: list[list[int]] = [[]]
    along: list[float] = [0.0]
    for i, s in enumerate(sizes):
        need = s[main] if not groups[-1] else along[-1] + gap + s[main]
        if groups[-1] and need > avail + 1e-9:
            groups.append([])
            along.append(0.0)
            need = s[main]
        groups[-1].append(i)
        along[-1] = need
    across = [max(sizes[i][1 - main] for i in g) for g in groups]
    return groups, along, across


@dataclass
class LegendColumn:
    """The strip that owns every legend in the figure.

    SCP composes multiple independent legends (group colors + lineage colors +
    PAGA + velocity + stat pies) by rendering each from a throwaway plot,
    binding the gtables and gluing the result onto the panel.  Here, artists
    register handles and the column lays them out once.

    Legends stack in registration order along the side named by
    ``position`` -- top to bottom for ``right``/``left``, left to right for
    ``top``/``bottom`` -- with :data:`SPACING` between them, and wrap into
    another column (or row) instead of running past the panel block.  Under
    :class:`PanelGrid` the column is measured and placed *inside* the figure.
    ``width_in`` is ignored and kept only so old calls still construct.
    """

    position: Literal["right", "left", "top", "bottom", "none"] = "right"
    width_in: float = 1.4
    entries: list[tuple[str, list]] = field(default_factory=list)
    colorbars: list[Any] = field(default_factory=list, repr=False)

    def add(self, title: str, handles: Sequence) -> None:
        self.entries.append((title, list(handles)))

    @property
    def horizontal(self) -> bool:
        return self.position in ("top", "bottom")

    def _build(self, fig: Figure, theme: Theme) -> list[_Box]:
        """Create one ``fig.legend`` per entry (unplaced) plus the colourbars."""
        boxes: list[_Box] = []
        for title, handles in self.entries:
            ncols = max(len(handles), 1) if self.horizontal else 1
            boxes.append(_LegendBox(self._legend(fig, theme, title, handles, ncols)))
        boxes.extend(self.colorbars)
        return boxes

    @staticmethod
    def _legend(fig: Figure, theme: Theme, title: str, handles: list, ncols: int) -> Legend:
        # ``fig.legend`` already registers the legend in ``fig.legends``;
        # adding it as an artist too draws it twice and breaks
        # ``leg.remove()``, which then only drops the artist copy.
        return fig.legend(
            handles=handles,
            title=f"{title}:" if title else None,
            loc="upper left",
            bbox_to_anchor=(0.0, 1.0),
            bbox_transform=fig.transFigure,
            borderaxespad=0.0,
            frameon=False,
            fontsize=theme.size("legend.text"),
            title_fontsize=theme.size("legend.title"),
            alignment="left",
            ncols=ncols,
        )

    def _measure(self, boxes: list[_Box], fig: Figure, renderer, theme: Theme,
                 avail_across: float) -> list[tuple[float, float]]:
        sizes = []
        k = 0
        for b in boxes:
            w, h = b.measure(fig, renderer)
            if isinstance(b, _LegendBox):
                title, handles = self.entries[k]
                k += 1
                if self.horizontal and w > avail_across:
                    # a horizontal legend wider than the panels gets fewer
                    # columns; a Legend's box is built once, so rebuild it
                    for ncols in range(len(handles) - 1, 0, -1):
                        at = fig.legends.index(b.legend)
                        b.legend.remove()
                        b.legend = self._legend(fig, theme, title, handles, ncols)
                        fig.legends.remove(b.legend)
                        fig.legends.insert(at, b.legend)  # keep registration order
                        w, h = b.measure(fig, renderer)
                        if w <= avail_across:
                            break
            sizes.append((w, h))
        return sizes

    def draw(self, fig: Figure, theme: Theme | None = None) -> None:
        """Lay the legends out against a figure this package did not size.

        For a caller-owned figure (``ax=`` usage): the legends stack down the
        figure's inside right edge.  :class:`PanelGrid` does not use this; it
        measures the column and grows the figure to hold it instead.
        """
        t = theme or Theme()
        if self.position == "none" or not (self.entries or self.colorbars):
            return
        boxes = self._build(fig, t)
        renderer = _renderer(fig)
        W, H = fig.get_size_inches()
        top = H - MARGIN
        for b in boxes:
            w, h = b.measure(fig, renderer)
            b.place(fig, W - MARGIN - w, top)
            top -= h + SPACING


class _PanelFixEngine(LayoutEngine):
    """Marks a figure as laid out by :class:`PanelGrid`.

    Positions are final once :meth:`PanelGrid.finish` has run, so ``execute``
    has nothing to do.  The engine exists so that ``fig.subplots_adjust`` --
    which would move the exact-size panels -- is refused with a warning, as it
    is for constrained layout.
    """

    _adjust_compatible = False
    _colorbar_gridspec = False

    def execute(self, fig) -> None:  # noqa: D102 - positions are already final
        return None


class PanelGrid:
    """Allocate a grid of equally sized panels at a known physical size.

    Port of the composition half of SCP's ``panel_fix`` (``SCP-plot.R:453``)
    by Hao Zhang, https://github.com/zhanghao-njmu/SCP.

    Every panel's axes box is exactly ``panel_size`` inches.  When the theme
    fixes ``aspect_ratio`` (panel height / width) and ``panel_size`` disagrees
    with it, the panel is the largest box of that ratio inside ``panel_size``
    -- ``panel_fix`` derives the missing dimension from the aspect in the same
    way.  The figure size is an *output*: :meth:`finish` measures what each
    panel draws outside its box and sizes every row and column gap, the outer
    margins and the legend column from those measurements.

    ``margin`` is the outer figure margin and ``wspace`` / ``hspace`` the pad
    between neighbouring panels' decorations, all in inches.  A zero
    ``wspace`` / ``hspace`` makes the panels abut exactly (a stacked violin);
    decorations crossing that seam are the caller's to avoid.

    A cell may be subdivided -- ``pg.axes[key].get_subplotspec().subgridspec``
    -- and the sub-axes are laid out inside the exact panel box.  Colourbars
    go through :meth:`colorbar`, never ``fig.colorbar``, which would steal
    width from the panel.

    Example
    -------
    >>> pg = PanelGrid(n=4, panel_size=(2.5, 2.5), theme=theme_scp(aspect_ratio=1))
    >>> for key, ax in pg:
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
        margin: float = MARGIN,
        wspace: float = SPACING,
        hspace: float = SPACING,
        polar: bool = False,
    ) -> None:
        self.theme = theme or Theme()
        self.legend = legend if legend is not None else LegendColumn(position=self.theme.legend_position)
        self.keys = list(keys) if keys is not None else [str(i) for i in range(n)]
        self.nrow, self.ncol = grid_shape(n, nrow, ncol, byrow)
        self.byrow = byrow
        self.margin, self.wspace, self.hspace = margin, wspace, hspace
        self.panel_w, self.panel_h = self._panel_dims(panel_size, self.theme.aspect_ratio)
        self._slot_bars: dict[tuple[int, int], list[_Box]] = {}
        self._slot_groups: dict[tuple[int, int], tuple[list[list[int]], list[float], list[float]]] = {}
        self._cbar_axes: set[Axes] = set()
        self._finished = False
        self._own: dict[tuple[int, int], list[float]] = {}

        # Initial canvas: exact-size cells with a generous guess at the gaps.
        # Decorations measured on it are in inches and do not depend on where
        # the panel sits, so the second pass only has to move things.
        g = 1.0
        W0 = 2 * g + self.ncol * self.panel_w + (self.ncol - 1) * g
        H0 = 2 * g + self.nrow * self.panel_h + (self.nrow - 1) * g
        with mpl.rc_context(theme_scp_rc(self.theme)):
            self.fig: Figure = plt.figure(figsize=(W0, H0))
        self.gs: GridSpec = self.fig.add_gridspec(
            2 * self.nrow - 1, 2 * self.ncol - 1,
            width_ratios=self._tracks(self.panel_w, [g] * (self.ncol - 1)),
            height_ratios=self._tracks(self.panel_h, [g] * (self.nrow - 1)),
            left=g / W0, right=1 - g / W0, bottom=g / H0, top=1 - g / H0,
            wspace=0.0, hspace=0.0,
        )
        self.axes: dict[str, Axes] = {}
        for i, key in enumerate(self.keys):
            r, c = (i // self.ncol, i % self.ncol) if byrow else (i % self.nrow, i // self.nrow)
            ax = self.fig.add_subplot(self.gs[2 * r, 2 * c], projection="polar" if polar else None)
            if not polar:
                apply_theme(ax, self.theme)
            else:
                # theme_scp's rectangular frame is meaningless on a polar axes;
                # coord_polar drops it in ggplot too.
                ax.set_facecolor(self.theme.background)
                ax.spines["polar"].set_visible(False)
            self.axes[key] = ax

    # ------------------------------------------------------------------ #
    @staticmethod
    def _panel_dims(size: tuple[float, float], aspect: float | None) -> tuple[float, float]:
        w, h = float(size[0]), float(size[1])
        if aspect and abs(h - w * aspect) > 1e-9:
            w = min(w, h / aspect)
            h = w * aspect
        return w, h

    @staticmethod
    def _tracks(panel: float, gaps: Sequence[float]) -> list[float]:
        out = [panel]
        for gp in gaps:
            out += [gp, panel]
        return out

    @property
    def _bar_side(self) -> Side:
        """Where a colourbar attached to one panel goes: the legend side."""
        pos = self.legend.position
        return "right" if pos == "none" else pos

    def __iter__(self):
        return iter(self.axes.items())

    def slot_of(self, ax: Axes) -> tuple[int, int] | None:
        """The (row, column) cell ``ax`` lives in, or ``None`` if it is not on the grid."""
        ss = ax.get_subplotspec()
        if ss is None:
            return None
        top = ss.get_topmost_subplotspec()
        if top.get_gridspec() is not self.gs:
            return None
        return top.rowspan.start // 2, top.colspan.start // 2

    # ------------------------------------------------------------------ #
    def colorbar(
        self,
        mappable,
        ax: Axes | Sequence[Axes] | None = None,
        *,
        length: float | None = None,
        thickness: float | None = None,
        **kwargs,
    ) -> Colorbar:
        """A ggplot colourbar guide, sized in inches rather than stolen from a panel.

        ``ax`` naming axes of a single panel attaches the bar beside that panel
        (each panel is its own ggplot in SCP, with its own guide); no ``ax``,
        or axes spread over several panels, put it in the legend column.  The
        bar is :data:`KEY` thick and ``5 * KEY`` long (ggplot2's default),
        on the side ``legend_position`` names.
        """
        axs = [ax] if isinstance(ax, Axes) else list(ax or [])
        slots = {self.slot_of(a) for a in axs}
        slot = next(iter(slots)) if len(slots) == 1 else None
        orientation = "horizontal" if self._bar_side in ("top", "bottom") else "vertical"
        thick = KEY if thickness is None else thickness
        long_ = 5 * KEY if length is None else length
        if slot is not None and orientation == "vertical":
            long_ = min(long_, self.panel_h)
        elif slot is not None:
            long_ = min(long_, self.panel_w)
        W, H = self.fig.get_size_inches()
        bw, bh = (thick, long_) if orientation == "vertical" else (long_, thick)
        cax = self.fig.add_axes((0.0, 0.0, bw / W, bh / H), label="<colorbar>")
        # ``location`` also puts the ticks on the side away from the panel
        cb = self.fig.colorbar(mappable, cax=cax, location=self._bar_side, **kwargs)
        box = _ColorbarBox(cb, orientation, thick, long_)
        self._cbar_axes.add(cax)
        if slot is None:
            self.legend.colorbars.append(box)
        else:
            self._slot_bars.setdefault(slot, []).append(box)
        return cb

    def legend_beside(self, ax: Axes, title: str, handles: Sequence) -> Legend:
        """A legend that belongs to one panel, stacked with that panel's
        colourbars on the legend side and centred on it.

        SCP draws a panel as its own ggplot and joins them with
        ``patchwork::wrap_plots`` *without* ``guides = "collect"``, so each
        panel keeps its own guides next to it; this is that guide.  A stack
        taller than the panel wraps into another column beside it.
        """
        slot = self.slot_of(ax)
        if slot is None:
            raise ValueError("legend_beside needs an axes on this grid")
        handles = list(handles)
        ncols = max(len(handles), 1) if self._bar_side in ("top", "bottom") else 1
        leg = LegendColumn._legend(self.fig, self.theme, title, handles, ncols)
        self._slot_bars.setdefault(slot, []).append(_LegendBox(leg))
        return leg

    # ------------------------------------------------------------------ #
    def _decorations(self, renderer) -> dict[tuple[int, int], list[float]]:
        """Per cell: how far its drawing reaches past the panel box (l, r, b, t)."""
        fig = self.fig
        members: dict[tuple[int, int], list[Axes]] = {}
        for a in fig.axes:
            if a in self._cbar_axes or not a.get_visible():
                continue
            s = self.slot_of(a)
            if s is not None:
                members.setdefault(s, []).append(a)
        out: dict[tuple[int, int], list[float]] = {}
        for s, axs in members.items():
            cell = _inches(self.gs[2 * s[0], 2 * s[1]].get_position(fig).transformed(fig.transFigure), fig)
            tight = _union(_inches(b, fig) for a in axs
                           if (b := a.get_tightbbox(renderer)) is not None)
            if tight is None:
                continue
            out[s] = [max(cell.x0 - tight.x0, 0.0), max(tight.x1 - cell.x1, 0.0),
                      max(cell.y0 - tight.y0, 0.0), max(tight.y1 - cell.y1, 0.0)]
        # an attached colourbar sits just outside the panel's own decorations
        self._own = {k: list(v) for k, v in out.items()}
        side = self._bar_side
        k = {"left": 0, "right": 1, "bottom": 2, "top": 3}[side]
        for s, bars in self._slot_bars.items():
            sizes = [b.measure(fig, renderer) for b in bars]
            vertical = side in ("left", "right")
            # a stack longer than the panel wraps into another column (row)
            panel = self.panel_h if vertical else self.panel_w
            avail = max(panel, *(sz[1] if vertical else sz[0] for sz in sizes))
            groups, along, across = _pack(sizes, avail, SPACING, vertical=vertical)
            self._slot_groups[s] = (groups, along, across)
            deco = out.setdefault(s, [0.0, 0.0, 0.0, 0.0])
            deco[k] += SPACING + sum(across) + SPACING * (len(groups) - 1)
            # a single key taller than the panel spills evenly past both ends
            spill = max(max(along) - panel, 0.0) / 2
            for e in ((2, 3) if vertical else (0, 1)):
                deco[e] = max(deco[e], spill)
        return out

    def _suplabels(self) -> dict[str, Text]:
        found = {}
        for name in ("_suptitle", "_supxlabel", "_supylabel"):
            t = getattr(self.fig, name, None)
            if t is not None and t.get_visible() and t.get_text():
                found[name] = t
        return found

    def finish(self, *, tight: bool = True) -> Figure:
        """Measure, place and size: the second pass of ``panel_fix``.

        ``tight`` is accepted for backward compatibility; the layout is always
        measured.
        """
        if self._finished:
            return self.fig
        self._finished = True
        fig, th = self.fig, self.theme
        nr, nc, pw, ph, m = self.nrow, self.ncol, self.panel_w, self.panel_h, self.margin

        fig.draw_without_rendering()  # resolve aspect, title offsets, tick labels
        renderer = _renderer(fig)
        deco = self._decorations(renderer)
        zero = [0.0, 0.0, 0.0, 0.0]
        L = [max(deco.get((r, c), zero)[0] for r in range(nr)) for c in range(nc)]
        R = [max(deco.get((r, c), zero)[1] for r in range(nr)) for c in range(nc)]
        B = [max(deco.get((r, c), zero)[2] for c in range(nc)) for r in range(nr)]
        T = [max(deco.get((r, c), zero)[3] for c in range(nc)) for r in range(nr)]
        col_gaps = [0.0 if self.wspace == 0 else R[c] + self.wspace + L[c + 1] for c in range(nc - 1)]
        row_gaps = [0.0 if self.hspace == 0 else B[r] + self.hspace + T[r + 1] for r in range(nr - 1)]
        block_w = L[0] + nc * pw + sum(col_gaps) + R[-1]
        block_h = T[0] + nr * ph + sum(row_gaps) + B[-1]

        sup = self._suplabels()
        sup_size = {k: _inches(t.get_window_extent(renderer), fig) for k, t in sup.items()}
        sup_t = sup_size["_suptitle"].height + SPACING / 2 if "_suptitle" in sup else 0.0
        sup_b = sup_size["_supxlabel"].height + SPACING / 2 if "_supxlabel" in sup else 0.0
        sup_l = sup_size["_supylabel"].width + SPACING / 2 if "_supylabel" in sup else 0.0
        content_w = sup_l + block_w
        content_h = sup_t + block_h + sup_b

        # legends: measured, then wrapped against the panel block
        pos = self.legend.position
        boxes: list[_Box] = []
        if pos != "none" or self.legend.colorbars:
            if pos == "none":  # a colourbar the caller asked for still needs a home
                self.legend.position = pos = "right"
            boxes = self.legend._build(fig, th)
        groups: list[list[int]] = []
        sizes: list[tuple[float, float]] = []
        leg_w = leg_h = 0.0
        if boxes:
            horizontal = self.legend.horizontal
            sizes = self.legend._measure(boxes, fig, renderer, th, block_w)
            groups, along, across = _pack(sizes, block_w if horizontal else content_h,
                                          SPACING, vertical=not horizontal)
            if horizontal:
                leg_w = max(along)
                leg_h = sum(across) + SPACING * (len(groups) - 1)
            else:
                leg_h = max(along)
                leg_w = sum(across) + SPACING * (len(groups) - 1)

        # figure size and the two anchors (content, legend), measured from the top-left
        if boxes and pos in ("left", "right"):
            inner_h = max(content_h, leg_h)
            W = 2 * m + content_w + SPACING + leg_w
            H = 2 * m + inner_h
            c_left = m + (leg_w + SPACING if pos == "left" else 0.0)
            c_top = m + (inner_h - content_h) / 2
            l_left = m if pos == "left" else m + content_w + SPACING
            l_top = m + (inner_h - leg_h) / 2
        elif boxes:
            inner_w = max(content_w, leg_w)
            W = 2 * m + inner_w
            H = 2 * m + content_h + SPACING + leg_h
            c_left = m + (inner_w - content_w) / 2
            c_top = m + (leg_h + SPACING if pos == "top" else 0.0)
            l_left = m + (inner_w - leg_w) / 2
            l_top = m if pos == "top" else m + content_h + SPACING
        else:
            W, H = 2 * m + content_w, 2 * m + content_h
            c_left, c_top, l_left, l_top = m, m, 0.0, 0.0
        fig.set_size_inches(W, H, forward=False)

        # panels: one GridSpec with gap tracks, so subdivided cells follow
        x0 = c_left + sup_l + L[0]
        y0 = c_top + sup_t + T[0]  # top of the first row, from the top edge
        grid_w = nc * pw + sum(col_gaps)
        grid_h = nr * ph + sum(row_gaps)
        self.gs.set_width_ratios(self._tracks(pw, col_gaps))
        self.gs.set_height_ratios(self._tracks(ph, row_gaps))
        self.gs.left, self.gs.right = x0 / W, (x0 + grid_w) / W
        self.gs.top, self.gs.bottom = 1 - y0 / H, 1 - (y0 + grid_h) / H
        for a in fig.axes:
            ss = a.get_subplotspec()
            if a not in self._cbar_axes and ss is not None and self.slot_of(a) is not None:
                a.set_position(ss.get_position(fig))
                a.set_in_layout(True)

        # colourbars attached to a panel, centred on it, past its decorations
        side = self._bar_side
        for (r, c), bars in self._slot_bars.items():
            cell = self.gs[2 * r, 2 * c].get_position(fig)
            cx0, cx1 = cell.x0 * W, cell.x1 * W
            ct, cbot = H - cell.y1 * H, H - cell.y0 * H  # top / bottom, from the top edge
            own = self._own.get((r, c), zero)
            s_groups, s_along, s_across = self._slot_groups[(r, c)]
            off = 0.0
            for grp, al, ac in zip(s_groups, s_along, s_across, strict=True):
                if side in ("left", "right"):
                    top = (ct + cbot) / 2 - al / 2
                    for i in grp:
                        b = bars[i]
                        x = (cx1 + own[1] + SPACING + off if side == "right"
                             else cx0 - own[0] - SPACING - off - b.w)
                        b.place(fig, x, H - top)
                        top += b.h + SPACING
                else:
                    x = (cx0 + cx1) / 2 - al / 2
                    for i in grp:
                        b = bars[i]
                        top = (cbot + own[2] + SPACING + off if side == "bottom"
                               else ct - own[3] - SPACING - off - b.h)
                        b.place(fig, x, H - top)
                        x += b.w + SPACING
                off += ac + SPACING

        # the legend column, group by group
        offset = 0.0
        for grp in groups:
            if self.legend.horizontal:
                x = l_left
                top = l_top + offset
                for i in grp:
                    boxes[i].place(fig, x, H - top)
                    x += sizes[i][0] + SPACING
                offset += max(sizes[i][1] for i in grp) + SPACING
            else:
                x = l_left + offset
                top = l_top
                for i in grp:
                    boxes[i].place(fig, x, H - top)
                    top += sizes[i][1] + SPACING
                offset += max(sizes[i][0] for i in grp) + SPACING

        # figure-level labels, centred on the panels they label
        mid_x = (x0 + grid_w / 2) / W
        mid_y = 1 - (y0 + grid_h / 2) / H
        if "_suptitle" in sup:
            sup["_suptitle"].set_position((mid_x, 1 - (c_top) / H))
            sup["_suptitle"].set_horizontalalignment("center")
            sup["_suptitle"].set_verticalalignment("top")
        if "_supxlabel" in sup:
            sup["_supxlabel"].set_position((mid_x, 1 - (c_top + content_h) / H))
            sup["_supxlabel"].set_horizontalalignment("center")
            sup["_supxlabel"].set_verticalalignment("bottom")
        if "_supylabel" in sup:
            sup["_supylabel"].set_position((c_left / W, mid_y))
            sup["_supylabel"].set_horizontalalignment("left")
            sup["_supylabel"].set_verticalalignment("center")

        fig.set_layout_engine(_PanelFixEngine())
        return fig


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
