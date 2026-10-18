"""PanelGrid's ``panel_fix`` contract: exact panels, measured gaps, legends inside.

These are geometric assertions in figure inches.  They are the regression
guard for the layout engine, which every multi-panel figure goes through.
"""

import io
import itertools

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox
from PIL import Image

import scp
from scp.layout import SPACING, LegendColumn, PanelGrid
from scp.theme import theme_scp

TOL = 0.01  # inches
POSITIONS = ["right", "left", "top", "bottom", "none"]
SHAPES = {  # name: (n, nrow, ncol)
    "1x1": (1, None, None),
    "2x2": (4, 2, 2),
    "1x3": (3, 1, 3),
    "uneven": (5, 2, 3),
}


def _inches(fig, bb: Bbox) -> Bbox:
    return bb.transformed(fig.dpi_scale_trans.inverted())


def _box(ax) -> Bbox:
    """The axes box in figure inches."""
    fig = ax.figure
    return _inches(fig, ax.get_window_extent(fig.canvas.get_renderer()))


def _tight(ax) -> Bbox:
    fig = ax.figure
    return _inches(fig, ax.get_tightbbox(fig.canvas.get_renderer()))


def _overlap(a: Bbox, b: Bbox, eps: float = 1e-3) -> bool:
    return (min(a.x1, b.x1) - max(a.x0, b.x0) > eps) and (min(a.y1, b.y1) - max(a.y0, b.y0) > eps)


def _legend(n_entries=(4, 7), position="right") -> LegendColumn:
    leg = LegendColumn(position=position)
    for k, n in enumerate(n_entries):
        leg.add(f"group{k}", [Line2D([], [], marker="o", linestyle="none", label=f"level {i} name")
                              for i in range(n)])
    return leg


def _grid(shape: str, position: str, *, legend: bool = True, size=(2.0, 1.6), **kw):
    n, nrow, ncol = SHAPES[shape]
    leg = _legend(position=position) if legend else LegendColumn(position=position)
    pg = PanelGrid(n, panel_size=size, nrow=nrow, ncol=ncol, theme=theme_scp(), legend=leg, **kw)
    for i, (_key, ax) in enumerate(pg):
        ax.plot([0, 1000 * (i + 1)], [0, 1])
        ax.set_xlabel("x label")
        ax.set_ylabel(f"y label {i}")
        ax.set_title(f"panel {i}")
    fig = pg.finish()
    fig.canvas.draw()
    return pg, fig


@pytest.mark.parametrize("shape,position,legend",
                         [(s, p, lg) for s, p, lg in itertools.product(SHAPES, POSITIONS, (True, False))])
def test_panels_are_exactly_panel_size(shape, position, legend):
    pg, fig = _grid(shape, position, legend=legend)
    for ax in pg.axes.values():
        b = _box(ax)
        assert b.width == pytest.approx(2.0, abs=TOL)
        assert b.height == pytest.approx(1.6, abs=TOL)


@pytest.mark.parametrize("shape,position", list(itertools.product(SHAPES, POSITIONS)))
def test_legends_inside_figure_and_clear_of_panels(shape, position):
    pg, fig = _grid(shape, position)
    W, H = fig.get_size_inches()
    r = fig.canvas.get_renderer()
    legs = [_inches(fig, lg.get_window_extent(r)) for lg in fig.legends]
    assert len(legs) == (0 if position == "none" else 2)
    panels = [_tight(ax) for ax in pg.axes.values()]
    for lb in legs:
        assert lb.x0 >= -TOL and lb.y0 >= -TOL and lb.x1 <= W + TOL and lb.y1 <= H + TOL
        assert not any(_overlap(lb, p) for p in panels)
    for a, b in itertools.combinations(legs, 2):
        assert not _overlap(a, b)
    # every panel's decorations are inside the figure too
    for p in panels:
        assert p.x0 >= -TOL and p.y0 >= -TOL and p.x1 <= W + TOL and p.y1 <= H + TOL


@pytest.mark.parametrize("shape", list(SHAPES))
def test_gaps_are_the_measured_decorations_plus_pad(shape):
    pg, fig = _grid(shape, "right")
    n, nrow, ncol = SHAPES[shape]
    nrow, ncol = pg.nrow, pg.ncol
    cells = {pg.slot_of(ax): ax for ax in pg.axes.values()}
    for c in range(ncol - 1):
        left = [cells[(r, c)] for r in range(nrow) if (r, c) in cells]
        right = [cells[(r, c + 1)] for r in range(nrow) if (r, c + 1) in cells]
        if not right:
            continue
        reach_r = max(_tight(a).x1 - _box(a).x1 for a in left)
        reach_l = max(_box(a).x0 - _tight(a).x0 for a in right)
        gap = _box(right[0]).x0 - _box(left[0]).x1
        assert gap <= reach_r + reach_l + SPACING + TOL
        assert gap >= reach_r + reach_l - TOL  # decorations never collide
    for r in range(nrow - 1):
        top = [cells[(r, c)] for c in range(ncol) if (r, c) in cells]
        bot = [cells[(r + 1, c)] for c in range(ncol) if (r + 1, c) in cells]
        reach_b = max(_box(a).y0 - _tight(a).y0 for a in top)
        reach_t = max(_tight(a).y1 - _box(a).y1 for a in bot)
        gap = _box(top[0]).y0 - _box(bot[0]).y1
        assert gap <= reach_b + reach_t + SPACING + TOL
        assert gap >= reach_b + reach_t - TOL


def test_zero_spacing_abuts_rows():
    pg = PanelGrid(3, panel_size=(2.0, 0.8), nrow=3, ncol=1, hspace=0.0,
                   legend=LegendColumn(position="none"))
    for _k, ax in pg:
        ax.set_xticks([])
    fig = pg.finish()
    fig.canvas.draw()
    boxes = [_box(ax) for ax in pg.axes.values()]
    for a, b in zip(boxes, boxes[1:], strict=False):
        assert a.y0 == pytest.approx(b.y1, abs=TOL)


def test_tall_legends_wrap_into_columns():
    leg = _legend(n_entries=(12, 12, 12))
    pg = PanelGrid(1, panel_size=(2.0, 2.0), legend=leg)
    fig = pg.finish()
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    legs = [_inches(fig, lg.get_window_extent(r)) for lg in fig.legends]
    xs = {round(b.x0, 3) for b in legs}
    assert len(xs) > 1  # wrapped
    for a, b in itertools.combinations(legs, 2):
        assert not _overlap(a, b)
    W, H = fig.get_size_inches()
    assert all(b.x1 <= W + TOL and b.y1 <= H + TOL and b.y0 >= -TOL for b in legs)


def test_wide_horizontal_legend_gets_fewer_columns():
    leg = _legend(n_entries=(15,), position="bottom")
    pg = PanelGrid(1, panel_size=(2.0, 2.0), legend=leg)
    fig = pg.finish()
    fig.canvas.draw()
    lb = _inches(fig, fig.legends[0].get_window_extent(fig.canvas.get_renderer()))
    assert lb.width <= _tight(pg.axes["0"]).width + TOL
    assert lb.y1 <= _box(pg.axes["0"]).y0


@pytest.mark.parametrize("position", ["right", "left", "top", "bottom"])
def test_colourbars_attached_and_in_the_column(position):
    leg = _legend(n_entries=(3,), position=position)
    pg = PanelGrid(4, panel_size=(2.0, 2.0), nrow=2, ncol=2, legend=leg)
    sm = ScalarMappable(norm=Normalize(0, 10), cmap="viridis")
    attached = pg.colorbar(sm, ax=pg.axes["0"])
    attached.set_label("attached")
    column = pg.colorbar(sm)
    column.set_label("shared")
    fig = pg.finish()
    fig.canvas.draw()
    W, H = fig.get_size_inches()
    for ax in pg.axes.values():
        assert _box(ax).width == pytest.approx(2.0, abs=TOL)
        assert _box(ax).height == pytest.approx(2.0, abs=TOL)
    bars = [_tight(cb.ax) for cb in (attached, column)]
    panels = [_tight(ax) for ax in pg.axes.values()]
    r = fig.canvas.get_renderer()
    legs = [_inches(fig, lg.get_window_extent(r)) for lg in fig.legends]
    for b in bars + legs:
        assert b.x0 >= -TOL and b.y0 >= -TOL and b.x1 <= W + TOL and b.y1 <= H + TOL
        assert not any(_overlap(b, p) for p in panels)
    for a, b in itertools.combinations(bars + legs, 2):
        assert not _overlap(a, b)


def test_subdivided_cell_fills_the_exact_panel_box():
    pg = PanelGrid(2, panel_size=(3.0, 3.0), ncol=2, legend=LegendColumn(position="none"))
    holder = pg.axes["1"]
    sub = holder.get_subplotspec().subgridspec(2, 2, wspace=0.0, hspace=0.0)
    holder.remove()
    inner = [pg.fig.add_subplot(sub[i, j]) for i in range(2) for j in range(2)]
    for ax in inner:
        ax.set_ylabel("y")
    fig = pg.finish()
    fig.canvas.draw()
    whole = Bbox.union([_box(a) for a in inner])
    assert whole.width == pytest.approx(3.0, abs=TOL)
    assert whole.height == pytest.approx(3.0, abs=TOL)
    assert whole.y0 == pytest.approx(_box(pg.axes["0"]).y0, abs=TOL)
    assert not _overlap(_tight(pg.axes["0"]), Bbox.union([_tight(a) for a in inner]))


def test_aspect_ratio_fits_inside_panel_size():
    pg = PanelGrid(1, panel_size=(3.0, 2.0), theme=theme_scp(aspect_ratio=1))
    fig = pg.finish()
    fig.canvas.draw()
    b = _box(pg.axes["0"])
    assert (b.width, b.height) == (pytest.approx(2.0, abs=TOL), pytest.approx(2.0, abs=TOL))


@pytest.mark.parametrize("tight", [False, True])
def test_savefig_keeps_everything(tight):
    pg, fig = _grid("uneven", "right")
    W, H = fig.get_size_inches()
    dpi = fig.dpi
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight" if tight else None)
    buf.seek(0)
    w, h = Image.open(buf).size
    if not tight:
        assert abs(w - W * dpi) <= 1 and abs(h - H * dpi) <= 1
    else:
        # nothing reaches outside the figure, and its margin is savefig's
        # default pad: the tight crop is the figure itself
        assert abs(w - W * dpi) <= 2 and abs(h - H * dpi) <= 2
    for ax in pg.axes.values():  # saving does not move the panels
        assert _box(ax).width == pytest.approx(2.0, abs=TOL)


def test_subplots_adjust_cannot_move_the_panels():
    pg, fig = _grid("2x2", "right")
    before = [tuple(ax.get_position().bounds) for ax in pg.axes.values()]
    with pytest.warns(UserWarning):
        fig.subplots_adjust(left=0.3)
    assert [tuple(ax.get_position().bounds) for ax in pg.axes.values()] == before


# --------------------------------------------------------------------------- #
# through the plot functions
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("position", POSITIONS)
def test_cell_dim_plot_split_panels_are_panel_size(adata, position):
    fig = scp.pl.cell_dim_plot(adata, "leiden", split_by="batch", legend_position=position,
                               panel_size=(2.4, 2.4))
    fig.canvas.draw()
    W, H = fig.get_size_inches()
    panels = [a for a in fig.axes if a.get_label() != "<colorbar>"]
    for ax in panels:
        assert _box(ax).width == pytest.approx(2.4, abs=TOL)
        assert _box(ax).height == pytest.approx(2.4, abs=TOL)
    r = fig.canvas.get_renderer()
    for lg in fig.legends:
        lb = _inches(fig, lg.get_window_extent(r))
        assert lb.x0 >= -TOL and lb.x1 <= W + TOL and lb.y0 >= -TOL and lb.y1 <= H + TOL
        assert not any(_overlap(lb, _tight(a)) for a in panels)


def test_feature_dim_plot_colourbars_do_not_shrink_panels(adata):
    fig = scp.pl.feature_dim_plot(adata, ["g0", "g1", "g2"], panel_size=(2.2, 2.2))
    fig.canvas.draw()
    panels = [a for a in fig.axes if a.get_label() != "<colorbar>"]
    bars = [a for a in fig.axes if a.get_label() == "<colorbar>"]
    assert len(bars) == 3
    for ax in panels:
        assert _box(ax).width == pytest.approx(2.2, abs=TOL)
    for b, p in itertools.product(bars, panels):
        assert not _overlap(_tight(b), _tight(p))


def test_projection_plot_goes_through_the_panel_grid(adata):
    ax = scp.pl.projection_plot(adata, adata, ref_group="leiden")
    assert ax.figure.get_layout_engine() is not None
    assert _box(ax).width == pytest.approx(3.5, abs=TOL)
    plt.close(ax.figure)


def test_borrowed_ax_is_left_alone(adata):
    fig, ax = plt.subplots(figsize=(4, 4))
    before = ax.get_position().bounds
    scp.pl.projection_plot(adata, adata, ax=ax)
    assert ax.get_position().bounds == before
    assert tuple(fig.get_size_inches()) == (4.0, 4.0)
    assert np.isclose(fig.get_size_inches(), (4, 4)).all()


def test_legend_beside_stacks_with_the_panel_and_wraps_when_taller():
    """A panel's own legends sit right of it, level with it, and a stack
    taller than the panel wraps into a second column instead of spilling
    over the neighbouring rows."""
    pg = PanelGrid(2, panel_size=(1.5, 1.0), ncol=1, legend=LegendColumn(position="right"))
    handles = [Line2D([], [], marker="o", linestyle="none", label=f"k{i}") for i in range(5)]
    a, b = pg.axes["0"], pg.axes["1"]
    first = pg.legend_beside(a, "One", handles)
    second = pg.legend_beside(a, "Two", handles)
    third = pg.legend_beside(b, "Three", handles[:2])
    fig = pg.finish()
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    pa, pb = a.get_window_extent(r), b.get_window_extent(r)
    l1, l2, l3 = (lg.get_window_extent(r) for lg in (first, second, third))
    assert l1.x0 >= pa.x1 and l3.x0 >= pb.x1
    assert l2.x0 >= l1.x1  # wrapped into a second column
    for lb, pb_ in ((l1, pa), (l3, pb)):
        assert pb_.y0 <= (lb.y0 + lb.y1) / 2 <= pb_.y1
    W, H = fig.get_size_inches() * fig.dpi
    assert max(l1.x1, l2.x1, l3.x1) <= W and min(l1.y0, l2.y0) >= 0 and max(l1.y1, l2.y1) <= H
    assert min(l1.y0, l2.y0) >= l3.y1
    plt.close(fig)
