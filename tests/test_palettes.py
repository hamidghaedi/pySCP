"""The verbatim-slice rule and the color algebra.

These are the tests that catch silent wrongness: an off-by-one palette or a
population-vs-sample standard deviation never raises, it just produces a
plausible, wrong figure.
"""

import numpy as np
import pytest

import scp
from scp.heatmap import color_limits, matrix_process
from scp.palettes import PALETTES, color_ramp


def test_all_palettes_load():
    assert len(scp.list_palettes()) == 229
    assert len(scp.list_palettes("discrete")) == 85
    assert len(scp.list_palettes("continuous")) == 144


def test_spectral_is_reversed_relative_to_brewer():
    # SCP reverses every Brewer diverging palette at build time, so low is
    # purple/blue and high is red. This is feature_dim_plot's default and it is
    # why SCP feature plots look the way they do.
    assert PALETTES["Spectral"].colors[0] == "#5E4FA2"
    assert PALETTES["Spectral"].colors[-1] == "#9E0142"


@pytest.mark.parametrize("n", [1, 2, 3, 8, 12])
def test_discrete_takes_first_n_verbatim(n):
    # The rule that matters: when n <= len(palette) the first n colors are used
    # as-is. Interpolating instead shifts every categorical figure.
    levels = [f"L{i}" for i in range(n)]
    got = scp.discrete_palette(levels, palette="Paired")
    assert list(got.values()) == list(PALETTES["Paired"].colors[:n])


def test_discrete_interpolates_beyond_palette_length():
    levels = [f"L{i}" for i in range(20)]
    got = scp.discrete_palette(levels, palette="Set1")  # 9 colors
    assert len(got) == 20
    assert got["L0"] == PALETTES["Set1"].colors[0]
    assert got["L19"] == PALETTES["Set1"].colors[-1]


def test_named_palcolor_is_honoured():
    got = scp.discrete_palette(["b", "a"], palcolor={"a": "#111111", "b": "#222222"})
    assert got == {"b": "#222222", "a": "#111111"}


def test_color_ramp_endpoints_and_midpoint():
    ramp = color_ramp(["#000000", "#FFFFFF"], 3)
    assert ramp == ["#000000", "#808080", "#FFFFFF"]


@pytest.mark.parametrize(
    "mode,expected",
    [("screen", "#FFFF40"), ("multiply", "#404040"), ("blend", "#AA5500")],
)
def test_blend_modes(mode, expected):
    assert scp.blendcolors(["#FF0000", "#00FF00"], mode) == expected


def test_blendcolors_drops_missing_and_passes_through_singletons():
    assert scp.blendcolors([None, "#FF0000"]) == "#FF0000"
    assert scp.blendcolors([None, None]) is None


def test_adjcolors_composites_over_white():
    assert scp.adjcolors("#FF0000", 0.1) == "#FFE6E6"
    assert scp.adjcolors("#FF0000", 1.0) == "#FF0000"


def test_zscore_uses_sample_sd():
    # R's scale() is ddof=1. np.std defaults to ddof=0, which would give
    # [-1.2247, 0, 1.2247] -- every heatmap off by sqrt(n/(n-1)).
    got = matrix_process(np.array([[1.0, 2.0, 3.0]]), "zscore")
    assert np.allclose(got, [[-1.0, 0.0, 1.0]])


def test_log2fc_and_fc():
    M = np.array([[1.0, 2.0, 3.0]])
    assert np.allclose(matrix_process(M, "fc"), M / 2.0)
    assert np.allclose(matrix_process(M, "log2fc"), np.log2(M / 2.0))


def test_color_limits_symmetric_branch_snaps_to_half():
    rng = np.random.default_rng(0)
    lo, hi = color_limits(rng.normal(size=(200, 200)), "zscore")
    assert lo == -hi
    assert (hi * 2) % 1 == 0  # rounded up to the nearest 0.5
