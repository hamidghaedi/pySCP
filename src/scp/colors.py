"""Color algebra ported from ``SCP/R/SCP-plot.R`` lines 1000--1129.

These four functions are what make ``FeatureDimPlot(compare_features=True)``
work.  They are pure and fully testable against R, so port them first and pin
them with golden fixtures.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Literal

import numpy as np

from .palettes import _hex_to_rgb, _rgb_to_hex

BlendMode = Literal["blend", "average", "screen", "multiply"]

__all__ = ["adjcolors", "blendcolors", "rgba_to_rgb", "blend2color", "blend_rgb_list", "luminance_text_color"]

_WHITE = np.array([1.0, 1.0, 1.0])


def rgba_to_rgb(rgb: Sequence[float], alpha: float, background: Sequence[float] = (1, 1, 1)) -> np.ndarray:
    """``RGBA2RGB``: flatten an RGBA color onto an opaque background."""
    rgb = np.asarray(rgb, dtype=float)
    bg = np.asarray(background, dtype=float)
    return rgb * alpha + bg * (1.0 - alpha)


def adjcolors(colors: str | Iterable[str], alpha: float) -> list[str] | str:
    """``adjcolors``: composite each color at ``alpha`` over white.

    Used for two things in SCP: the faint end of each feature's private ramp in
    the blend path (``adjcolors(hue, 0.1)``) and the washed-out background
    tiles of ``add_bg`` heatmaps.
    """
    scalar = isinstance(colors, str)
    seq = [colors] if scalar else list(colors)
    out = [_rgb_to_hex(rgba_to_rgb(_hex_to_rgb(c), alpha)) for c in seq]
    return out[0] if scalar else out


def blend2color(
    c1: Sequence[float], a1: float, c2: Sequence[float], a2: float, mode: BlendMode = "blend"
) -> tuple[np.ndarray, float]:
    """``Blend2Color``: composite two RGBA colors under one of four modes."""
    c1 = np.asarray(c1, dtype=float)
    c2 = np.asarray(c2, dtype=float)
    a = 1.0 - (1.0 - a1) * (1.0 - a2)
    if a < 1e-6:
        return np.zeros(3), 1.0
    if mode == "blend":
        return (c1 * a1 + c2 * a2 * (1.0 - a1)) / a, 1.0
    if mode == "average":
        return np.clip((c1 + c2) / 2.0, 0.0, 1.0), a
    if mode == "screen":
        return 1.0 - (1.0 - c1) * (1.0 - c2), a
    if mode == "multiply":
        return c1 * c2, a
    raise ValueError(f"unknown blend mode {mode!r}")


def blend_rgb_list(
    clist: Sequence[tuple[Sequence[float], float]],
    mode: BlendMode = "blend",
    background: Sequence[float] = (1, 1, 1),
) -> np.ndarray:
    """``BlendRGBList``: fold a list of RGBA colors down to one RGB triple.

    The reduction is unusual and must be reproduced exactly: while more than
    one element remains, the *last* element is folded into every earlier one
    with weights ``a_i * (1 - 1/N)`` and ``a_last * (1/N)``, where ``N`` is the
    current list length.  So with three inputs the last contributes 1/3, then
    the (new) last contributes 1/2.
    """
    cur = [(np.asarray(c, dtype=float), float(a)) for c, a in clist]
    n = len(cur)
    while n != 1:
        temp = cur
        cur = []
        last_c, last_a = temp[-1]
        for c, a in temp[:-1]:
            cur.append(blend2color(c, a * (1.0 - 1.0 / n), last_c, last_a * (1.0 / n), mode=mode))
        n = len(cur)
    return rgba_to_rgb(cur[0][0], cur[0][1], background)


def blendcolors(colors: Iterable[str | float | None], mode: BlendMode = "blend") -> str | None:
    """``blendcolors``: blend hex colors, dropping missing entries.

    ``None``/NaN entries are dropped; zero survivors return ``None``; one
    survivor passes through untouched.
    """
    seq = [c for c in colors if isinstance(c, str) and c]
    if not seq:
        return None
    if len(seq) == 1:
        return seq[0]
    return _rgb_to_hex(blend_rgb_list([(_hex_to_rgb(c), 1.0) for c in seq], mode=mode))


def luminance_text_color(color: str, dark: str = "black", light: str = "white") -> str:
    """SCP's rule for label text over a filled swatch.

    R: ``ifelse(colSums(col2rgb(x)) > 255 * 2, "black", "white")`` — i.e. the
    sum of the 0..255 channels, thresholded at 510.
    """
    return dark if sum(_hex_to_rgb(color)) * 255 > 510 else light
