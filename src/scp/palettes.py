"""Palette registry and resolver — the Python port of SCP's ``palette_scp()``.

All 229 palettes shipped by the R package were extracted verbatim from
``SCP/data/palette_list.rda`` and stored in ``data/palette_list.json``.  Each
entry carries the ``type`` attribute R used (``"discrete"`` or
``"continuous"``), because that attribute changes how ``palette_scp`` behaves.

Reference: ``SCP/R/SCP-plot.R`` lines 168--297.
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from typing import Literal

import numpy as np
import pandas as pd

PaletteType = Literal["discrete", "continuous"]

NA_COLOR_DEFAULT = "#CCCCCC"  # R's "grey80"

__all__ = [
    "Palette",
    "PALETTES",
    "list_palettes",
    "get_palette",
    "color_ramp",
    "palette_scp",
    "discrete_palette",
    "continuous_palette",
    "NA_COLOR_DEFAULT",
]


@dataclass(frozen=True)
class Palette:
    name: str
    colors: tuple[str, ...]
    type: PaletteType

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.colors)


@lru_cache(maxsize=1)
def _load() -> dict[str, Palette]:
    raw = json.loads(
        resources.files(__package__).joinpath("data/palette_list.json").read_text()
    )
    return {
        name: Palette(name=name, colors=tuple(v["colors"]), type=v["type"])
        for name, v in raw.items()
    }


class _PaletteRegistry(Mapping[str, Palette]):
    """Lazy, dict-like view over the bundled palettes."""

    def __getitem__(self, key: str) -> Palette:
        try:
            return _load()[key]
        except KeyError:
            raise KeyError(
                f"Unknown palette {key!r}. Use scp.pl.list_palettes() to see the "
                f"{len(_load())} available names, or pass explicit colors via "
                f"`palcolor=`."
            ) from None

    def __iter__(self):
        return iter(_load())

    def __len__(self) -> int:
        return len(_load())


PALETTES = _PaletteRegistry()


def list_palettes(type: PaletteType | None = None) -> list[str]:
    """Names of the bundled palettes, optionally filtered by kind."""
    return [n for n, p in _load().items() if type is None or p.type == type]


def get_palette(name: str) -> Palette:
    return PALETTES[name]


# --------------------------------------------------------------------------- #
# color ramp
# --------------------------------------------------------------------------- #
def _hex_to_rgb(c: str) -> tuple[float, float, float]:
    c = c.lstrip("#")
    if len(c) == 8:  # #RRGGBBAA -> drop alpha, SCP palettes are opaque
        c = c[:6]
    return tuple(int(c[i : i + 2], 16) / 255.0 for i in (0, 2, 4))  # type: ignore[return-value]


def _rgb_to_hex(rgb: Sequence[float]) -> str:
    r, g, b = (int(round(float(np.clip(v, 0.0, 1.0)) * 255)) for v in rgb[:3])
    return f"#{r:02X}{g:02X}{b:02X}"


def color_ramp(colors: Sequence[str], n: int) -> list[str]:
    """Port of ``grDevices::colorRampPalette``.

    Linear interpolation in sRGB across the control points, evaluated at ``n``
    equally spaced positions inclusive of both ends.  R's default
    ``colorRampPalette(space = "rgb")`` does exactly this, so values match to
    within one 8-bit quantisation step.
    """
    if n <= 0:
        return []
    stops = np.asarray([_hex_to_rgb(c) for c in colors], dtype=float)
    if len(stops) == 1:
        return [_rgb_to_hex(stops[0])] * n
    if n == 1:
        return [_rgb_to_hex(stops[0])]
    xp = np.linspace(0.0, 1.0, len(stops))
    x = np.linspace(0.0, 1.0, n)
    out = np.stack([np.interp(x, xp, stops[:, k]) for k in range(3)], axis=1)
    return [_rgb_to_hex(row) for row in out]


# --------------------------------------------------------------------------- #
# the resolver
# --------------------------------------------------------------------------- #
def _resolve_colors(
    palette: str, palcolor: Sequence[str] | Mapping[str, str] | None
) -> tuple[list[str], PaletteType, Mapping[str, str] | None]:
    """Return (colors, declared type, name->color map if palcolor was named)."""
    if palcolor is None or (hasattr(palcolor, "__len__") and len(palcolor) == 0):
        p = PALETTES[palette]
        return list(p.colors), p.type, None
    if isinstance(palcolor, Mapping):
        return list(palcolor.values()), "discrete", palcolor
    return list(palcolor), "discrete", None


def discrete_palette(
    levels: Sequence[str],
    palette: str = "Paired",
    palcolor: Sequence[str] | Mapping[str, str] | None = None,
    *,
    reverse: bool = False,
    na_color: str = NA_COLOR_DEFAULT,
    include_na: bool = False,
) -> dict[str, str]:
    """Map an ordered level list to colors, reproducing SCP's discrete branch.

    The rule that matters: if ``len(levels) <= len(palette)`` the *first*
    ``len(levels)`` palette colors are taken **verbatim** (no interpolation);
    otherwise the palette is ramped to exactly ``len(levels)`` colors.  Getting
    this wrong changes every categorical figure.

    A *named* ``palcolor`` is honoured directly for the levels it covers, which
    is how ``adata.uns['<key>_colors']`` should be threaded through.
    """
    levels = [str(x) for x in levels]
    colors, declared, named = _resolve_colors(palette, palcolor)

    if named is not None and all(lv in named for lv in levels):
        out = {lv: named[lv] for lv in levels}
    else:
        n = len(levels)
        if declared == "continuous":
            # R: attr(palcolor,"type") == "continuous" -> always interpolate
            picked = color_ramp(colors, n)
        elif n <= len(colors):
            picked = list(colors[:n])
        else:
            picked = color_ramp(colors, n)
        out = dict(zip(levels, picked, strict=True))

    if include_na:
        out["NA"] = na_color
    if reverse:
        keys = list(out)
        out = dict(zip(keys, list(out.values())[::-1], strict=True))
    return out


def continuous_palette(
    palette: str = "Spectral",
    palcolor: Sequence[str] | None = None,
    *,
    n: int = 100,
    reverse: bool = False,
) -> list[str]:
    """``n`` colors spanning the palette — SCP's continuous branch, cleaned up.

    SCP's R implementation cuts the literal vector ``1:100`` rather than the
    data when ``matched = FALSE``, which is a bug that happens to be harmless
    because the result is only ever used as an evenly spaced ramp.  We return
    the ramp directly.
    """
    colors, _declared, _named = _resolve_colors(palette, palcolor)
    ramp = color_ramp(colors, n)
    return ramp[::-1] if reverse else ramp


def palette_scp(
    x: Iterable | None = None,
    *,
    n: int = 100,
    palette: str = "Paired",
    palcolor: Sequence[str] | Mapping[str, str] | None = None,
    type: Literal["auto", "discrete", "continuous"] = "auto",
    matched: bool = False,
    reverse: bool = False,
    na_keep: bool = False,
    na_color: str = NA_COLOR_DEFAULT,
):
    """Faithful entry point mirroring ``palette_scp()``.

    Prefer :func:`discrete_palette` / :func:`continuous_palette` in new code —
    they have narrower, clearer contracts.  This function exists so that ported
    R call sites can be transcribed one-to-one during the initial port and
    diffed against R output.

    Returns
    -------
    ``dict[str, str]`` for the discrete branch, ``list[str]`` for the
    continuous branch, or ``np.ndarray[str]`` of per-element colors when
    ``matched=True``.
    """
    if x is None:
        x = np.arange(1, n + 1)
        type = "continuous"

    s = pd.Series(list(x)) if not isinstance(x, pd.Series) else x
    if type == "auto":
        type = "continuous" if pd.api.types.is_numeric_dtype(s) else "discrete"

    has_na = bool(s.isna().any())

    if type == "discrete":
        from .fetch import as_ordered_categorical  # local import, avoids a cycle

        cat = as_ordered_categorical(s)
        mapping = discrete_palette(
            list(cat.cat.categories),
            palette=palette,
            palcolor=palcolor,
            reverse=reverse,
            na_color=na_color,
            include_na=has_na and na_keep,
        )
        if matched:
            return np.asarray([mapping.get(str(v), na_color) for v in s], dtype=object)
        if has_na and not na_keep:
            mapping.pop("NA", None)
        return mapping

    # continuous
    vals = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
    ramp = continuous_palette(palette, palcolor, n=n, reverse=reverse)  # type: ignore[arg-type]
    if not matched:
        return ramp
    finite = np.isfinite(vals)
    out = np.full(vals.shape, na_color, dtype=object)
    if finite.any():
        lo, hi = float(np.nanmin(vals[finite])), float(np.nanmax(vals[finite]))
        if hi == lo:
            out[finite] = ramp[0]
        else:
            idx = np.clip(((vals[finite] - lo) / (hi - lo) * (n - 1)).round().astype(int), 0, n - 1)
            out[finite] = np.asarray(ramp, dtype=object)[idx]
    if has_na and not na_keep:
        warnings.warn("NA values present; they are colored with `na_color`.", stacklevel=2)
    return out
