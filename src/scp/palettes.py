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
    "uns_palcolor",
    "level_colors",
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
    """Names of the bundled palettes, optionally filtered by kind.

    The name listing of ``show_palettes(return_names = TRUE)``
    (``SCP-plot.R:298``) from SCP by Hao Zhang (GPL-3),
    https://github.com/zhanghao-njmu/SCP; the palettes themselves are SCP's
    ``palette_list``.
    """
    return [n for n, p in _load().items() if type is None or p.type == type]


def get_palette(name: str) -> Palette:
    return PALETTES[name]


# --------------------------------------------------------------------------- #
# color ramp
# --------------------------------------------------------------------------- #
def _channels(c: str) -> tuple[int, int, int]:
    """``col2rgb(c)``: the 8-bit channels, alpha dropped (``alpha = FALSE``)."""
    s = str(c)
    if not s.startswith("#"):
        from matplotlib.colors import to_hex

        s = to_hex(s)
    s = s.lstrip("#")
    if len(s) == 3:
        s = "".join(ch * 2 for ch in s)
    return int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)


def _hex_to_rgb(c: str) -> tuple[float, float, float]:
    r, g, b = _channels(c)
    return r / 255.0, g / 255.0, b / 255.0


def _rgb_to_hex(rgb: Sequence[float]) -> str:
    r, g, b = (int(round(float(np.clip(v, 0.0, 1.0)) * 255)) for v in rgb[:3])
    return f"#{r:02X}{g:02X}{b:02X}"


def color_ramp(colors: Sequence[str], n: int) -> list[str]:
    """``grDevices::colorRampPalette(colors)(n)``, to the bit.

    R's recipe, operation for operation, so every channel agrees exactly:

    * ``colorRamp`` takes ``col2rgb(colors) / 255`` and builds one
      ``approxfun`` per channel over ``seq.int(0, 1, length.out = k)``;
    * the ramp is evaluated at ``seq.int(0, 1, length.out = n)`` with
      ``approx1``'s arithmetic (an exact hit on a stop returns the stop), then
      clamped to [0, 1] and multiplied by 255;
    * ``rgb(..., maxColorValue = 255)`` coerces to integer, i.e. it
      **truncates**.  Rounding instead lands one 8-bit step above R on most
      interior positions (87 of Spectral's 100), which is what this port did
      until the palette goldens in ``tests/fixtures/parity/palettes`` pinned it.

    A single colour is repeated (``colorRamp`` duplicates it), and alpha is
    dropped, as ``colorRampPalette``'s default ``alpha = FALSE`` does.
    """
    if n <= 0:
        return []
    stops = np.asarray([_channels(c) for c in colors], dtype=float) / 255.0
    if len(stops) == 1:
        stops = np.vstack([stops, stops])
    xp = np.linspace(0.0, 1.0, len(stops))
    x = np.linspace(0.0, 1.0, n)
    # approx1(): bisection to x[i] <= v <= x[i + 1], then y[i] + dy * (dv / dx)
    i = np.clip(np.searchsorted(xp, x, side="right") - 1, 0, len(xp) - 2)
    j = i + 1
    with np.errstate(invalid="ignore"):
        frac = (x - xp[i]) / (xp[j] - xp[i])
    chan = stops[i] + (stops[j] - stops[i]) * frac[:, None]
    chan = np.where((x == xp[j])[:, None], stops[j], chan)
    chan = np.where((x == xp[i])[:, None], stops[i], chan)
    ints = np.trunc(np.clip(chan, 0.0, 1.0) * 255.0).astype(int)
    return ["#{:02X}{:02X}{:02X}".format(*row) for row in ints]


def _take_or_ramp(colors: Sequence[str], n: int) -> list[str]:
    """``ifelse(n <= pal_n, palcolor[1:n], colorRampPalette(palcolor)(n))``.

    The rule both of ``palette_scp``'s branches apply to a palette not
    declared continuous, and the continuous branch to every palette: up to its
    length a palette is used **verbatim**, from the start.  So
    ``palette_scp(palette = "viridis", n = 100)`` is viridis' own 100 colours
    (alpha suffix included), not a re-interpolation of them.
    """
    return list(colors[:n]) if n <= len(colors) else color_ramp(colors, n)


def _r_cut_codes(vals: np.ndarray, n: int) -> np.ndarray:
    """``cut(x, seq(min, max, length.out = n + 1), include.lowest = TRUE)``.

    The 0-based bin of every finite value, ``-1`` for NA: ``.bincode`` with
    right-closed bins ``(b[k-1], b[k]]``, the first closed on both sides.  The
    breaks are ``seq``'s, which ``np.linspace`` reproduces bit for bit.
    """
    out = np.full(vals.shape, -1, dtype=int)
    finite = np.isfinite(vals)
    if not finite.any():
        return out
    breaks = np.linspace(np.min(vals[finite]), np.max(vals[finite]), n + 1)
    code = np.searchsorted(breaks, vals[finite], side="left")
    out[finite] = np.clip(code, 1, n) - 1
    return out


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
    this wrong changes every categorical figure.  A palette declared
    ``"continuous"`` is always ramped, as R checks ``attr(palcolor, "type")``.

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
        else:
            picked = _take_or_ramp(colors, n)
        out = dict(zip(levels, picked, strict=True))

    if include_na:
        out["NA"] = na_color
    if reverse:
        keys = list(out)
        out = dict(zip(keys, list(out.values())[::-1], strict=True))
    return out


def uns_palcolor(adata, key: str, levels: Sequence[str] | None = None) -> dict[str, str] | None:
    """scanpy's ``adata.uns[f"{key}_colors"]`` as a level -> colour mapping.

    scanpy stores one colour per category of ``obs[key]``, in category order.
    The list is used only when it is aligned: ``obs[key]`` is categorical and
    the list has exactly one colour per category (otherwise it is ignored
    with a warning -- positions would pair the wrong colours).  Categories
    are keyed by ``str``, as every plot labels them.  ``levels`` narrows the
    mapping to those levels.  Returns ``None`` when there is nothing usable.

    Nothing is written back to ``adata.uns``: that would make the palette of
    one call stick to every later one, since a stored list takes precedence.
    """
    uns = getattr(adata, "uns", None)
    obs = getattr(adata, "obs", None)
    if uns is None or obs is None or key not in getattr(obs, "columns", ()):
        return None
    stored = uns.get(f"{key}_colors")
    if stored is None:
        return None
    col = obs[key]
    if not isinstance(col.dtype, pd.CategoricalDtype):
        return None
    cats = [str(c) for c in col.cat.categories]
    stored = [str(c) for c in np.atleast_1d(np.asarray(stored, dtype=object))]
    if len(stored) != len(cats):
        warnings.warn(
            f"adata.uns['{key}_colors'] has {len(stored)} colours for {len(cats)} "
            f"categories of obs['{key}']; it is ignored.",
            stacklevel=3,
        )
        return None
    out = dict(zip(cats, stored, strict=True))
    if levels is not None:
        out = {str(lv): out[str(lv)] for lv in levels if str(lv) in out}
    return out or None


def level_colors(
    adata,
    key: str,
    levels: Sequence[str],
    *,
    palette: str,
    palcolor: Sequence[str] | Mapping[str, str] | None,
    default_palette: str,
    use_uns_colors: bool = True,
) -> dict[str, str]:
    """The colours a plot gives the levels of ``obs[key]``.

    SCP's ``palette_scp(levels, palette, palcolor)`` -- unless the caller
    left both ``palette`` (at ``default_palette``) and ``palcolor`` alone and
    scanpy has stored colours for ``key``, in which case those win for the
    levels they cover (:func:`uns_palcolor`).  Levels they do not cover,
    such as ``show_na``'s ``"NA"``, keep the palette's colour.
    ``use_uns_colors=False`` restores R's colours exactly.
    """
    out = discrete_palette(levels, palette=palette, palcolor=palcolor)
    if use_uns_colors and palcolor is None and palette == default_palette:
        stored = uns_palcolor(adata, key, levels)
        if stored:
            out.update(stored)
    return out


def continuous_palette(
    palette: str = "Spectral",
    palcolor: Sequence[str] | None = None,
    *,
    n: int = 100,
    reverse: bool = False,
) -> list[str]:
    """``palette_scp(palette = palette, n = n)``: SCP's continuous branch.

    ``n`` colours for an evenly spaced scale.  As in R, a palette with at
    least ``n`` colours is used verbatim from the start (the 100-colour
    viridis family at ``n = 100`` is itself; a 5-colour request takes its
    first five), and a shorter one is ramped with :func:`color_ramp`.

    SCP cuts the literal vector ``1:100`` rather than any data here, which
    only ever yields ``n`` levels, so the result is returned directly.
    """
    colors, _declared, _named = _resolve_colors(palette, palcolor)
    ramp = _take_or_ramp(colors, n)
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
    """Faithful entry point mirroring ``palette_scp()`` (SCP-plot.R:168), from
    SCP by Hao Zhang (GPL-3), https://github.com/zhanghao-njmu/SCP.

    Prefer :func:`discrete_palette` / :func:`continuous_palette` in new code —
    they have narrower, clearer contracts.  This function exists so that ported
    R call sites can be transcribed one-to-one and diffed against R output;
    ``tests/fixtures/parity/palettes`` pins it against SCP 0.5.6.

    ``matched=True`` returns one colour per element of ``x``.  For continuous
    ``x`` that is R's ``cut(x, seq(min, max, length.out = n + 1),
    include.lowest = TRUE)``: ``n`` equal-width right-closed bins, bin ``k``
    taking the ``k``-th colour of the ``n``-colour scale.  NA elements come
    back as ``None`` — what R's ``NA_keep = FALSE`` filter leaves there — or
    ``na_color`` with ``na_keep=True``.

    Two deliberate divergences, both R bugs that only show at the edges:

    * a constant ``x`` makes R return a *single* colour (the palette's first)
      whatever the length of ``x``; here it is repeated per element;
    * ``reverse=TRUE`` with ``matched=TRUE`` reverses R's per-element vector,
      so colours no longer line up with their elements; here the *scale* is
      reversed, as for an unmatched palette.

    Returns
    -------
    ``dict[str, str]`` for the discrete branch, ``list[str]`` for the
    continuous branch, or ``np.ndarray[object]`` of per-element colours when
    ``matched=True``.
    """
    if x is None:
        x = np.arange(1, n + 1)
        type = "continuous"

    s = pd.Series(list(x)) if not isinstance(x, pd.Series) else x
    if type == "auto":
        type = "continuous" if pd.api.types.is_numeric_dtype(s) else "discrete"

    has_na = bool(s.isna().any())
    na_out = na_color if na_keep else None

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
            return np.asarray([na_out if pd.isna(v) else mapping.get(str(v), na_out)
                               for v in s], dtype=object)
        return mapping

    # continuous
    vals = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(vals)
    colors, _declared, _named = _resolve_colors(palette, palcolor)  # type: ignore[arg-type]
    constant = finite.any() and np.unique(vals[finite]).size == 1
    if not finite.any() or constant:
        # R: a one-level factor, so a one-colour scale -- the palette's first
        ramp = list(colors[:1])
    else:
        ramp = continuous_palette(palette, palcolor, n=n)  # type: ignore[arg-type]
    if reverse:
        ramp = ramp[::-1]
    if not matched:
        return ramp + [na_color] if (has_na and na_keep) else ramp
    out = np.full(vals.shape, na_out, dtype=object)
    if constant:
        out[finite] = ramp[0]
    elif finite.any():
        out[finite] = np.asarray(ramp, dtype=object)[_r_cut_codes(vals, n)[finite]]
    return out
