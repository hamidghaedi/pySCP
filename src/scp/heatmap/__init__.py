"""Heatmap engine: a declarative spec, a GridSpec renderer, and the pure
transforms behind them (matrix processing, feature splitting and ordering,
enrichment word boxes, query/reference similarity)."""

from . import cluster, enrichment, glyphs, similarity
from .matrix import ExpMethod, clean_nonfinite, color_limits, lib_normalize, matrix_process
from .render import render
from .spec import Dendrogram, HeatmapResult, HeatmapSpec, Layer, LegendSpec, PanelSpec, Track

__all__ = [
    "HeatmapSpec",
    "PanelSpec",
    "Track",
    "Layer",
    "LegendSpec",
    "Dendrogram",
    "HeatmapResult",
    "render",
    "matrix_process",
    "lib_normalize",
    "clean_nonfinite",
    "color_limits",
    "ExpMethod",
    "cluster",
    "enrichment",
    "glyphs",
    "similarity",
]
