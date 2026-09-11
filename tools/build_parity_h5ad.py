#!/usr/bin/env python
"""Assemble the files written by ``tools/export_scp_data.R`` into ``.h5ad``.

Run the R side first::

    Rscript tools/export_scp_data.R notebooks/data/_raw
    python tools/build_parity_h5ad.py notebooks/data/_raw notebooks/data

The two halves are split so that the R dependency is confined to the export.
Once the ``.h5ad`` files exist the parity notebook's Python cells, and the
package's own tests, need no R at all.

The AnnData produced here is **scanpy-conventional**, so that pySCP works on it
out of the box: ``X`` holds the log-normalised ``data`` slot, raw counts live in
``layers['counts']``, and embeddings are renamed to ``X_pca`` / ``X_umap``
rather than kept as Seurat's verbatim ``PCA`` / ``UMAP``.  Every factor column
is restored as a Categorical preserving R's level order, which drives palette
index and legend order on both sides.

The original Seurat names are recorded in ``uns['scp']['reduction_name_map']``,
because the R side of the parity notebook must still be addressed with them.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.io import mmread
from scipy.sparse import csr_matrix


def scanpy_obsm_key(seurat_name: str) -> str:
    """Seurat reduction name -> scanpy obsm key: ``UMAP`` -> ``X_umap``.

    SCP's own ``srt_to_adata`` keeps the name verbatim, which leaves ``PCA`` and
    ``UMAP`` in obsm and makes the object unlike anything scanpy produces.  The
    point of this package is to work on ordinary scanpy objects, so the export
    normalises to the scanpy convention and ``adata_to_srt``'s own
    ``sub("^X_", "", k)`` is exactly the inverse.
    """
    return seurat_name if seurat_name.startswith("X_") else f"X_{seurat_name.lower()}"


def _sanitize(df: pd.DataFrame, what: str) -> pd.DataFrame:
    """Make object columns writable to HDF5.

    R writes a logical column holding NA as a mix of bool and NaN, which h5py
    refuses ("Can't implicitly convert non-string objects to strings").  Such
    columns become Categoricals of strings, with missing values left missing.
    """
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = pd.Categorical(df[col].map(lambda v: None if pd.isna(v) else str(v)))
    return df


def _read_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.astype(str)
    return df


def build(raw_dir: Path, out_dir: Path) -> Path:
    meta = json.loads((raw_dir / "meta.json").read_text())
    name = meta["dataset"]

    X = csr_matrix(mmread(raw_dir / "X_data.mtx"))
    obs = _read_table(raw_dir / "obs.csv")
    var = _read_table(raw_dir / "var.csv")

    # R wrote obs/var in object order; assert rather than reindex, because a
    # silent reorder here would misalign every cell against its coordinates.
    obs_names = [str(x) for x in meta["obs_names"]]
    var_names = [str(x) for x in meta["var_names"]]
    assert list(obs.index) == obs_names, f"{name}: obs.csv order does not match meta.json"
    assert list(var.index) == var_names, f"{name}: var.csv order does not match meta.json"
    assert X.shape == (len(obs_names), len(var_names)), f"{name}: X is {X.shape}"

    # Restore factor level order exactly as R had it. jsonlite's auto_unbox
    # collapses a single-level factor to a bare string, so a scalar has to be
    # re-wrapped -- iterating it as a sequence would yield its characters.
    for col, levels in (meta.get("obs_levels") or {}).items():
        if col not in obs.columns:
            continue
        if isinstance(levels, str):
            levels = [levels]
        cats = [str(x) for x in levels]
        values = obs[col].astype(str)
        unseen = set(values.unique()) - set(cats)
        assert not unseen, f"{name}: {col} has values outside its R levels: {sorted(unseen)[:5]}"
        obs[col] = pd.Categorical(values, categories=cats, ordered=False)

    adata = ad.AnnData(X=X, obs=_sanitize(obs, "obs"), var=_sanitize(var, "var"))
    adata.obs_names = obs_names
    adata.var_names = var_names

    for mtx in sorted(raw_dir.glob("layer_*.mtx")):
        key = mtx.stem[len("layer_"):]
        layer = csr_matrix(mmread(mtx))
        if layer.shape != adata.shape:
            print(f"   layer {key}: shape {layer.shape} != {adata.shape}, skipped")
            continue
        adata.layers[key] = layer

    name_map: dict[str, str] = {}
    for csv in sorted(raw_dir.glob("obsm_*.csv")):
        seurat_name = csv.stem[len("obsm_"):]
        key = scanpy_obsm_key(seurat_name)
        emb = _read_table(csv)
        assert list(emb.index) == obs_names, f"{name}: obsm {seurat_name} not in object order"
        adata.obsm[key] = np.asarray(emb, dtype=np.float32)
        name_map[key] = seurat_name

    # Provenance and the Seurat @key values, which have no AnnData counterpart.
    adata.uns["scp"] = {
        "source": "SCP (Hao Zhang, NJMU, GPL-3) -- https://github.com/zhanghao-njmu/SCP",
        "dataset": name,
        "scp_version": meta.get("scp_version"),
        "seurat_version": meta.get("seurat_version"),
        "x_slot": "data",
        "reduction_keys": meta.get("reduction_keys", {}),
        "reduction_name_map": name_map,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{name}.h5ad"
    adata.write_h5ad(out, compression="gzip")
    print(f"{name}: {adata.n_obs} x {adata.n_vars} | layers={list(adata.layers)} "
          f"| obsm={list(adata.obsm)} | {out.stat().st_size / 1e6:.1f} MB -> {out}")
    return out


def main(argv: list[str]) -> int:
    raw_root = Path(argv[1] if len(argv) > 1 else "notebooks/data/_raw")
    out_dir = Path(argv[2] if len(argv) > 2 else "notebooks/data")
    dirs = [d for d in sorted(raw_root.iterdir()) if (d / "meta.json").exists()]
    if not dirs:
        print(f"no exported datasets under {raw_root}; run tools/export_scp_data.R first")
        return 1
    for d in dirs:
        build(d, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
