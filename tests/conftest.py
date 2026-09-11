import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "DejaVu Sans"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402


@pytest.fixture(scope="session")
def adata():
    """A small seeded AnnData with four well-separated clusters.

    Deliberately synthetic: runs in milliseconds, needs no network, and the
    cluster structure is known, so label placement can be asserted.
    """
    anndata = pytest.importorskip("anndata")
    rng = np.random.default_rng(0)
    n, g = 600, 30
    obs_names = pd.Index([f"c{i}" for i in range(n)])
    leiden = pd.Categorical(rng.choice(list("ABCD"), n))
    centers = {c: rng.normal(size=2) * 5 for c in "ABCD"}
    ad = anndata.AnnData(
        X=rng.gamma(2.0, 1.0, size=(n, g)).astype(np.float32),
        obs=pd.DataFrame(
            {
                "leiden": leiden,
                "batch": pd.Categorical(rng.choice(["b1", "b2"], n)),
                "score": rng.normal(size=n),
            },
            index=obs_names,
        ),
        var=pd.DataFrame(index=pd.Index([f"g{i}" for i in range(g)])),
    )
    ad.obsm["X_umap"] = np.vstack([centers[c] + rng.normal(size=2) for c in leiden])
    ad.obsm["X_pca"] = rng.normal(size=(n, 50))
    ad.layers["counts"] = rng.poisson(3, size=(n, g)).astype(np.float32)
    return ad
