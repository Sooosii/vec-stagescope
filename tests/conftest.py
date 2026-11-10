from __future__ import annotations

import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp

from synth import GENES, make_stage


@pytest.fixture(scope="session")
def stages():
    return {name: make_stage(name, 3000, seed) for seed, name in enumerate(["S1", "S2", "S3"])}


def write_h5ad(path, x: np.ndarray, genes=GENES):
    a = ad.AnnData(X=sp.csc_matrix(x))
    a.var_names = list(genes)
    a.obs_names = [f"c{i}" for i in range(x.shape[0])]
    a.write_h5ad(path)
    return path
