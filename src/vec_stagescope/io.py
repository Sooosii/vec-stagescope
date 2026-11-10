"""Reading .h5ad files, aligning gene axes and sampling rows."""
from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp


class InputError(ValueError):
    pass


def load(path: Path) -> tuple[sp.csr_matrix, list[str]]:
    path = Path(path)
    if not path.is_file():
        raise InputError(f"File not found: {path}")
    try:
        adata = ad.read_h5ad(path)
    except Exception as exc:  # anndata raises many types for malformed files
        raise InputError(f"Cannot read {path}: {exc}") from exc
    x = adata.X
    x = sp.csr_matrix(x, dtype=np.float32) if sp.issparse(x) else sp.csr_matrix(
        np.asarray(x, dtype=np.float32))
    return x, [str(g) for g in adata.var_names]


def align(pred_genes: list[str], stage_genes: dict[str, list[str]]):
    """Genes shared by the prediction and every stage, in prediction order, plus warnings."""
    common = set(pred_genes)
    for genes in stage_genes.values():
        common &= set(genes)
    order = [g for g in pred_genes if g in common]
    warnings = []
    for label, genes in stage_genes.items():
        missing = len(set(pred_genes) - set(genes))
        if missing:
            warnings.append(f"{missing} prediction genes not in {label}; using the shared genes.")
    if not order:
        raise InputError("No genes shared between the prediction and the training stages.")
    return order, warnings


def subset_genes(x: sp.csr_matrix, genes: list[str], keep: list[str]) -> sp.csr_matrix:
    index = {g: i for i, g in enumerate(genes)}
    return x[:, [index[g] for g in keep]].tocsr()


def sample_dense(x, n: int, rng: np.random.Generator, idx: np.ndarray | None = None) -> np.ndarray:
    if idx is None:
        idx = np.sort(rng.choice(x.shape[0], min(n, x.shape[0]), replace=False))
    rows = x[idx]
    return rows.toarray() if sp.issparse(rows) else np.asarray(rows)
