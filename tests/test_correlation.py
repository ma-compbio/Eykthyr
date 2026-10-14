"""Tests for eykthyr.correlation, the numba replacement for velocyto's kernels.

If velocyto is importable, the kernels are also checked against it directly.
"""
import importlib.util
import pathlib

import numpy as np
import pytest

# Load the module by path so this test does not need the full eykthyr import chain.
_spec = importlib.util.spec_from_file_location(
    "eykthyr_correlation",
    pathlib.Path(__file__).resolve().parents[1] / "eykthyr" / "correlation.py",
)
correlation = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(correlation)


def _data(seed=0, genes=50, cells=80):
    rng = np.random.RandomState(seed)
    e = rng.rand(genes, cells)
    d = rng.randn(genes, cells)
    e[:, 5] = e[:, 3]  # identical cells -> NaN, as in velocyto
    d[:, 7] = 0.0  # zero displacement -> NaN row
    ixs = np.stack([rng.choice(cells, 10, replace=False) for _ in range(cells)])
    ixs[:, 0] = np.arange(cells)  # self is a neighbour, as in the caller
    return e, d, ixs


def _reference_full(e, d):
    n = e.shape[1]
    out = np.zeros((n, n))
    with np.errstate(invalid="ignore", divide="ignore"):
        for i in range(n):
            for j in range(n):
                out[i, j] = np.corrcoef(e[:, j] - e[:, i], d[:, i])[0, 1]
    return out


def test_full_matches_numpy_reference():
    e, d, _ = _data()
    got = correlation.colDeltaCor(e, d, threads=2)
    np.testing.assert_allclose(got, _reference_full(e, d), rtol=1e-10, atol=1e-12, equal_nan=True)


def test_partial_is_full_restricted_to_neighbours():
    e, d, ixs = _data()
    full = correlation.colDeltaCor(e, d)
    part = correlation.colDeltaCorpartial(e, d, ixs)
    mask = np.zeros_like(full, dtype=bool)
    mask[np.arange(len(ixs))[:, None], ixs] = True
    np.testing.assert_array_equal(part[~mask], 0)
    np.testing.assert_allclose(part[mask], full[mask], rtol=0, atol=0, equal_nan=True)


def test_accepts_float32_and_fortran_order():
    e, d, ixs = _data()
    ref = correlation.colDeltaCor(e, d)
    got = correlation.colDeltaCor(np.asfortranarray(e.astype(np.float32)), np.asfortranarray(d))
    np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-6, equal_nan=True)


@pytest.mark.skipif(importlib.util.find_spec("velocyto") is None, reason="velocyto not installed")
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_velocyto(seed):
    from velocyto.estimation import colDeltaCor, colDeltaCorpartial

    e, d, ixs = _data(seed)
    np.testing.assert_allclose(
        correlation.colDeltaCor(e, d), colDeltaCor(e, d), rtol=1e-10, atol=1e-12, equal_nan=True
    )
    np.testing.assert_allclose(
        correlation.colDeltaCorpartial(e, d, ixs),
        colDeltaCorpartial(e, d, ixs),
        rtol=1e-10,
        atol=1e-12,
        equal_nan=True,
    )
