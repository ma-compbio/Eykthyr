"""Numba replacements for the ``velocyto.estimation`` correlation kernels.

velocyto implements ``colDeltaCor`` and ``colDeltaCorpartial`` as Cython/OpenMP
extensions and is no longer maintained. These functions compute the same
quantities with the same signatures, so velocyto is not needed at runtime.
"""

import multiprocessing

import numba
import numpy as np
from numba import njit, prange


@njit(inline="always")
def _pearson_delta(e, d, i, j, A):
    """Pearson correlation between ``e[:, j] - e[:, i]`` and ``d[:, i]``.

    Matches velocyto's floating point behaviour: identical cells (or a zero
    displacement) give 0/0 = NaN.
    """
    rows = e.shape[0]
    muA = 0.0
    muB = 0.0
    for k in range(rows):
        A[k] = e[k, j] - e[k, i]
        muA += A[k]
        muB += d[k, i]
    muA /= rows
    muB /= rows
    num = 0.0
    sA = 0.0
    sB = 0.0
    for k in range(rows):
        a = A[k] - muA
        b = d[k, i] - muB
        num += a * b
        sA += a * a
        sB += b * b
    return num / np.sqrt(sA * sB)


@njit(parallel=True)
def _colDeltaCor(e, d, out):
    rows, cols = e.shape
    for i in prange(cols):
        A = np.empty(rows)
        for j in range(cols):
            out[i, j] = _pearson_delta(e, d, i, j, A)


@njit(parallel=True)
def _colDeltaCorpartial(e, d, out, ixs):
    rows = e.shape[0]
    for i in prange(ixs.shape[0]):
        A = np.empty(rows)
        for n in range(ixs.shape[1]):
            j = ixs[i, n]
            out[i, j] = _pearson_delta(e, d, i, j, A)


def _run_with_threads(threads, fn, *args):
    available = numba.config.NUMBA_NUM_THREADS
    if threads is None:
        threads = max(multiprocessing.cpu_count() // 2, 1)
    threads = int(min(max(threads, 1), available))
    previous = numba.get_num_threads()
    numba.set_num_threads(threads)
    try:
        fn(*args)
    finally:
        numba.set_num_threads(previous)


def colDeltaCor(emat: np.ndarray, dmat: np.ndarray, threads: int = None) -> np.ndarray:
    """Calculate the correlation between the displacement (d[:,i])
    and the difference between a cell and every other (e - e[:, i])

    Arguments
    ---------
    emat: np.ndarray (ngenes, ncells)
        gene expression matrix
    dmat: np.ndarray (ngenes, ncells)
        gene velocity/displacement matrix
    threads: int
        number of parallel threads to use (default: half of the CPUs)

    Returns
    -------
    np.ndarray (ncells, ncells) where out[i, j] = corr(e[:, j] - e[:, i], d[:, i])
    """
    emat = np.ascontiguousarray(emat, dtype=np.float64)
    dmat = np.ascontiguousarray(dmat, dtype=np.float64)
    out = np.zeros((emat.shape[1], emat.shape[1]))
    _run_with_threads(threads, _colDeltaCor, emat, dmat, out)
    return out


def colDeltaCorpartial(
    emat: np.ndarray,
    dmat: np.ndarray,
    ixs: np.ndarray,
    threads: int = None,
) -> np.ndarray:
    """Calculate the correlation between the displacement (d[:,i])
    and the difference between a cell and its neighbours (e[:, ixs[i]] - e[:, i])

    Arguments
    ---------
    emat: np.ndarray (ngenes, ncells)
        gene expression matrix
    dmat: np.ndarray (ngenes, ncells)
        gene velocity/displacement matrix
    ixs: the neighborhood matrix (ncells, nneighbours)
        ixs[i, k] is the kth neighbour to the cell i
    threads: int
        number of parallel threads to use (default: half of the CPUs)

    Returns
    -------
    np.ndarray (ncells, ncells), zero except at out[i, ixs[i, k]]
    """
    emat = np.ascontiguousarray(emat, dtype=np.float64)
    dmat = np.ascontiguousarray(dmat, dtype=np.float64)
    ixs = np.ascontiguousarray(ixs, dtype=np.intp)
    out = np.zeros((emat.shape[1], emat.shape[1]))
    _run_with_threads(threads, _colDeltaCorpartial, emat, dmat, out, ixs)
    return out
