from dataclasses import dataclass

import numpy as np
from scipy.sparse import spmatrix


@dataclass
class Embedding:
    """Container for all data associated with a single cell embedding (e.g. spatial or UMAP).

    Stores raw coordinates, transition probabilities, flow vectors, and grid-level
    summaries used during TF perturbation simulation and visualization.

    Attributes:
        embedding (np.ndarray): 2D cell coordinates, shape ``(n_cells, 2)``.
        embedding_knn (spmatrix): Sparse KNN graph over cells in embedding space,
            shape ``(n_cells, n_cells)``.
        sampling_ixs (np.ndarray): Integer indices of cells selected for the
            transition-probability computation.
        corrcoef (np.ndarray): Pearson correlation between observed and
            simulated expression per cell, shape ``(n_cells,)``.
        corrcoef_random (np.ndarray): Same as ``corrcoef`` but computed with a
            randomized perturbation graph (null model).
        transition_prob (np.ndarray): Cell-to-cell transition probabilities under
            the TF perturbation, shape ``(n_cells, n_cells)``.
        transition_prob_random (np.ndarray): Transition probabilities under the
            randomized null model.
        delta_embedding (np.ndarray): Per-cell velocity vectors in embedding space
            (observed), shape ``(n_cells, 2)``.
        delta_embedding_random (np.ndarray): Per-cell velocity vectors under the
            null model, shape ``(n_cells, 2)``.
        total_p_mass (np.ndarray): Kernel density estimate of cell mass at each
            grid point, shape ``(n_grid**2,)``.
        flow_embedding (np.ndarray): Grid point coordinates used for flow
            visualization, shape ``(n_grid**2, 2)``.
        flow_grid (np.ndarray): Meshgrid coordinates (same as ``flow_embedding``
            but kept for compatibility), shape ``(n_grid**2, 2)``.
        flow (np.ndarray): Smoothed flow vectors on the grid (observed),
            shape ``(n_grid**2, 2)``.
        flow_norm (np.ndarray): Unit-normalized version of ``flow``.
        flow_norm_magnitude (np.ndarray): Magnitude of ``flow`` at each grid point,
            shape ``(n_grid**2,)``.
        flow_rndm (np.ndarray): Smoothed flow vectors on the grid (null model).
        flow_norm_rndm (np.ndarray): Unit-normalized null-model flow.
        flow_norm_magnitude_rndm (np.ndarray): Magnitude of null-model flow.
        min_mass (float): Minimum probability mass threshold; grid points below this
            are masked out during visualization.
        mass_filter (np.ndarray): Boolean mask of shape ``(n_grid**2,)``; ``True``
            where mass is below ``min_mass``.
        colorandum (np.ndarray): Per-cell color values used for scatter plots,
            shape ``(n_cells,)`` or ``(n_cells, 4)`` (RGBA).
    """

    embedding: np.ndarray = None
    embedding_knn: spmatrix = None
    sampling_ixs: np.ndarray = None
    corrcoef: np.ndarray = None
    corrcoef_random: np.ndarray = None
    transition_prob: np.ndarray = None
    transition_prob_random: np.ndarray = None
    delta_embedding: np.ndarray = None
    delta_embedding_random: np.ndarray = None
    total_p_mass: np.ndarray = None
    flow_embedding: np.ndarray = None
    flow_grid: np.ndarray = None
    flow: np.ndarray = None
    flow_norm: np.ndarray = None
    flow_norm_magnitude: np.ndarray = None
    flow_rndm: np.ndarray = None
    flow_norm_rndm: np.ndarray = None
    flow_norm_magnitude_rndm: np.ndarray = None
    min_mass: float = None
    mass_filter: np.ndarray = None
    colorandum: np.ndarray = None
