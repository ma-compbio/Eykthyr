from dataclasses import dataclass

import numpy as np
from scipy.sparse import spmatrix


@dataclass
class Embedding:

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
