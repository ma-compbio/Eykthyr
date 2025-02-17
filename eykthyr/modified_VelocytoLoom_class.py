import itertools
import logging
import warnings
from copy import deepcopy
from typing import Any, Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
from numba import jit
from scipy import sparse
from scipy.spatial.distance import pdist, squareform
from scipy.stats import norm as normal
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from velocyto.diffusion import Diffusion
from velocyto.estimation import (
    colDeltaCor,
    colDeltaCorLog10,
    colDeltaCorLog10partial,
    colDeltaCorpartial,
    colDeltaCorSqrt,
    colDeltaCorSqrtpartial,
)

from .util import (
    CONFIG,
    _adata_to_matrix,
    _get_clustercolor_from_anndata,
    _obsm_to_matrix,
    plot_background,
    plot_cluster_cells_use,
)


class modified_VelocytoLoom:
    """
    ########################################################################################
    ###  This class is taken from CellOracle (https://github.com/morris-lab/CellOracle)  ###
    ###  and modified for our use                                                        ###
    ########################################################################################
    """

    def __init__(self):

        pass

    def calculate_p_mass(
        self,
        embeddings,
        smooth=0.8,
        n_grid=40,
        n_neighbors=itertools.repeat(200),
        n_jobs=-1,
    ):
        for i, embedding in enumerate(embeddings):
            self.calculate_grid_arrows(
                embedding,
                smooth=0.8,
                steps=(n_grid, n_grid),
                n_neighbors=n_neighbors[i],
                n_jobs=-1,
            )

    def suggest_mass_thresholds(self, n_suggestion=12, s=1, n_col=4):

        min_ = self.total_p_mass.min()
        max_ = self.total_p_mass.max()
        suggestions = np.linspace(min_, max_ / 2, n_suggestion)

        n_rows = math.ceil(n_suggestion / n_col)

        fig, ax = plt.subplots(n_rows, n_col, figsize=[5 * n_col, 5 * n_rows])
        if n_rows == 1:
            ax = ax.reshape(1, -1)

        row = 0
        col = 0
        for i in range(n_suggestion):

            ax_ = ax[row, col]

            col += 1
            if col == n_col:
                col = 0
                row += 1

            idx = self.total_p_mass > suggestions[i]

            ax_.scatter(self.embedding[:, 0], self.embedding[:, 1], c="lightgray", s=s)
            ax_.scatter(
                self.flow_grid[idx, 0],
                self.flow_grid[idx, 1],
                c="black",
                s=s,
            )
            ax_.set_title(f"min_mass: {suggestions[i]: .2g}")
            ax_.axis("off")

    def calculate_mass_filter(
        self,
        embedding_names,
        min_mass=itertools.repeat(0.01),
        plot=False,
    ):
        for dataset_num in range(len(self.perturbed_X)):
            for i, embedding_name in enumerate(embedding_names):
                self.embeddings[dataset_num][embedding_name].min_mass = min_mass[i]
                self.embeddings[dataset_num][embedding_name].mass_filter = (
                    self.embeddings[dataset_num][embedding_name].total_p_mass
                    < min_mass[i]
                )

                if plot:
                    fig, ax = plt.subplots(figsize=[5, 5])

                    # ax_.scatter(gridpoints_coordinates[mass_filter, 0], gridpoints_coordinates[mass_filter, 1], s=0)
                    ax.scatter(
                        self.embeddings[dataset_num][embedding_name].embedding[:, 0],
                        self.embeddings[dataset_num][embedding_name].embedding[:, 1],
                        c="lightgray",
                        s=10,
                    )
                    ax.scatter(
                        self.embeddings[dataset_num][embedding_name].flow_grid[
                            ~self.embeddings[dataset_num][embedding_name].mass_filter,
                            0,
                        ],
                        self.embeddings[dataset_num][embedding_name].flow_grid[
                            ~self.embeddings[dataset_num][embedding_name].mass_filter,
                            1,
                        ],
                        c="black",
                        s=0.5,
                    )
                    ax.set_title("Grid points selected")
                    ax.axis("off")

    def plot_cluster_whole(
        self,
        embedding_name="",
        cluster_name="",
        dataset_num=0,
        ax=None,
        s=CONFIG["s_scatter"],
        args=CONFIG["default_args"],
    ):

        if ax is None:
            ax = plt

        if not hasattr(self, "colorandum"):
            self.colorandum = []

        # update color information
        print(dataset_num)
        col_dict = _get_clustercolor_from_anndata(
            adata=self.perturbed_X[dataset_num],
            cluster_name=cluster_name,
            return_as="dict",
        )
        self.colorandum.append(
            np.array(
                [col_dict[i] for i in self.perturbed_X[dataset_num].obs[cluster_name]],
            ),
        )

        c = self.colorandum[dataset_num]
        ax.scatter(
            self.embeddings[dataset_num][embedding_name].embedding[:, 0],
            self.embeddings[dataset_num][embedding_name].embedding[:, 1],
            c=c,
            s=s,
            **args,
        )
        ax.axis("off")

    def plot_simulation_flow_on_grid(
        self,
        embedding_name,
        dataset_num,
        ax=None,
        scale=CONFIG["scale_simulation"],
        show_background=True,
        s=CONFIG["s_scatter"],
        args=CONFIG["default_args_quiver"],
    ):
        self._plot_simulation_flow_on_grid(
            embedding_name=embedding_name,
            dataset_num=dataset_num,
            ax=ax,
            scale=scale,
            show_background=show_background,
            s=s,
            data_random=False,
            args=args,
        )

    def plot_simulation_flow_random_on_grid(
        self,
        embedding_name,
        dataset_num,
        ax=None,
        scale=CONFIG["scale_simulation"],
        show_background=True,
        s=CONFIG["s_scatter"],
        args=CONFIG["default_args_quiver"],
    ):
        self._plot_simulation_flow_on_grid(
            embedding_name=embedding_name,
            dataset_num=dataset_num,
            ax=ax,
            scale=scale,
            show_background=show_background,
            s=s,
            data_random=True,
            args=args,
        )

    def _plot_simulation_flow_on_grid(
        self,
        embedding_name="",
        dataset_num=0,
        ax=None,
        scale=CONFIG["scale_simulation"],
        show_background=True,
        s=CONFIG["s_scatter"],
        data_random=False,
        args=CONFIG["default_args_quiver"],
    ):

        if ax is None:
            ax = plt

        if show_background:
            plot_background(
                self=self,
                embedding_name=embedding_name,
                dataset_num=dataset_num,
                ax=ax,
                s=s,
                args=CONFIG["default_args"],
            )
        else:
            plot_cluster_cells_use(
                self=self,
                embedding_name=embedding_name,
                dataset_num=dataset_num,
                ax=ax,
                s=0,
                color=None,
                show_background=False,
                args={},
            )

        # mass filter selection
        if hasattr(
            self.embeddings[dataset_num][embedding_name],
            "mass_filter_simulation",
        ):
            mass_filter = self.embeddings[dataset_num][
                embedding_name
            ].mass_filter_simulation
        elif hasattr(self.embeddings[dataset_num][embedding_name], "mass_filter"):
            mass_filter = self.embeddings[dataset_num][embedding_name].mass_filter

        # Gridpoint cordinate selection
        if hasattr(
            self.embeddings[dataset_num][embedding_name],
            "gridpoints_coordinates",
        ):
            gridpoints_coordinates = self.embeddings[dataset_num][
                embedding_name
            ].gridpoints_coordinates
        elif hasattr(self.embeddings[dataset_num][embedding_name], "mass_filter"):
            gridpoints_coordinates = self.embeddings[dataset_num][
                embedding_name
            ].flow_grid

        # Arrow selection
        if data_random:
            flow = self.embeddings[dataset_num][embedding_name].flow_rndm
        else:
            flow = self.embeddings[dataset_num][embedding_name].flow

        ax.quiver(
            gridpoints_coordinates[~mass_filter, 0],
            gridpoints_coordinates[~mass_filter, 1],
            flow[~mass_filter, 0],
            flow[~mass_filter, 1],  # zorder=20000,
            scale=scale,
            **args,
        )

        ax.axis("off")

    #############
    def estimate_transition_probs(
        self,
        embedding_names: List[str],
        tf_name: str,
        n_neighbors: List[int] = [None, None],
        knn_random: bool = True,
        sampled_fraction: float = 0.3,
        sampling_probs: Tuple[float, float] = (0.5, 0.1),
        n_jobs: int = 4,
        threads: int = None,
        calculate_randomized: bool = True,
        random_seed: int = 15071990,
        cell_idx_use=None,
    ) -> None:
        for embedding, n_neigh in zip(embedding_names, n_neighbors):
            self.estimate_transition_prob(
                embedding_name=embedding,
                tf_name=tf_name,
                n_neighbors=n_neigh,
                knn_random=knn_random,
                sampled_fraction=sampled_fraction,
                sampling_probs=sampling_probs,
                n_jobs=n_jobs,
                threads=threads,
                calculate_randomized=calculate_randomized,
                random_seed=random_seed,
                cell_idx_use=cell_idx_use,
            )

    def estimate_transition_prob(
        self,
        embedding_name: str,
        tf_name: str,
        n_neighbors: int = None,
        knn_random: bool = True,
        sampled_fraction: float = 0.3,
        sampling_probs: Tuple[float, float] = (0.5, 0.1),
        n_jobs: int = 4,
        threads: int = None,
        calculate_randomized: bool = True,
        random_seed: int = 15071990,
        cell_idx_use=None,
    ) -> None:
        """Use correlation to estimate transition probabilities for every cells
        to its embedding neighborhood.

        Arguments
        ---------
        embed: str, default="ts"
            The name of the attribute containing the embedding. It will be retrieved as getattr(self, embed)
        transform: str, default="sqrt"
            The transformation that is applies on the high dimensional space.
            If None the raw data will be used

        n_sight: int, default=None (also n_neighbors)
            The number of neighbors to take into account when performing the projection
        knn_random: bool, default=True
            whether to random sample the neighborhoods to speedup calculation
        sampling_probs: Tuple, default=(0.5, 1)
        max_dist_embed: float, default=None
            CURRENTLY NOT USED
            The maximum distance allowed
            If None it will be set to 0.25 * average_distance_two_points_taken_at_random
        n_jobs: int, default=4
            number of jobs to calculate knn
            this only applies to the knn search, for the more time consuming correlation computation see threads
        threads: int, default=None
            The threads will be used for the actual correlation computation by default half of the total.
        calculate_randomized: bool, default=True
            Calculate the transition probabilities with randomized residuals.
            This can be plotted downstream as a negative control and can be used to adjust the visualization scale of the velocity field.
        random_seed: int, default=15071990
            Random seed to make knn_random mode reproducible

        Returns
        -------

        """

        numba_random_seed(random_seed)

        for dataset_num in range(len(self.perturbed_X)):

            X = _obsm_to_matrix(
                self.perturbed_X[dataset_num],
                f"normalized_X_{tf_name}_dropout",
            )  # [:, :ndims]
            delta_X = X - _obsm_to_matrix(self.perturbed_X[dataset_num], "normalized_X")
            embedding = self.perturbed_X[dataset_num].obsm[embedding_name]
            self.embeddings[dataset_num][embedding_name].embedding = embedding

            if n_neighbors is None:
                n_neighbors = int(self.perturbed_X[dataset_num].shape[0] / 5)

            if knn_random:
                np.random.seed(random_seed)
                self.corr_calc = "knn_random"

                if calculate_randomized:
                    delta_X_rndm = np.copy(delta_X)
                    permute_rows_nsign(delta_X_rndm)

                logging.debug("Calculate KNN in the embedding space")

                if cell_idx_use is None:
                    nn = NearestNeighbors(n_neighbors=n_neighbors + 1, n_jobs=n_jobs)
                    nn.fit(embedding)  # NOTE should support knn in high dimensions
                    self.embeddings[dataset_num][embedding_name].embedding_knn = (
                        nn.kneighbors_graph(
                            mode="connectivity",
                        )
                    )

                else:
                    self.embeddings[dataset_num][embedding_name].embedding_knn = (
                        calculate_embedding_knn_with_cell_idx(
                            embedding_original=self.embeddings[dataset_num][
                                embedding_name
                            ].embedding,
                            cell_idx_use=cell_idx_use,
                            n_neighbors=n_neighbors,
                            n_jobs=n_jobs,
                        )
                    )

                # Pick random neighbours and prune the rest
                neigh_ixs = self.embeddings[dataset_num][
                    embedding_name
                ].embedding_knn.indices.reshape(
                    (-1, n_neighbors + 1),
                )
                p = np.linspace(
                    sampling_probs[0],
                    sampling_probs[1],
                    neigh_ixs.shape[1],
                )
                p = p / p.sum()

                # There was a problem of API consistency because the random.choice can pick the diagonal value (or not)
                # resulting self.corrcoeff with different number of nonzero entry per row.
                # Not updated yet not to break previous analyses
                # Fix is substituting below `neigh_ixs.shape[1]` with `np.arange(1,neigh_ixs.shape[1]-1)`
                # I change it here since I am doing some breaking changes
                sampling_ixs = np.stack(
                    [
                        np.random.choice(
                            neigh_ixs.shape[1],
                            size=(int(sampled_fraction * (n_neighbors + 1)),),
                            replace=False,
                            p=p,
                        )
                        for i in range(neigh_ixs.shape[0])
                    ],
                    0,
                )
                self.embeddings[dataset_num][embedding_name].sampling_ixs = sampling_ixs
                neigh_ixs = neigh_ixs[
                    np.arange(neigh_ixs.shape[0])[:, None],
                    sampling_ixs,
                ]
                nonzero = neigh_ixs.shape[0] * neigh_ixs.shape[1]
                self.embeddings[dataset_num][embedding_name].embedding_knn = (
                    sparse.csr_matrix(
                        (
                            np.ones(nonzero),
                            neigh_ixs.ravel(),
                            np.arange(0, nonzero + 1, neigh_ixs.shape[1]),
                        ),
                        shape=(
                            neigh_ixs.shape[0],
                            neigh_ixs.shape[0],
                        ),
                    )
                )

                logging.debug(f"Correlation Calculation '{self.corr_calc}'")

                ###
                ###
                self.embeddings[dataset_num][embedding_name].corrcoef = (
                    colDeltaCorpartial(
                        X,
                        delta_X,
                        neigh_ixs,
                        threads=threads,
                    )
                )
                if calculate_randomized:
                    logging.debug(f"Correlation Calculation for negative control")
                    self.embeddings[dataset_num][embedding_name].corrcoef_random = (
                        colDeltaCorpartial(
                            X,
                            delta_X_rndm,
                            neigh_ixs,
                            threads=threads,
                        )
                    )
                ######

                if np.any(
                    np.isnan(self.embeddings[dataset_num][embedding_name].corrcoef),
                ):
                    self.embeddings[dataset_num][embedding_name].corrcoef[
                        np.isnan(self.embeddings[dataset_num][embedding_name].corrcoef)
                    ] = 1
                    logging.debug(
                        "Nans encountered in corrcoef and corrected to 1s. If not identical cells were present it is probably a small isolated cluster converging after imputation.",
                    )
                if calculate_randomized:
                    np.fill_diagonal(
                        self.embeddings[dataset_num][embedding_name].corrcoef_random,
                        0,
                    )
                    if np.any(
                        np.isnan(
                            self.embeddings[dataset_num][
                                embedding_name
                            ].corrcoef_random,
                        ),
                    ):
                        self.embeddings[dataset_num][embedding_name].corrcoef_random[
                            np.isnan(
                                self.embeddings[dataset_num][
                                    embedding_name
                                ].corrcoef_random,
                            )
                        ] = 1
                        logging.debug(
                            "Nans encountered in corrcoef_random and corrected to 1s. If not identical cells were present it is probably a small isolated cluster converging after imputation.",
                        )
                logging.debug(f"Done Correlation Calculation")
            else:
                self.corr_calc = "full"

                if calculate_randomized:
                    delta_X_rndm = np.copy(delta_X)
                    permute_rows_nsign(delta_X_rndm)

                logging.debug("Calculate KNN in the embedding space")
                nn = NearestNeighbors(n_neighbors=n_neighbors + 1, n_jobs=n_jobs)
                nn.fit(embedding)
                self.embeddings[dataset_num][embedding_name].embedding_knn = (
                    nn.kneighbors_graph(
                        mode="connectivity",
                    )
                )

                logging.debug("Correlation Calculation 'full'")
                #####
                self.embeddings[dataset_num][embedding_name].corrcoef = colDeltaCor(
                    X,
                    delta_X,
                    threads=threads,
                )
                if calculate_randomized:
                    logging.debug(f"Correlation Calculation for negative control")
                    self.embeddings[dataset_num][embedding_name].corrcoef_random = (
                        colDeltaCor(
                            X,
                            delta_X_rndm,
                            threads=threads,
                        )
                    )

                #####
                np.fill_diagonal(
                    self.embeddings[dataset_num][embedding_name].corrcoef,
                    0,
                )
                if calculate_randomized:
                    np.fill_diagonal(
                        self.embeddings[dataset_num][embedding_name].corrcoef_random,
                        0,
                    )

    ############
    def calculate_embedding_shifts(
        self,
        embedding_names: List[str],
        sigma_corr: float = 0.05,
    ) -> None:
        for embedding_name in embedding_names:
            self.calculate_embedding_shift(
                embedding_name=embedding_name,
                sigma_corr=sigma_corr,
            )

    def calculate_embedding_shift(
        self,
        embedding_name: str,
        sigma_corr: float = 0.05,
    ) -> None:
        """Use the transition probability to project the velocity direction on
        the embedding.

        Arguments
        ---------
        sigma_corr: float, default=0.05
            the kernel scaling

        Returns
        -------
        Nothing but it creates the following attributes:
        transition_prob: np.ndarray
            the transition probability calculated using the exponential kernel on the correlation coefficient
        delta_embedding: np.ndarray
            The resulting vector

        """
        # Kernel evaluation
        logging.debug("Calculate transition probability")

        # NOTE maybe sparse matrix here are slower than dense
        # NOTE if knn_random this could be made much faster either using sparse matrix or neigh_ixs

        for dataset_num in range(len(self.perturbed_X)):

            self.embeddings[dataset_num][embedding_name].transition_prob = (
                np.exp(
                    self.embeddings[dataset_num][embedding_name].corrcoef / sigma_corr,
                )
                * self.embeddings[dataset_num][embedding_name].embedding_knn.todense().A
            )  # naive
            self.embeddings[dataset_num][
                embedding_name
            ].transition_prob /= self.embeddings[dataset_num][
                embedding_name
            ].transition_prob.sum(
                1,
            )[
                :,
                None,
            ]
            if hasattr(self.embeddings[dataset_num][embedding_name], "corrcoef_random"):
                logging.debug("Calculate transition probability for negative control")
                self.embeddings[dataset_num][embedding_name].transition_prob_random = (
                    np.exp(
                        self.embeddings[dataset_num][embedding_name].corrcoef_random
                        / sigma_corr,
                    )
                    * self.embeddings[dataset_num][embedding_name]
                    .embedding_knn.todense()
                    .A
                )  # naive
                self.embeddings[dataset_num][
                    embedding_name
                ].transition_prob_random /= self.embeddings[dataset_num][
                    embedding_name
                ].transition_prob_random.sum(
                    1,
                )[
                    :,
                    None,
                ]

            unitary_vectors = (
                self.embeddings[dataset_num][embedding_name].embedding.T[:, None, :]
                - self.embeddings[dataset_num][embedding_name].embedding.T[:, :, None]
            )  # shape (2,ncells,ncells)
            with np.errstate(divide="ignore", invalid="ignore"):
                unitary_vectors /= np.linalg.norm(
                    unitary_vectors,
                    ord=2,
                    axis=0,
                )  # divide by L2
                np.fill_diagonal(unitary_vectors[0, ...], 0)  # fix nans
                np.fill_diagonal(unitary_vectors[1, ...], 0)

            self.embeddings[dataset_num][embedding_name].delta_embedding = (
                self.embeddings[dataset_num][embedding_name].transition_prob
                * unitary_vectors
            ).sum(2)
            self.embeddings[dataset_num][embedding_name].delta_embedding -= (
                self.embeddings[dataset_num][embedding_name].embedding_knn.todense().A
                * unitary_vectors
            ).sum(2) / self.embeddings[dataset_num][embedding_name].embedding_knn.sum(
                1,
            ).A.T
            self.embeddings[dataset_num][embedding_name].delta_embedding = (
                self.embeddings[dataset_num][embedding_name].delta_embedding.T
            )

            if hasattr(self.embeddings[dataset_num][embedding_name], "corrcoef_random"):
                self.embeddings[dataset_num][embedding_name].delta_embedding_random = (
                    self.embeddings[dataset_num][embedding_name].transition_prob_random
                    * unitary_vectors
                ).sum(2)
                self.embeddings[dataset_num][embedding_name].delta_embedding_random -= (
                    self.embeddings[dataset_num][embedding_name]
                    .embedding_knn.todense()
                    .A
                    * unitary_vectors
                ).sum(2) / self.embeddings[dataset_num][
                    embedding_name
                ].embedding_knn.sum(
                    1,
                ).A.T
                self.embeddings[dataset_num][embedding_name].delta_embedding_random = (
                    self.embeddings[dataset_num][
                        embedding_name
                    ].delta_embedding_random.T
                )

    def calculate_grid_arrows(
        self,
        embedding,
        smooth: float = 0.5,
        steps: Tuple = (40, 40),
        n_neighbors: int = 100,
        n_jobs: int = 4,
        xylim: Tuple = ((None, None), (None, None)),
    ) -> None:
        """Calculate the velocity using a points on a regular grid and a
        gaussian kernel.

        Note: the function should work also for n-dimensional grid

        Arguments
        ---------
        embed: str, default=embedding
            The name of the attribute containing the embedding. It will be retrieved as getattr(self, embed)
            The difference vector is getattr(self, 'delta' + '_' + embed)
        smooth: float, smooth=0.5
            Higher value correspond to taking in consideration further points
            the standard deviation of the gaussian kernel is smooth * stepsize
        steps: tuple, default
            the number of steps in the grid for each axis
        n_neighbors:
            number of neighbors to use in the calculation, bigger number should not change too much the results..
            ...as soon as smooth is small
            Higher value correspond to slower execution time
        n_jobs:
            number of processes for parallel computing
        xymin:
            ((xmin, xmax), (ymin, ymax))

        Returns
        -------
        Nothing but it sets the attributes:
        flow_embedding: np.ndarray
            the coordinates of the embedding
        flow_grid: np.ndarray
            the gridpoints
        flow: np.ndarray
            vector field coordinates
        flow_magnitude: np.ndarray
            magnitude of each vector on the grid
        total_p_mass: np.ndarray
            density at each point of the grid

        """
        for dataset_num in range(len(self.perturbed_X)):
            delta_embedding = getattr(
                self.embeddings[dataset_num][embedding],
                f"delta_embedding",
            )
            emb = self.embeddings[dataset_num][embedding].embedding

            if hasattr(self.embeddings[dataset_num][embedding], "corrcoef_random"):
                delta_embedding_random = getattr(
                    self.embeddings[dataset_num][embedding],
                    f"delta_embedding_random",
                )

            # Prepare the grid
            grs = []
            for dim_i in range(emb.shape[1]):
                m, M = np.min(emb[:, dim_i]), np.max(emb[:, dim_i])

                if xylim[dim_i][0] is not None:
                    m = xylim[dim_i][0]
                if xylim[dim_i][1] is not None:
                    M = xylim[dim_i][1]

                m = m - 0.025 * np.abs(M - m)
                M = M + 0.025 * np.abs(M - m)
                gr = np.linspace(m, M, steps[dim_i])
                grs.append(gr)

            meshes_tuple = np.meshgrid(*grs)
            gridpoints_coordinates = np.vstack([i.flat for i in meshes_tuple]).T

            nn = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=n_jobs)
            nn.fit(emb)
            dists, neighs = nn.kneighbors(gridpoints_coordinates)

            std = np.mean([(g[1] - g[0]) for g in grs])
            # isotropic gaussian kernel
            gaussian_w = normal.pdf(loc=0, scale=smooth * std, x=dists)
            self.embeddings[dataset_num][embedding].total_p_mass = gaussian_w.sum(1)

            UZ = (delta_embedding[neighs] * gaussian_w[:, :, None]).sum(1) / np.maximum(
                1,
                self.embeddings[dataset_num][embedding].total_p_mass,
            )[
                :,
                None,
            ]  # weighed average
            magnitude = np.linalg.norm(UZ, axis=1)
            # Assign attributes
            self.embeddings[dataset_num][embedding].flow_embedding = emb
            self.embeddings[dataset_num][embedding].flow_grid = gridpoints_coordinates
            self.embeddings[dataset_num][embedding].flow = UZ
            self.embeddings[dataset_num][embedding].flow_norm = UZ / np.percentile(
                magnitude,
                99.5,
            )
            self.embeddings[dataset_num][embedding].flow_norm_magnitude = (
                np.linalg.norm(
                    self.embeddings[dataset_num][embedding].flow_norm,
                    axis=1,
                )
            )

            if hasattr(self.embeddings[dataset_num][embedding], "corrcoef_random"):
                UZ_rndm = (delta_embedding_random[neighs] * gaussian_w[:, :, None]).sum(
                    1,
                ) / np.maximum(1, self.embeddings[dataset_num][embedding].total_p_mass)[
                    :,
                    None,
                ]  # weighed average
                magnitude_rndm = np.linalg.norm(UZ, axis=1)
                # Assign attributes
                self.embeddings[dataset_num][embedding].flow_rndm = UZ_rndm
                self.embeddings[dataset_num][embedding].flow_norm_rndm = (
                    UZ_rndm
                    / np.percentile(
                        magnitude_rndm,
                        99.5,
                    )
                )
                self.embeddings[dataset_num][embedding].flow_norm_magnitude_rndm = (
                    np.linalg.norm(
                        self.embeddings[dataset_num][embedding].flow_norm_rndm,
                        axis=1,
                    )
                )


def scatter_viz(x: np.ndarray, y: np.ndarray, *args: Any, **kwargs: Any) -> Any:
    """A wrapper of scatter plot that guarantees that every point is visible in
    a very crowded scatterplot.

    Args
    ----
    x: np.ndarray
        x axis coordinates
    y: np.ndarray
        y axis coordinates
    args and kwargs:
        positional and keyword arguments as in matplotplib.pyplot.scatter

    Returns
    -------
    Plots the graph and returns the axes object

    """
    ix_x_sort = np.argsort(x, kind="mergesort")
    ix_yx_sort = np.argsort(y[ix_x_sort], kind="mergesort")
    args_new = []
    kwargs_new = {}
    for arg in args:
        if type(arg) is np.ndarray:
            args_new.append(arg[ix_x_sort][ix_yx_sort])
        else:
            args_new.append(arg)
    for karg, varg in kwargs.items():
        if type(varg) is np.ndarray:
            kwargs_new[karg] = varg[ix_x_sort][ix_yx_sort]
        else:
            kwargs_new[karg] = varg
    ax = plt.scatter(
        x[ix_x_sort][ix_yx_sort],
        y[ix_x_sort][ix_yx_sort],
        *args_new,
        **kwargs_new,
    )
    return ax


@jit(nopython=True)
def numba_random_seed(value: int) -> None:
    """Same as np.random.seed but for numba."""
    np.random.seed(value)


@jit(nopython=True)
def permute_rows_nsign(A: np.ndarray) -> None:
    """Permute in place the entries and randomly switch the sign for each row of
    a matrix independently."""
    plmi = np.array([+1, -1])
    for i in range(A.shape[0]):
        np.random.shuffle(A[i, :])
        A[i, :] = A[i, :] * np.random.choice(plmi, size=A.shape[1])


def gaussian_kernel(X: np.ndarray, mu: float = 0, sigma: float = 1) -> np.ndarray:
    """Compute gaussian kernel."""
    return np.exp(-((X - mu) ** 2) / (2 * sigma**2)) / np.sqrt(2 * np.pi * sigma**2)
