from typing import Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
from matplotlib.colors import LinearSegmentedColormap

from .embedding import Embedding
from .eykthyr import Eykthyr


def paga_spatial_simulation(
    eykthyr: Eykthyr,
    TFs: Sequence[str],
    cluster_name: str,
):
    """Run spatial perturbation simulations for a list of TFs and show flow plots.

    Convenience wrapper around :func:`umap_spatial_simulations` that uses the
    default spatial and graph embeddings.  Call :func:`prep_paga` first to
    compute PAGA connectivity and the force-directed graph layout.

    Parameters:
        eykthyr (Eykthyr): A fully-trained Eykthyr instance with
            ``perturbed_X`` populated by :meth:`~eykthyr.Eykthyr.run_all_perturbations`.
        TFs (Sequence[str]): TF names to simulate.  Each name must appear as a
            column in ``eykthyr.TF[i].var_names``.
        cluster_name (str): Key in ``perturbed_X[i].obs`` that stores cell-type /
            cluster labels (used for colouring the background scatter plot).
    """
    umap_spatial_simulations(
        eykthyr,
        TFs,
        cluster_name=cluster_name,
    )


def prep_paga(
    eykthyr: Eykthyr,
    groups: str,
):
    """Compute PAGA graph connectivity and force-directed layout for each dataset.

    Runs the standard Scanpy PAGA/draw_graph pipeline on each ``perturbed_X``
    AnnData object so that the force-directed graph embedding
    (``X_draw_graph_fr``) is available for downstream simulation plotting.
    Results are stored in-place on each AnnData.

    Parameters:
        eykthyr (Eykthyr): Eykthyr instance whose ``perturbed_X`` list will be
            modified in-place.
        groups (str): Key in ``perturbed_X[i].obs`` used to define PAGA groups
            (typically ``'original_leiden'`` or a cell-type annotation column).
    """
    for d in eykthyr.perturbed_X:
        sc.pp.neighbors(d, use_rep="normalized_X")
        sc.tl.umap(d)
        sc.tl.paga(d, groups=groups)
        sc.tl.draw_graph(d, init_pos="X_umap", random_state=123, layout="fr")
        sc.pl.draw_graph(d, color=groups, legend_loc="on data")


def umap_spatial_simulations(
    eykthyr,
    TFs,
    n_grid=40,
    min_masses=[0.27, 0.007],
    scales=[30, 1.2],
    embeddings=["spatial", "X_draw_graph_fr"],
    n_neighbors=[20, 25],
    cluster_name="original_leiden",
    show_plots=[True, True],
):
    """Run :func:`umap_spatial_simulation` for each TF in *TFs*.

    Iterates over the list of TF names and calls :func:`umap_spatial_simulation`
    for each one, sharing the same grid / embedding settings.

    Parameters:
        eykthyr: Eykthyr instance with ``perturbed_X`` populated.
        TFs (list[str]): List of TF names to simulate.
        n_grid (int): Number of grid points along each axis for flow-field
            interpolation. Default ``40``.
        min_masses (list[float]): Per-embedding minimum probability-mass threshold
            for masking low-density grid points. Default ``[0.27, 0.007]``.
        scales (list[float]): Quiver arrow scale per embedding. Default
            ``[30, 1.2]``.
        embeddings (list[str]): ``obsm`` keys to visualize. Default
            ``['spatial', 'X_draw_graph_fr']``.
        n_neighbors (list[int]): Number of neighbors for KDE mass estimation per
            embedding. Default ``[20, 25]``.
        cluster_name (str): ``obs`` key for cluster labels used in background
            scatter. Default ``'original_leiden'``.
        show_plots (list[bool]): Whether to display the side-by-side observed /
            randomized flow panels per embedding. Default ``[True, True]``.
    """
    for rna in eykthyr.RNA:
        rna.X = rna.X.astype("double")

    for TF in TFs:
        umap_spatial_simulation(
            eykthyr,
            TF,
            n_grid=n_grid,
            min_masses=min_masses,
            scales=scales,
            embeddings=embeddings,
            n_neighbors=n_neighbors,
            cluster_name=cluster_name,
            show_plots=show_plots,
        )


def umap_spatial_simulation(
    eykthyr,
    TF,
    n_grid=40,
    min_masses=[0.27, 0.007],
    scales=[30, 1.2],
    embeddings=["spatial", "X_draw_graph_fr"],
    n_neighbors=[20, 25],
    cluster_name="original_leiden",
    show_plots=[True, True],
):
    """Simulate the effect of knocking out a single TF and plot flow fields.

    Performs the full simulation pipeline for one TF:

    1. Estimates cell-to-cell transition probabilities using perturbed expression.
    2. Computes embedding-space velocity vectors (delta embeddings).
    3. Interpolates the velocity field onto a regular grid.
    4. Optionally plots observed and randomized flow fields side-by-side for each
       requested embedding, plus a combined cluster + flow overlay figure.

    Parameters:
        eykthyr: Eykthyr instance with ``perturbed_X`` populated.
        TF (str): Name of the TF to simulate (knockout).
        n_grid (int): Grid resolution for flow-field interpolation. Default ``40``.
        min_masses (list[float]): Per-embedding mass threshold below which grid
            points are masked. Default ``[0.27, 0.007]``.
        scales (list[float]): Quiver arrow scales per embedding. Default
            ``[30, 1.2]``.
        embeddings (list[str]): ``obsm`` keys to visualize. Default
            ``['spatial', 'X_draw_graph_fr']``.
        n_neighbors (list[int]): Neighbors for KDE mass estimation per embedding.
            Default ``[20, 25]``.
        cluster_name (str): ``obs`` key for cluster labels. Default
            ``'original_leiden'``.
        show_plots (list[bool]): Whether to display the side-by-side panels per
            embedding. Default ``[True, True]``.
    """

    eykthyr.embeddings = []
    for dataset_num in range(len(eykthyr.perturbed_X)):
        eykthyr.embeddings.append({})
    for embedding in embeddings:
        for dataset_num in range(len(eykthyr.perturbed_X)):
            eykthyr.embeddings[dataset_num][embedding] = Embedding()
            eykthyr.perturbed_X[dataset_num].obsm[embedding] = (
                eykthyr.perturbed_X[dataset_num].obsm[embedding].astype(float)
            )
    eykthyr.estimate_transition_probs(
        embedding_names=embeddings,
        tf_name=TF,
        n_neighbors=n_neighbors,
        sampled_fraction=1,
    )
    eykthyr.calculate_embedding_shifts(embedding_names=embeddings, sigma_corr=0.05)
    eykthyr.calculate_p_mass(
        embeddings,
        smooth=0.8,
        n_grid=n_grid,
        n_neighbors=n_neighbors,
    )
    eykthyr.calculate_mass_filter(embeddings, min_mass=min_masses, plot=False)
    # eykthyr.suggest_mass_thresholds(n_suggestion=12)
    for embedding, show_plot, scale in zip(embeddings, show_plots, scales):
        for dataset_num in range(len(eykthyr.perturbed_X)):
            if show_plot == True:
                fig, ax = plt.subplots(1, 2, figsize=[13, 6])

                eykthyr.plot_simulation_flow_on_grid(
                    embedding,
                    dataset_num,
                    scale=scale,
                    ax=ax[0],
                )
                ax[0].set_title(f"Simulated cell identity shift vector: {TF} KO")

                # Show quiver plot that was calculated with randomized graph.
                eykthyr.plot_simulation_flow_random_on_grid(
                    embedding,
                    dataset_num,
                    scale=scale,
                    ax=ax[1],
                )
                ax[1].set_title(f"Randomized simulation vector")

                plt.show()
    width = 8 * len(embeddings)
    height = 6 * len(eykthyr.perturbed_X)
    fig2, ax2 = plt.subplots(
        len(eykthyr.perturbed_X),
        len(embeddings),
        figsize=[width, height],
    )
    ax2 = np.atleast_2d(ax2)
    for dataset_num in range(len(eykthyr.perturbed_X)):
        for i, embedding in enumerate(embeddings):

            eykthyr.plot_cluster_whole(
                embedding,
                cluster_name,
                dataset_num,
                ax=ax2[dataset_num][i],
                s=5,
            )
            eykthyr.plot_simulation_flow_on_grid(
                embedding,
                dataset_num,
                scale=scales[i],
                ax=ax2[dataset_num][i],
                show_background=False,
            )
            ax2[dataset_num][i].set_title(
                f"Simulated cell identity shift {embedding}: {TF} KO",
            )

    plt.show()


def development_simulation(
    eykthyr,
    TFs,
    n_grid=40,
    min_masses=[1, 0.003],
    # scales=[50,1.1],
    scales=[30, 0.8],
    cluster_name="original_leiden",
    embeddings=["spatial", "X_draw_graph_fr"],
    n_neighbors=[20, 25],
    show_plots=[True, True],
    vm=2,
    spotsize=40,
    arrow_args={},
    save_figs=False,
    fig_prefix=None,
):
    """Compute developmental trajectory scores for each TF and plot inner products.

    For every TF in *TFs*, this function:

    1. Runs the full spatial perturbation simulation via
       :func:`umap_spatial_simulation`.
    2. Fits a pseudotime gradient (``ventricle_distance`` must be precomputed in
       ``perturbed_X[0].obs``) onto the spatial grid using polynomial regression.
    3. Calculates the inner product (perturbation strength, PS) between each
       cell's simulated flow vector and the pseudotime gradient direction.
    4. Plots: (a) the reference differentiation flow, (b) PS on the spatial grid
       (observed vs. null), (c) PS overlaid with simulation arrows.

    Parameters:
        eykthyr: Eykthyr instance with ``perturbed_X`` populated and a
            ``ventricle_distance`` column in ``perturbed_X[0].obs``.
        TFs (list[str]): TF names to score.
        n_grid (int): Grid resolution. Default ``40``.
        min_masses (list[float]): Per-embedding mass thresholds. Default
            ``[1, 0.003]``.
        scales (list[float]): Quiver arrow scales. Default ``[30, 0.8]``.
        cluster_name (str): ``obs`` key for cluster labels. Default
            ``'original_leiden'``.
        embeddings (list[str]): ``obsm`` keys to use. Default
            ``['spatial', 'X_draw_graph_fr']``.
        n_neighbors (list[int]): Neighbors for KDE. Default ``[20, 25]``.
        show_plots (bool or list[bool]): Whether to display plots. A single
            ``False`` suppresses all plots including the inner product figures.
            A list is passed per-embedding to :func:`umap_spatial_simulation`,
            with ``any(show_plots)`` controlling the inner product figures.
            Default ``[True, True]``.
        vm (float): Colormap saturation limit for the PS grid plot. Default ``2``.
        spotsize (int): Marker size for scatter overlays. Default ``40``.
        arrow_args (dict): Extra keyword arguments forwarded to the quiver plot.
        save_figs (bool): If ``True``, save SVG figures to the working directory.
        fig_prefix (str or None): Prefix for saved SVG filenames. If ``None``,
            files are named ``differentiation.svg``, ``{TF}_inner_product_abs.svg``,
            and ``{TF}_inner_product.svg``. If set, files are named
            ``{fig_prefix}_differentiation.svg``, etc.

    Returns:
        list[tuple[str, float]]: List of ``(TF_name, perturbation_strength_sum)``
        tuples, one per TF.  Positive values indicate the TF promotes
        differentiation towards the reference trajectory.
    """

    from .development import Pseudotime_module
    from .pseudotime import Gradient_calculator

    if isinstance(show_plots, bool):
        _show = show_plots
        show_plots_list = [show_plots] * len(embeddings)
    else:
        _show = any(show_plots)
        show_plots_list = show_plots

    prefix = f"{fig_prefix}_" if fig_prefix else ""
    embedding_name = embeddings[0]
    ips = []

    for TF in TFs:
        umap_spatial_simulation(
            eykthyr,
            TF,
            n_grid=n_grid,
            min_masses=min_masses,
            scales=scales,
            embeddings=embeddings,
            n_neighbors=n_neighbors,
            cluster_name=cluster_name,
            show_plots=show_plots_list,
        )

        n_grid_grad = n_grid
        min_mass_grad = 1
        gradient = Gradient_calculator(
            adata=eykthyr.perturbed_X[0],
            pseudotime_key="ventricle_distance",
            obsm_key=embedding_name,
        )
        gradient.calculate_p_mass(smooth=0.8, n_grid=n_grid_grad, n_neighbors=200)
        gradient.calculate_mass_filter(min_mass=min_mass_grad, plot=False)
        gradient.transfer_data_into_grid(
            args={"method": "polynomial", "n_poly": 7},
            plot=False,
        )
        gradient.calculate_gradient()

        dev = Pseudotime_module()
        # Load development flow
        dev.load_differentiation_reference_data(gradient_object=gradient)

        # Load simulation result
        dev.load_perturb_simulation_data(
            embedding_object=eykthyr.embeddings[0][embedding_name],
        )
        my_gradient = LinearSegmentedColormap.from_list(
            "my_gradient",
            (
                # Edit this gradient at https://eltos.github.io/gradient/#008837-A6DBA0-FFFFFF-A6DBA0-008837
                (0.000, (0.000, 0.533, 0.216)),
                (0.250, (0.651, 0.859, 0.627)),
                (0.500, (1.000, 1.000, 1.000)),
                (0.750, (0.651, 0.859, 0.627)),
                (1.000, (0.000, 0.533, 0.216)),
            ),
        )

        # Calculate inner product scores
        dev.calculate_inner_product()
        dev.calculate_digitized_ip(n_bins=10)
        fig, ax = plt.subplots(1, 1, figsize=[4, 6])
        dev.plot_reference_flow_on_grid(
            ax=ax,
            scale=scales[0],
            s=spotsize,
            args=arrow_args,
        )
        if save_figs:
            plt.savefig(f"{prefix}differentiation.svg", bbox_inches="tight")
        if _show:
            plt.show()
        else:
            plt.close()

        fig, ax = plt.subplots(1, 2, figsize=[9, 6])
        dev.plot_inner_product_on_grid(vm=vm, s=spotsize, ax=ax[0], cmap=my_gradient)
        ax[0].set_title(f"PS")
        dev.plot_inner_product_random_on_grid(vm=vm, s=spotsize, ax=ax[1])
        ax[1].set_title(f"PS calculated with Randomized simulation vector")
        if save_figs:
            plt.savefig(f"{prefix}{TF}_inner_product_abs.svg", bbox_inches="tight")
        if _show:
            plt.show()
        else:
            plt.close()

        fig, ax = plt.subplots(figsize=[5, 6])
        dev.plot_inner_product_on_grid(
            vm=vm,
            s=spotsize,
            ax=ax,
            show_background=False,
            cmap=my_gradient,
        )
        dev.plot_simulation_flow_on_grid(
            scale=scales[0],
            show_background=False,
            ax=ax,
            args=arrow_args,
        )
        if save_figs:
            plt.savefig(f"{prefix}{TF}_inner_product.svg", bbox_inches="tight")
        if _show:
            plt.show()
        else:
            plt.close()

        ips.append((TF, dev.inner_product[~dev.mass_filter_simulation].sum()))
    return ips
