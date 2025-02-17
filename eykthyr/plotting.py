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
    """"""
    umap_spatial_simulations(
        eykthyr,
        TFs,
        cluster_name=cluster_name,
    )


def prep_paga(
    eykthyr: Eykthyr,
    groups: str,
):
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
):

    from .development import Pseudotime_module
    from .pseudotime import Gradient_calculator

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
            show_plots=show_plots,
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

        # Calculate inner produc scores
        dev.calculate_inner_product()
        dev.calculate_digitized_ip(n_bins=10)
        fig, ax = plt.subplots(1, 1, figsize=[4, 6])
        dev.plot_reference_flow_on_grid(
            ax=ax,
            scale=scales[0],
            s=spotsize,
            args=arrow_args,
        )
        if save_figs == True:
            plt.savefig("differentiation.svg", bbox_inches="tight")

        fig, ax = plt.subplots(1, 2, figsize=[9, 6])
        dev.plot_inner_product_on_grid(vm=vm, s=spotsize, ax=ax[0], cmap=my_gradient)
        ax[0].set_title(f"PS")

        dev.plot_inner_product_random_on_grid(vm=vm, s=spotsize, ax=ax[1])
        ax[1].set_title(f"PS calculated with Randomized simulation vector")
        if save_figs == True:
            plt.savefig(f"{TF}_inner_product_abs.svg", bbox_inches="tight")
        plt.show()
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
        if save_figs == True:
            plt.savefig(f"{TF}_inner_product.svg", bbox_inches="tight")

        ips.append((TF, dev.inner_product[~dev.mass_filter_simulation].sum()))
    return ips
