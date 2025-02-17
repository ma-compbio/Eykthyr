import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from tqdm import tqdm

CONFIG = {
    "default_args": {"lw": 0.3, "rasterized": True},
    "s_scatter": 5,
    "s_grid": 20,
    "scale_simulation": 30,
    "scale_dev": 30,
    "cmap_ps": "PiYG",
    "default_args_quiver": {"linewidths": 0.25, "width": 0.004},
}


def get_predecessors(g, links):
    return links[links["target"] == g]["source"].unique()


def get_upstream_genes(target_gene, links, num_hops=3):
    upstream_genes = set()
    new_genes = [target_gene]
    for i in range(num_hops):
        curr_genes = new_genes
        new_genes = []
        for g in curr_genes:
            preds = get_predecessors(g, links)
            for pred in preds:
                if pred not in upstream_genes:
                    upstream_genes.add(pred)
                    new_genes.append(pred)
    return upstream_genes


def get_descendents(g, links):
    return links[links["source"] == g]["target"].unique()


def add_signs(links):
    links["sign"] = np.zeros(links.shape[0], dtype=int)
    links.loc[links["coef_mean"] > 0, "sign"] = 1
    links.loc[links["coef_mean"] < 0, "sign"] = -1

    return links


def get_signs(source, links, targets):
    signs = []
    coefs = []
    source_links = links[links["source"] == source]
    target_links = source_links[source_links["target"].isin(targets)]
    return list(target_links.loc[:, "sign"]), list(target_links.loc[:, "coef_mean"])


def get_downstream_genes(target_gene, links, num_hops=3, use_coefs=True):
    # This could change to also return the coefficients from the dictionary and do some kind of calculation on them.
    # Maybe just start with the sign
    downstream_genes = set()
    downstream_genes_signs = {}
    downstream_genes_signs[target_gene] = 1
    if use_coefs == True:
        downstream_genes_coefs = {}
        downstream_genes_coefs[target_gene] = math.log(1)
    new_genes = [target_gene]
    add_signs(links)
    for i in range(num_hops):
        curr_genes = new_genes
        new_genes = []
        for g in curr_genes:
            descs = get_descendents(g, links)
            signs, coefs = get_signs(g, links, descs)
            for j, desc in enumerate(descs):
                if desc not in downstream_genes:
                    downstream_genes.add(desc)
                    new_genes.append(desc)
                    if use_coefs == True:
                        """If coefs[j] != 0.0: downstream_genes_coefs[desc] =
                        downstream_genes_signs[g] + math.log(abs(coefs[j]))

                        else:
                            downstream_genes_coefs[desc] = downstream_genes_signs[g] - 800

                        """
                        downstream_genes_coefs[desc] = coefs[j]
                    downstream_genes_signs[desc] = downstream_genes_signs[g] * signs[j]
                else:
                    if (
                        downstream_genes_signs[desc]
                        != downstream_genes_signs[g] * signs[j]
                    ):
                        print(
                            f"for gene {desc}, it has a different sign coming from {g} than from previous. It has a coef of {coefs[j]}.",
                        )
    for t in downstream_genes_signs:
        downstream_genes_signs[t] = (
            downstream_genes_signs[t] * downstream_genes_coefs[t]
        )
    return downstream_genes, downstream_genes_signs


def get_metagenes_score(
    adata,
    regulators,
    m_name,
    links,
    links_dict_num="0",
    negative_signs=True,
    use_coefs=True,
):
    regdf = pd.DataFrame(
        index=regulators,
        columns=adata.uns["spicemix_genes"].astype(str),
        dtype=int,
    )
    regdf.loc[:, :] = 0
    for regnum, reg in enumerate(regulators):
        dsg, signs = get_downstream_genes(
            reg,
            links.links_dict[links_dict_num],
            num_hops=1,
            use_coefs=use_coefs,
        )
        for g in dsg:
            if g in regdf.columns:
                if negative_signs == False:
                    regdf.loc[reg, g] = abs(signs[g])
                else:
                    regdf.loc[reg, g] = signs[g]
    regdf = regdf.loc[regdf.sum(axis=1) != 0]
    metagenes = adata.uns["M"][m_name]
    mval_nonorm = np.matmul(regdf, metagenes).astype(float)
    # normalize
    mval = mval_nonorm.div(mval_nonorm.abs().sum(axis=1), axis=0).astype(float)
    return mval, mval_nonorm


def get_intersecting(ms_nonorm, metagene_scores, k=10):
    intersect_lists = []
    top_percent_lists = []
    negative_intersect_lists = []
    negative_top_percent_lists = []
    for i in range(len(metagene_scores.columns)):
        topk = ms_nonorm.loc[ms_nonorm.loc[:, i].sort_values()[-k:].index, :]
        topknorm = metagene_scores.loc[
            metagene_scores.loc[:, i].sort_values()[-k:].index,
            :,
        ]
        intersect = set(topk.index).intersection(set(topknorm.index))
        intersect_lists.append(list(intersect))
        top_percent_lists.append(list(topknorm.index)[-2:])

        negtopk = ms_nonorm.loc[ms_nonorm.loc[:, i].sort_values()[:k].index, :]
        negtopknorm = metagene_scores.loc[
            metagene_scores.loc[:, i].sort_values()[:k].index,
            :,
        ]
        negintersect = set(negtopk.index).intersection(set(negtopknorm.index))
        negative_intersect_lists.append(list(negintersect))
        negative_top_percent_lists.append(list(negtopknorm.index)[:2])
    return (
        top_percent_lists,
        intersect_lists,
        negative_top_percent_lists,
        negative_intersect_lists,
    )


from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import Ridge


def get_edges_window(ad_ex, ad_motif, target_gene, grn, ad_pop, num_hops=1):

    # subset the ad_ex and ad_motif by the neighbors
    tfs = intersect(grn[target_gene], ad_motif.var_names)
    retdf = pd.DataFrame(index=tfs)
    for i, cell in enumerate(ad_ex.obs_names):
        #         neighbors_bool = np.asarray(ad_pop.obsp['adjacency_matrix'][i,:].todense().astype(bool)).flatten()
        neighbors_bool = get_nhop_neighbors(ad_pop, i, num_hops=num_hops)
        neighbors = ad_ex.obs_names[neighbors_bool]
        if len(neighbors) == 0:
            retdf[cell] = np.zeros((len(tfs), 1))
            continue
        data = ad_motif[neighbors, tfs].to_df()
        label = ad_ex[neighbors, target_gene].to_df()
        model = BaggingRegressor(
            base_estimator=Ridge(
                alpha=1,
                solver="auto",
                random_state=123,
            ),
            n_estimators=10,
            bootstrap=True,
            max_features=0.8,
            verbose=False,
            random_state=123,
        )
        model.fit(data, label)
        ans = _get_coef_matrix(model, tfs).mean(axis=0)
        retdf[cell] = ans
    return retdf.T


def get_neighbors(ad_pop, i, nn_indices, cell_type, cluster_id, num_within, num_total):
    cell_indices = nn_indices[i]
    within_cluster_indices = [
        ind
        for ind, cell in enumerate(ad_pop.obs_names)
        if ad_pop.obs[cluster_id][cell] == cell_type
    ]
    cell_type_list = [t for t in ad_pop.obs[cluster_id].values()]

    num_total += 1  # This is done so you always return yourself plus the number of required neighbors
    neighbors = []
    num_without_added = 0
    cell_idx = 0
    while len(neighbors) < num_total:
        if cell_type_list[cell_indices[cell_idx]] != cell_type:
            if num_without_added < (num_total - num_within):
                neighbors.append(cell_indices[cell_idx])
                num_without_added += 1
        else:
            neighbors.append(cell_indices[cell_idx])
        cell_idx += 1
    # now to get the boolean mask
    neighbors_bool = [
        True if c_ind in neighbors else False for c_ind in range(len(ad_pop.obs_names))
    ]
    return neighbors_bool


def get_nearest_neighbors(ad, n_neighbors=None):
    from sklearn.neighbors import NearestNeighbors

    X = ad.obsm["spatial"]
    if n_neighbors == None:
        n_neighbors = len(ad.obs_names)
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm="ball_tree").fit(X)
    distances, indices = nbrs.kneighbors(X)
    return indices


def get_metagene_edges_window(
    ad_ex,
    ad_motif,
    target_metagene,
    ad_pop,
    num_hops=1,
    cluster_id=None,
    num_within=50,
    num_total=100,
):

    # subset the ad_ex and ad_motif by the neighbors
    tfs = ad_motif.var_names
    retdf = pd.DataFrame(index=tfs, columns=ad_ex.obs_names)
    if cluster_id != None:
        nn_indices = get_nearest_neighbors(ad_pop)
    for i, cell in tqdm(enumerate(ad_ex.obs_names)):
        #         neighbors_bool = np.asarray(ad_pop.obsp['adjacency_matrix'][i,:].todense().astype(bool)).flatten()
        if cluster_id == None:
            neighbors_bool = get_nhop_neighbors(ad_pop, i, num_hops=num_hops)
        else:
            neighbors_bool = get_neighbors(
                ad_pop,
                i,
                nn_indices,
                ad_ex.obs[cluster_id][cell],
                cluster_id,
                num_within,
                num_total,
            )
        neighbors = ad_ex.obs_names[neighbors_bool]
        if len(neighbors) == 0:
            retdf[cell] = np.zeros((len(tfs), 1))
            continue
        data = ad_motif[neighbors, tfs].to_df()
        # label = ad_pop.obsm['normalized_X'][neighbors_bool,target_metagene]
        label = ad_pop.obsm["X"][neighbors_bool, target_metagene]
        model = BaggingRegressor(
            estimator=Ridge(
                alpha=1,
                solver="auto",
                random_state=123,
            ),
            n_estimators=10,
            bootstrap=True,
            max_features=0.8,
            verbose=False,
            random_state=123,
        )
        model.fit(data, label)
        ans = _get_coef_matrix(model, tfs).mean(axis=0)
        retdf[cell] = ans
    return retdf.T


def get_metagene_edges_smoothed(ad_ex, ad_motif, metagene_num, ad_pop, num_hops=1):

    # subset the ad_ex and ad_motif by the neighbors
    tfs = ad_motif.var_names
    retdf = pd.DataFrame(index=tfs)
    for target_metagene in tqdm(range(metagene_num)):
        data = pd.DataFrame(index=tfs, dtype=np.float64)
        label = pd.DataFrame(index=[target_metagene], dtype=np.float64)
        for i, cell in enumerate(ad_ex.obs_names):
            #         neighbors_bool = np.asarray(ad_pop.obsp['adjacency_matrix'][i,:].todense().astype(bool)).flatten()
            neighbors_bool = get_nhop_neighbors(ad_pop, i, num_hops=num_hops)
            neighbors = ad_ex.obs_names[neighbors_bool]
            if len(neighbors) == 0:
                data[cell] = np.zeros((len(tfs), 1))
                label[cell] = 0
                continue
            data[cell] = np.asarray(ad_motif[neighbors, tfs].X.sum(axis=0))
            # label[cell] = np.asarray(ad_pop[neighbors,:].obsm['normalized_X'][:,target_metagene].sum())
            label[cell] = np.asarray(
                ad_pop[neighbors, :].obsm["X"][:, target_metagene].sum(),
            )
        data = data.T - data.T.min(axis=0)
        data /= data.max(axis=0)
        label = label.T - label.T.min()
        label /= label.max()
        model = BaggingRegressor(
            base_estimator=Ridge(
                alpha=1,
                solver="auto",
                random_state=123,
            ),
            n_estimators=100,
            bootstrap=True,
            max_features=0.8,
            verbose=False,
            random_state=123,
        )
        model.fit(data, label)
        ans = _get_coef_matrix(model, tfs).mean(axis=0)
        retdf[str(target_metagene)] = ans
    return retdf


def get_nhop_neighbors(ad, cellidx, num_hops=1):
    neighbors = ad.obsp["adjacency_matrix"][cellidx, :]
    for i in range(num_hops - 1):
        neighbors = neighbors.multiply(ad.obsp["adjacency_matrix"][cellidx, :])
        neighbors[neighbors > 1] = 1
    return np.asarray(neighbors.todense().astype(bool)).flatten()


from multiprocessing import Pool

from popari import Popari, tl
from scipy.stats import zscore


def in_silico_perturb(ad_pop, ad_tf, ad_edge, tf, K=16, multiplier=10, useX=False):
    if useX == False:
        dropout_X = ad_pop.obsm["normalized_X"].copy()
    else:
        dropout_X = ad_pop.obsm["X"].copy()
    # multiply the tf by the edge-weight for each cell for each metagene
    for metagene in range(K):
        perturbation = ad_tf[:, tf].X * ad_edge[:, tf].layers[f"M_{metagene}"]
        perturbation *= multiplier
        dropout_X[:, metagene] = dropout_X[:, metagene] - perturbation.flatten()
    ad_pop.obsm[f"X_{tf}_dropout"] = dropout_X

    normalized_embeddings = zscore(ad_pop.obsm[f"X_{tf}_dropout"])
    nan_mask = np.isnan(normalized_embeddings)
    normalized_embeddings[nan_mask] = 0

    ad_pop.obsm[f"normalized_X_{tf}_dropout"] = normalized_embeddings
    sc.pp.neighbors(ad_pop, use_rep=f"normalized_X_{tf}_dropout")


def align_leiden(d, tf):
    # find the best matching between 'original_leiden' clusters and 'leiden' clusters
    # Can be solved with minimum-weight perfect matching for a bipartite graph of the old and new clusters where the edges are the number of matches between clusters
    # Can use the scipy algorithm for linear_sum_assignment()
    from scipy.optimize import linear_sum_assignment

    cost_matrix = np.zeros(
        (len(d.obs["original_leiden"].unique()), len(d.obs["leiden"].unique())),
    )
    for cluster in d.obs["leiden"].unique():
        vals = d.obs[d.obs["leiden"] == cluster]["original_leiden"].value_counts()
        inds = d.obs[d.obs["leiden"] == cluster]["original_leiden"].value_counts().index
        for ind, val in zip(inds, vals):
            cost_matrix[int(ind), int(cluster)] = val
    row_matches, col_matches = linear_sum_assignment(cost_matrix, maximize=True)
    remapped_col_matches = [
        row_matches[col_matches.tolist().index(i)] for i in range(len(col_matches))
    ]
    newleiden = [str(remapped_col_matches[int(obs)]) for obs in d.obs["leiden"]]
    # d.obs[f'leiden_{tf}_dropout'] = newleiden
    return newleiden


def test_all_true(l):
    if sum(l) < len(l):
        return False
    else:
        return True


def run_all_perturbations_parallel(
    pop,
    ad_tfs,
    ad_edges,
    K=16,
    multiplier=1,
    useX=False,
    target_clusters=10,
    num_processes=4,
):
    tfs = ad_tfs[0].var_names
    # cluster_changes = {}
    new_columns = [[] for d in pop.datasets]
    checkpoints = 100
    with Pool(processes=num_processes) as pool:
        new_columns.append(
            pool.starmap(
                run_perturbation,
                [(tf, pop, ad_tfs, ad_edges, K, multiplier, useX) for tf in tfs],
            ),
        )

    tfcolumns = [f"leiden_{tf}_dropout" for tf in tfs]
    new_columns_pds = [
        pd.DataFrame(
            new_columns[i],
            index=tfcolumns,
            columns=pop.datasets[i].obs_names,
        ).T
        for i in range(len(pop.datasets))
    ]
    for d, ncpd in zip(pop.datasets, new_columns_pds):
        d.obs = d.obs.join(ncpd)
    # return cluster_changes


def run_perturbation(tf, pop, ad_tfs, ad_edges, K, multiplier, useX):
    for pop_ad, ad_tf, ad_edge in zip(pop.datasets, ad_tfs, ad_edges):
        in_silico_perturb(
            pop_ad,
            ad_tf,
            ad_edge,
            tf,
            K=K,
            multiplier=multiplier,
            useX=useX,
        )
    # tl.leiden(pop, use_rep=f"X_{tf}_dropout", target_clusters=10)
    tl.leiden(
        pop,
        use_rep=f"normalized_X_{tf}_dropout",
        target_clusters=target_clusters,
    )
    j = target_clusters
    len_satisfies = test_all_true(
        [len(d.obs["leiden"].unique()) <= target_clusters for d in pop.datasets],
    )
    while len_satisfies == False:
        j -= 1
        tl.leiden(pop, use_rep=f"normalized_X_{tf}_dropout", target_clusters=j)
        len_satisfies = test_all_true(
            [len(d.obs["leiden"].unique()) <= target_clusters for d in pop.datasets],
        )
    return [align_leiden(d, tf) for d in pop.datasets]


def run_all_perturbations(
    pop,
    ad_tfs,
    ad_edges,
    K=16,
    multiplier=1,
    useX=False,
    target_clusters=10,
    get_leiden=False,
):
    new_columns = [[] for d in pop.datasets]
    perturbed_datasets = [d.copy() for d in pop.datasets]
    for d in perturbed_datasets:
        for name in ["Sigma_x_inv", "popari_hyperparameters", "losses", "sigma_yx"]:
            if name in d.uns:
                del d.uns[name]
        if "adjacency_list" in d.obs.columns:
            del d.obs["adjacency_list"]
    for pop_ad, ad_tf, ad_edge, ite in zip(
        perturbed_datasets,
        ad_tfs,
        ad_edges,
        range(len(pop.datasets)),
    ):
        tfs = ad_tf.var_names
        # cluster_changes = {}
        for it, tf in tqdm(enumerate(tfs)):
            in_silico_perturb(
                pop_ad,
                ad_tf,
                ad_edge,
                tf,
                K=K,
                multiplier=multiplier,
                useX=useX,
            )
        # tl.leiden(pop, use_rep=f"X_{tf}_dropout", target_clusters=10)
    if get_leiden == True:
        for tf in tfs:
            tl.leiden(
                pop,
                use_rep=f"normalized_X_{tf}_dropout",
                target_clusters=target_clusters,
            )
            j = target_clusters
            len_satisfies = test_all_true(
                [
                    len(d.obs["leiden"].unique()) <= target_clusters
                    for d in pop.datasets
                ],
            )
            while len_satisfies == False:
                j -= 1
                # tl.leiden(pop, use_rep=f"X_{tf}_dropout", target_clusters=j)
                tl.leiden(pop, use_rep=f"normalized_X_{tf}_dropout", target_clusters=j)
                len_satisfies = test_all_true(
                    [
                        len(d.obs["leiden"].unique()) <= target_clusters
                        for d in pop.datasets
                    ],
                )
            for ite, pop_ad in enumerate(pop.datasets):
                new_columns[ite].append(align_leiden(pop_ad, tf))

        tfcolumns = [f"leiden_{tf}_dropout" for tf in tfs]
        new_columns_pds = [
            pd.DataFrame(
                new_columns[i],
                index=tfcolumns,
                columns=pop.datasets[i].obs_names,
            ).T
            for i in range(len(pop.datasets))
        ]
        for d, ncpd in zip(pop.datasets, new_columns_pds):
            d.obs = d.obs.join(ncpd)
    return perturbed_datasets


def find_tf_causing_cluster(pop_ad, cluster_num):
    tf_leidens = [c for c in pop_ad.obs.columns if "dropout" in c]
    changes = {}
    pop_subset = pop_ad[pop_ad.obs["original_leiden"] == cluster_num]
    for tf_leiden in tf_leidens:
        change_num = (pop_subset.obs[tf_leiden] != cluster_num).sum()
        changes[tf_leiden] = change_num
    return changes


#  The functions below were taken from CellOracle (https://github.com/morris-lab/CellOracle)
#  and modified for our use
def _adata_to_matrix(adata, layer_name, transpose=True):
    """Extract an numpy array from adata and returns as numpy matrix.

    Args:
        adata (anndata): anndata

        layer_name (str): name of layer in anndata

        trabspose (bool) : if True, it returns transposed array.

    Returns:
        2d numpy array: numpy array

    """
    if isinstance(adata.layers[layer_name], np.ndarray):
        matrix = adata.layers[layer_name].copy()
    else:
        matrix = adata.layers[layer_name].todense().A.copy()

    if transpose:
        matrix = matrix.transpose()

    return matrix.copy(order="C")


def _obsm_to_matrix(adata, obsm_name, transpose=True):
    """Extract an numpy array from adata and returns as numpy matrix.

    Args:
        adata (anndata): anndata

        layer_name (str): name of layer in anndata

        trabspose (bool) : if True, it returns transposed array.

    Returns:
        2d numpy array: numpy array

    """
    if isinstance(adata.obsm[obsm_name], np.ndarray):
        matrix = adata.obsm[obsm_name].copy()
    else:
        matrix = adata.obsm[obsm_name].todense().A.copy()

    if transpose:
        matrix = matrix.transpose()

    return matrix.copy(order="C")


def plot_background(
    self,
    embedding_name="",
    dataset_num=0,
    ax=None,
    s=CONFIG["s_scatter"],
    args=CONFIG["default_args"],
):

    if ax is None:
        ax = plt

    ax.scatter(
        self.embeddings[dataset_num][embedding_name].embedding[:, 0],
        self.embeddings[dataset_num][embedding_name].embedding[:, 1],
        c="lightgray",
        s=s,
        **args,
    )

    # ax.set_title("Pseudotime")
    ax.axis("off")


def plot_cluster_cells_use(
    self,
    embedding_name="",
    dataset_num=0,
    ax=None,
    s=CONFIG["s_scatter"],
    color=None,
    show_background=True,
    args=CONFIG["default_args"],
):

    if ax is None:
        ax = plt

    if s == 0:
        color = "white"

    if show_background:
        plot_background(self=self, dataset_num=dataset_num, ax=ax, s=s, args=args)

    if not hasattr(self.embeddings[dataset_num][embedding_name], "cell_idx_use"):
        self.embeddings[dataset_num][embedding_name].cell_idx_use = None

    if self.embeddings[dataset_num][embedding_name].cell_idx_use is None:
        if color is None:
            ax.scatter(
                self.embeddings[dataset_num][embedding_name].embedding[:, 0],
                self.embeddings[dataset_num][embedding_name].embedding[:, 1],
                c=self.colorandum,
                s=s,
                **args,
            )
        else:
            ax.scatter(
                self.embeddings[dataset_num][embedding_name].embedding[:, 0],
                self.embeddings[dataset_num][embedding_name].embedding[:, 1],
                c=color,
                s=s,
                **args,
            )

    else:
        if color is None:
            ax.scatter(
                self.embeddings[dataset_num][embedding_name].embedding[
                    self.embeddings[dataset_num][embedding_name].cell_idx_use,
                    0,
                ],
                self.embeddings[dataset_num][embedding_name].embedding[
                    self.embeddings[dataset_num][embedding_name].cell_idx_use,
                    1,
                ],
                c=self.embeddings[dataset_num][embedding_name].colorandum[
                    self.embeddings[dataset_num][embedding_name].cell_idx_use,
                    :,
                ],
                s=s,
                **args,
            )
        else:
            ax.scatter(
                self.embeddings[dataset_num][embedding_name].embedding[
                    self.embeddings[dataset_num][embedding_name].cell_idx_use,
                    0,
                ],
                self.embeddings[dataset_num][embedding_name].embedding[
                    self.embeddings[dataset_num][embedding_name].cell_idx_use,
                    1,
                ],
                c=color,
                s=s,
                **args,
            )

    ax.axis("off")


def _get_clustercolor_from_anndata(adata, cluster_name, return_as):
    """Extract clor information from adata and returns as palette (pandas data
    frame) or dictionary.

    Args:
        adata (anndata): anndata

        cluster_name (str): cluster name in anndata.obs

        return_as (str) : "palette" or "dict"

    Returns:
        2d numpy array: numpy array

    """

    # return_as: "palette" or "dict"
    def float2rgb8bit(x):
        x = (x * 255).astype("int")
        x = tuple(x)

        return x

    def rgb2hex(rgb):
        return "#%02x%02x%02x" % rgb

    def float2hex(x):
        x = float2rgb8bit(x)
        x = rgb2hex(x)
        return x

    def hex2rgb(c):
        return (int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16), 255)

    pal = get_palette(adata, cluster_name)
    if return_as == "palette":
        return pal
    elif return_as == "dict":
        col_dict = {}
        for i in pal.index:
            col_dict[i] = np.array(hex2rgb(pal.loc[i, "palette"])) / 255
        return col_dict
    else:
        raise ValueErroe("return_as")
    return 0


def get_palette(adata, cname):
    c = [i.upper() for i in adata.uns[f"{cname}_colors"]]
    # c = sns.cubehelix_palette(24)
    """
    col = adata.obs[cname].unique()
    col = list(col)
    col.sort()
    """
    try:
        col = adata.obs[cname].cat.categories
        pal = pd.DataFrame({"palette": c}, index=col)
    except:
        col = adata.obs[cname].cat.categories
        c = c[: len(col)]
        pal = pd.DataFrame({"palette": c}, index=col)
    return pal


# this is a function to extract coef information from sklearn ensemble_model.
def _get_coef_matrix(ensemble_model, feature_names):
    # ensemble_model: trained ensemble model. e.g. BaggingRegressor
    # feature_names: list or numpy array of feature names. e.g. feature_names=X_train.columns
    feature_names = np.array(feature_names)
    n_estimater = len(ensemble_model.estimators_features_)
    coef_list = [
        pd.Series(
            ensemble_model.estimators_[i].coef_,
            index=feature_names[ensemble_model.estimators_features_[i]],
        )
        for i in range(n_estimater)
    ]

    coef_df = pd.concat(coef_list, axis=1, sort=False).transpose()

    return coef_df


def intersect(list1, list2):
    """Intersect two list and get components that exists in both list.

    Args:
        list1 (list): input list.
        list2 (list): input list.

    Returns:
        list: intersected list.

    """
    inter_list = list(set(list1).intersection(list2))
    return inter_list
