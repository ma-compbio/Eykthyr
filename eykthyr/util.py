import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from tqdm.notebook import tqdm

import seaborn as sns


from scipy.stats import zscore
from typing import List, Optional
from scipy.sparse import issparse
from scipy import sparse
from sklearn.neighbors import NearestNeighbors

CONFIG = {
    "default_args": {"lw": 0.3, "rasterized": True},
    "s_scatter": 5,
    "s_grid": 20,
    "scale_simulation": 30,
    "scale_dev": 30,
    "cmap_ps": "PiYG",
    "default_args_quiver": {"linewidths": 0.25, "width": 0.004},
}


class RollingRidge:
    def __init__(self, n_features, alpha=1.0):
        self.n_features = n_features
        self.alpha = alpha
        
        # The "State" matrices
        # XT_X corresponds to the covariance (shape: features x features)
        self.XT_X = np.zeros((n_features, n_features))
        # XT_y corresponds to the correlation with target (shape: features,)
        self.XT_y = np.zeros(n_features)
        
        # Track number of samples for variance calc
        self.n_samples = 0

    def update(self, X_add, y_add, X_remove, y_remove):
        """
        Updates the regression state by adding new cells and removing old ones.
        """
        # 1. Add new cells (rank-k update)
        if len(X_add) > 0:
            self.XT_X += X_add.T @ X_add
            self.XT_y += X_add.T @ y_add
            self.n_samples += len(X_add)

        # 2. Remove old cells
        if len(X_remove) > 0:
            self.XT_X -= X_remove.T @ X_remove
            self.XT_y -= X_remove.T @ y_remove
            self.n_samples -= len(X_remove)

    def fit_and_get_robust_weights(self):
        """
        Solves Ridge regression and calculates analytical variance 
        (replacing BaggingRegressor).
        """
        if self.n_samples < 2:
            return np.zeros(self.n_features)

        # 1. Solve Beta (Coefficients)
        # (XT_X + alpha*I)^-1 * XT_y
        regularized_XT_X = self.XT_X + np.eye(self.n_features) * self.alpha
        
        # Using linalg.solve is faster than inverting
        try:
            beta = np.linalg.solve(regularized_XT_X, self.XT_y)
        except np.linalg.LinAlgError:
            return np.zeros(self.n_features)

        # 2. Calculate Analytical Variance (The "Robust" part)
        # Residual Variance (sigma^2) = Sum(y_true - y_pred)^2 / (n - p)
        # Note: Calculating full residuals is expensive. We approximate 
        # variance based on the diagonal of the inverse covariance matrix.
        
        # Covariance of betas = sigma^2 * (XT_X + alpha*I)^-1
        # We need the inverse diagonal for standard errors
        inv_cov = np.linalg.inv(regularized_XT_X)
        std_errors = np.sqrt(np.diag(inv_cov))
        
        # 3. Apply "One-Standard-Error" Rule (Mean / Std)
        # This mimics your Bagging logic: signal / noise
        epsilon = 1e-6
        robust_weights = beta / (std_errors + epsilon)
        
        return robust_weights

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


# def get_neighbors(ad_pop, i, nn_indices, cell_type, cluster_id, num_within, num_total):
#     cell_indices = nn_indices[i]
#     within_cluster_indices = [
#         ind
#         for ind, cell in enumerate(ad_pop.obs_names)
#         if ad_pop.obs[cluster_id][cell] == cell_type
#     ]
#     cell_type_list = [t for t in ad_pop.obs[cluster_id].values]

#     num_total += 1  # This is done so you always return yourself plus the number of required neighbors
#     neighbors = []
#     num_without_added = 0
#     cell_idx = 0
#     while len(neighbors) < num_total:
#         if cell_type_list[cell_indices[cell_idx]] != cell_type:
#             if num_without_added < (num_total - num_within):
#                 neighbors.append(cell_indices[cell_idx])
#                 num_without_added += 1
#         else:
#             neighbors.append(cell_indices[cell_idx])
#         cell_idx += 1
#     # now to get the boolean mask
#     neighbors_bool = [
#         True if c_ind in neighbors else False for c_ind in range(len(ad_pop.obs_names))
#     ]
#     return neighbors_bool


def get_neighbors(ad_pop, i, nn_indices, cell_type, cluster_id, num_within, num_total, nn_indices_same=None):
    """
    Selects neighbors based on cluster identity constraints.
    
    If nn_indices_same is provided:
      1. Fills up to 'num_within' using nn_indices_same (guaranteed same cluster).
      2. Fills the remainder (up to num_total + 1) using nn_indices (global), 
         strictly picking 'different type' neighbors.
    """
    # 1. Adjust total to include self
    target_count = num_total + 1
    
    # 2. Extract Labels (Optimized)
    if isinstance(ad_pop, np.ndarray): 
        all_labels = ad_pop 
    else:
        all_labels = ad_pop.obs[cluster_id].values

    final_indices = []

    # --- STRATEGY A: Use Pre-calculated Same-Cluster Indices (Requested) ---
    if nn_indices_same is not None:
        # A. Get Same-Cluster Neighbors (Priority)
        # These are pre-calculated to be the nearest spatial neighbors that share the cluster ID
        candidates_same = nn_indices_same[i]
        
        # Remove -1 padding if any (from the matrix generation step)
        candidates_same = candidates_same[candidates_same != -1]
        
        # We take up to 'num_within' of these, plus the cell itself is likely in here
        # So we actually take up to num_within + 1 (for self) or stick to num_within logic
        # Let's aim to fill 'num_within' slots with same-type.
        # Note: target_count includes self. 
        limit_same = num_within + 1 # Allow slightly more to ensure self is included
        
        selected_same = candidates_same[:limit_same]
        
        # B. Get Different-Cluster Neighbors (Fill the rest)
        # We need to fill the gap: target_count - len(selected_same)
        # But specifically, we want 'diff' types to fill the 'limit_diff' quota.
        
        current_count = len(selected_same)
        needed_diff = target_count - current_count
        
        if needed_diff > 0:
            candidates_global = nn_indices[i]
            candidate_labels = all_labels[candidates_global]
            
            # Identify different types
            is_diff_type = (candidate_labels != cell_type)
            
            # Select the nearest ones that are different
            selected_diff = candidates_global[is_diff_type][:needed_diff]
            
            # Combine
            final_indices = np.concatenate([selected_same, selected_diff])
        else:
            # We have enough (or too many) same-cluster neighbors
            # Just truncate the same-cluster list to target_count
            final_indices = selected_same[:target_count]

    # --- STRATEGY B: Original Logic (Fallback) ---
    else:
        candidates = nn_indices[i]
        candidate_labels = all_labels[candidates]
        is_diff_type = (candidate_labels != cell_type)
        limit_diff = target_count - num_within
        cumsum_diff = np.cumsum(is_diff_type)
        keep_mask = (~is_diff_type) | (cumsum_diff <= limit_diff)
        final_indices = candidates[keep_mask][:target_count]
    
    # 7. Create Boolean Mask (Vectorized)
    n_cells = len(all_labels)
    neighbors_bool = np.zeros(n_cells, dtype=bool)
    neighbors_bool[final_indices.astype(int)] = True
    
    return neighbors_bool


def get_nearest_neighbors(ad, n_neighbors=None):
    X = ad.obsm["spatial"]
    if n_neighbors == None:
        n_neighbors = len(ad.obs_names)
    # Ensure n_neighbors doesn't exceed sample size
    n_neighbors = min(n_neighbors, X.shape[0])
    
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm="ball_tree").fit(X)
    distances, indices = nbrs.kneighbors(X)
    return indices

def get_cluster_nearest_neighbors(ad, cluster_id, n_neighbors):
    """
    Generates a matrix (N_cells x n_neighbors) where row i contains 
    indices of the nearest neighbors to cell i that share the SAME cluster_id.
    """
    X = ad.obsm["spatial"]
    if isinstance(X, pd.DataFrame): X = X.values
    
    labels = ad.obs[cluster_id].values
    unique_labels = np.unique(labels)
    
    # Initialize with -1 (padding)
    n_cells = ad.shape[0]
    result_indices = np.full((n_cells, n_neighbors), -1, dtype=int)
    
    # Iterate over each cluster to compute local neighbors
    for label in unique_labels:
        # Get global indices for cells in this cluster
        group_mask = (labels == label)
        global_indices_in_group = np.where(group_mask)[0]
        
        if len(global_indices_in_group) == 0:
            continue
            
        # Subset spatial data
        X_subset = X[global_indices_in_group, :]
        
        # Fit NN for this cluster specifically
        # Cap k at the number of cells in the cluster
        k = min(len(global_indices_in_group), n_neighbors)
        
        nbrs = NearestNeighbors(n_neighbors=k, algorithm="ball_tree").fit(X_subset)
        _, local_neighbor_indices = nbrs.kneighbors(X_subset)
        
        # Map local indices (0..k) back to global indices (0..N)
        # local_neighbor_indices is (M, k)
        # global_indices_in_group is (M,)
        global_neighbor_indices = global_indices_in_group[local_neighbor_indices]
        
        # Store in the result matrix
        # We fill the rows corresponding to this group
        # If k < n_neighbors, the remaining columns stay -1
        result_indices[global_indices_in_group, :k] = global_neighbor_indices
        
    return result_indices

def get_neighbors_cluster(ad_pop, i, nn_indices, cell_type, cluster_id):
    """
    Return a boolean mask over ad_pop.obs_names marking:
      - the focal cell i, and
      - all of its kNN neighbors whose ad_pop.obs[cluster_id] == cell_type.
    """
    # neighbors from the precomputed kNN list
    cell_indices = nn_indices[i]

    # Build the set of neighbor indices that match the focal cell's cluster
    # (include self explicitly)
    same_cluster_neighbor_indices = set()
    same_cluster_neighbor_indices.add(i)
    obs_col = ad_pop.obs[cluster_id]
    for idx in cell_indices:
        # obs is a pandas Series/DataFrame column, so use iloc for speed/safety
        if obs_col.iloc[idx] == cell_type:
            same_cluster_neighbor_indices.add(idx)

    # boolean mask over all cells
    neighbors_bool = [j in same_cluster_neighbor_indices for j in range(len(ad_pop.obs_names))]
    return neighbors_bool




def _visualize_local_regression(
    center_cell_name, 
    neighbor_names, 
    all_coords, 
    global_cell_names, # <--- NEW ARGUMENT
    tf_data_df, 
    metagene_label_array, 
    target_mg_idx
):
    """
    Visualizes the input data for the regression of a single cell.
    """
    # 1. Identify the most relevant TF in this window
    corrs = tf_data_df.corrwith(pd.Series(metagene_label_array, index=tf_data_df.index))
    best_tf = corrs.idxmax()
    
    fig = plt.figure(figsize=(18, 5))
    gs = fig.add_gridspec(1, 4)
    
    # --- Panel 1: Global Spatial Context ---
    # FIX: Map string names to global integer indices to slice all_coords correctly
    global_idx_map = pd.Index(global_cell_names)
    nbr_indices = global_idx_map.get_indexer(neighbor_names)
    center_index = global_idx_map.get_loc(center_cell_name)
    
    nbr_coords = all_coords[nbr_indices]
    center_coord = all_coords[center_index]

    ax1 = fig.add_subplot(gs[0, 0])
    # Plot all cells grey
    ax1.scatter(all_coords[:, 0], all_coords[:, 1], c='lightgrey', s=1, alpha=0.5)
    # Plot neighbors blue
    ax1.scatter(nbr_coords[:, 0], nbr_coords[:, 1], c='blue', s=10, label='Neighbors')
    # Plot center red
    ax1.scatter([center_coord[0]], [center_coord[1]], c='red', s=50, marker='*', label='Center')
    
    ax1.set_title(f"Cell {center_cell_name}\n(Global Context)")
    ax1.axis('off')
    ax1.legend(fontsize='x-small')

    # --- Panel 2: Local Metagene Expression ---
    ax2 = fig.add_subplot(gs[0, 1])
    # Note: nbr_coords and metagene_label_array are aligned because they both come from 'neighbors' list
    sc2 = ax2.scatter(nbr_coords[:, 0], nbr_coords[:, 1], c=metagene_label_array, cmap='viridis', s=30)
    ax2.set_title(f"Target: Metagene {target_mg_idx}\n(Local Expression)")
    ax2.axis('off')
    plt.colorbar(sc2, ax=ax2, label='Score')

    # --- Panel 3: Local TF Expression (Best TF) ---
    ax3 = fig.add_subplot(gs[0, 2])
    tf_vals = tf_data_df[best_tf].values
    sc3 = ax3.scatter(nbr_coords[:, 0], nbr_coords[:, 1], c=tf_vals, cmap='magma', s=30)
    ax3.set_title(f"Feature: {best_tf}\n(Local Expression)")
    ax3.axis('off')
    plt.colorbar(sc3, ax=ax3, label='Counts')

    # --- Panel 4: The Regression View (X vs Y) ---
    ax4 = fig.add_subplot(gs[0, 3])
    sns.regplot(x=tf_vals, y=metagene_label_array, ax=ax4, scatter_kws={'s': 20}, line_kws={'color': 'red'})
    ax4.set_xlabel(f"{best_tf} Expression")
    ax4.set_ylabel(f"Metagene {target_mg_idx}")
    ax4.set_title(f"Regression Task\n(Locally weighted)")

    plt.tight_layout()
    plt.show()



def _visualize_local_regression_robust(
    center_cell_name, 
    neighbor_names, 
    all_coords, 
    global_cell_names, 
    tf_data_df, 
    metagene_label_array, 
    target_mg_idx,
    model,             # <--- NEW: The fitted BaggingRegressor
    final_weight_vector # <--- NEW: The calculated robust weights
):
    """
    Visualizes the regression stability and robust metrics.
    """
    # 1. Identify the TF with the highest 'Raw' correlation to focus on
    # (We can't plot 16 dimensions, so we pick the most relevant one)
    corrs = tf_data_df.corrwith(pd.Series(metagene_label_array, index=tf_data_df.index))
    best_tf = corrs.abs().idxmax()
    tf_idx_in_model = list(tf_data_df.columns).index(best_tf)
    print(tf_idx_in_model)
    
    # Get stats for this specific TF
    tf_vals = tf_data_df[best_tf].values
    
    # 2. Extract specific stats for the Scorecard
    if hasattr(model, 'oob_score_'):
        oob = model.oob_score_
    else:
        oob = np.nan
    
    sparsity = (tf_vals > 0).mean() # Fraction of cells expressing this TF
    
    # Extract coefficients from the ensemble for this specific TF
    # (Note: estimators might not have selected this feature if max_features < 1.0)
    coefs = []
    for est, feats in zip(model.estimators_, model.estimators_features_):
        # feats is indices of features used by this estimator
        if tf_idx_in_model in feats:
            # Find where tf_idx_in_model is located in the estimator's feature list
            internal_idx = np.where(feats == tf_idx_in_model)[0][0]
            coefs.append(est.coef_[internal_idx])
        else:
            coefs.append(0.0)
    
    mu = np.mean(coefs)
    sigma = np.std(coefs)
    final_w = final_weight_vector[best_tf]

    # --- PLOTTING ---
    fig = plt.figure(figsize=(20, 5))
    gs = fig.add_gridspec(1, 4)
    
    # Panel 1: Global Context (Same as before)
    global_idx_map = pd.Index(global_cell_names)
    nbr_indices = global_idx_map.get_indexer(neighbor_names)
    center_index = global_idx_map.get_loc(center_cell_name)
    nbr_coords = all_coords[nbr_indices]
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(all_coords[:, 0], all_coords[:, 1], c='lightgrey', s=1, alpha=0.5)
    ax1.scatter(nbr_coords[:, 0], nbr_coords[:, 1], c='blue', s=10)
    ax1.scatter(all_coords[center_index, 0], all_coords[center_index, 1], c='red', s=50, marker='*')
    ax1.set_title(f"Cell {center_cell_name}\nGlobal Context")
    ax1.axis('off')

    # Panel 2: Local Metagene (Same as before)
    ax2 = fig.add_subplot(gs[0, 1])
    sc2 = ax2.scatter(nbr_coords[:, 0], nbr_coords[:, 1], c=metagene_label_array, cmap='viridis', s=30)
    ax2.set_title(f"Target: Metagene {target_mg_idx}\n(Local Expression)")
    ax2.axis('off')
    plt.colorbar(sc2, ax=ax2, label='Score')

    # Panel 3: "Spaghetti" Regression Plot
    ax3 = fig.add_subplot(gs[0, 2])
    
    # A. Scatter Data
    ax3.scatter(tf_vals, metagene_label_array, color='black', s=20, label='Data')
    
    # B. Plot Individual Estimators (Grey Lines)
    x_range = np.linspace(tf_vals.min(), tf_vals.max(), 100)
    
    # We plot the lines assuming y = coef*x (fit_intercept=False)
    for c in coefs:
        y_est = c * x_range
        ax3.plot(x_range, y_est, color='grey', alpha=0.3, linewidth=1)
        
    # C. Plot Final Robust Weight (Red Line)
    y_final = final_w * x_range
    ax3.plot(x_range, y_final, color='red', linewidth=3, label='Final Robust Weight')
    
    ax3.set_xlabel(f"{best_tf} Expression")
    ax3.set_ylabel(f"Metagene {target_mg_idx}")
    ax3.set_title(f"Stability Check\n(Grey=Bootstrap, Red=Final)")
    ax3.legend(fontsize='small')

    # Panel 4: The Scorecard (Text Stats)
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.axis('off')
    
    # Color code the judgment
    status_color = "green" if abs(final_w) > 0.01 else "red"
    
    text_str = (
        f"Feature: {best_tf}\n"
        f"-----------------------\n"
        f"OOB Score (R2):  {oob:.3f}\n"
        f"Sparsity (Support): {sparsity:.2f}\n"
        f"-----------------------\n"
        f"Raw Mean (μ):    {mu:.3f}\n"
        f"Stability (σ):   {sigma:.3f}\n"
        f"μ - σ:           {abs(mu)-sigma:.3f}\n"
        f"-----------------------\n"
        f"FINAL WEIGHT:    {final_w:.4f}"
    )
    
    ax4.text(0.1, 0.5, text_str, fontsize=14, family='monospace', va='center', color='black',
             bbox=dict(boxstyle="round,pad=0.5", fc="white", ec=status_color, lw=2))
    ax4.set_title("Robust Weight Scorecard")

    plt.tight_layout()
    plt.show()

def get_metagene_edges_window(
    ad_ex,
    ad_motif,
    target_metagene,
    ad_pop,
    num_hops=1,
    cluster_id=None,
    num_within=50,
    num_total=100,
    cluster_only=False,
    verbose=False, 
):

    tfs = ad_motif.var_names
    retdf = pd.DataFrame(index=tfs, columns=ad_ex.obs_names)

    # Precompute kNN indices if we'll need them
    nn_indices = None
    if (cluster_id is not None) or cluster_only:
        nn_indices = get_nearest_neighbors(ad_pop, n_neighbors=num_total)

    nn_indices_same = None
    if cluster_id is not None:
        # We fetch enough same-cluster neighbors to satisfy 'num_within'
        # +1 to account for self
        nn_indices_same = get_cluster_nearest_neighbors(
            ad_pop, 
            cluster_id=cluster_id, 
            n_neighbors=num_within + 10 # small buffer
        )

    # Convenience: cluster series (if provided)
    cluster_series = ad_ex.obs[cluster_id] if cluster_id is not None else None
    labels_array = ad_pop.obs[cluster_id].values.astype(str) if cluster_id is not None else None

    # --- VERBOSE SETUP ---
    debug_indices = set()
    if verbose:
        # Pick 3 random cells to visualize
        num_cells = len(ad_ex.obs_names)
        if num_cells > 0:
            debug_indices = set(np.random.choice(range(num_cells), min(10, num_cells), replace=False))
        
        # Get global coords for plotting context
        if 'spatial' in ad_pop.obsm:
            all_coords = ad_pop.obsm['spatial']
            if isinstance(all_coords, pd.DataFrame): all_coords = all_coords.values
        else:
            print("Warning: No spatial coords found in ad_pop.obsm['spatial']. Skipping plots.")
            debug_indices = set()
    # ---------------------

    for i, cell in enumerate(ad_ex.obs_names):
        # Choose neighbor mask logic
        if cluster_only:
            if cluster_id is None:
                raise ValueError("cluster_only=True requires a valid cluster_id.")
            neighbors_bool = get_neighbors_cluster(
                ad_pop=ad_pop,
                i=i,
                nn_indices=nn_indices,
                cell_type=cluster_series.loc[cell],
                cluster_id=cluster_id,
            )
        else:
            if cluster_id is None:
                neighbors_bool = get_nhop_neighbors(ad_pop, i, num_hops=num_hops)
            else:
                neighbors_bool = get_neighbors(
                    ad_pop=labels_array, 
                    i=i,
                    nn_indices=nn_indices,
                    cell_type=cluster_series.iloc[i],
                    cluster_id=cluster_id,
                    num_within=num_within,
                    num_total=num_total,
                    nn_indices_same=nn_indices_same 
                )
        
        neighbors = ad_pop.obs_names[neighbors_bool]
        
        if len(neighbors) == 0:
            retdf[cell] = np.zeros((len(tfs), 1))
            continue
            
        data = ad_motif[neighbors, tfs].to_df()
        label = ad_pop.obsm["X"][neighbors_bool, target_metagene]
            
        # # # --- VISUALIZATION HOOK ---
        # # if verbose and (i in debug_indices):
        # #     print(f"--- Debugging Cell {i}: {cell} ---")
        # #     print(f"Number of neighbors: {len(neighbors)}")
        # #     _visualize_local_regression(
        # #         center_cell_name=cell,
        # #         neighbor_names=neighbors,
        # #         all_coords=all_coords,
        # #         global_cell_names=ad_pop.obs_names, # <--- NEW ARG PASSED HERE
        # #         tf_data_df=data,
        # #         metagene_label_array=label,
        # #         target_mg_idx=target_metagene
        # #     )
        # # # --------------------------

        # # Use the new robust function (Attempt 2)
        # weights = fit_and_get_robust_weights(data, label, tfs)
        
        # # 2. Visualization Hook
        # if verbose and (i in debug_indices):
        #     print(f"--- Debugging Cell {i}: {cell} ---")
        #     print(weights.reindex(tfs, fill_value=0.0).values)
            
        #     # Access the internal fitted model from inside your helper function
        #     # NOTE: fit_and_get_robust_weights needs to return (weights, model) 
        #     # for this visualization to work fully. 
        #     # Alternatively, re-fit locally just for the plot (inefficient but easy):
            
        #     # Re-creating model state just for visualization clarity:
        #     debug_model = BaggingRegressor(
        #         estimator=Ridge(alpha=1, fit_intercept=False),
        #         n_estimators=10, max_features=0.8, bootstrap=True, oob_score=True, random_state=123
        #     )
        #     try:
        #         debug_model.fit(data, label)
        #     except:
        #         pass # Visualization might fail on degenerate data

        #     _visualize_local_regression_robust(
        #         center_cell_name=cell,
        #         neighbor_names=neighbors,
        #         all_coords=all_coords,
        #         global_cell_names=ad_pop.obs_names,
        #         tf_data_df=data,
        #         metagene_label_array=label,
        #         target_mg_idx=target_metagene,
        #         model=debug_model,     # Pass the model
        #         final_weight_vector=weights # Pass the result
        #     )
        
        # retdf[cell] = weights.reindex(tfs, fill_value=0.0).values

        # Attempt 1
        # model = BaggingRegressor(
        #     estimator=Ridge(
        #         alpha=1,
        #         solver="auto",
        #         random_state=123,
        #         fit_intercept=False,
        #     ),
        #     n_estimators=10,
        #     bootstrap=True,
        #     max_features=0.8,
        #     verbose=False,
        #     random_state=123,
        # )
        # model.fit(data, label)
        
        # # # ans = _get_coef_matrix(model, tfs).mean(axis=0)

        # # coef_df = _get_coef_matrix(model, tfs)
        # # mean = coef_df.mean(axis=0)
        # # std = coef_df.std(axis=0)
        
        # # # Add a small epsilon to avoid division by zero
        # # epsilon = 1e-6 
        # # ans = mean / (std + epsilon)
        # # retdf[cell] = ans

        # stats = get_ensemble_stats(model, tfs)

        # # 1. Start with the conservative lower bound of magnitude (from Quantiles)
        # # If the 5th percentile is negative and 95th is positive, this will span 0.
        # # We look for where the *entire* confidence interval is away from 0.
        # lower = stats["lower_ci"]
        # upper = stats["upper_ci"]
        
        # # Check if CI excludes zero
        # significant_mask = (lower > 0) | (upper < 0)
        
        # # 2. Get the Median Effect (more robust than Mean)
        # median_effect = stats["raw_coefficients"].median(axis=0)
        
        # # 3. Apply Penalties
        # # Only keep if significant, and scale by how often it was selected
        # final_weight = median_effect * significant_mask * stats["selection_frequency"]
        
        # # Save this
        # retdf[cell] = final_weight

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

def fit_and_get_robust_weights(data, label, tfs):
    """
    Fits BaggingRegressor and returns weights penalized for 
    instability, poor model fit, and sparsity.
    """
    # 1. Initialize Bagging with OOB Score enabled
    # We must use oob_score=True to get the generalization metric
    model = BaggingRegressor(
        estimator=Ridge(
            alpha=1, 
            solver="auto", 
            random_state=123, 
            fit_intercept=False
        ),
        n_estimators=10, 
        bootstrap=True, 
        max_features=0.8, 
        verbose=False, 
        random_state=123,
        oob_score=True  # <--- CRITICAL FOR YOUR REQUEST
    )
    
    # 2. Fit the model
    # Note: If n_samples is very small (< 20), OOB might warn/fail. 
    # In that case, we assume OOB=0 (untrustworthy).
    try:
        model.fit(data, label)
        # Clip OOB score at 0. Negative R^2 means "worse than guessing mean"
        model_confidence = max(0, model.oob_score_)
    except Exception:
        model_confidence = 0.0

    # If the model is trash, return zeros immediately
    if model_confidence < 1e-3:
        return pd.Series(0.0, index=tfs)

    # 3. Get Coefficient Stats
    coef_df = _get_coef_matrix(model, tfs)
    mu = coef_df.mean(axis=0)
    sigma = coef_df.std(axis=0)
    
    # 4. Calculate "Robust Effect" (One-Standard-Error Rule)
    # This specifically kills TFs that are active in only a few cells
    # because their coefficients will bounce between High and 0.0 across bags.
    robust_effect = mu.abs() - sigma
    robust_effect[robust_effect < 0] = 0
    robust_effect = robust_effect * np.sign(mu)
    
    # 5. Calculate "Support" (Explicit Sparsity Penalty)
    # Fraction of neighbors where TF expression > 0
    # We use sqrt to penalize extreme sparsity but not punish 50% vs 100% too hard.
    support = (data > 0).mean(axis=0)
    sparsity_penalty = np.sqrt(support)

    # 6. Combine
    # Weight = (Stable Effect) * (How good is the model?) * (Do enough cells have it?)
    final_weights = robust_effect * model_confidence * sparsity_penalty
    
    return final_weights

# def get_metagene_edges_window(
#     ad_ex,
#     ad_motif,
#     target_metagene,
#     ad_pop,
#     num_hops=1,
#     cluster_id=None,
#     num_within=50,
#     num_total=100,
#     cluster_only=False,
# ):

#     tfs = ad_motif.var_names
#     retdf = pd.DataFrame(index=tfs, columns=ad_ex.obs_names)

#     # Precompute kNN indices if we'll need them
#     nn_indices = None
#     if (cluster_id is not None) or cluster_only:
#         nn_indices = get_nearest_neighbors(ad_pop)

#     # Convenience: cluster series (if provided)
#     cluster_series = ad_ex.obs[cluster_id] if cluster_id is not None else None

#     for i, cell in enumerate(ad_ex.obs_names):
#         # Choose neighbor mask logic
#         if cluster_only:
#             # Require cluster_id to be defined for cluster_only behavior
#             # (fall back to error if absent to prevent silent misuse)
#             if cluster_id is None:
#                 raise ValueError("cluster_only=True requires a valid cluster_id.")
#             neighbors_bool = get_neighbors_cluster(
#                 ad_pop=ad_pop,
#                 i=i,
#                 nn_indices=nn_indices,
#                 cell_type=cluster_series.loc[cell],
#                 cluster_id=cluster_id,
#             )
#         else:
#             if cluster_id is None:
#                 neighbors_bool = get_nhop_neighbors(ad_pop, i, num_hops=num_hops)
#             else:
#                 neighbors_bool = get_neighbors(
#                     ad_pop=ad_pop,
#                     i=i,
#                     nn_indices=nn_indices,
#                     cell_type=cluster_series.loc[cell],
#                     cluster_id=cluster_id,
#                     num_within=num_within,
#                     num_total=num_total,
#                 )
#         neighbors = ad_ex.obs_names[neighbors_bool]
#         if len(neighbors) == 0:
#             retdf[cell] = np.zeros((len(tfs), 1))
#             continue
#         data = ad_motif[neighbors, tfs].to_df()
#         # label = ad_pop.obsm['normalized_X'][neighbors_bool,target_metagene]
#         label = ad_pop.obsm["X"][neighbors_bool, target_metagene]
#         model = BaggingRegressor(
#             estimator=Ridge(
#                 alpha=1,
#                 solver="auto",
#                 random_state=123,
#             ),
#             n_estimators=10,
#             bootstrap=True,
#             max_features=0.8,
#             verbose=False,
#             random_state=123,
#         )
#         model.fit(data, label)
#         ans = _get_coef_matrix(model, tfs).mean(axis=0)
#         retdf[cell] = ans
#     return retdf.T


def get_metagene_edges_sliding(
    ad_ex,
    ad_motif,
    target_metagene,
    ad_pop,
    num_hops=1,
    cluster_id=None,
    num_within=50,
    num_total=100,
    cluster_only=False,
    verbose=False,
):
    tfs = ad_motif.var_names
    n_tfs = len(tfs)
    n_cells = len(ad_ex.obs_names)
    
    # 1. Prepare Output Array
    retdf = pd.DataFrame(index=tfs, columns=ad_ex.obs_names, dtype=float)

    # 2. Initialize Rolling Regression
    # We maintain ONE model instance that evolves as we loop
    roller = RollingRidge(n_features=n_tfs, alpha=1.0)
    
    # Track the set of neighbor indices from the PREVIOUS iteration
    prev_neighbors_set = set()

    # 3. Precompute kNN / Clusters (Same as before)
    nn_indices = None
    if (cluster_id is not None) or cluster_only:
        nn_indices = get_nearest_neighbors(ad_pop)

    cluster_series = ad_ex.obs[cluster_id] if cluster_id is not None else None

    # --- MAIN SLIDING LOOP ---
    for i, cell in enumerate(ad_ex.obs_names):
        
        # A. Determine Neighbors (Indices)
        # --------------------------------
        if cluster_only:
             neighbors_bool = get_neighbors_cluster(
                ad_pop=ad_pop, i=i, nn_indices=nn_indices,
                cell_type=cluster_series.iloc[i], cluster_id=cluster_id
            )
        else:
            if cluster_id is None:
                neighbors_bool = get_nhop_neighbors(ad_pop, i, num_hops=num_hops)
            else:
                neighbors_bool = get_neighbors(
                    ad_pop=ad_pop, i=i, nn_indices=nn_indices,
                    cell_type=cluster_series.iloc[i], cluster_id=cluster_id,
                    num_within=num_within, num_total=num_total
                )
        
        # Convert boolean mask to integer indices efficiently
        current_neighbors_idx = np.where(neighbors_bool)[0]
        current_neighbors_set = set(current_neighbors_idx)
        
        # B. Calculate Diff (Who entered? Who left?)
        # ------------------------------------------
        # Indices to Add: In Current but not Previous
        idx_to_add = list(current_neighbors_set - prev_neighbors_set)
        
        # Indices to Remove: In Previous but not Current
        idx_to_remove = list(prev_neighbors_set - current_neighbors_set)
        
        # C. Fetch Data for the Difference
        # --------------------------------
        # Note: We use ad_motif.X directly with integer slicing for speed
        if idx_to_add:
            X_add = ad_motif.X[idx_to_add, :]
            y_add = ad_pop.obsm["X"][idx_to_add, target_metagene]
        else:
            X_add = np.empty((0, n_tfs))
            y_add = np.empty((0,))

        if idx_to_remove:
            X_remove = ad_motif.X[idx_to_remove, :]
            y_remove = ad_pop.obsm["X"][idx_to_remove, target_metagene]
        else:
            X_remove = np.empty((0, n_tfs))
            y_remove = np.empty((0,))
            
        # D. Update Model State
        # ---------------------
        roller.update(X_add, y_add, X_remove, y_remove)
        
        # E. Solve (No re-fitting!)
        # -----------------------
        weights = roller.fit_and_get_robust_weights()
        retdf[cell] = weights
        
        # F. Update history for next iteration
        prev_neighbors_set = current_neighbors_set

    return retdf.T


def get_gene_edges_by_cluster_only(
    ad_ex,
    ad_motif,
    target_gene,            # gene name (str) or integer index
    *,
    cluster_id: str,        # REQUIRED: column in ad_ex.obs defining clusters
    alpha: float = 1.0,
    n_estimators: int = 10,
    max_features: float = 0.8,
):
    """
    Compute TF->gene weights once per cluster.

    Returns
    -------
    pd.DataFrame
        index = cluster labels
        columns = TFs
        values = TF coefficients predicting the target gene within that cluster
    """
    # Resolve gene name
    if isinstance(target_gene, int):
        gname = ad_ex.var_names[target_gene]
    else:
        gname = str(target_gene)
        if gname not in ad_ex.var_names:
            raise ValueError(f"Gene '{gname}' not found in ad_ex.var_names")

    # Checks
    if cluster_id not in ad_ex.obs.columns:
        raise ValueError(f"cluster_id '{cluster_id}' not found in ad_ex.obs")
    if ad_ex.n_obs != ad_motif.n_obs:
        raise ValueError("ad_ex and ad_motif must share the same cells (obs).")

    tfs = list(ad_motif.var_names)

    # Prepare output: one row per cluster
    out = pd.DataFrame(index=[], columns=tfs, dtype=float)

    clusters = ad_ex.obs[cluster_id].astype(str)
    for clabel, idx in clusters.groupby(clusters).groups.items():
        idx = list(idx)  # LABELS (cell IDs)

        # Features and target within this cluster
        X = ad_motif[idx, tfs].to_df()
        y = ad_ex[idx, gname].X
        try:
            y = np.asarray(y).ravel()
        except Exception:
            y = y.A.ravel()  # sparse -> dense

        # Fit once per cluster
        model = BaggingRegressor(
            estimator=Ridge(alpha=alpha, solver="auto", random_state=123),
            n_estimators=n_estimators,
            bootstrap=True,
            max_features=max_features,
            verbose=False,
            random_state=123,
        )
        model.fit(X, y)

        # Average TF weights across bags → a single TF vector for this cluster
        coef_series = _get_coef_matrix(model, tfs).mean(axis=0)  # pd.Series

        # Store under the cluster label
        out.loc[clabel] = coef_series.values

    return out


def get_gene_edges_window_cluster_only(
    ad_ex,
    ad_motif,
    target_gene,        # gene name (str) or integer index
    ad_pop,             # kept for signature parity; not used here
    *,
    cluster_id: str,    # REQUIRED: column in ad_ex.obs defining clusters
    alpha: float = 1.0,
    n_estimators: int = 10,
    max_features: float = 0.8,
):
    """
    Cluster-aggregated TF->gene weights.

    For each cluster (defined by `cluster_id`), fit a single Bagging(Ridge) model
    predicting the expression of `target_gene` from TF activities, using all cells
    in that cluster as the training set. Then assign the *same* TF-weight vector
    to every cell in that cluster.

    Returns
    -------
    pd.DataFrame
        (cells x TFs) with identical rows for cells within the same cluster.
    """

    # Resolve gene name
    if isinstance(target_gene, int):
        gname = ad_ex.var_names[target_gene]
    else:
        gname = str(target_gene)
        if gname not in ad_ex.var_names:
            raise ValueError(f"Gene '{gname}' not found in ad_ex.var_names")

    # Basic checks
    if cluster_id not in ad_ex.obs.columns:
        raise ValueError(f"cluster_id '{cluster_id}' not found in ad_ex.obs")
    if ad_ex.n_obs != ad_motif.n_obs:
        raise ValueError("ad_ex and ad_motif must share the same cells (obs).")

    # Setup
    tfs = list(ad_motif.var_names)
    # We'll fill cell x TFs (same as original function's return shape)
    out = pd.DataFrame(index=ad_ex.obs_names, columns=tfs, dtype=float)

    clusters = ad_ex.obs[cluster_id].astype(str)
    # Group once; iterate clusters instead of cells
    for clabel, idx in clusters.groupby(clusters).groups.items():
        idx = np.asarray(list(idx))

        # Features: TF activities for cells in this cluster (cells x TFs)
        X = ad_motif[idx, tfs].to_df()

        # Target: expression vector for the target gene in these cells (length = n_cells_cluster)
        y = ad_ex[idx, gname].X
        try:
            y = np.asarray(y).ravel()
        except Exception:
            y = y.A.ravel()  # if sparse

        # Fit once per cluster
        model = BaggingRegressor(
            estimator=Ridge(alpha=alpha, solver="auto", random_state=123),
            n_estimators=n_estimators,
            bootstrap=True,
            max_features=max_features,
            verbose=False,
            random_state=123,
        )
        model.fit(X, y)

        # Extract averaged TF weights (shape: TFs,)
        coef_series = _get_coef_matrix(model, tfs).mean(axis=0)  # pd.Series indexed by TFs

        # Assign the same vector to all cells in the cluster
        out.loc[idx, :] = coef_series.values

    # Return (cells x TFs) like the original .T return (which was cells x TFs)
    return out




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
    adj = ad.obsp["adjacency_matrix"]
    
    # 1. Create a sparse vector representing the starting cell
    # shape: (1, n_cells)
    n_cells = adj.shape[0]
    current_shell = sparse.lil_matrix((1, n_cells)) # LIL is fast for setting items
    current_shell[0, cellidx] = 1
    current_shell = current_shell.tocsr() # CSR is fast for math
    
    # This will hold the union of all neighbors found so far
    # (Initialize with the starting cell, or set to empty if you don't want the center included)
    accumulated_neighbors = current_shell.copy()

    # 2. Iteratively expand
    for _ in range(num_hops):
        # Matrix Multiplication is the key here:
        # It says: "For every cell I am currently touching, grab their neighbors"
        current_shell = current_shell @ adj
        
        # Add new neighbors to our total set
        accumulated_neighbors = accumulated_neighbors + current_shell

    # 3. Convert to boolean array
    # Any cell with a value > 0 was visited
    visited_mask = np.asarray(accumulated_neighbors.todense()).flatten() > 0
    
    # Optional: If you strictly want "neighbors" and not the cell itself:
    # visited_mask[cellidx] = False
    
    return visited_mask


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


# def run_all_perturbations(
#     pop,
#     ad_tfs,
#     ad_edges,
#     K=16,
#     multiplier=1,
#     useX=False,
#     target_clusters=10,
#     get_leiden=False,
# ):
#     new_columns = [[] for d in pop.datasets]
#     perturbed_datasets = [d.copy() for d in pop.datasets]
#     for d in perturbed_datasets:
#         for name in ["Sigma_x_inv", "popari_hyperparameters", "losses", "sigma_yx"]:
#             if name in d.uns:
#                 del d.uns[name]
#         if "adjacency_list" in d.obs.columns:
#             del d.obs["adjacency_list"]
#     for pop_ad, ad_tf, ad_edge, ite in zip(
#         perturbed_datasets,
#         ad_tfs,
#         ad_edges,
#         range(len(pop.datasets)),
#     ):
#         tfs = ad_tf.var_names
#         # cluster_changes = {}
#         for it, tf in tqdm(enumerate(tfs)):
#             in_silico_perturb(
#                 pop_ad,
#                 ad_tf,
#                 ad_edge,
#                 tf,
#                 K=K,
#                 multiplier=multiplier,
#                 useX=useX,
#             )
#         # tl.leiden(pop, use_rep=f"X_{tf}_dropout", target_clusters=10)
#     if get_leiden == True:
#         for tf in tfs:
#             tl.leiden(
#                 pop,
#                 use_rep=f"normalized_X_{tf}_dropout",
#                 target_clusters=target_clusters,
#             )
#             j = target_clusters
#             len_satisfies = test_all_true(
#                 [
#                     len(d.obs["leiden"].unique()) <= target_clusters
#                     for d in pop.datasets
#                 ],
#             )
#             while len_satisfies == False:
#                 j -= 1
#                 # tl.leiden(pop, use_rep=f"X_{tf}_dropout", target_clusters=j)
#                 tl.leiden(pop, use_rep=f"normalized_X_{tf}_dropout", target_clusters=j)
#                 len_satisfies = test_all_true(
#                     [
#                         len(d.obs["leiden"].unique()) <= target_clusters
#                         for d in pop.datasets
#                     ],
#                 )
#             for ite, pop_ad in enumerate(pop.datasets):
#                 new_columns[ite].append(align_leiden(pop_ad, tf))

#         tfcolumns = [f"leiden_{tf}_dropout" for tf in tfs]
#         new_columns_pds = [
#             pd.DataFrame(
#                 new_columns[i],
#                 index=tfcolumns,
#                 columns=pop.datasets[i].obs_names,
#             ).T
#             for i in range(len(pop.datasets))
#         ]
#         for d, ncpd in zip(pop.datasets, new_columns_pds):
#             d.obs = d.obs.join(ncpd)
#     return perturbed_datasets


# ---------- Helper: assemble cluster→(genes×TFs) weights from gene-mode edge AnnData ----------
def _stack_gene_weights_by_cluster(ad_edge):
    """
    Build a dict mapping cluster_label -> W (genes x TFs) from a gene-mode edge AnnData.

    Returns
    -------
    cluster_to_W : dict[str, np.ndarray]
        For each cluster label (row in ad_edge.obs_names), a dense array of shape (n_genes, n_tfs).
    genes_list : list[str]
        Order of genes corresponding to the rows of W (taken from ad_edge.uns["M_index_to_gene"]).
    tfs_list : list[str]
        Order of TFs corresponding to the columns of W (ad_edge.var_names).
    """
    if ad_edge.uns.get("edge_target_type") != "gene":
        raise ValueError("Expected gene-mode edge weights with uns['edge_target_type'] == 'gene'.")

    # Genes are stored as a list under this key (string names; indices are 0..n_genes-1)
    mig = ad_edge.uns.get("M_index_to_gene", None)
    if mig is None:
        # Fall back to number of layers; create generic names
        m_keys = [k for k in ad_edge.layers.keys() if k.startswith("M_")]
        m_keys_sorted = sorted(m_keys, key=lambda s: int(s.split("_", 1)[1]))
        genes_list = [f"gene_{int(k.split('_',1)[1])}" for k in m_keys_sorted]
    elif isinstance(mig, dict):
        # keys may be "0","1",... or 0,1,...
        try:
            # sort by numeric key
            genes_list = [mig[str(k)] for k in sorted(map(int, mig.keys()))]
        except Exception:
            # if keys aren’t numeric, just sort by key as strings
            genes_list = [mig[k] for k in sorted(mig.keys())]
    else:
        # list/tuple/Index/ndarray-like
        genes_list = list(mig)
    tfs_list = list(ad_edge.var_names)
    clusters = list(ad_edge.obs_names)

    n_clusters = len(clusters)
    n_genes = len(genes_list)
    n_tfs = len(tfs_list)

    # Stack layers [M_0, M_1, ..., M_{n_genes-1}] each with shape (n_clusters x n_tfs)
    # We’ll build a tensor T of shape (n_genes, n_clusters, n_tfs) to allow fast slicing by cluster.
    T = np.empty((n_genes, n_clusters, n_tfs), dtype=float)
    for k in range(n_genes):
        layer = ad_edge.layers[f"M_{k}"]
        # layer may be sparse; ensure dense
        L = layer.A if hasattr(layer, "A") else np.asarray(layer)
        T[k] = L  # (n_clusters x n_tfs)

    # Convert to dict: cluster_label -> (n_genes x n_tfs)
    cluster_to_W = {
        clusters[c]: T[:, c, :]  # (n_genes x n_tfs)
        for c in range(n_clusters)
    }
    return cluster_to_W, genes_list, tfs_list


# ---------- Helper: apply TF knockout in gene-mode (cluster-specific weights) ----------
def _in_silico_perturb_gene_mode(
    ad_pop,        # cells x genes (AnnData)
    ad_tf,         # cells x TFs   (AnnData)
    ad_edge,       # clusters x TFs, layers M_k = clusters x TFs (gene-mode Edges)
    tf: str,
    *,
    K: int = 16,               # PCA components to save as embedding
    multiplier: float = 1.0,
    cluster_id: Optional[str] = None,  # if None, taken from ad_edge.uns["cluster_id"]
):
    """
    Modify gene expression based on a TF knockout using cluster-specific TF→gene weights,
    then compute a K-dim PCA embedding and store it like the metagene pipeline:
      - ad_pop.obsm[f"X_{tf}_dropout"] (K-dim PCA)
      - ad_pop.obsm[f"normalized_X_{tf}_dropout"] (z-scored PCA)
    """
    # Resolve cluster_id and build weights
    if cluster_id is None:
        cluster_id = ad_edge.uns.get("cluster_id", None)
    if cluster_id is None:
        raise ValueError("cluster_id must be provided either as argument or in ad_edge.uns['cluster_id'].")

    cluster_to_W, genes_list, tfs_list = _stack_gene_weights_by_cluster(ad_edge)

    # Check TF present in both ad_tf and ad_edge
    if tf not in ad_tf.var_names:
        raise ValueError(f"TF '{tf}' not found in ad_tf.var_names")
    if tf not in tfs_list:
        raise ValueError(f"TF '{tf}' not found in edge weights TF list")
    tf_idx = tfs_list.index(tf)

    # Build mapping from genes_list (edge order) to expression matrix columns
    # Only update genes that exist in ad_pop.var_names.
    # print(ad_pop.var_names)
    # print(genes_list)
    gene_to_expr_idx = {}
    for g in genes_list:
        if g in ad_pop.var_names:
            gene_to_expr_idx[g] = ad_pop.var_names.get_loc(g)
    if not gene_to_expr_idx:
        raise ValueError("None of the edge-weight genes are present in ad_pop.var_names")

    # Start from current expression (dense)
    X_expr = ad_pop.X
    X_expr = X_expr.A if issparse(X_expr) else np.asarray(X_expr, dtype=float)
    n_cells = X_expr.shape[0]

    # Cluster labels per cell
    if cluster_id not in ad_pop.obs.columns:
        raise ValueError(f"cluster_id '{cluster_id}' not found in ad_pop.obs")
    cell_clusters = ad_pop.obs[cluster_id].astype(str).values

    # TF activity vector for this TF (cells,)
    a = ad_tf[:, tf].X
    a = a.A.ravel() if issparse(a) else np.asarray(a).ravel()

    # For each cluster, apply the outer-product update:
    # ΔX[cells_in_cluster, gene_idx] -= multiplier * a[cells] * W_c[gene, tf_idx]
    # Vectorized by cluster.
    # Prebuild the list of expr column indices (in the same order as genes_list),
    # and a mask for which genes exist in this dataset.
    genes_exist_mask = np.array([g in gene_to_expr_idx for g in genes_list], dtype=bool)
    expr_cols = np.array([gene_to_expr_idx[g] if g in gene_to_expr_idx else -1 for g in genes_list])

    # Iterate clusters appearing in the dataset (and present in edge weights)
    for c_label in np.unique(cell_clusters):
        if c_label not in cluster_to_W:
            # skip clusters not present in edge weights
            continue

        cell_mask = (cell_clusters == c_label)
        if not np.any(cell_mask):
            continue

        W_c = cluster_to_W[c_label]           # (n_genes x n_tfs)
        w_vec_all = W_c[:, tf_idx]            # (n_genes,)
        # keep only genes that exist in this dataset
        w_vec = w_vec_all[genes_exist_mask]   # (n_kept_genes,)
        cols = expr_cols[genes_exist_mask]    # (n_kept_genes,)

        if w_vec.size == 0:
            continue

        # Outer product: (n_cells_c x 1) @ (1 x n_kept_genes) -> (n_cells_c x n_kept_genes)
        a_c = a[cell_mask][:, None]           # (n_cells_c x 1)
        delta = multiplier * (a_c @ w_vec[None, :])  # dense

        # Subtract from expression
        X_expr[np.ix_(cell_mask, cols)] -= delta

    # Save the modified expression as a layer (optional, but handy)
    # ad_pop.layers[f"expr_{tf}_dropout"] = X_expr

    # Compute PCA (K comps) from the modified expression and store in obsm
    # We do this in a tiny temporary AnnData to avoid mutating ad_pop.X.
    _tmp = sc.AnnData(X_expr, obs=ad_pop.obs.copy(), var=ad_pop.var.copy())
    # You can add scaling/normalization here if desired before PCA.
    sc.tl.pca(_tmp, n_comps=K, use_highly_variable=False, svd_solver="arpack")
    ad_pop.obsm[f"X_{tf}_dropout"] = _tmp.obsm["X_pca"]

    # Normalize (z-score over cells, feature-wise), fill NaNs -> 0 (match previous behavior)
    norm_embed = zscore(ad_pop.obsm[f"X_{tf}_dropout"], axis=0)
    norm_embed = np.nan_to_num(norm_embed, nan=0.0, posinf=0.0, neginf=0.0)
    ad_pop.obsm[f"normalized_X_{tf}_dropout"] = norm_embed

    # Optional: recompute neighbors on the normalized embedding (kept from old pipeline)
    sc.pp.neighbors(ad_pop, use_rep=f"normalized_X_{tf}_dropout")


# ---------- Original metagene helper retained (unchanged) ----------
def _in_silico_perturb_metagene_mode(
    ad_pop, ad_tf, ad_edge, tf, K=16, multiplier=1.0, useX=False
):
    if useX is False:
        dropout_X = ad_pop.obsm["normalized_X"].copy()
    else:
        dropout_X = ad_pop.obsm["X"].copy()

    for metagene in range(K):
        
        perturbation = ad_tf[:, tf].X * ad_edge[:, tf].layers[f"M_{metagene}"]
        perturbation *= multiplier
        dropout_X[:, metagene] = dropout_X[:, metagene] - perturbation.flatten()

    ad_pop.obsm[f"X_{tf}_dropout"] = dropout_X

    normalized_embeddings = zscore(ad_pop.obsm[f"X_{tf}_dropout"])
    normalized_embeddings[np.isnan(normalized_embeddings)] = 0
    ad_pop.obsm[f"normalized_X_{tf}_dropout"] = normalized_embeddings
    sc.pp.neighbors(ad_pop, use_rep=f"normalized_X_{tf}_dropout")


# ---------- Main: works for both metagene & gene modes ----------
def run_all_perturbations(
    pop,
    ad_tfs,
    ad_edges,
    ad_rnas,
    K: int = 16,           # metagene dim OR PCA dim (gene-mode)
    multiplier: float = 1.0,
    useX: bool = False,    # only used in metagene mode
    target_clusters: int = 10,
    get_leiden: bool = False,
):
    """
    If ad_edge.uns['edge_target_type'] == 'gene', run gene-mode:
      - modify gene expression via cluster-specific TF->gene weights,
      - compute PCA (K components) per TF knockout,
      - store results under 'X_{tf}_dropout' and 'normalized_X_{tf}_dropout'.

    Otherwise (metagene mode), use the original embedding-based perturbation.
    """
    new_columns = [[] for _ in pop.datasets]
    perturbed_datasets = [d.copy() for d in pop.datasets]

    # Clean up heavy stuff
    for d in perturbed_datasets:
        for name in ["Sigma_x_inv", "popari_hyperparameters", "losses", "sigma_yx"]:
            if name in d.uns:
                del d.uns[name]
        if "adjacency_list" in d.obs.columns:
            del d.obs["adjacency_list"]

    # Iterate datasets
    for pop_ad, ad_tf, ad_edge, ite, ad_rna in zip(
        perturbed_datasets, ad_tfs, ad_edges, range(len(pop.datasets)), ad_rnas
    ):
        tfs = list(ad_tf.var_names)
        mode = ad_edge.uns.get("edge_target_type", "metagene")

        # main loop over TFs
        for _, tf in tqdm(enumerate(tfs), total=len(tfs)):
            if mode == "gene":
                _in_silico_perturb_gene_mode(
                    ad_pop=ad_rna,
                    ad_tf=ad_tf,
                    ad_edge=ad_edge,
                    tf=tf,
                    K=K,  # PCA components to match metagene dim (e.g., 16)
                    multiplier=multiplier,
                    cluster_id=ad_edge.uns.get("cluster_id", None),
                )
            else:
                _in_silico_perturb_metagene_mode(
                    ad_pop=pop_ad,
                    ad_tf=ad_tf,
                    ad_edge=ad_edge,
                    tf=tf,
                    K=K,
                    multiplier=multiplier,
                    useX=useX,
                )

    # Optional Leiden pass remains the same (works on normalized_X_{tf}_dropout)
    if get_leiden:
        for tf in tfs:
            tl.leiden(pop, use_rep=f"normalized_X_{tf}_dropout", target_clusters=target_clusters)
            j = target_clusters
            len_satisfies = test_all_true(
                [len(d.obs["leiden"].unique()) <= target_clusters for d in pop.datasets]
            )
            while len_satisfies is False:
                j -= 1
                tl.leiden(pop, use_rep=f"normalized_X_{tf}_dropout", target_clusters=j)
                len_satisfies = test_all_true(
                    [len(d.obs["leiden"].unique()) <= target_clusters for d in pop.datasets]
                )
            for ite, pop_ad in enumerate(pop.datasets):
                new_columns[ite].append(align_leiden(pop_ad, tf))

        tfcolumns = [f"leiden_{tf}_dropout" for tf in tfs]
        new_columns_pds = [
            pd.DataFrame(new_columns[i], index=tfcolumns, columns=pop.datasets[i].obs_names).T
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


def _get_coef_matrix(ensemble_model, feature_names):
    feature_names = np.array(feature_names)
    n_estimater = len(ensemble_model.estimators_features_)
    coef_list = [
        pd.Series(
            ensemble_model.estimators_[i].coef_,
            index=feature_names[ensemble_model.estimators_features_[i]],
        )
        for i in range(n_estimater)
    ]

    # Fill NaN with 0.0 because "not selected" implies "0 influence"
    coef_df = pd.concat(coef_list, axis=1, sort=False).transpose().fillna(0.0)

    return coef_df
# this is a function to extract coef information from sklearn ensemble_model.
# def _get_coef_matrix(ensemble_model, feature_names):
#     # ensemble_model: trained ensemble model. e.g. BaggingRegressor
#     # feature_names: list or numpy array of feature names. e.g. feature_names=X_train.columns
#     feature_names = np.array(feature_names)
#     n_estimater = len(ensemble_model.estimators_features_)
#     coef_list = [
#         pd.Series(
#             ensemble_model.estimators_[i].coef_,
#             index=feature_names[ensemble_model.estimators_features_[i]],
#         )
#         for i in range(n_estimater)
#     ]

#     coef_df = pd.concat(coef_list, axis=1, sort=False).transpose()

#     return coef_df


def get_ensemble_stats(ensemble_model, feature_names):
    feature_names = np.array(feature_names)
    n_estimators = len(ensemble_model.estimators_features_)
    
    # 1. Extract Coefficients (Slope) - What you had before
    coef_list = [
        pd.Series(
            ensemble_model.estimators_[i].coef_,
            index=feature_names[ensemble_model.estimators_features_[i]],
        )
        for i in range(n_estimators)
    ]
    # Cells x Features (FillNaN with 0.0)
    coef_df = pd.concat(coef_list, axis=1, sort=False).transpose().fillna(0.0)

    # 2. Extract Intercepts (Basal Expression)
    # Check if the estimator actually fitted an intercept
    if hasattr(ensemble_model.estimators_[0], 'intercept_'):
        intercepts = np.array([est.intercept_ for est in ensemble_model.estimators_])
    else:
        intercepts = np.zeros(n_estimators)

    # --- DERIVED STATISTICS ---

    # 3. Sign Stability (0.0 to 1.0)
    # 1.0 = All estimators agree on sign (all + or all -)
    # 0.0 = Perfect disagreement (half +, half -)
    # We use a small epsilon to ignore exact zeros
    signs = np.sign(coef_df[np.abs(coef_df) > 1e-9])
    sign_stability = np.abs(signs.mean(axis=0)) 

    # 4. Selection Frequency (0.0 to 1.0)
    # Fraction of estimators where coeff != 0
    selection_freq = (np.abs(coef_df) > 1e-9).mean(axis=0)

    # 5. Non-Parametric Confidence Interval (Lower Bound)
    # The 5th percentile. If this crosses 0, the effect is likely not significant.
    lower_ci = coef_df.quantile(0.05, axis=0)
    upper_ci = coef_df.quantile(0.95, axis=0)

    return {
        "raw_coefficients": coef_df,       # (N_estimators x N_TFs)
        "raw_intercepts": intercepts,      # (N_estimators,)
        "sign_stability": sign_stability,  # (N_TFs,)
        "selection_frequency": selection_freq, # (N_TFs,)
        "lower_ci": lower_ci,              # (N_TFs,)
        "upper_ci": upper_ci               # (N_TFs,)
    }


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
