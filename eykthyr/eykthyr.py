import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union, Tuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from tqdm.notebook import tqdm
import torch
from popari import pl, tl
from popari.components import PopariDataset
from popari.io import save_anndata
from popari.model import Popari, load_trained_model
from scipy.sparse import spmatrix

from .embedding import Embedding
from .modified_VelocytoLoom_class import modified_VelocytoLoom
from .util import get_metagene_edges_window, run_all_perturbations, get_gene_edges_by_cluster_only


class Eykthyr(modified_VelocytoLoom):
    """Main class for inferring TF regulatory influences on spatial gene programs.

    Eykthyr integrates spatial transcriptomics (RNA-seq) with spatial
    chromatin accessibility (ATAC-seq) to:

    1. Learn low-dimensional *metagene* representations of spatial gene expression
       via the Popari model.
    2. Compute per-cell TF activity scores from ArchR peak/motif outputs.
    3. Infer cell-level TF → metagene regulatory edge weights using spatial
       sliding-window ridge regression.
    4. Simulate TF knockout perturbations and project the resulting expression
       shifts onto any 2-D embedding (spatial coordinates, UMAP, force-directed
       graph, etc.).
    5. Score each TF's developmental relevance by aligning its simulated flow
       with a pseudotime gradient.

    Typical usage::

        e = Eykthyr()
        e.set_RNA([adrna])
        e.preprocess_rna(make_plots=True)
        e.compute_metagenes()
        e.analyze_metagenes()
        e.compute_TF_activity(peak_tsvs=[...], archr_dataset_names=[...], motif_tsvs=[...])
        e.compute_TF_metagene_weights()
        e.run_all_perturbations()
        e.save_anndata('results.h5ad')

    Attributes:
        RNA (list[AnnData]): Preprocessed spatial RNA AnnData objects, one per
            dataset.  Cells must carry spatial coordinates in ``.obsm['spatial']``.
        popari (Popari | None): Fitted Popari model holding the metagene factors
            and spatial affinities.
        TF (list[AnnData]): Per-dataset TF activity AnnData objects (cells × TFs).
        edge_weights (list[AnnData]): Per-dataset AnnData objects storing
            TF → metagene edge weights in layers ``M_0`` … ``M_{K-1}``.
        perturbed_X (list[AnnData]): Simulated post-perturbation expression
            datasets, one per dataset.
        datasetnames (list[str]): Human-readable names for each dataset.
        cluster_annotation (list[str]): ``obs`` keys to overlay on UMAPs.
        num_metagenes (int): Number of metagenes ``K`` used by Popari.
        embeddings (list[dict[str, Embedding]]): Per-dataset dicts mapping
            embedding name → :class:`~eykthyr.embedding.Embedding` instance.
    """

    def __init__(
        self,
        RNA: Optional[List[sc.AnnData]] = None,
        popari: Optional[Popari] = None,
        TF: Optional[List[sc.AnnData]] = None,
        edge_weights: Optional[List[sc.AnnData]] = None,
        perturbed_X: Optional[List[sc.AnnData]] = None,
        names: Optional[List[str]] = ["eykthyr_dataset"],
        cluster_annotation: Sequence[str] = [],
        num_metagenes: int = -1,
        embeddings: Optional[List[Dict[str, Embedding]]] = [],
    ):
        """Initialize an Eykthyr instance.

        All parameters are optional; a bare ``Eykthyr()`` is valid and you can
        populate fields later via the ``set_*`` helper methods.

        Parameters:
            RNA (list[AnnData] | None): Spatial RNA datasets.
            popari (Popari | None): Pre-trained Popari model.
            TF (list[AnnData] | None): TF activity datasets.
            edge_weights (list[AnnData] | None): GRN edge-weight datasets.
            perturbed_X (list[AnnData] | None): Simulated perturbed datasets.
            names (list[str]): Human-readable dataset names used internally by
                Popari and for file naming in :meth:`save_anndata`.
            cluster_annotation (Sequence[str]): ``obs`` keys to overlay when
                calling :meth:`preprocess_rna` or :meth:`analyze_metagenes` with
                ``make_plots=True``.
            num_metagenes (int): Pre-set ``K``; inferred automatically by
                :meth:`compute_metagenes`.
            embeddings (list[dict[str, Embedding]] | None): Pre-populated
                embedding containers; normally built by the plotting helpers.
        """

        self.RNA = RNA if RNA is not None else []
        self.popari = popari
        self.TF = TF if TF is not None else []
        self.edge_weights = edge_weights if edge_weights is not None else []
        self.perturbed_X = perturbed_X if perturbed_X is not None else []

        self.datasetnames = names
        self.rna_preprocessed = False

        self.cluster_annotation = cluster_annotation
        self.num_metagenes = num_metagenes
        self.embeddings = embeddings

    def preprocess_rna(
        self,
        make_plots: bool = False,
        cluster_annotation: Optional[Sequence[str]] = [],
    ):
        """Preprocesses RNA data by filtering genes and cells, normalizing, and
        log-transforming data.

        Parameters:
            make_plots (bool): If True, performs PCA, neighbors, UMAP, and Leiden clustering for visualization.
            cluster_annotation (Optional[Sequence[str]]): Optional list of cluster annotations to use for visualization.

        Returns:
            None

        """
        for RNA in self.RNA:
            if self.rna_preprocessed:
                print("RNA appears to already be preprocessed, doing nothing")
                return

            sc.pp.filter_genes(RNA, min_cells=5)
            sc.pp.filter_cells(RNA, min_counts=10)
            RNA.layers["raw"] = RNA.X
            sc.pp.normalize_total(RNA, inplace=True, target_sum=10000)
            sc.pp.log1p(RNA)
            if len(cluster_annotation) > 0:
                self.cluster_annotation = cluster_annotation
            if make_plots:
                sc.pp.pca(RNA)
                sc.pp.neighbors(RNA)
                sc.tl.umap(RNA)
                sc.tl.leiden(RNA, key_added="clusters")
                sc.pl.umap(RNA, color=["clusters"] + self.cluster_annotation)
        # subset to the common genes
        common_genes = set(self.RNA[0].var_names)
        for ad in self.RNA:
            common_genes.intersection_update(ad.var_names)

        # Convert common_genes back to a list
        common_genes = list(common_genes)
        self.RNA = [ad[:, common_genes].copy() for ad in self.RNA]
        self.rna_preprocessed = True

    def compute_metagenes(
        self,
        K: int = 16,
        lambda_Sigma_x_inv: float = 1e-4,
        torch_context: dict = dict(device="cuda:0", dtype=torch.float64),
        initial_iterations: int = 10,
        spatial_iterations: int = 200,
    ):
        """Computes metagenes using the Popari model with specified initial and
        spatial iterations.

        Parameters:
            K (int): Number of metagenes to compute.
            lambda_Sigma_x_inv (float): Regularization parameter for the Popari model.
            torch_context (dict): Device and data type for torch operations.
            initial_iterations (int): Number of initial iterations without spatial affinities.
            spatial_iterations (int): Number of iterations with spatial affinities.

        Returns:
            None

        """
        if not self.rna_preprocessed:
            print(
                "RNA appears to not be preprocessed. Please preprocess RNA using Eykthyr.preprocess_rna() or set Eykthyr.rna_preprocessed = True",
            )
            return

        self.num_metagenes = K
        popari_datasets = []
        for RNA, name in zip(self.RNA, self.datasetnames):
            popari_d = PopariDataset(RNA, name)
            popari_d.compute_spatial_neighbors()
            RNA.obs["adjacency_list"] = popari_d.obs["adjacency_list"]
            RNA.obsp["adjacency_matrix"] = popari_d.obsp["adjacency_matrix"]
            if isinstance(RNA.X, spmatrix):
                RNA.X = RNA.X.todense()
            RNA.X = np.asarray(RNA.X)
            altRNA = RNA.copy()
            if "X_diffmap" in altRNA.obsm.keys():
                del altRNA.obsm["X_diffmap"]
            if "X_pca" in altRNA.obsm.keys():
                del altRNA.obsm["X_pca"]
            popari_datasets.append(altRNA)
        print(popari_datasets)
        self.popari = Popari(
            K=K,
            replicate_names=self.datasetnames,
            datasets=popari_datasets,
            lambda_Sigma_x_inv=lambda_Sigma_x_inv,
            torch_context=torch_context,
            initial_context=torch_context,
            verbose=0,
        )

        for iteration in range(initial_iterations):
            self.popari.estimate_parameters(update_spatial_affinities=False)
            self.popari.estimate_weights(use_neighbors=False)

        for iteration in range(spatial_iterations):
            self.popari.estimate_parameters()
            self.popari.estimate_weights()

    def analyze_metagenes(
        self,
        num_leiden_clusters: int = 10,
    ):
        """Analyzes computed metagenes by performing Leiden clustering and UMAP
        visualization of embeddings.

        Parameters:
            num_leiden_clusters (int): Target number of clusters for Leiden clustering.

        Returns:
            None

        """
        if not self.popari:
            print("Popari has not been run. Please run compute_metagenes() first.")
            return

        tl.preprocess_embeddings(self.popari)
        tl.leiden(
            self.popari,
            use_rep="normalized_X",
            target_clusters=num_leiden_clusters,
        )

        for dataset in self.popari.datasets:
            sc.pp.neighbors(
                dataset,
                use_rep="normalized_X",
                key_added="norm_X_neighbors",
            )
            sc.tl.umap(dataset, neighbors_key="norm_X_neighbors")
            sc.pl.umap(dataset, color=["leiden"] + self.cluster_annotation)

    def save_anndata(
        self,
        dirpath: str,
    ):
        """Saves RNA, TF, edge weights, and perturbed_X AnnData objects to a
        specified directory.

        Parameters:
            dirpath (str): Path to the directory where AnnData objects will be saved.

        Returns:
            None

        """
        dirpath = Path(dirpath)
        path_without_extension = dirpath.parent / dirpath.stem
        path_without_extension.mkdir(exist_ok=True)

        for i, RNA in enumerate(self.RNA):
            RNA.uns["datasetname"] = self.datasetnames[i]
            RNA.uns["rna_preprocessed"] = self.rna_preprocessed
            RNA.uns["cluster_annotation"] = self.cluster_annotation
            RNA.uns["num_metagenes"] = self.num_metagenes
            if "adjacency_list" in RNA.obs.columns:
                del RNA.obs["adjacency_list"]
            RNA.write(f"{path_without_extension}/RNA_{i}.h5ad")

        if self.popari:# and not os.path.isfile(f"{path_without_extension}/popari.h5ad"):
            self.popari.save_results(f"{path_without_extension}/popari.h5ad")

        for i, TF in enumerate(self.TF):
            TF.write(f"{path_without_extension}/TF_{i}.h5ad")

        for i, edge_weight in enumerate(self.edge_weights):
            edge_weight.write(f"{path_without_extension}/edge_weights_{i}.h5ad")

        for i, perturbed_X in enumerate(self.perturbed_X):
            if "adjacency_list" in perturbed_X.obs.columns:
                del perturbed_X.obs["adjacency_list"]
            perturbed_X.write(f"{path_without_extension}/perturbed_X_{i}.h5ad")

    def set_RNA(
        self,
        RNA: List[sc.AnnData],
    ):
        """Sets the RNA datasets for the Eykthyr instance.

        Parameters:
            RNA (List[sc.AnnData]): List of RNA AnnData objects to assign to the instance.

        Returns:
            None

        """
        self.RNA = RNA

    def set_popari(
        self,
        popari: Popari,
    ):
        """Sets the Popari model instance for Eykthyr.

        Parameters:
            popari (Popari): A Popari model instance to use for metagene computation.

        Returns:
            None

        """
        self.popari = popari

    def set_TF(
        self,
        TF: List[sc.AnnData],
    ):
        """Sets the transcription factor (TF) activity datasets for Eykthyr.

        Parameters:
            TF (List[sc.AnnData]): List of AnnData objects with TF activity data.

        Returns:
            None

        """
        self.TF = TF

    def set_edge_weights(
        self,
        edge_weights: List[sc.AnnData],
    ):
        """Sets the edge weights datasets for gene regulatory network analysis
        in Eykthyr.

        Parameters:
            edge_weights (List[sc.AnnData]): List of AnnData objects representing GRN edge weights.

        Returns:
            None

        """
        self.edge_weights = edge_weights

    def set_perturbed_X(
        self,
        perturbed_X: List[sc.AnnData],
    ):
        """Sets the perturbed gene expression datasets for Eykthyr.

        Parameters:
            perturbed_X (List[sc.AnnData]): List of AnnData objects with perturbed gene expression data.

        Returns:
            None

        """
        self.perturbed_X = perturbed_X

    # Adjust the rest of the methods similarly

    def compute_TF_activity(
        self,
        peak_tsvs: List[str],
        archr_dataset_names: List[str],
        motif_tsvs: List[str],
        archr_suffix: str = "",
    ):
        """Compute per-cell TF activity scores from ArchR peak and motif outputs.

        Reads the peak-by-cell and motif-by-peak TSV files produced by ArchR,
        multiplies them to obtain a cell × TF activity matrix, normalizes each
        TF to [0, 1] range, and stores the result in ``self.TF``.

        Cells present in the RNA data but absent from the ArchR output are
        imputed with the mean TF activity across observed cells.

        Parameters:
            peak_tsvs (list[str]): Paths to the peak-by-cell count TSV files
                (one per dataset), as exported by ArchR's ``getMatrixFromProject``.
            archr_dataset_names (list[str]): ArchR project/sample names
                corresponding to each dataset.  Used to strip the sample-name
                prefix that ArchR appends to barcode column names.
            motif_tsvs (list[str]): Paths to the peak-by-motif binary TSV files
                (one per dataset), as exported by ArchR's ``getMatches``.
            archr_suffix (str): Optional suffix appended to each barcode after
                stripping the sample-name prefix.  Leave empty when barcodes
                already match ``RNA.obs_names``.

        Returns:
            None.  Populates ``self.TF`` with one AnnData per dataset, where
            ``.X`` is a cells × TFs float array of normalized TF activity scores.
        """

        if not isinstance(self.RNA, list):
            raise ValueError("self.RNA should be a list of AnnData objects.")

        if not (
            len(self.RNA)
            == len(peak_tsvs)
            == len(archr_dataset_names)
            == len(motif_tsvs)
        ):
            raise ValueError(
                "The lengths of RNA, peak_tsvs, archr_dataset_names, and motif_tsvs must match.",
            )

        self.TF = []

        for i, RNA in enumerate(self.RNA):
            peak_tsv = peak_tsvs[i]
            archr_dataset_name = archr_dataset_names[i]
            motif_tsv = motif_tsvs[i]

            archr_name_len = len(archr_dataset_name) + 1
            tfpeaks = pd.read_csv(peak_tsv, sep=" ")
            new_col_names = [
                f"{c[archr_name_len:-2]}{archr_suffix}" for c in tfpeaks.columns
            ]
            tfpeaks.rename(
                columns={c: new_c for c, new_c in zip(tfpeaks.columns, new_col_names)},
                inplace=True,
            )
            tfpeaks = tfpeaks.T

            tfmotifs = pd.read_csv(motif_tsv, sep=" ")
            tfmotifs.index = range(1, tfmotifs.shape[0] + 1)
            tfmotifs = tfmotifs.rename(
                columns={c: c.split("_")[0] for c in tfmotifs.columns},
            )
            tfmotifs = tfmotifs.astype(int)

            cellmotifs = sc.AnnData(tfpeaks.dot(tfmotifs))
            cellmotifs.layers["raw"] = cellmotifs.X

            sc.pp.normalize_total(cellmotifs, exclude_highly_expressed=True)
            sc.pp.scale(cellmotifs, zero_center=False)

            included = RNA.obs.index[RNA.obs.index.isin(cellmotifs.obs.index)]
            excluded = RNA.obs.index[~RNA.obs.index.isin(cellmotifs.obs.index)]
            cellmotifs = cellmotifs[included, :]
            tf_archr = cellmotifs.to_df()
            for exclude in excluded:
                tf_archr.loc[exclude, :] = tf_archr.mean(axis=0)
            tf_archr = tf_archr.rename(
                columns={c: c.split("_")[0] for c in tf_archr.columns},
            )
            tf_archr = tf_archr.loc[:, ~tf_archr.columns.duplicated()]
            tf_adata = sc.AnnData(tf_archr)
            tf_adata = tf_adata[RNA.obs_names, :]
            tf_adata.obsm["spatial"] = RNA.obsm["spatial"]
            tf_adata.layers["raw"] = tf_adata.X
            tf_adata.X -= tf_adata.X.min(axis=0)
            tf_adata.X /= tf_adata.X.max(axis=0)

            if (
                np.isnan(tf_adata.X).mean() * 100
            ) > 90:  # More than 90% of tf_adata.X is nan
                print(
                    "TF activity is mostly NaN. Are you sure you have the correct archr suffix",
                )
                print(
                    f"RNA obs name: {RNA.obs_names[0]}, archr obs name: {new_col_names[0]}",
                )
            self.TF.append(tf_adata)

    def compute_TF_metagene_weights(
        self,
        num_hops: int = 2,
        cluster_only: bool = False,
        cluster_id: str = None,
        verbose: bool = False,
        num_within: int = 50,
        num_total: int = 100,
        *,
        target_type: str = "metagene",
        genes: Optional[List[str]] = None,
    ):
        """Infer TF → metagene (or TF → gene) regulatory edge weights.

        For each target (metagene index or gene name), a spatial sliding-window
        ridge regression is run: for every cell, a neighborhood of spatially
        proximal cells is assembled and a ridge regression of TF activity against
        target expression is fitted.  The resulting regression coefficients become
        that cell's TF edge weights for the target.  Results are stored in
        ``self.edge_weights`` as an AnnData with layers ``M_0`` … ``M_{K-1}``.

        Parameters:
            num_hops (int): Spatial graph radius (in hops) used to build each
                cell's regression neighborhood. Default ``2``.
            cluster_only (bool): If ``True`` and ``cluster_id`` is set, restrict
                neighbors to cells in the same cluster as the focal cell. Default
                ``False``.
            cluster_id (str | None): ``obs`` key that stores cluster labels.
                Required when ``cluster_only=True`` or ``target_type='gene'``.
            verbose (bool): If ``True``, print regression diagnostics for the
                first metagene / gene. Default ``False``.
            num_within (int): Target number of same-cluster neighbors to include
                in each regression window (used when ``cluster_only=False`` to
                balance cluster composition). Default ``50``.
            num_total (int): Total neighborhood size for each regression window.
                Default ``100``.
            target_type (str): ``'metagene'`` (default) to regress TF activity
                against metagene embeddings, or ``'gene'`` to regress against raw
                gene expression (cluster-level).
            genes (list[str] | None): Explicit list of gene targets when
                ``target_type='gene'``.  If ``None``, all genes in RNA are used.

        Returns:
            None.  Populates ``self.edge_weights``.
        """
        if not self.popari or not self.TF:
            print("Popari/TF activity not computed.")
            return

        if len(self.TF) != len(self.popari.datasets):
            print("Number of TF datasets must match.")
            return

        if target_type not in {"metagene", "gene"}:
            raise ValueError("target_type must be 'metagene' or 'gene'")

        self.edge_weights = []
        self._gene_targets = [] if target_type == "gene" else None

        for i, TF, popdata, RNA in zip(
            range(len(self.TF)),
            self.TF,
            self.popari.datasets,
            self.RNA,
        ):
            # Targets
            if target_type == "metagene":
                target_ids = list(range(self.num_metagenes))
                num_targets = self.num_metagenes
            else:
                if genes is None:
                    gene_list = list(RNA.var_names)
                else:
                    missing = [g for g in genes if g not in RNA.var_names]
                    if missing:
                        raise ValueError(f"genes not found: {missing}")
                    gene_list = list(genes)
                target_ids = gene_list
                num_targets = len(gene_list)
                self._gene_targets.append(gene_list)
                if cluster_id is None:
                    raise ValueError("cluster_id required for gene target.")

            M_edges = []
            for k, target in tqdm(enumerate(target_ids), total=num_targets, position=0):
                if target_type == "metagene":
                    
                    # Only show plots for the first metagene to avoid spamming 
                    # (Or pass verbose=verbose to see samples for all MGs)
                    current_verbose = verbose and (k == 0) 
                    
                    # cell-level (as before)
                    edges = get_metagene_edges_window(
                        RNA, TF, target, popdata,
                        num_hops=num_hops,
                        cluster_only=cluster_only,
                        cluster_id=cluster_id,
                        num_within=num_within,
                        num_total=num_total,
                        verbose=current_verbose # <--- PASS FLAG
                    )
                    if current_verbose:
                        print(edges)
                    Madata = sc.AnnData(edges)
                    Madata = Madata[popdata.obs_names, :]
                else:
                    # cluster-level 
                    edges = get_gene_edges_by_cluster_only(
                        RNA, TF, target, cluster_id=cluster_id
                    )
                    Madata = sc.AnnData(
                        X=edges.values,
                        obs=pd.DataFrame(index=edges.index),     
                        var=pd.DataFrame(index=edges.columns),   
                    )

                M_edges.append(Madata)

            # Assemble (unchanged)
            self.edge_weights.append(M_edges[0])
            
            if target_type == "gene":
                self.num_metagenes = num_targets 

            for k in range(num_targets):
                self.edge_weights[i].layers[f"M_{k}"] = M_edges[k].X

            self.edge_weights[i].uns["edge_target_type"] = target_type
            if target_type == "gene":
                self.edge_weights[i].uns["edges_axis"] = "clusters" 
                self.edge_weights[i].uns["M_index_to_gene"] = list(map(str, target_ids)) 
                self.edge_weights[i].uns["cluster_id"] = cluster_id
    # def compute_TF_metagene_weights(
    #     self,
    #     num_hops: int = 2,
    #     cluster_only: bool = False,
    #     cluster_id: str = None,
    #     *,
    #     target_type: str = "metagene",       # "metagene" (default) or "gene"
    #     genes: Optional[List[str]] = None,   # Only used when target_type="gene"
    # ):
    #     """
    #     Computes TF edge weights.
    
    #     - 'metagene': original behavior (cell-level; aligned to popdata.obs_names).
    #     - 'gene': cluster-level outputs (rows=clusters, cols=TFs), packed into an AnnData
    #       with layers M_k; no spatial stored, no reindexing to cells.
    #     """
    #     if not self.popari or not self.TF:
    #         print(
    #             "Popari needs to be run first, please run compute_metagenes().\n"
    #             "TF activity also needs to be computed by ArchR. Please follow the jupyter notebook for\n"
    #             "preprocessing ATAC-seq data and then run compute_TF_activity().",
    #         )
    #         return
    
    #     if len(self.TF) != len(self.popari.datasets):
    #         print("Number of TF datasets must match the number of datasets in Popari.")
    #         return
    
    #     if target_type not in {"metagene", "gene"}:
    #         raise ValueError("target_type must be 'metagene' or 'gene'")
    
    #     self.edge_weights = []
    #     self._gene_targets = [] if target_type == "gene" else None
    
    #     for i, TF, popdata, RNA in zip(
    #         range(len(self.TF)),
    #         self.TF,
    #         self.popari.datasets,
    #         self.RNA,
    #     ):
    #         # Targets
    #         if target_type == "metagene":
    #             target_ids = list(range(self.num_metagenes))
    #             num_targets = self.num_metagenes
    #         else:
    #             if genes is None:
    #                 gene_list = list(RNA.var_names)
    #             else:
    #                 missing = [g for g in genes if g not in RNA.var_names]
    #                 if missing:
    #                     raise ValueError(f"genes not found in RNA.var_names: {missing}")
    #                 gene_list = list(genes)
    #             target_ids = gene_list
    #             num_targets = len(gene_list)
    #             self._gene_targets.append(gene_list)
    #             if cluster_id is None:
    #                 raise ValueError("cluster_id is required for target_type='gene' (cluster-level edges).")
    
    #         M_edges = []
    #         for k, target in tqdm(enumerate(target_ids), total=num_targets, position=0):
    #             if target_type == "metagene":
    #                 # cell-level (as before)
    #                 edges = get_metagene_edges_window(
    #                     RNA, TF, target, popdata,
    #                     num_hops=num_hops,
    #                     cluster_only=cluster_only,
    #                     cluster_id=cluster_id,
    #                 )
    #                 Madata = sc.AnnData(edges)
    #                 # Keep original behavior: align obs to popdata (cells)
    #                 Madata = Madata[popdata.obs_names, :]
    #                 # NOTE: no spatial requirement change here; if you were
    #                 # previously setting Madata.obsm['spatial'], you can keep it
    #                 # or omit—it won’t be used downstream per your note.
    #             else:
    #                 # cluster-level (rows=clusters, cols=TFs); NO spatial, NO reindex to cells
    #                 edges = get_gene_edges_by_cluster_only(
    #                     RNA, TF, target, cluster_id=cluster_id
    #                 )
    #                 Madata = sc.AnnData(
    #                     X=edges.values,
    #                     obs=pd.DataFrame(index=edges.index),     # clusters
    #                     var=pd.DataFrame(index=edges.columns),   # TFs
    #                 )
    
    #             M_edges.append(Madata)
    
    #         # Assemble: first target as base, others in layers M_k
    #         self.edge_weights.append(M_edges[0])
    
    #         # Preserve downstream loops that expect num_metagenes and layers "M_k"
    #         if target_type == "gene":
    #             self.num_metagenes = num_targets  # alias: number of genes modeled
    
    #         for k in range(num_targets):
    #             self.edge_weights[i].layers[f"M_{k}"] = M_edges[k].X
    
    #         # Minimal provenance
    #         self.edge_weights[i].uns["edge_target_type"] = target_type
    #         if target_type == "gene":
    #             self.edge_weights[i].uns["edges_axis"] = "clusters"  # rows are clusters
    #             # self.edge_weights[i].uns["M_index_to_gene"] = dict(enumerate(target_ids))
    #             self.edge_weights[i].uns["M_index_to_gene"] = list(map(str, target_ids)) 
    #             self.edge_weights[i].uns["cluster_id"] = cluster_id
    
    def compute_TF_gene_influence_for_region(
        self,
        dataset_idx: int,
        region_key: str = "leiden",
        region_value: Optional[Union[str, int]] = None,
        agg: str = "mean",                        # 'mean' or 'median'
        normalize: Optional[str] = None,          # None, 'rowsum', or 'rows_z'
        return_matrix: bool = False,
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, np.ndarray]]:
        """
        Compute region-level TF→gene influences by chaining:
          (TF→metagene aggregated over region cells) × (metagene→gene).
    
        Assumes:
          - self.edge_weights[dataset_idx].layers['M_{k}'] stores cell×TF weights for metagene k
          - self.edge_weights[dataset_idx].obs_names align to self.popari.datasets[dataset_idx].obs_names
          - self.popari.datasets[dataset_idx].uns['M']['A1'] is genes×K (metagene→gene)
    
        Parameters
        ----------
        dataset_idx : int
            Which dataset in self.popari.datasets / self.edge_weights to use.
        region_key : str
            Column in the dataset's obs to select the region (e.g. 'leiden' or your cluster key).
        region_value : str|int or None
            Specific region value to compute. If None, raises with available values.
        agg : {'mean','median'}
            Aggregation across cells in the region.
        normalize : {None,'rowsum','rows_z'}
            Optional per-TF normalization of TF→gene influences.
        return_matrix : bool
            If True, also return the raw np.ndarray (n_TF × n_genes).
    
        Returns
        -------
        df : pd.DataFrame
            TF→gene influences (rows: TFs, cols: genes).
        mat : np.ndarray (optional)
            Raw matrix (n_TF × n_genes) if return_matrix=True.
        """
        # sanity checks
        if self.popari is None:
            raise RuntimeError("compute_metagenes() must be run first (self.popari is None).")
        if not self.edge_weights:
            raise RuntimeError("compute_TF_metagene_weights() must be run first (edge_weights empty).")
    
        ew = self.edge_weights[dataset_idx]
        dset = self.popari.datasets[dataset_idx]
        dname = self.datasetnames[dataset_idx]
    
        if "M" not in dset.uns or dname not in dset.uns["M"]:
            raise RuntimeError(f"Expected metagene→gene matrix at dset.uns['M'][{dname}].")
    
        if region_key not in dset.obs.columns:
            raise KeyError(f"'{region_key}' not found in dataset.obs.")
    
        # pick cells in region
        if region_value is None:
            vals = dset.obs[region_key].astype(str).unique().tolist()
            raise ValueError(f"region_value is None. Available {region_key} values: {vals}")
    
        mask = (dset.obs[region_key].astype(str).values == str(region_value))
        if not np.any(mask):
            vals = dset.obs[region_key].astype(str).unique().tolist()
            raise ValueError(f"No cells for {region_key} == {region_value}. Available: {vals}")
    
        # ensure obs alignment between edge_weights and dataset
        if not np.array_equal(ew.obs_names, dset.obs_names):
            # try to align by reindexing
            ew = ew[dset.obs_names, :].copy()
    
        # stack TF→metagene across K layers
        # layers['M_k']: (cells × TF). We aggregate over cells in the region -> (TF,)
        K = self.num_metagenes
        if K is None or K <= 0:
            # fallback from shape of A1
            K = dset.uns["M"][dname].shape[1]
    
        tf_names = ew.var_names.to_numpy()
        print(tf_names)
        agg_vecs = []
        for k in range(K):
            layer_key = f"M_{k}"
            if layer_key not in ew.layers:
                raise RuntimeError(f"Missing layer '{layer_key}' in edge_weights for dataset {dataset_idx}.")
            mat_ck = ew.layers[layer_key]  # (cells × TF)
            # subset to region cells
            sub = mat_ck[mask, :]
            print(sub)
            if agg == "mean":
                v = np.asarray(sub).mean(axis=0)
            elif agg == "median":
                v = np.asarray(sub).median(axis=0)
            else:
                raise ValueError("agg must be 'mean' or 'median'")
            agg_vecs.append(v)
    
        # W_tf_mg: n_TF × K
        W_tf_mg = np.stack(agg_vecs, axis=1)  # currently K vectors of length n_TF -> (K × n_TF), but we stacked axis=1 so shape is (n_TF, K)
        if W_tf_mg.shape != (tf_names.shape[0], K):
            # if it's transposed (K × n_TF), fix it
            if W_tf_mg.shape == (K, tf_names.shape[0]):
                W_tf_mg = W_tf_mg.T
            else:
                raise RuntimeError(f"Unexpected TF×MG shape: {W_tf_mg.shape}, expected ({tf_names.shape[0]}, {K})")
    
        # metagene→gene: genes × K  ->  transpose to K × genes
        MG_gene = dset.uns["M"][dname]  # (n_genes × K)
        if MG_gene.shape[1] != K:
            raise RuntimeError(f"Metagene count mismatch: M['A1'] has {MG_gene.shape[1]}, TF→MG has {K}.")
        W_mg_gene = MG_gene.T  # (K × n_genes)
    
        # chain: (n_TF × K) @ (K × n_genes) = (n_TF × n_genes)
        W_tf_gene = W_tf_mg @ W_mg_gene  # dense multiply
    
        # optional normalization per TF (row)
        if normalize is None:
            pass
        elif normalize == "rowsum":
            rs = W_tf_gene.sum(axis=1, keepdims=True) + 1e-12
            W_tf_gene = W_tf_gene / rs
        elif normalize == "rows_z":
            mu = W_tf_gene.mean(axis=1, keepdims=True)
            sd = W_tf_gene.std(axis=1, keepdims=True) + 1e-12
            W_tf_gene = (W_tf_gene - mu) / sd
        else:
            raise ValueError("normalize must be None, 'rowsum', or 'rows_z'")
    
        gene_names = dset.var_names.to_numpy()
        df = pd.DataFrame(W_tf_gene, index=tf_names, columns=gene_names)
        return (df, W_tf_gene) if return_matrix else df


    def compute_TF_gene_influence_by_region(
        self,
        dataset_idx: int,
        region_key: str = "leiden",
        agg: str = "mean",
        normalize: Optional[str] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Compute TF → gene influence matrices for every region in a dataset.

        Iterates over all unique values of ``region_key`` and calls
        :meth:`compute_TF_gene_influence_for_region` for each one.

        Parameters:
            dataset_idx (int): Index into ``self.popari.datasets`` /
                ``self.edge_weights`` to use.
            region_key (str): ``obs`` column used to define regions (e.g.
                ``'leiden'``). Default ``'leiden'``.
            agg (str): Aggregation method across cells in each region — either
                ``'mean'`` or ``'median'``. Default ``'mean'``.
            normalize (str | None): Optional per-TF row normalization applied to
                the final TF × gene matrix.  ``None`` (no normalization),
                ``'rowsum'`` (divide by row sum), or ``'rows_z'`` (z-score rows).

        Returns:
            dict[str, pd.DataFrame]: Mapping of ``region_value`` → DataFrame of
            shape ``(n_TFs, n_genes)`` with TF → gene influence scores.
        """
        dset = self.popari.datasets[dataset_idx]
        values = dset.obs[region_key].astype(str).unique().tolist()
        out = {}
        for v in values:
            out[v] = self.compute_TF_gene_influence_for_region(
                dataset_idx=dataset_idx,
                region_key=region_key,
                region_value=v,
                agg=agg,
                normalize=normalize,
                return_matrix=False,
            )
        return out

    # def compute_TF_metagene_weights(
    #     self,
    #     num_hops: int = 2,
    #     cluster_only: bool = False,
    #     cluster_id: str = None,
    # ):
    #     """Computes the edge weights between transcription factors and metagenes
    #     over spatial neighborhoods.

    #     Parameters:
    #         num_hops (int): Number of spatial hops for computing metagene edges in the regulatory network.

    #     Returns:
    #         None

    #     """

    #     if not self.popari or not self.TF:
    #         print(
    #             "Popari needs to be run first, please run compute_metagenes().\n"
    #             "TF activity also needs to be computed by ArchR. Please follow the jupyter notebook for\n"
    #             "preprocessing ATAC-seq data and then run compute_TF_activity().",
    #         )
    #         return

    #     if len(self.TF) != len(self.popari.datasets):
    #         print(
    #             "Number of TF datasets must match the number of datasets in Popari.",
    #         )
    #         return

    #     self.edge_weights = []
    #     for i, TF, popdata, RNA in zip(
    #         range(len(self.TF)),
    #         self.TF,
    #         self.popari.datasets,
    #         self.RNA,
    #     ):
    #         M_edges = []
    #         for j in range(self.num_metagenes):
    #             temp = []
    #             edges = get_metagene_edges_window(
    #                 RNA,
    #                 TF,
    #                 j,
    #                 popdata,
    #                 num_hops=num_hops,
    #                 cluster_only=cluster_only,
    #                 cluster_id=cluster_id,
    #             )
    #             Madata = sc.AnnData(edges)
    #             Madata = Madata[popdata.obs_names, :]
    #             Madata.obsm["spatial"] = popdata.obsm["spatial"]
    #             temp.append(Madata)
    #             M_edges.append(temp)
    #         self.edge_weights.append(M_edges[0][0])
    #         for k in range(self.num_metagenes):
    #             self.edge_weights[i].layers[f"M_{k}"] = M_edges[k][0].X

    def run_all_perturbations(
        self,
    ):
        """Runs all perturbation analyses based on computed TF activity and edge
        weights.

        Parameters:
            None

        Returns:
            None

        """

        if not self.popari:
            print("Popari needs to be run first, please run compute_metagenes().")
            return

        if not self.TF:
            print(
                "TF activity needs to be computed by ArchR. Please follow the jupyter notebook for\n"
                "preprocessing ATAC-seq data and then run compute_TF_activity().",
            )
            return

        if not self.edge_weights:
            print(
                "GRN edge weights need to be computed first, please run compute_TF_metagene_weights().",
            )
            return

        for d in self.popari.datasets:
            d.obs["original_leiden"] = d.obs["leiden"]
        self.perturbed_X = run_all_perturbations(
            self.popari,
            self.TF,
            self.edge_weights,
            self.RNA,
            K=self.num_metagenes,
            useX=True,
        )


def load_anndata(dirpath: str) -> Eykthyr:
    """Load a previously saved Eykthyr session from disk.

    Reads all ``RNA_*.h5ad``, ``TF_*.h5ad``, ``edge_weights_*.h5ad``, and
    ``perturbed_X*.h5ad`` files from the directory derived from *dirpath*, and
    reloads the Popari model from ``popari.h5ad`` if present.  Dataset-level
    metadata (names, preprocessing flag, cluster annotations, ``K``) is restored
    from the ``uns`` of the first RNA object.

    Parameters:
        dirpath (str): Path used when calling :meth:`Eykthyr.save_anndata`
            (e.g. ``'results.h5ad'``).  The function strips the extension and
            treats the resulting name as a directory.

    Returns:
        Eykthyr: Fully restored Eykthyr instance ready for visualization or
        continued analysis.
    """

    dirpath = Path(dirpath)
    path_without_extension = dirpath.parent / dirpath.stem
    RNA_list = []
    TF_list = []
    edge_weights_list = []
    perturbed_X_list = []
    popari = None

    # Load RNA datasets
    for file in sorted(path_without_extension.glob("RNA_*.h5ad")):
        RNA_list.append(sc.read(file))

    # Load TF datasets
    for file in sorted(path_without_extension.glob("TF_*.h5ad")):
        TF_list.append(sc.read(file))

    # Load edge weights datasets
    for file in sorted(path_without_extension.glob("edge_weights_*.h5ad")):
        edge_weights_list.append(sc.read(file))

    # Load perturbed_X dataset
    for file in sorted(path_without_extension.glob("perturbed_X*.h5ad")):
        perturbed_X_list.append(sc.read(file))

    # Load popari model
    popari_file = path_without_extension / "popari.h5ad"
    if popari_file.is_file():
        popari = load_trained_model(popari_file)
    elif (path_without_extension / "popari").is_dir():
        popari = load_trained_model(popari_file)

    # Create Eykthyr instance
    eykthyr = Eykthyr(
        RNA=RNA_list,
        TF=TF_list,
        edge_weights=edge_weights_list,
        perturbed_X=perturbed_X_list,
        popari=popari,
    )

    # Optionally, restore attributes from the saved RNA dataset
    if RNA_list:
        eykthyr.datasetnames = [R.uns.get("datasetname", "unknown") for R in RNA_list]
        eykthyr.rna_preprocessed = RNA_list[0].uns.get("rna_preprocessed", False)
        eykthyr.cluster_annotation = RNA_list[0].uns.get("cluster_annotation", [])
        eykthyr.num_metagenes = RNA_list[0].uns.get("num_metagenes", -1)

    return eykthyr
