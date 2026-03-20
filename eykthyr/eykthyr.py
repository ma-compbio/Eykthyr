import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from popari.train import TrainParameters, Trainer
from popari import tl
import anndata as ad
from popari.components import PopariDataset
from popari.io import save_anndata
from popari.model import Popari, load_trained_model
from scipy.sparse import spmatrix
import scipy.sparse as sp
from .embedding import Embedding
from .modified_VelocytoLoom_class import modified_VelocytoLoom
from .util import get_metagene_edges_window, run_all_perturbations





class Eykthyr(modified_VelocytoLoom):

    def __init__(
        self,
        RNA: Optional[List[sc.AnnData]] = None,
        popari: Optional[Popari] = None,
        TF: Optional[List[sc.AnnData]] = None,
        edge_weights: Optional[List[sc.AnnData]] = None,
        perturbed_X: Optional[List[sc.AnnData]] = None,
        names: Optional[List[str]] = ["eykthyr_dataset"],
        cluster_annotation: Optional[Sequence[str]] = None,
        num_metagenes: int = -1,
        embeddings: Optional[List[Dict[str, Embedding]]] = [],
    ):

        self.RNA = RNA if RNA is not None else []
        self.popari = popari
        self.TF = TF if TF is not None else []
        self.edge_weights = edge_weights if edge_weights is not None else []
        self.perturbed_X = perturbed_X if perturbed_X is not None else []

        self.datasetnames = names
        self.rna_preprocessed = False

        if cluster_annotation is None:
            self.cluster_annotation = []
        else:
            # convert numpy/pandas sequences safely to a plain python list
            self.cluster_annotation = list(cluster_annotation)
        # keep only strings (obs keys)
        self.cluster_annotation = [x for x in self.cluster_annotation if isinstance(x, str)]

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
        lambda_Sigma_bar: float = 1e-4,
        torch_context: dict = None,
        initial_iterations: int = 10,
        spatial_iterations: int = 200,
        spatial_affinity_mode: str = "differential lookup",
        save_dir: str = "./popari_runs",
        run_name: str = "popari_results",
        # ---- hierarchical popari options ----
        hierarchical_levels: Optional[int] = None,          
        downsampling_method: str = "partition",             
        superresolution_lr: float = 1e-1,                   
    ):
        """
        Popari (new API) metagene computation via:
          1) Build a merged .h5ad with obs['batch'] and uns['adjacency_matrix'] (dict keyed by replicate)
          2) Popari(dataset_path=...)
          3) Trainer.train()
    
        Requires these module-level imports in eykthyr.py:
          from popari.components import PopariDataset
          from popari.model import Popari
          from popari.train import TrainParameters, Trainer
        """
        if not self.rna_preprocessed:
            print(
                "RNA appears to not be preprocessed. Please preprocess RNA using "
                "Eykthyr.preprocess_rna() or set Eykthyr.rna_preprocessed = True"
            )
            return

        if not isinstance(self.datasetnames, list) or len(self.datasetnames) == 0:
            # fallback: auto-generate names if missing
            self.datasetnames = [f"rep{i}" for i in range(len(self.RNA))]
        
        if len(self.datasetnames) != len(self.RNA):
            raise ValueError(
                f"datasetnames length ({len(self.datasetnames)}) must match RNA length ({len(self.RNA)}). "
                "Otherwise zip(self.RNA, self.datasetnames) will drop replicates."
            )
            
        self.num_metagenes = K

    
        # default CUDA context (matches Popari docs style; helps avoid simplex assert)
        if torch_context is None:
            torch_context = {"device": "cuda:0", "dtype": torch.float64}
    
        save_dir_p = Path(save_dir)
        save_dir_p.mkdir(parents=True, exist_ok=True)
    
        dataset_path = save_dir_p / "preprocessed_dataset.h5ad"
        results_path = save_dir_p / f"{run_name}.h5ad"
    
        prepared = []
        adjacency_by_rep = {}
    
        for RNA, name in zip(self.RNA, self.datasetnames):
            rep = str(name)
            a = RNA.copy()
    
            # ensure X is sparse CSR
            if sp.issparse(a.X):
                a.X = a.X.tocsr()
            else:
                a.X = sp.csr_matrix(a.X)
    
            # require spatial coords
            if "spatial" not in a.obsm:
                raise KeyError(f"Replicate '{rep}' missing obsm['spatial'].")
    
            # mark replicate membership for Popari
            a.obs["batch"] = rep
    
            # compute spatial neighbors via PopariDataset
            ds = PopariDataset(a, rep)
            ds.compute_spatial_neighbors()
    
            if "adjacency_matrix" not in ds.obsp:
                raise KeyError(f"PopariDataset '{rep}' did not produce obsp['adjacency_matrix'].")
    
            # store adjacency for Popari loader (expects merged.uns['adjacency_matrix'][rep])
            adjacency_by_rep[rep] = ds.obsp["adjacency_matrix"].tocsr()
    
            # remove fields that can complicate concatenation; Popari will initialize internally
            for k in ["X_umap", "X_pca", "X_diffmap"]:
                if k in a.obsm:
                    del a.obsm[k]
            if "PCs" in a.varm:
                del a.varm["PCs"]
    
            # do NOT keep adjacency artifacts in the AnnData during concat
            if "adjacency_matrix" in a.obsp:
                del a.obsp["adjacency_matrix"]
            if "adjacency_list" in a.obsm:
                del a.obsm["adjacency_list"]
            if "adjacency_list" in a.obs:
                del a.obs["adjacency_list"]
    
            # remove scanpy neighbor graphs if present
            for k in ["distances", "connectivities"]:
                if k in a.obsp:
                    del a.obsp[k]
    
            # layers can complicate merging; Popari uses X
            if hasattr(a, "layers") and len(a.layers.keys()) > 0:
                for lk in list(a.layers.keys()):
                    del a.layers[lk]
    
            prepared.append(a)
    
        # merge replicates; keep outer join on genes
        merged = ad.concat(
            prepared,
            join="outer",
            merge="unique",
            uns_merge="unique",
            index_unique="-",
        )
    
        if "batch" not in merged.obs:
            raise KeyError("Merged AnnData missing obs['batch']; Popari multi-replicate requires this.")

        merged.uns["num_metagenes"] = K
        merged.uns["adjacency_matrix"] = adjacency_by_rep
        merged.write_h5ad(dataset_path)
    
        # instantiate Popari from dataset_path
        model_parameters = {
            "K": K,
            "dataset_path": dataset_path,
            "lambda_Sigma_x_inv": lambda_Sigma_x_inv,
            "lambda_Sigma_bar": lambda_Sigma_bar,
            "initial_context": torch_context,
            "torch_context": torch_context,
            "verbose": 0,
            "spatial_affinity_mode": spatial_affinity_mode,
        }

        # Enable hierarchical mode if requested
        use_hier = (hierarchical_levels is not None) and (int(hierarchical_levels) > 1)
        if use_hier:
            model_parameters.update(
                {
                    "downsampling_method": downsampling_method,
                    "hierarchical_levels": int(hierarchical_levels),
                    "superresolution_lr": float(superresolution_lr),
                }
            )

        self.popari = Popari(**model_parameters)
    
        # train
        train_parameters = TrainParameters(
            nmf_iterations=initial_iterations,
            iterations=spatial_iterations,
            savepath=results_path,
        )
        trainer = Trainer(parameters=train_parameters, model=self.popari, verbose=1)
        trainer.train()

        #trainer.superresolve(n_epochs=10000, tol=1e-6)
    
        self.popari_dataset_path = str(dataset_path)
        self.popari_results_path = str(results_path)

        self.popari.save_results(str(results_path))

    


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
            joint=True,
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

        if self.popari and not os.path.isfile(f"{path_without_extension}/popari.h5ad"):
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
        """Computes TF activity across multiple RNA datasets."""

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

    def compute_TF_metagene_weights(self, num_hops: int = 2):
        """Computes TF→metagene edge weights over spatial neighborhoods."""
    
        if not self.popari or not self.TF:
            print(
                "Popari needs to be run first, please run compute_metagenes().\n"
                "TF activity also needs to be computed by ArchR. Please follow the jupyter notebook for\n"
                "preprocessing ATAC-seq data and then run compute_TF_activity().",
            )
            return
    
        # If you load Popari from disk, num_metagenes may not be set on Eykthyr
        if self.num_metagenes is None or self.num_metagenes <= 0:
            K = getattr(self.popari, "K", None)
            if K is None:
                # fallback: set manually if your Popari object doesn't expose K
                raise ValueError(
                    f"num_metagenes must be > 0. Got {self.num_metagenes}. "
                    "Set e.num_metagenes = <K> after loading Popari."
                )
            self.num_metagenes = int(K)
    
        if len(self.TF) != len(self.popari.datasets):
            print("Number of TF datasets must match the number of datasets in Popari.")
            return
    
        if len(self.RNA) != len(self.popari.datasets):
            print("Number of RNA datasets must match the number of datasets in Popari.")
            return
    
        self.edge_weights = []
    
        # iterate by index (clearer than zip(range(...), ...))
        for i in range(len(self.popari.datasets)):
            TF = self.TF[i]
            popdata = self.popari.datasets[i]
            RNA = self.RNA[i]
    
            # Popari obs_names look like "<barcode>-1-<rep>"; edges often keyed by "<barcode>-1"
            pop_base = pd.Index(popdata.obs_names).str.replace(r"-\d+$", "", regex=True)
    
            M_edges: List[sc.AnnData] = []
    
            for j in range(self.num_metagenes):
                edges = get_metagene_edges_window(
                    RNA,
                    TF,
                    j,
                    popdata,
                    num_hops=num_hops,
                )
    
                Madata = sc.AnnData(edges)
    
                # safety check: ensure alignment is possible
                missing = pop_base[~pop_base.isin(Madata.obs_names)]
                if len(missing) > 0:
                    raise KeyError(
                        f"[rep {i}] {len(missing)} cells in popdata not found in edges output after "
                        f"base-barcode stripping. Example missing: {missing[0]}. "
                        f"Example pop obs: {popdata.obs_names[0]}, example edges obs: {Madata.obs_names[0]}"
                    )
    
                # reorder to Popari order, then rename to match Popari exactly
                Madata = Madata[pop_base, :].copy()
                Madata.obs_names = popdata.obs_names
                Madata.obsm["spatial"] = popdata.obsm["spatial"]
    
                M_edges.append(Madata)
    
            if len(M_edges) == 0:
                raise ValueError(
                    f"[rep {i}] No metagene edge matrices were created. num_metagenes={self.num_metagenes}."
                )
    
            # store the first as the main X, and the rest in layers
            ew = M_edges[0].copy()
            for k in range(self.num_metagenes):
                ew.layers[f"M_{k}"] = M_edges[k].X
    
            self.edge_weights.append(ew)


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
            K=self.num_metagenes,
            useX=True,
        )


def load_anndata(dirpath: str) -> Eykthyr:
    """Loads Eykthyr datasets from the specified directory."""

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
        ca = RNA_list[0].uns.get("cluster_annotation", [])
        # normalize
        if ca is None:
            ca = []
        elif not isinstance(ca, list):
            ca = list(ca)
        ca = [x for x in ca if isinstance(x, str)]
        eykthyr.cluster_annotation = ca

        eykthyr.num_metagenes = RNA_list[0].uns.get("num_metagenes", -1)

    return eykthyr
