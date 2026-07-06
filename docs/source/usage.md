# Usage

The Eykthyr pipeline consists of three phases covered in the tutorials below.
Run them in order: ATAC preprocessing must be done before the RNA/training steps,
and the RNA preprocessing results feed into the training notebook.

## Tutorials

```{toctree}

tutorial_gallery/atac_preprocess
tutorial_gallery/rna_preprocess
tutorial_gallery/training_analysis

```

## Quick-start example

```python
from eykthyr.eykthyr import Eykthyr, load_anndata
import eykthyr.plotting as pl
import scanpy as sc

# --- 1. Load and preprocess RNA ---
e = Eykthyr()
adrna = sc.read("data/mouse_embryo2_rna.h5ad")
e.set_RNA([adrna])
e.preprocess_rna(make_plots=True)

# --- 2. Learn spatial metagenes ---
e.compute_metagenes(K=16, spatial_iterations=200)
e.analyze_metagenes()

# --- 3. Compute TF activity from ArchR outputs ---
e.compute_TF_activity(
    peak_tsvs=["data/spatialATACRNAmouseembryo2_peaks.tsv"],
    archr_dataset_names=["spatialATACRNAmouseembryo2"],
    motif_tsvs=["data/spatialATACRNAmouseembryo2_motifs.tsv"],
)

# --- 4. Infer TF → metagene regulatory weights ---
e.compute_TF_metagene_weights(num_hops=2)

# --- 5. Simulate TF perturbations ---
e.run_all_perturbations()

# --- 6. Save session ---
e.save_anndata("results.h5ad")

# --- Reload later ---
e = load_anndata("results.h5ad")

# --- 7. Visualize ---
pl.prep_paga(e, "original_leiden")
pl.paga_spatial_simulation(e, ["Msx1"], "original_leiden")
```

## Re-loading a saved session

```python
from eykthyr.eykthyr import load_anndata

e = load_anndata("results.h5ad")
# e is a fully restored Eykthyr object
```

## Computing region-specific TF → gene influences

After running the full pipeline you can chain the TF → metagene weights with
the metagene → gene decoder to get per-region TF → gene influence matrices:

```python
# All regions at once
influences = e.compute_TF_gene_influence_by_region(
    dataset_idx=0,
    region_key="leiden",
    normalize="rows_z",
)
# influences is a dict: leiden_cluster_id -> DataFrame (TFs × genes)

# Single region
df = e.compute_TF_gene_influence_for_region(
    dataset_idx=0,
    region_key="leiden",
    region_value="3",
)
```

## Developmental perturbation scoring

After computing a pseudotime proxy (e.g. distance to a reference cell type),
score each TF's alignment with the differentiation trajectory:

```python
import scanpy as sc

# Compute ventricle distance as pseudotime proxy
sc.pp.neighbors(e.perturbed_X[0], use_rep="spatial", key_added="spatial_neighbors")
nns = sc.Neighbors(e.perturbed_X[0], neighbors_key="spatial_neighbors")
nns.compute_neighbors(knn=False, use_rep="spatial", method="gauss")
e.perturbed_X[0].obs["ventricle_distance"] = (
    nns.distances[:, e.perturbed_X[0].obs["original_leiden"] == "7"].min(axis=1)
)

# Score TF perturbation strength along the trajectory
ips = pl.development_simulation(e, ["Msx1", "Sox9"])
for tf, score in ips:
    print(f"{tf}: perturbation strength = {score:.4f}")
```
