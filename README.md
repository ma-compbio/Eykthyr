# Overview of EYKTHYR

<img width="649" alt="EYKTHYR overview" src="https://github.com/user-attachments/assets/bbc155b5-a2fe-4479-b38f-37d2740bb7b9">

EYKTHYR is the first method developed to infer region-specific TF influences on spatial gene programs (or metagenes) using spatial multiome data. EYKTHYR addresses high technical dropout by introducing a novel combination of linear embeddings for gene expression and chromatin accessibility, denoising the data while maintaining interpretability, as shifts in metagene embeddings map directly back to input gene expression. Using information from spatially proximal neighbors, EYKTHYR learns a linear relationship between TF activity and metagene expression in each cell, where these two layers of linear mappings—from TF activity to metagenes, and from metagenes to gene expression—enable reasoning about how TF activity changes affect gene expression.

# Running EYKTHYR
## Input Data Format

EYKTHYR requires spatial transcriptomics and spatial chromatin accessibility data to be provided as AnnData objects, structured as follows:

**Spatial coordinates:**
The spatial coordinates of each cell should be included in the `.obsm` attribute of the AnnData object. The coordinates must be stored under `.obsm['spatial']` and formatted as an array with dimensions `[number of cells, 2]` (representing x and y coordinates for each cell).

**Gene expression data:**
Gene expression data should be stored in `.X` as a sparse matrix (recommended for large datasets) or a dense matrix, with dimensions `[number of cells, number of genes]`.

While not required, any additional metadata (e.g., cell types, batch labels) can be stored in `.obs`.

**Chromatin accessibility data:**
Chromatin accessibility data can be in the form of a fragments file, where each fragment corresponds to a cell, using the same cell IDs as in the gene expression data.

## System Requirements

The package was tested on Linux operating systems. Theoretically any OS that can run Python 3.10+ should be compatible, however extensive testing has not yet occurred.

A GPU is essentially required for running Popari, which is required for running EYKTHYR.

Expected installation time is around 10 minutes.

Expected runtime varies by dataset, but should take less than an hour including running plotting tools.

## Installation

### Step 1: Create a Conda Environment

Before installing any Python packages, we strongly recommend using Anaconda (please refer to the Anaconda webpage for conda installation instructions) to create a Python 3.12 environment using the following command:

```
conda create --name eykthyr python=3.12
conda activate eykthyr
```

### Step 2: Install PyTorch

If you have an NVIDIA GPU and want to use CUDA for acceleration, install PyTorch with your desired CUDA version. You can use `light-the-torch` for an easier install:

```
pip install light-the-torch
ltt install --pytorch-computation-backend=cu121 torch torchvision torchaudio
```

For a CPU-only installation:

```
conda install pytorch -c pytorch
```

### Step 3: Install EYKTHYR

EYKTHYR is available as a PyPI package:

```
pip install eykthyr[with-velocyto,simulation]
```

## Running the Pipeline

The EYKTHYR pipeline has three phases that must be run in order:

1. **ATAC preprocessing** — process spatial ATAC-seq fragments into a peak matrix and annotate peaks with TF motifs
2. **RNA preprocessing** — normalize and embed the spatial transcriptomic data
3. **Training and analysis** — combine TF activity with gene expression for inference and perturbation simulation

We provide tutorial notebooks for each of these steps (see `docs/source/tutorial_gallery/`), as well as a full walkthrough demo in `eykthyr_walkthrough_demo.ipynb`.

## Quick-Start Example

```python
from eykthyr.eykthyr import Eykthyr, load_anndata
import eykthyr.plotting as pl
import scanpy as sc

# 1. Load and preprocess RNA
e = Eykthyr()
adrna = sc.read("data/mouse_embryo2_rna.h5ad")
e.set_RNA([adrna])
e.preprocess_rna(make_plots=True)

# 2. Learn spatial metagenes
e.compute_metagenes(K=16, spatial_iterations=200)
e.analyze_metagenes()

# 3. Compute TF activity from ArchR outputs
e.compute_TF_activity(
    peak_tsvs=["data/spatialATACRNAmouseembryo2_peaks.tsv"],
    archr_dataset_names=["spatialATACRNAmouseembryo2"],
    motif_tsvs=["data/spatialATACRNAmouseembryo2_motifs.tsv"],
)

# 4. Infer TF → metagene regulatory weights
e.compute_TF_metagene_weights(num_hops=2)

# 5. Simulate TF perturbations
e.run_all_perturbations()

# 6. Save session
e.save_anndata("results.h5ad")

# Reload later
e = load_anndata("results.h5ad")

# 7. Visualize
pl.prep_paga(e, "original_leiden")
pl.paga_spatial_simulation(e, ["Msx1"], "original_leiden")
```

# Reproducing Paper Figures

The source data to recreate the papers is available through Zenodo at https://doi.org/10.5281/zenodo.20529989.
The following notebooks reproduce the figures in the EYKTHYR manuscript:

| Notebook | Contents |
|---|---|
| `Eykthyr_fig2.ipynb` | Figure 2 panels (spatial simulation, supplementary panels) |
| `Eykthyr_fig3.ipynb` | Figure 3 panels (pseudotime, TF perturbation simulation, TF ranking, GSEA) |
| `Eykthyr_fig4.ipynb` | Figure 4 panels |
| `Eykthyr_fig5.ipynb` | Figure 5 panels |
| `Ablation_plots.ipynb` | Ablation study results (Figure 2C) |
| `Nrg1_isoform_spatial.ipynb` | Nrg1 isoform spatial analysis (Figure 2H) |
