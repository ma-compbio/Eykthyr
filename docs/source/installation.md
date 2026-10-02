(installation)=

# Installation

`eykthyr` requires Python **≥ 3.10**.

## Quick install

```bash
pip install eykthyr
```

## Recommended: conda environment

We strongly recommend creating a dedicated conda environment to avoid dependency conflicts.

### Step 1 — Create and activate the environment

```bash
conda create -n eykthyr python=3.10
conda activate eykthyr
```

### Step 2 — Install PyTorch (optional GPU acceleration)

If you have an NVIDIA GPU, install PyTorch with the matching CUDA version **before** installing `eykthyr`.  Replace `cu121` with your installed CUDA version.

```bash
# GPU (CUDA 12.1)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# CPU only
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

We recommend installing PyTorch from the official pip wheels above.

### Step 3 — Install Eykthyr

```bash
pip install eykthyr
```

### Step 4 — (Optional) Install ATAC preprocessing dependencies

The ArchR-based ATAC preprocessing notebook requires **R** and the following R packages:

```r
install.packages("BiocManager")
BiocManager::install(c("ArchR", "Seurat", "Signac", "BSgenome.Mmusculus.UCSC.mm10"))
```

Refer to the [ArchR documentation](https://www.archrproject.com/) for full installation instructions.

## Development install

To install the latest development version directly from the repository:

```bash
git clone https://github.com/gkrieg/eykthyr.git
cd eykthyr
pip install -e .
```

## Dependencies

Core dependencies installed automatically by `pip`:

| Package | Minimum version |
|---------|----------------|
| numpy | ≥ 1.20.1 |
| scipy | ≥ 1.7.1 |
| scikit-learn | ≥ 0.24.1 |
| pandas | ≥ 1.5.2 |
| anndata | ≥ 0.9.1 |
| torch | ≥ 1.13.0 |
| scanpy | ≥ 1.9.2 |
| popari | ≥ 0.0.72 |
| squidpy | ≥ 1.2.3 |
| umap-learn | ≥ 0.5.1 |
| tqdm | ≥ 4.60.0 |
| matplotlib | ≥ 3.7.0 |
| seaborn | ≥ 0.11.1 |
| louvain | ≥ 0.8.0 |
| cython | ≥ 3.0.11 |
| leidenalg | ≥ 0.9.0, < 0.11 |
| awkward | ≥ 2.0 |
| setuptools | < 81 |

The figure notebooks need a few more packages (`gseapy`, `pysam`, `fa2_modified`, `jupyterlab`), installed with `pip install -e ".[figures]"` from a clone of the repository.

## Verifying the installation

```python
import eykthyr
print(eykthyr.__version__)
```
