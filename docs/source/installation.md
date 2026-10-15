(installation)=

# Installation

`eykthyr` requires Python **≥ 3.10**. The steps below were tested on Linux with Python 3.12 and NumPy 2.

## Step 1 — Create and activate a conda environment

```bash
conda create -n eykthyr -c conda-forge python=3.12
conda activate eykthyr
```

## Step 2 — Install PyTorch

Install PyTorch with pip **before** installing `eykthyr`. Choose the wheel that matches your hardware (see the [PyTorch install selector](https://pytorch.org/get-started/locally/) for other CUDA versions):

```bash
# GPU (CUDA 12.1)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# CPU only
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

> **Note:** Do not use `conda install pytorch -c pytorch`. That channel is no longer updated, and the build it installs can fail to import (`libmkl_intel_lp64.so.2` missing).

## Step 3 — Install Eykthyr

```bash
pip install "eykthyr[simulation]"
```

To run the tutorial and figure notebooks, install from the GitHub repository instead:

```bash
git clone https://github.com/ma-compbio/Eykthyr.git
cd Eykthyr
pip install -e ".[simulation,figures]"
```

The extras are optional:

* `simulation` — JupyterLab and ipycanvas, for the interactive notebooks.
* `figures` — JupyterLab, gseapy, pysam and fa2_modified, used by the figure notebooks (the notebooks themselves are only in the repository).

## Step 4 — (Optional) Install ATAC preprocessing dependencies

The ArchR-based ATAC preprocessing notebook requires **R** and the following R packages:

```r
install.packages("BiocManager")
BiocManager::install(c("ArchR", "Seurat", "Signac", "BSgenome.Mmusculus.UCSC.mm10"))
```

Refer to the [ArchR documentation](https://www.archrproject.com/) for full installation instructions.

## Dependencies

Core dependencies installed automatically by `pip` (see `pyproject.toml`):

| Package | Minimum version |
|---------|----------------|
| numpy | ≥ 1.24 (NumPy 2 supported) |
| numba | ≥ 0.59 |
| scipy | ≥ 1.7.1 |
| scikit-learn | ≥ 0.24.1 |
| pandas | ≥ 1.5.2 |
| anndata | ≥ 0.9.1 |
| torch | ≥ 1.13.0 |
| scanpy | ≥ 1.9.2 |
| popari | ≥ 0.0.72, < 0.1 |
| squidpy | ≥ 1.2.3 |
| umap-learn | ≥ 0.5.1 |
| tqdm | ≥ 4.60.0 |
| matplotlib | ≥ 3.7.0, < 3.11 |
| seaborn | ≥ 0.11.1 |
| louvain | ≥ 0.8.0 |
| leidenalg | ≥ 0.9.0, < 0.11 |
| awkward | ≥ 2.0 |
| setuptools | < 81 |

The figure notebooks need a few more packages (`gseapy`, `pysam`, `fa2_modified`, `jupyterlab`), installed with the `figures` extra (Step 3).

Eykthyr no longer depends on velocyto. The two transition-probability kernels it used (`colDeltaCor`, `colDeltaCorpartial`) are reimplemented in `eykthyr/correlation.py`.

## Verifying the installation

```python
import eykthyr
import eykthyr.eykthyr
print(eykthyr.__version__)
```
