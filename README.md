# Overview of EYKTHYR

<img width="649" alt="EYKTHYR overview" src="https://github.com/user-attachments/assets/bbc155b5-a2fe-4479-b38f-37d2740bb7b9">

EYKTHYR is the first method developed to infer region-specific TF influences on spatial gene programs (or metagenes) using spatial multiome data. EYKTHYR addresses high technical dropout by introducing a novel combination of linear embeddings for gene expression and chromatin accessibility, denoising the data while maintaining interpretability, as shifts in metagene embeddings map directly back to input gene expression. Using information from spatially proximal neighbors, EYKTHYR learns a linear relationship between TF activity and metagene expression in each cell, where these two layers of linear mappings—from TF activity to metagenes, and from metagenes to gene expression—enable reasoning about how TF activity changes affect gene expression.

# Running EYKTHYR
## Input Data Format

EYKTHYR requires spatial transcriptomics and spatial chromatin accessibility data to be provided as AnnData objects, structured as follows:

Spatial coordinates:
The spatial coordinates of each cell should be included in the .obsm attribute of the AnnData object.
The coordinates must be stored under .obsm['spatial'] and formatted as an array with dimensions [number of cells, 2] (representing x and y coordinates for each cell)

Gene expression data:
Gene expression data should be stored in .X as a sparse matrix (recommended for large datasets) or a dense matrix, with dimensions [number of cells, number of genes]

While not required, any additional metadata (e.g., cell types, batch labels) can be stored in .obs

Chromatin accessibility data:
Chromatin accessibility data can be in the form of a fragments file, where each fragment corresponds to a cell, using the same cell ids as in the gene expression data.

## Installation

### Step 1: Create a Conda Environment

Before installing any Python packages, we strongly recommend using Anaconda (please refer to the Anaconda webpage for conda installation instructions) to create a python 3.10 environment using the following command:

conda install --name eykthyr python=3.10

After creating the environment, activate it using:

conda activate eykthyr

### Step 2: Install Dependencies

Install PyTorch with CUDA (optional)

If you have an NVIDIA GPU and want to use CUDA for acceleration, install PyTorch with the desired CUDA version. For example, to install PyTorch 2.1.0 with CUDA 11.8, run:

conda install pytorch==2.1.0 cudatoolkit=11.8 -c pytorch

Note: For a CPU-only installation, you can omit the cudatoolkit argument.

### Step 3: Install EYKTHYR

EYKTHYR is available as a pypi package, and can be installed using:

pip install eykthyr

## Running Code

To run our method, you need to run the pipeline in two parts: first process the spatial ATAC-seq data into a peak matrix, and create a file that annotates peaks with TF motifs present in the peak region. The second part of the pipeline preprocesses the spatial transcriptomic data and then combines this with the TF activity for inference.

We provide tutorial notebooks for each of these processes.

