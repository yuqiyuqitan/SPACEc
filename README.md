# SPatial Analysis for CodEX data (SPACEc)

## Installation notes

**Note**: Due to some dependencies, we currently only support Python up to `3.10`.

We generally recommend to use a `conda` environment. It makes installing requirements like `graphviz` a lot easier.

Install.

```bash
# setup `conda` repository
conda create -n spacec python==3.10
conda activate spacec

# install `graphviz`
conda install graphviz

# install `SPACEc` from pypi
pip install spacec

# install `SPACEc` from cloned repo
#pip install -e .

# on Apple M1/M2
#conda install tensorflow=2.10.0
```

Run tests.

```bash
pip install pytest pytest-cov
pytest
```


```bash
# conda create -n sap python==3.8.0
# pip install deepcell cellpose

# conda install glob2 matplotlib numpy pandas scanpy seaborn scipy networkx tensorly statsmodels scikit-learn yellowbrick joblib tifffile tensorflow
# conda install anaconda::graphviz
# conda install -c conda-forge scikit-image
# pip install leidenalg concave-hull==0.0.6
```

## General outline of SPACEc analysis

![SPACEc](https://github.com/yuqiyuqitan/SAP/tree/master/docs/overview.png?raw=true)


### Tissue extraction
	Step 1: Set up the environment
	Step 2: Downscale the whole tissue image
	Step 3: Rename tissue number (optional)
	Step 4: Extract individual labeled tissues into separate tiff stack

### Cell segmentation & visualization
	Step 5: Visualize segmentation channel (optional)
	Step 6: Run cell segmentation
	Step 7: Visually inspect segmentation results

### Data preprocessing & normalization
	Step 8: Load the segmented results
	Step 9: Initial filtering of artifacts/debris by DAPI intensity and cell size
	Step 10: Normalize data
	Step 11: Second filtering for noisy cell

### Cell type annotation
	Step 12: Cell type annotation via clustering
	Step 13: Visualize & annotate clustering results
	Step 14: Visualize clustering results in the original tissue coordinates
	Step 15: Compute basic statistics of the cell type composition
	Step 16: Loading in training data
	Step 17: Train the SVM model
	Step 18: Predict cell type labels using the trained model

### Interactive data inspection and exploration
	Step 19: Initialize interactive data inspection session via TissUUmaps
	Step 20: Additional data exploration via TissUUmaps (optional)

### Spatial analysis
	Step 21: Compute cellular neighborhoods
	Step 22: Visualize & annotate the cellular neighborhoods analysis results
	Step 23: Generate spatial context maps
	Step 24: Compute the cellular neighborhood interface via barycentric coordinate system
	Step 25: Compute patch proximity analysis
	Step 26: Compute for distance permutation analysis
