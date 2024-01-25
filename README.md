# SPatial Analysis for CodEX data (SPACEc)

## Installation notes

**Note**: Due to some dependencies, we currently only support Python up to `3.10`.

We generally recommend to use a `conda` environment. It makes installing requirements like `graphviz` a lot easier.

Install `conda` repository.

```bash
conda create -n sap python==3.10
```

Install `graphviz`.

```bash
conda install graphviz
```

Install `SPACEc`
```bash
pip install spacec
``````

```bash
# conda create -n sap python==3.8.0
# pip install deepcell cellpose

# conda install glob2 matplotlib numpy pandas scanpy seaborn scipy networkx tensorly statsmodels scikit-learn yellowbrick joblib tifffile tensorflow
# conda install anaconda::graphviz
# conda install -c conda-forge scikit-image
# pip install leidenalg concave-hull==0.0.6
```

## General outline of SPACEc analysis

### I.	Cell segmentation & visualization
	a.	Mesmer
	b.	CellPose
	c.	CellSeg [Under Development]
### II.	Data prepcoessing
	a.  First filtering based on size and DAPI
	b.	Normalization
	c.	Second filtering based on noisy signals
	d.	Data type conversion (df --> anndata)
### III.	Clustering & cell type annotation
	a.	Clustering [GPU implementation UD]
	b.	ML annotation [e.g. STELLAR, UD]
	c.	Cell type statistics and visualization
### IV.	Downstream spatial analysis
	a.	Cellular neighborhood analysis
	b.	Tissue schematic analysis
	c.	Distance permutation analysis
	d.	Neighbor permutation analysis [UD]
	e.	Patch proximity analysis
	e.	Shannon diversity function
	f.	(optional) Mario or MaxFuse integration[UD]
