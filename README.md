# SAP

## Installation notes
conda create -n sap python==3.8.0
pip install deepcell
pip install cellpose

conda install glob2 matplotlib numpy pandas scanpy seaborn scipy networkx tensorly statsmodels scikit-learn yellowbrick joblib tifffile tensorflow

conda install -c conda-forge scikit-image leidenalg

## General outline of CODEX analysis
### I.	Image preprocessing (Matlab) | python version possible?! [Matlab AWS machine] [CPU Matlab container?]
	a.	Deconvolution
	b.	Title stitch
	c.	Co-registration of all data
	d.	Background subtraction
	e.	Concatenate all the stacks into a hyperstack
	f.	(optional) H&E staining
### II.	Cell segmentation
	a.	CellSeg (CPU version is faster than the GPU version) [Trying to get it working] [Andrew Dockerfile]
	b.	CellPose (different preprocessing in python) 
	c.	Mesmer
### III.	Data normalization 
	a.	Z normalization
	b.	Double Z normalization
	c.	Min Max normalization
	d.	Arcsinh normalization
### IV.	Cell type annotation
	a.	Clustering [Leiden clustering; GPU requirement] [Andrew Dockerfile]
	b.	SVM annotation
	c.	STELLAR annotation [RapidAIs] [Done, try to test it] [Yuqi Dockerfile]
### V.	Downstream spatial analysis
	a.	Cell type composition
	b.	Cellular neighborhood analysis [Jupyter notebook]
	c.	Community analysis [Jupyter notebook]
	d.	Cell density analysis
	e.	Shannon diversity function
	f.	Tissue schematic analysis (R) [Jupyter notebook]
	g.	Distance permutation analysis (R) [Andrew]
	h.	Neighbor permutation analysis [long time]
	i.	Compositional differences for CN?
	j.	(optional) Mario or MaxFuse integration
	k.	Transcriptomics? [DEG]

