#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 16:11:15 2023

@author: timnoahkempchen
"""

# AnnData
import numpy as np
import pandas as pd
import anndata as ad
from scipy.sparse import csr_matrix
print(ad.__version__)

#############################################################
# Filepaths 
#############################################################

input_file = "/Users/timnoahkempchen/Library/CloudStorage/GoogleDrive-timkem@stanford.edu/Meine Ablage/Datasets/data/22_08_09_CellAtlas_only_Processed_Metadata.csv"

output_dir = "/Users/timnoahkempchen/Downloads/Output_testnew/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

#############################################################
# Reading the data 
#############################################################
df = pd.read_csv(input_file)

df['ID'] = [f'abc_{n+1:02}' for n in range(len(df))]
df['ID'] = 'cell_' + (df.index + 1).astype(str).str.zfill(2)

counts = df[['CHGA', 'MMP9', 'CD36', 'CK7', 'PDL1', 'Bcatenin',
       'Vimentin', 'FoxP3', 'CD56', 'CD31', 'pH2AX', 'CD90', 'CD15', 'PD1',
       'aSMA', 'CD25', 'Ki67', 'Cytokeratin', 'CD20', 'CD4', 'CD11c',
       'AnnexA1', 'Nestin', 'p53', 'CD73', 'EGFR', 'MUC5AC', 'HLADR', 'COX2',
       'BCL2', 'p63', 'CD3', 'MUC2', 'CD8', 'CD45', 'PGA3', 'CD57', 'CD68',
       'aDef5', 'CD34', 'Podoplanin', 'CD38', 'CD11b', 'CD163', 'MUC1',
       'CD138', 'Arginase1', 'PP', 'CD79a', 'MUC6','CD206', 'CollIV', "ID"]]

cell_types = df[['Cell Type', "ID"]]
spatial =df[["x", "y", "ID"]]

metadata = df.drop(labels = ['CHGA', 'MMP9', 'CD36', 'CK7', 'PDL1', 'Bcatenin',
       'Vimentin', 'FoxP3', 'CD56', 'CD31', 'pH2AX', 'CD90', 'CD15', 'PD1',
       'aSMA', 'CD25', 'Ki67', 'Cytokeratin', 'CD20', 'CD4', 'CD11c',
       'AnnexA1', 'Nestin', 'p53', 'CD73', 'EGFR', 'MUC5AC', 'HLADR', 'COX2',
       'BCL2', 'p63', 'CD3', 'MUC2', 'CD8', 'CD45', 'PGA3', 'CD57', 'CD68',
       'aDef5', 'CD34', 'Podoplanin', 'CD38', 'CD11b', 'CD163', 'MUC1',
       'CD138', 'Arginase1', 'PP', 'CD79a', 'MUC6','CD206', 'CollIV', 'Cell Type',
       "x", "y"], axis = 1)



counts.set_index('ID')
cell_types.set_index('ID')
spatial.set_index('ID')
metadata.set_index('ID')

adata = ad.AnnData(counts)
                                                                                                                                                                                                          
adata.X

print(adata.obs_names[:10])                                                        
                                                     

adata.obs["cell_type"] = cell_types  # Categoricals are preferred for efficiency
adata.obs



