# load required packages 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import zscore
from scipy.stats import norm
import os
import glob
import scanpy as sc
from itertools import product


def pp_read_data(path, reg_list, nuc_thres = 1, size_thres = 1):
    #Change working directory to where data is stored
    all_files = glob.glob(path + "*.csv")

    #Concatenate CSV files to one frame
    dftest = pd.concat((pd.read_csv(filename, index_col=None, header=0).assign(filename = os.path.basename(filename)) for filename in all_files)) 

    #Separate out File Names and Cell Types
    dftest[['region_num','x1','y1','z1','comp']] = dftest.filename.apply(lambda x: pd.Series(str(x).split("_"))) 

    #Drop redudnent columns
    df_rename=dftest.drop(['x1','y1','z1','comp', 'filename',], axis = 1)

    #See resultant dataframe
    df_rename.columns = df_rename.columns.str.split(':').str[-1].tolist()
    df_rename = df_rename.reset_index().rename(columns={'index':'first_index'})

    #Remove problematic regions
    df_regionout = df_rename.loc[~(df_rename.region_num.isin(reg_list))]

    #Plot scatter plot by region
    #plt.rcParams["legend.markerscale"] = 1
    #plt.figure(figsize=(7,7))
    #g = sns.scatterplot(data=df_regionout, x='DAPI', y='size', hue='region_num', size=1)
    #g.set_xscale('log')
    #g.set_yscale('log')
    #ticks = [0.1, 1, 10, 100,1000]
    #g.set_yticks(ticks)
    #g.set_yticklabels(ticks)
    #g.set_xticks(ticks)
    #g.set_xticklabels(ticks)
    #plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)

    df_nuc = df_regionout[(df_regionout.DAPI > nuc_thres) * df_regionout['size'] > size_thres]
    per_keep = len(df_nuc)/len(df_regionout)
    print(per_keep)
    
    return df_nuc, per_keep

def pp_format(data, list_out, list_keep, method = "zscore", ArcSin_cofactor = 150):

    # original function:
    # #Drop column list
    # list1 = [col for col in data.columns if 'blank' in col]
    # list_out1 = list1+list_out
    
    # #Drop columns not interested in
    # dfin = data.drop(list_out1, axis = 1)

    # #save columns for later
    # df_loc = dfin.loc[:,list_keep]

    # #dataframe for normalization
    # dfz = dfin.drop(list_keep, axis = 1)

    # #zscore of the column markers 
    # dfz1 = pd.DataFrame(zscore(dfz,0),index = dfz.index,columns = [i for i in dfz.columns])

    # #Add back labels for normalization type
    # dfz_all = pd.concat([dfz1, df_loc], axis=1, join='inner')
    
    # print("the number of regions = "+str(len(dfz_all.region_num.unique())))
    
    # return dfz_all
    list = ["zscore", "double_zscore", "MinMax", "ArcSin"]

    if method not in list:
        print("Please select methods from zscore, double_zscore, MinMax, ArcSin!")
        exit()
    
    ##ArcSin transformation
    if(method == "ArcSin"):       
        #Drop column list
        list1 = [col for col in data.columns if 'blank' in col]
        list_out1 = list1+list_out
        
        #Drop columns not interested in
        dfin = data.drop(list_out1, axis = 1)

        #save columns for later
        df_loc = dfin.loc[:,list_keep]

        #dataframe for normalization
        dfas = dfin.drop(list_keep, axis = 1)
        
        #parameters seit in function
        #Only decrease the background if the median is higher than the background
        dfa = dfas.apply(lambda x: np.arcsinh(x/ArcSin_cofactor))
        
        #Add back labels for normalization type
        dfz_all = pd.concat([dfa, df_loc], axis=1, join='inner')
        
        return dfz_all

    ##Double Z normalization
    elif(method == "double_zscore"):
        
        #Drop column list
        list1 = [col for col in data.columns if 'blank' in col]
        list_out1 = list1+list_out
        
        #Drop columns not interested in
        dfin = data.drop(list_out1, axis = 1)

        #save columns for later
        df_loc = dfin.loc[:,list_keep]

        #dataframe for normalization
        dfz = dfin.drop(list_keep, axis = 1)

        #zscore of the column markers 
        dfz1 = pd.DataFrame(zscore(dfz,0),index = dfz.index,columns = [i for i in dfz.columns])
        
        #zscore rows 
        dfz2 = pd.DataFrame(zscore(dfz1,1),index = dfz1.index,columns = [i for i in dfz1.columns])
        
        #Take cumulative density function to find probability of z score across a row
        dfz3 = pd.DataFrame(norm.cdf(dfz2),index = dfz2.index, columns = [i for i in dfz2.columns])
        
        #First 1-probability and then take negative logarithm so greater values demonstrate positive cell type
        dflog = dfz3.apply(lambda x: -np.log(1-x))
        
        #Add back labels for normalization type
        dfz_all = pd.concat([dflog, df_loc], axis=1, join='inner')
        
        #print("the number of regions = "+str(len(dfz_all.region_num.unique())))
        
        return dfz_all
        
    #Min Max normalization
    elif(method == "MinMax"):
        #Drop column list
        list1 = [col for col in data.columns if 'blank' in col]
        list_out1 = list1+list_out
        
        #Drop columns not interested in
        dfin = data.drop(list_out1, axis = 1)

        #save columns for later
        df_loc = dfin.loc[:,list_keep]

        #dataframe for normalization
        dfmm = dfin.drop(list_keep, axis = 1)
        
        for col in dfmm.columns:
            max_value = dfmm[col].quantile(.99) 
            min_value = dfmm[col].quantile(.01)
            dfmm[col].loc[dfmm[col] > max_value] = max_value
            dfmm[col].loc[dfmm[col] < min_value] = min_value
            dfmm[col] = (dfmm[col] - min_value) / (max_value - min_value) 
            
        #Add back labels for normalization type
        dfz_all = pd.concat([dfmm, df_loc], axis=1, join='inner')
        
        return dfz_all
    
    ## Z normalization
    else:
        #Drop column list
        list1 = [col for col in data.columns if 'blank' in col]
        list_out1 = list1+list_out
        
        #Drop columns not interested in
        dfin = data.drop(list_out1, axis = 1)

        #save columns for later
        df_loc = dfin.loc[:,list_keep]

        #dataframe for normalization
        dfz = dfin.drop(list_keep, axis = 1)

        #zscore of the column markers 
        dfz1 = pd.DataFrame(zscore(dfz,0),index = dfz.index,columns = [i for i in dfz.columns])

        #Add back labels for normalization type
        dfz_all = pd.concat([dfz1, df_loc], axis=1, join='inner')
        
        #print("the number of regions = "+str(len(dfz_all.region_num.unique())))
            
        return dfz_all

   




# Only useful for "classic CODEX" where samples are covered by multiple regions 
# Could also be used for montages of multiple samples (tiles arraged in grid)
def pp_xycorr(data, y_rows, x_columns, X_pix, Y_pix):
    
    #Make a copy for xy correction
    df_XYcorr = data.copy()
    df_XYcorr["Xcorr"] = 0
    df_XYcorr["Ycorr"] = 0
    dict_test = dict(enumerate(df_XYcorr.region_num.unique()))
    dict_map = {v:k+1 for k,v in dict_test.items()}
    df_XYcorr['regloop'] = df_XYcorr['region_num'].map(dict_map)
    region_num = df_XYcorr.regloop.max()

    #first value of tuple is y and second is x
    d = list(product(range(0,y_rows,1),range(0,x_columns,1)))
    e = list(range(1,region_num+1,1))
    dict_corr = {}
    dict_corr = dict(zip(e, d)) 

    #Adding the pixels with the dictionary
    for reg_num in list(df_XYcorr['regloop'].unique()):
        df_XYcorr["Xcorr"].loc[df_XYcorr["regloop"]== reg_num] = df_XYcorr['x'].loc[df_XYcorr['regloop']==reg_num] +dict_corr[reg_num][1]*X_pix

    for reg_num in list(df_XYcorr['regloop'].unique()):
        df_XYcorr["Ycorr"].loc[df_XYcorr["regloop"]== reg_num] = df_XYcorr['y'].loc[df_XYcorr['regloop']==reg_num] +dict_corr[reg_num][0]*Y_pix

    df_XYcorr.drop(columns=['regloop'],inplace=True)
    return df_XYcorr




#Get rid of noisy cells from dataset
def pp_remove_noise(df, col_num=24, z_sum_thres=22, z_count_thres=20):
    df_z_1_copy = df.copy()
    df_z_1_copy['Count']=df_z_1_copy.iloc[:,:col_num+1].ge(0).sum(axis=1)
    df_z_1_copy['z_sum']=df_z_1_copy.iloc[:,:col_num+1].sum(axis=1)
    cc = df_z_1_copy[(df_z_1_copy['z_sum']>z_sum_thres) & (df_z_1_copy['Count']>z_count_thres)]
    df_want = df_z_1_copy[~((df_z_1_copy['z_sum']>z_sum_thres) & (df_z_1_copy['Count']>z_count_thres))]
    percent_removed = 1- len(df_want)/len(df_z_1_copy)
    print(str(percent_removed) + " percent of cells are removed.")
    #ee = df_z_1_copy['Count'].plot.hist(bins=50, alpha=0.8,logy = False)
    #dd = df_z_1_copy['z_sum'].plot.hist(bins=50, alpha=0.8,logy = False)
    df_want.drop(columns=['Count','z_sum'], inplace=True)
    df_want.reset_index(inplace=True, drop=True)
    return df_want, cc

def pp_clust_leid(adata, res=1, Matrix_plot=True):
    #Compute the neighborhood relations of single cells the range 2 to 100 and usually 10
    sc.pp.neighbors(adata, n_neighbors=10)
    
    #Perform leiden clustering - improved version of louvain clustering
    sc.tl.leiden(adata, resolution = res, key_added = "leiden")
    
    #UMAP computation
    sc.tl.umap(adata)
    
    plt.rcParams["legend.markerscale"] = 1
    sc.pl.umap(adata, color=['leiden'])
    
    m_list = adata.var.index.to_list()
    
    #Create matrix plot with mean expression per each cluster
    if Matrix_plot==True:
        sc.pl.matrixplot(adata, m_list, 'leiden',standard_scale='var')
    
    return adata, m_list

