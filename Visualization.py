#!/usr/bin/env python
"""
In this script we visualize data for clusters formed.
Given: 
Data -> contains features (normal and PCA) and corresponding 
        cluster values and risk scores
Feature names -> cluster column, risk score column
PCA meta data -> info on weights for each features

output:
1. TSne plot showing the cluster formation
2. Hist on proportion of each cluster's PCA dim
3. Hist on proportion on each clusters Risk scores
4. Risk level of each feature in each cluster

"""
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from matplotlib.pyplot import scatter
from sklearn.manifold import TSNE
from sklearn.preprocessing import scale
import glob
import os
import sys
import argparse

from scipy.stats import f_oneway
from scipy.stats import kruskal
#import statsmodels.stats.multicomp.pairwise_tukeyhsd as TukeyHSD
#import scikit_posthocs as sp
class Visualization:
    """
    methods:
    Labels_to_color: Gives color labels 
    TSNE: 
    PCA_Dist: 
    """
    
    def __init__ (s,df,pcaWeights,features,cluster_col,risk_score_col,results_directory):
        
        s.features = features
        s.df = df
        s.pca_W = pcaWeights   
        s.clust_col= cluster_col
        s.risk_col = risk_score_col
        s.resultsdir = results_directory
        
    #-------------------------------------------------------------------------
    def Labels_to_color (s,label_col):
        """
        For a dataframe (df) and its classification (classif) it 
        turns each class label to numerical values and returns the dataframe
        """
        classif = s.clust_col
        s.label = label_col
        ClassList= s.df[classif].unique()
        id = 0
        color = ['b', 'g', 'r', 'y',  'w', 'c','k','m']
        marker = ['o','x','+','^','s','v']
        ClassDict = {}
        for ClassVal in ClassList:
            
            ClassDict[ClassVal]= color[id]
            id = id+1
        for ClassNum in ClassDict.keys():
            s.df.ix[s.df[classif]==ClassNum,label_col] = ClassDict[ClassNum]
        #df=df.drop(classif,axis =1)
            
        return ClassDict
    
    def TSNE (s,png_name,features = None):
        """
        In this method TSNE dimensionality reduction is applied to the features in the 
        dataframe.
        If features are given then the features are used, otherwise all features 
        except any labels are used
        """
        col_clust_label= s.Labels_to_color('marker')
        df_plot = s.df.dropna()
        if features == None:
            col_label = df_plot['marker']
            df_plot = df_plot[features]#df_plot.drop([s.clust_col,s.risk_col,'marker'],axis=1)
        else:
            col_label = df_plot['marker']
            df_plot = df_plot[features]
        
        X_tsne = TSNE(learning_rate=50).fit_transform(df_plot)
        
        plt.figure(figsize=(10, 5))
        plt.scatter(X_tsne[:, 0], X_tsne[:, 1],color=col_label,s=80,alpha = 0.5,edgecolors ='black')
        plt.legend()
        plt.savefig(s.resultsdir + 'TSNE_plot'+png_name+'.png')
        plt.close() 
    #------------------------------------------------------------------------------------------------    
    #need to review this
    def PCA_Dist (s):
        """
        Visualizes the distribution of PCA features in each cluster
        """
        PCA_feats = s.pca_W
        print(list(s.df.columns))
        df_grp = s.df[PCA_feats+[s.clust_col]].groupby(s.clust_col).mean().plot(kind='bar',figsize=(5,10))
        plt.xlabel('Clusters')
        plt.ylabel('Mean risk scores')
        plt.savefig(s.resultsdir + 'PCA_dist_ClusterCenters.png')
        plt.close()
        #plt.show()
    
    #------------------------------------------------------------------------------------------------
    
    def Risk_Scores(s):
        """
        Given dataframe and maingroup: group of columns used
        for grouping and making pivot table
        """
        n = len(list(s.df[s.clust_col].unique()))
        s.df[s.risk_col] = pd.cut(s.df[s.risk_col],bins = n) 
        
        maingroup = [s.risk_col,s.clust_col,'SupplierID']
        df= s.df[maingroup].groupby([s.clust_col,s.risk_col]).count().reset_index()
        df=df.rename(columns={'SupplierID':'count'})
        df.pivot(s.clust_col,s.risk_col, "count").plot(kind='bar',figsize=(5,10))
        plt.xlabel('Risk Cluster')
        plt.ylabel('Risk Tag counts')
        plt.savefig(s.resultsdir + 'Risk_Score_CLusterCenters.png')
        plt.close()
    #------------------------------------------------------------------------------------------------
        
    def Feats_Risk_Scores(s):
        """
        Given: dataframe with Id and clusters and dataframe with original features
        Outputs : Mean risk level of each features in each clusters
        """
        # normalizes the risk sum according to the number of clusters formed

        df = s.df[s.features+[s.clust_col]]
        df = df.groupby([s.clust_col]).median().T.reset_index()
        # adding the statistical test column
        p_val_kruskal_list = []
        f_val_kruskal_list = []
        f_val_anova_list = []
        p_val_anova_list = []
        for feats in s.features:
            f_val_kruskal, p_val_kruskal,f_val_anova,p_val_anova = s.stat_test(feats)
            p_val_kruskal_list.append(p_val_kruskal)
            f_val_kruskal_list.append(f_val_kruskal)
            f_val_anova_list.append(f_val_anova)
            p_val_anova_list.append(p_val_anova)
        df['p_val_kruskal'] = p_val_kruskal_list
        df['f_val_kruskal'] = f_val_kruskal_list
        df['p_val_anova'] = p_val_anova_list
        df['f_val_anova'] = f_val_anova_list
        
        df.to_csv(s.resultsdir+'cluster_feat_means_pvals.csv')
    #------------------------------------------------------------------------------------------------   
    def feat_clust_dist(s):
        """
        From each cluster shows the distribution of the each features 
        """
        df_gr = s.df.groupby(s.clust_col)
        results_hist_dir = s.resultsdir+ 'dists'
        if not os.path.isdir(results_hist_dir):
                os.mkdir(results_hist_dir)        
        
        for feature in s.features:
            df_gr[feature].plot(kind='hist', bins=20, figsize=[12,6], alpha=.4, legend=True)
            plt.savefig(results_hist_dir+'/'+feature)
            plt.close()
        

    #------------------------------------------------------------------------------------------------    
    def stat_test(s,feat = 'mean_unexpcost'):
        """ Runs One way ANOVA test for each of the means across the clusters
        """
        clust_labels = s.df[s.clust_col].unique()
        df_gr_list = []
        for c in clust_labels:
            df_gr = s.df[s.df[s.clust_col]==c][feat]
            df_gr_list.append(df_gr)
        f_val_kruskal,p_val_kruskal = kruskal(*df_gr_list)
        f_val_anova,p_val_anova = f_oneway(*df_gr_list)
        
        
        # TODO: Work on finding post hoc analysis of statistical test
        # TODO: Run clustering again 
        #sp.posthoc_
        
        return f_val_kruskal,p_val_kruskal,f_val_anova,p_val_anova 
       # print('p_val ', p_val)
    
    def run_viz(s):
        tsne_feats = [x for x in s.df.columns if 'comp' in x]
        s.TSNE('_pca_comp_',tsne_feats)
        s.PCA_Dist()
        s.feat_clust_dist()
        s.Risk_Scores()
        s.Feats_Risk_Scores()
        s.stat_test()
        s.TSNE('_raw_data_',s.features)
        

#----------------------------------------------------------------------------------------------------

def run_outputs(args,folder_list):
    """ Runs visualization on all the 
        cluster files
    """

    
    for folder in folder_list:
        #directory = args.o + folder
        input_filepath = args.i + folder + '/*clust.csv'
        files = glob.glob(input_filepath)
        for file in files:
            out_fname = file.split('.csv')[0]
            out_fname = out_fname.split('\\')[-1]
            out_fname = out_fname.split('_Harris')[0]
            directory = args.o + folder + '/'+ out_fname +'/'
            if not os.path.isdir(directory):
                os.mkdir(directory)
                    
            df_all = pd.read_csv(file)
            mean_feats = [x for x in df_all.columns if 'mean' in x]
            median_feats = [x for x in df_all.columns if 'median' in x]
            pca_weights = [x for x in df_all.columns if 'comp' in x]
            
            Clust_Viz = Visualization(df=df_all, pcaWeights=pca_weights, features= mean_feats, cluster_col=args.c,
                                      risk_score_col=args.r, results_directory=directory)
            Clust_Viz.run_viz()            
            
        

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
      
    
    parser.add_argument('-i', help='input, ex:example/test.csv', type=str, 
                        default='../output3/')
    parser.add_argument('-c', help='cluster label, ex:example/test.csv', type=str, 
                        default='clust_CH')   
    parser.add_argument('-r', help='Risk Score, ex:example/test.csv', type=str, 
                        default='RiskSum')   
    parser.add_argument('-o', help='results directory, ex:example/test.csv', type=str, 
                        default='../Results_all/Results5/')    
    
    args = parser.parse_args()
    #df_all = pd.read_csv(args.i)
    
    #mean_feats = [x for x in df_all.columns if 'mean' in x]
    #median_feats = [x for x in df_all.columns if 'median' in x]
    #pca_weights = [x for x in df_all.columns if 'comp' in x]
    
    #dir_name = args.o+args.c +'/'
    #Clust_Viz = Visualization(df=df_all, pcaWeights=pca_weights, features= mean_feats, cluster_col=args.c,
                              #risk_score_col=args.r, results_directory=dir_name)
    #Clust_Viz.run_viz()
    folders = ['Feat_3','Feat_5','Feat_bin']
    run_outputs(args,folders)
    
    
