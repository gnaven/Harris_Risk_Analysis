#!/usr/bin/env python
"""
------------------------------------------------------------------------
  Given:
    frames.csv 
    k range
    
  creates:
    clusters_k.csv
    clusters_k.png
    scores.csv 

  example:
      $ python -i 'example/test.csv'
------------------------------------------------------------------------
"""
from __future__ import print_function
import csv
import numpy as np
import pandas as pd
import math
from scipy import linalg
from scipy.stats import multivariate_normal
import operator 
import concurrent.futures as cf

#import compare
from sklearn import cluster
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabaz_score
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn import mixture

import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from sklearn.preprocessing import scale

import glob
import os
import sys
import argparse
import logging
logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)

from collections import defaultdict
from itertools import combinations
import itertools

# you can set the logging level from a command-line option such as:
#   --log=INFO


#-------------------------------------------------------------------------------
class ClusterSearch:
    """ 
    
    public methods:
    :ClusterSearch(inputfile) - specify file containing openface frames
    :feature_search - runs clustering for subsets of specified features, k_range 
    
    :k_search   - runs sklearn kmeans over range of k values
    :gmm_search - runs sklearn gmm over range of k
    :bin_search - counts frequencies for all possible binary subsets
    
    :write_clusters
    :write_scores
    :write_plots
    """
    
    def __init__(s, df,k_num=None):
        log.info('...ClusterSearch initialized')
        s.df = df
        s.CH = 0 
        s.ncpu = 3
        s.y_clust = None
        s.k_range = k_num
        s.df_clustResult = None
    
    #-------------------------------------------
    def feature_search(s, features_cluster, k_range, max_dim, f_search):
        """ runs f_search within k_range for all subsets of features up 
        to max_dim. Output results are saved by f_search.
        """
        df_list=[]
        
        for num_features in range(1, max_dim+1):
            feature_combos = list(combinations(features_cluster, num_features))
            feature_combo_list = [x for x in feature_combos if len(list(x))>5]
            if len(feature_combo_list)==0:
                continue
            if s.ncpu !=1:
                with cf.ProcessPoolExecutor(max_workers=s.ncpu) as ex:
                    args = feature_combo_list
                    for arg,df in zip(args, ex.map(f_search,args)):
                        features = [arg]*df.shape[0]
                        df['features']=features
                        df_list.append(df)                        
            else:
                for feature_subset in feature_combo_list:
                    
                    #if len(feature_subset) >1 :
                    df= f_search(feature_subset)
                    features = [feature_subset]*df.shape[0]
                    df['features']=features
                    df_list.append(df)
        s.df_clustResult = pd.concat(df_list)
        #s.df['Risk_Clust'] = s.y_clust
        #return df_final
      
    #-----------------------------------------------------------------------------
    def k_search(s,features):
        """ Runs kmeans, saves cluster file, saves plot file,
            returns silhoutte and Calinski scores 
        """
        logging.info('starting cluster search' )
        logging.info('\tfeatures=' + str(features))
        k_range= s.k_range
        cluster_list = []
        sil_scores = []
        ch_scores = []
        X = s.df.loc[:,features].dropna().values
        #X = scale(X)
        logging.info('\tX.shape' + str(X.shape))
        score_fname = '../output_allFrames/scores_' + \
            '_'.join(features).replace(' ','')+ '.csv'
        df_clust = pd.DataFrame(columns=['#clusters','features','CH_score','Sil_score'])         
        CH_now = s.CH
        y_clust = None
        for k in k_range:
            png_fname = '../output_allFrames/clusters_' + '_'.join(features).replace(' ','')\
                + '_' + str(k) + '.png'
            cluster_fname = 'output/risk_clusters_' + \
                '_'.join(features).replace(' ','') + '_' + str(k) + '.csv'
            #if os.path.isfile(png_fname):
                #logging.info('file exists, skipping...' + png_fname)
                #continue
            logging.info('\tk=' + str(k) )
            k_means = cluster.KMeans(n_clusters=k+1, max_iter=1000, n_jobs=-1)
            k_means.fit(X)
            y = k_means.predict(X)
            sil_score = s.calc_sil_score(X,y)
            ch_score = calinski_harabaz_score(X,y)
            df_clust.loc[k,'#clusters'] = k+1
            df_clust.loc[k,'CH_score'] = ch_score
            df_clust.loc[k,'Sil_score'] = sil_score
            
            logging.info('silhouette score with ' + str(k) + ' clusters: ' + \
                             '{0:.3f}'.format(sil_score))
            logging.info('CH score with ' + str(k) + ' clusters: ' + \
                             '{0:.3f}'.format(ch_score))
            
            clusters = k_means.cluster_centers_
                
            # write the clusters to a csv file
            s.write_clusters(cluster_fname, features, clusters)
            

            
            #if ch_score >= CH_now:
                #CH_now = ch_score
                #s.y_clust = y
                
            
            #subtitle = 'sil score: ' + '{:.3f}'.format(sil_score)
            #subtitle += ', CH score: ' + '{:.2E}'.format(ch_score)
            #s.write_plots(png_fname, features, X,[clusters], subtitle)  
            #s.write_scores(score_fname, features, [k], [sil_score], [ch_score])

            #cluster_list.append(clusters)
            #sil_scores.append(sil_score)
            #ch_scores.append(ch_score)
        
        #if len(cluster_list) > 1:
            #k_start = cluster_list[0].shape[0]
            #k_end = cluster_list[-1].shape[0]
            #fname = 'output_allFrames/clusters_' + '_'.join(features).replace(' ','') + '_R' + \
                                    #str(k_start) + 'to' + str(k_end) + '.png'
            #s.save_score_plot(score_fname.replace('.csv','.png'), k_range, ch_scores)
        #s.CH = CH_now    
        return df_clust
    
    #-------------------------------------------------------------------------------
    def bin_search(s, k_range, features):
        """ Counts all possible binary combinations of features in data. 
        """
        log.info('starting bin search' )
        log.info('\tfeatures=' + str(features))

        cluster_list = []
        sil_scores = []
        ch_scores = []
        X = s.df.loc[:,features].dropna().values
        log.info('\tX.shape' + str(X.shape))
        count_dict = {}
        for i in list(itertools.product([0, 1], repeat=len(features))):    
            count_dict[i] = 0
            
        count_tensor = np.zeros(len(features), dtype=int)
        for i in range(X.shape[0]):
            #count_tensor[X.iloc[i]]] += 1
            count_dict[tuple(X[i])] += 1
        
        sorted_counts = sorted(count_dict.items(), key=operator.itemgetter(1))
        total = sum(count_dict.values())
        print(features)
        for v,k in sorted_counts:
            print(v ,  ' {:.3}'.format(k/total))
            
        return    
    
    #-------------------------------------------------------------------------
    
    def Choose_feats():
        """ Finds the best cluster for each cluster combination and then sends the df to 
            best_clust to output the data.
        """
        
    
    def best_clust(s):
        """
        Runs clustering algorithm again on the best cluster number and features 
        used combination
        
        returns the best clusters formed based on.
        if best cluster consists of the same features and 
        number of clusters formed, it wont make a new k_means fit
        """
        
        
        df_CH = s.df_clustResult[s.df_clustResult['CH_score'] == 
                                 s.df_clustResult['CH_score'].max()]
        df_sill = s.df_clustResult[s.df_clustResult['Sil_score'] == 
                                 s.df_clustResult['Sil_score'].max()]
        
        C_CH = df_CH.iloc[0]['#clusters']
        C_sill = df_sill.iloc[0]['#clusters']
        
        feats_CH =  list(df_CH.iloc[0]['features'])
        feats_sill = list(df_sill.iloc[0]['features'])
                
        if df_CH.index[0] == df_sill.index[0] and set(feats_CH) == set(feats_sill):
            X = s.df.loc[:,feats_CH].dropna().values
            k_means = cluster.KMeans(n_clusters=C_CH, max_iter=1000, n_jobs=-1)
            k_means.fit(X)
            y_CH = k_means.predict(X)
            y_sill = y_CH
        else:
            X = s.df.loc[:,feats_sill].dropna().values
            
            k_means_CH = cluster.KMeans(n_clusters=C_CH, max_iter=1000, n_jobs=-1)
            k_means_CH.fit(X)
            y_CH = k_means_CH.predict(X)
            
            k_means_sill = cluster.KMeans(n_clusters=C_sill, max_iter=1000, n_jobs=-1)
            k_means_sill.fit(X)
            y_sill = k_means_sill.predict(X)   
            
        s.df['clust_CH'] = y_CH
        s.df['clust_sill'] = y_sill
    
    #-------------------------------------------------------------------------
    def write_clusters(s, outfile, features, clusters):
        """ Writes cluster means to a csv file. 
        """
        
        with open(outfile, 'w') as f:
            feature_data = []
            writer = csv.writer(f)
            writer.writerow(features)
        
            for cluster_i in clusters:
                row_txt = [str(x) for x in list(cluster_i)]
    
                writer.writerow(row_txt)        
        
    #-------------------------------------------
    def calc_sil_score(s,X,y):
        """ Calculate silhoutte score with efficiency mods.
        """
        if(X.shape[0] > 5000):
            # due to efficiency reasons, we need to only use subsample
            sil_score_list = []
            for i in range (0,100):
                X_train, X_test, y_train, y_test = train_test_split(X, y, \
                                                     test_size=2000./X.shape[0])        
                sil_score = silhouette_score(X_test,y_test)
                sil_score_list.append(sil_score)
            sil_score_avg = np.nanmean(sil_score_list)
        else:   
            sil_score_avg = silhouette_score(X,y)

        return sil_score_avg
        
#------------------------------------------------------------------------

def run_datasets(args):
    filepath = args.i
    files = glob.glob(filepath)
    
    for file in files:
        do_all(args,file)
        

def do_all(args,file):
    
    if not os.path.isdir('output'):
        os.mkdir('output')
        
    if 'csv' in file:
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)
    c_search = ClusterSearch(df,k_num=range(2,args.k+1))    
    
    if 'bin' in file:
        output_folder = 'Feat_bin/'
    elif '3' in file:
        output_folder = 'Feat_3/'
    elif '5' in file:
        output_folder = 'Feat_5/'
    else:
        output_folder =''
        
    if not os.path.isdir('../output3/'+output_folder):
        os.mkdir('../output3/'+output_folder)  
        
    output_fname = file.split('.csv')[0]
    output_fname = output_fname.split('\\')[-1]
    print('folder ',output_fname)
    output_dir = '../output3/' + output_folder + output_fname
    #features=['var_log_pocost', '%_early', 'mean_log_return_spend', 'mean_log_costsave', 
                  #'median_log_return_spend', 'sum_log_return_spend', 'sum_log_costsave', 
                  #'median_log_leadtime', 'variance_log_leadtime', 'mean_log_spend', 
                  #'var_log_spend', 'mean_log_qty', 'mean_priority', 'mean_log_return_qty', 
                  #'sum_log_spend', 'var_log_return_qty', '%_late', 'mean_log_stdcost', 
                  #'var_log_return_spend', 'mean_log_leadtime', 'sum_log_qty', 'median_log_spend', 
                  #'sum_log_return_qty', 'num_returns_purch', 'median_log_qty', 'avg_early', 
                  #'mean_log_pocost', '%_critical', 'median_log_return_qty', 'num_transactions_expo', 
                  #'var_log_stdcost', 'num_orders_purch', 'median_log_stdcost', 'var_log_costsave', 
                  #'var_log_qty', '%_singlesource', 'median_log_costsave', 'median_log_pocost', 'avg_late']
    features = ['mean_log_costsave', 'mean_log_spend', 
                'mean_log_qty', 'mean_priority', 'mean_log_stdcost', 
                'mean_log_leadtime', 'mean_log_pocost']
    features2=['var_log_pocost', '%_early', 'mean_log_return_spend', 'mean_log_costsave', 
              'median_log_return_spend', 'sum_log_return_spend', 'sum_log_costsave', 
              'median_log_leadtime', 'variance_log_leadtime', 'mean_log_spend', 
              'var_log_spend', 'mean_log_qty', 'mean_priority', 'mean_log_return_qty', 
              'sum_log_spend', 'var_log_return_qty', '%_late', 'mean_log_stdcost', 
              'var_log_return_spend', 'mean_log_leadtime', 'sum_log_qty', 'median_log_spend', 
              'sum_log_return_qty', 'num_returns_purch', 'median_log_qty', 'avg_early', 
              'mean_log_pocost', '%_critical', 'median_log_return_qty', 'num_transactions_expo', 
              'var_log_stdcost', 'num_orders_purch', 'median_log_stdcost', 'var_log_costsave', 
              'var_log_qty', '%_singlesource', 'median_log_costsave', 'median_log_pocost', 'avg_late']
    features3 = [x for x in df.columns if 'comp' in x]

    max_d = len(features3)

    if args.t == 'km':
        c_search.feature_search(features3, range(2,args.k+1), max_d, c_search.k_search)
        c_search.best_clust()
        c_search.df_clustResult.to_csv(output_dir+'_Cluster_Output.csv')
        c_search.df.to_csv(output_dir+'_Harris_data_clust.csv')
        
    elif args.t == 'gmm':
        c_search.feature_search(features, range(1,args.k+1), max_d, c_search.gmm_search)
    elif args.t == 'bin':
        features=['var_log_pocost', '%_early', 'mean_log_return_spend', 'mean_log_costsave', 
                  'median_log_return_spend', 'sum_log_return_spend', 'sum_log_costsave', 
                  'median_log_leadtime', 'variance_log_leadtime', 'mean_log_spend', 
                  'var_log_spend', 'mean_log_qty', 'mean_priority', 'mean_log_return_qty', 
                  'sum_log_spend', 'var_log_return_qty', '%_late', 'mean_log_stdcost', 
                  'var_log_return_spend', 'mean_log_leadtime', 'sum_log_qty', 'median_log_spend', 
                  'sum_log_return_qty', 'num_returns_purch', 'median_log_qty', 'avg_early', 
                  'mean_log_pocost', '%_critical', 'median_log_return_qty', 'num_transactions_expo', 
                  'var_log_stdcost', 'num_orders_purch', 'median_log_stdcost', 'var_log_costsave', 
                  'var_log_qty', '%_singlesource', 'median_log_costsave', 'median_log_pocost', 'avg_late']
        c_search.feature_search(features, range(1,args.k+1), 3, c_search.bin_search)        
    
#------------------------------------------------------------------------
if __name__ == '__main__':

    # Setup commandline parser
    help_intro = 'Program for running clustering on features subsets.'
    help_intro += '  example usage:\n\t$ ./cluster.py -i \'example/test.csv\''
    help_intro += ' -t gmm -k 6'
    parser = argparse.ArgumentParser(description=help_intro)

    parser.add_argument('-i', help='inputs, ex:example/test.csv', type=str, 
                        default='Harris Data/new_data/PCA_results/*.csv')
    parser.add_argument('-t', help='type: gmm, km', type=str, 
                        default='km')
    parser.add_argument('-k', help='maximum k', type=int, 
                        default=20)
    
    args = parser.parse_args()
    

    run_datasets(args)
    log.info('PROGRAM COMPLETE')
