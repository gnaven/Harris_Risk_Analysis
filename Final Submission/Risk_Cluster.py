#!/usr/bin/env python3

"""
This script will run some clustering frameworks on the Harris Dataset 
to search optimal number of clusters and what features in teh cluster

"""
from __future__ import print_function
import csv
import numpy as np
import pandas as pd
import math
from scipy import linalg
from scipy.stats import multivariate_normal
import operator 

#import compare
from sklearn import cluster
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabaz_score
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn import mixture

import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

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

class ClusterSearch:
    
    
    def __init__(s,df):
        s.df= df
    
    def k_search(s, k_range,features):
        """
        runs k-means algorithm 
        """
        
        logging.info('starting cluster search' )
        logging.info('\tfeatures=' + str(features))

        cluster_list = []
        sil_scores = []
        ch_scores = []
        X_n = s.df.loc[:,features].dropna().values
        X_n_mean = X_n.mean()
        X_n_var  = X_n.var()        
        X= (X_n - X_n_mean)/(X_n_var**.5)
        logging.info('\tX.shape' + str(X.shape))
        score_fname = '../output/scores_' + \
            '_'.join(features).replace(' ','')+ '.csv'
        for k in k_range:
            png_fname = '../output_allFrames/clusters_' + '_'.join(features).replace(' ','')\
                + '_' + str(k) + '.png'
            cluster_fname = '../output_allFrames/face_clusters_' + \
                '_'.join(features).replace(' ','') + '_' + str(k) + '.csv'
            if os.path.isfile(png_fname):
                logging.info('file exists, skipping...' + png_fname)
                #continue
            logging.info('\tk=' + str(k) )
            k_means = cluster.KMeans(n_clusters=5, max_iter=1, n_jobs=-1)
            k_means.fit(X)
            y = k_means.predict(X)
            
            s.df['risk']= pd.Series(y)
            print(s.df.columns)
            #sil_score = s.calc_sil_score(X,y)
            #ch_score = calinski_harabaz_score(X,y)
            
            #logging.info('silhouette score with ' + str(k) + ' clusters: ' + \
                             #'{0:.3f}'.format(sil_score))
            #logging.info('CH score with ' + str(k) + ' clusters: ' + \
                             #'{0:.3f}'.format(ch_score))
            
            #clusters = k_means.cluster_centers_
                
            ## write the clusters to a csv file
            #s.write_clusters(cluster_fname, features, clusters)
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
            
        
        return s.df#sil_scores, ch_scores
    
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
    #-------------------------------------------
    
    def write_plots(s, fname, header, data, cluster_list,  subtitle=None):
        """ Save png files for plots of data with clusters.
            TODO: plot tsne when d > 2.
        """
        num_clusters = len(cluster_list)
        if num_clusters == 0:
            return
        num_col = math.ceil(num_clusters/2.)
        num_row = math.ceil(num_clusters/float(num_col))
        plt.figure(figsize=(8,8))
        
        for i,clusters in enumerate(cluster_list):
            k = clusters.shape[0]
            d = clusters.shape[1]
            if d > 2:
                continue
            myTitle = 'k=' + str(k) + ' '
            if subtitle:
                myTitle += ', ' + subtitle

            if d==2:
                if len(cluster_list) > 1:
                    plt.subplot(num_row, num_col,i+1)                                
                    plt.scatter(data[:,0], data[:,1], alpha = 0.01)
                    plt.scatter(clusters[:,0], clusters[:,1], s=500, \
                                c='red', alpha =0.5)
                    plt.xlabel(header[0])
                    plt.ylabel(header[1])
                else:
                    nullfmt = NullFormatter()         # no labels
                    
                    # definitions for the axes
                    left, width = 0.1, 0.65
                    bottom, height = 0.1, 0.65
                    bottom_h = left_h = left + width + 0.05
                    
                    rect_scatter = [left, bottom, width, height]
                    rect_histx = [left, bottom_h, width, 0.2]
                    rect_histy = [left_h, bottom, 0.2, height]
                    
                    axScatter = plt.axes(rect_scatter)
                    axHistx = plt.axes(rect_histx)
                    axHisty = plt.axes(rect_histy)
                    
                    # no labels
                    axHistx.xaxis.set_major_formatter(nullfmt)
                    axHisty.yaxis.set_major_formatter(nullfmt)
                    
                    # the scatter plot:
                    axScatter.scatter(data[:,0], data[:,1], alpha = 0.01)
                    axScatter.scatter(clusters[:,0], clusters[:,1], s=500, \
                                      c='red', alpha =0.5)
                    
                    # now determine nice limits by hand:
                    binwidth = 0.1
                    xymax = np.max([np.max(np.fabs(data[:,0])), \
                                    np.max(np.fabs(data[:,1]))])
                    lim = (int(xymax/binwidth) + 1) * binwidth
                    
                    axScatter.set_xlim((0, lim))
                    axScatter.set_ylim((0, lim))
                    
                    bins = np.arange(0, lim + binwidth, binwidth)
                    axHistx.hist(data[:,0], bins=bins, color='green', \
                                 histtype='bar', ec='black', normed=1)
                    axHisty.hist(data[:,1], bins=bins, orientation='horizontal',\
                                 histtype='bar', ec='black' , normed=1)
                    
                    axHistx.set_xlim(axScatter.get_xlim())
                    axHisty.set_ylim(axScatter.get_ylim())                
                    axScatter.set_title(myTitle)
                    axScatter.set_xlabel(header[0])
                    axScatter.set_ylabel(header[1])
                
            else:
                if len(cluster_list) > 1:
                    plt.subplot(num_row, num_col,i+1)        
                plt.hist(data[:,0], 50, color='green', \
                     histtype='bar', ec='black', normed=1)                    
                #plt.scatter(data[:,0], np.ones(data.shape[0]), alpha = 0.01)
                plt.scatter(clusters[:,0], np.zeros(clusters.shape[0]), s=500, \
                            c='red')
                plt.xlabel(header[0])
                    
                plt.title(myTitle)
                #plt.tight_layout()            
        plt.savefig(fname)
        plt.close()        

#-------------------------------------------
    def write_scores(s, score_fname, features, k_range, sil_scores, ch_scores):
        df_scores= pd.DataFrame.from_items([
                ('features', '_'.join(features).replace(' ','')),
                ('sil score', sil_scores),
                ('CH score', ch_scores)])
        df_scores.index = k_range[0:len(sil_scores)]
        # append to score file if exists, otherwise create new
        if os.path.isfile(score_fname):
            with open(score_fname, 'a') as f:
                df_scores.to_csv(f, header=False,index_label='k')        
        else:
            df_scores.to_csv(score_fname, index_label='k')   
#----------------------------------------------------------------------------
        
def do_clust():
    
    DATA = 'Harris_Data/Expo_Purchasing_join.xlsx'
    df = pd.read_excel(DATA)
    c_search = ClusterSearch(df)
    
    features= ['Late Need', 'Late Performance','StdCost','PoCost','Qty','InternalCostSavings',
               'Spend','PurchLeadTime','ABC']
    

    clust_df = c_search.k_search(range(1,2),features)
    
    clust_df.to_csv('../output/Risk_clusters.csv')

if __name__ == '__main__':
    
    do_clust()
