Files:

How to run code:
Data Prep:
	Run data_prep.py
Dimensionality Reduction:
	Run PCA_VM_FA_script_join.ipynb

- Final Submission
	- Final Figures: Figures generated during the project
	- Harris Data
		- Results5: All the clustering results for our models -- Details Belo
		- old_data: data from Harris:
			- 3 files: scorecard, purchasing tool, and expo archive

		- data_prep.py: MAIN DATA PREP
			- Cleans the data for modeling
			- Bins the data values
			- Does a 3-way join on the data and aggregates values(no unique key)
			- Engineers the feature unexp_cost
			- Engineers the "mean" and "median" features for aggregated values
			- Produces 3 models with binning structures of 2,3, and 5 bins.
		- distribution_analysis.ipynb: Plots the distributions of the data after preparation

		- prepared_data: 3 csv dataframe files -- Results of data prep

		- PCA_VM_FA_script_join.ipynb: MAIN DIMENSIONALITY REDUCTION
			- Generates new risk sum
			- Generates somponent features with:
				- Principal Component Analysis
				- PCA+Varimax
				- Factor Analysis models.
		- PCA_NoiseReduceFigure.ipynb: Generates an example graphic of pca noise reduction.
		- PLOT_PCA.ipynb: Plots figures for the distributions of the data on the PCA model
		- PLOT_Varimax.ipynb: Plots figures for the distributions of the data on the PCA + Varimax Model.

		- PCA_VM_FA_results: 12 csv dataframe files -- Results of Dimensionality Reduction

Clustering :
	Risk_CLuster_1:
	How to Run : python Risk_CLuster_1.py -i 'filename for the datafile'
									   -t 'km' <- this indicates using k-means algorithm
									   -k 7 <- max number of clusters to be used for k
	This algorithm has the ability to iterate over 2 to k number of cluster on the selected features.
	It is also possible to decide if the algorithm needs to iterate over combinations of features instead.
	That option exists in the feature_search() method. 
	The program outputs a csv file, with the results of CH and Silhouette score for a cluster number 
	and feature combination. 
	It also outputs a csv file with cluster labels and the full data for the best clustering option
	The program also uses prarallel programming to speed up running tests. It will use all available 
	cores of the computer.
Results:
	Visualization:
	How to Run: python Visualization.py -i 'folder that contains output file(s) with the cluster labels'
										-c 'name of the cluster column'
										-r 'name of the Risk Sum column'
										-o 'name of the directory where all the results should be stored'
	This algorithm has methods to output visualization and results. The following are outputted:
	1) csv file that contains raw features mapped back to cluster labels with statistical test results and 
		mean value of each raw features in each clusters
	2) Outputs folder with all the distribution of each raw feature in each cluster
	3) Bar graph of risk sum values mapped back to each clusters 
	4) Histogram on proportion of each cluster's PCA/FA dimensions
	5) TSNE plot that shows overall cluster seperation 
		- TSNE fit on new componenets
		- TSNE fit on raw features
