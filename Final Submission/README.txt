Files:

How to run code:
Data Prep:
	Run data_prep.py
Dimensionality Reduction:
	Run PCA_VM_FA_script_join.ipynb

- Final Submission
	- Final Figures: Figures generated during the project
	- Harris Data
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
