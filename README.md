This repository is an attempt to emulate the w-Kmeans algorithm from the papers:
Horvath, B., Issa, Z., Muguruza, A. (2021). Clustering Market Regimes using the Wasserstein Distance. arXiv preprint arXiv:2110.11848. 

I am currently working on a multivariate version based on:
Luan, Q., Hamp, J. (2023). Automated regime detection in multidimensional time series data using
  sliced Wasserstein k-means clustering. arXiv preprint arXiv:2310.01285.

The class operates similarly to the sklearn api in its functionality, using fit() and predict() methods. Initialization parameters for the class WKMeans are k (number of clusters), tol (loss tolerance), gamma (kernel sensitivity), and MMD pairs, or the number of pairs used in MMD calculation. 

*****IN PROGRESS***


