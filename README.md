# CS4641
Machine Learning for Pairs Trading

## Files and Descriptions

- **dbscan.ipynb**: 
  Performs DBSCAN clustering on market data, evaluating clusters and analyzing the stock patterns identified by this density-based approach.

- **k_mean_clustering.ipynb**: 
  Applies the K-Means clustering algorithm to  market data, including feature standardization, cluster analysis, and evaluation metrics (e.g., Silhouette score) for assessing clustering quality.

- **clustered_data.csv**: 
  Contains data resulting from the clustering process, providing an organized view of stock data grouped into clusters.

- **correlation_between_clusters.ipynb**: 
  Explores relationships between clusters formed in the clustering analysis by calculating and visualizing correlations between them.

- **pca_representation.ipynb**: 
  Applies Principal Component Analysis (PCA) to reduce the dimensionality of stock market data, offering a visual representation of clusters in a lower-dimensional space for easier interpretation.

- **summary_statistics_2015_2017.csv**: 
  Includes summary statistics of data for the years 2015-2017, providing a historical overview that aids in feature engineering and data preprocessing.

- **feature_engineering.ipynb**: 
  Performs feature engineering on stock market data by calculating additional metrics (e.g., moving averages, Bollinger Bands) to improve the quality of inputs for clustering


- **kalman_filtering.ipynb**:
  Implements Kalman Filtering for predicting stock price movements and smoothing noisy time series data to identify meaningful trends

- **linear_regression.ipynb**:
  Develops a linear regression model to identify relationships between selected stock features and predict stock prices, analyzing feature importance and model performance

## Requirements

To run these notebooks, ensure you have the necessary Python packages installed:
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `seaborn`
