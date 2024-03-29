import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics import silhouette_samples

def k_means(X, y, k=3):
    """
    Perform k-means clustering on the given input X and label y
    """

    # Perform k-means clustering
    kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
    labels = kmeans.labels_
    X['cluster'] = labels
    data = pd.concat([X, y], axis=1)
    clustered_dfs = [data[data['cluster'] == i].drop('cluster', axis=1) for i in range(k)]

    return clustered_dfs

def draw_samples(clustered_dfs, sample_fraction = 0.01, random_state = 42):
    sample_dfs = []
    for df in clustered_dfs:
        sample_size = int(len(df) * sample_fraction)
        sample_dfs.append(df.sample(n=sample_size, random_state=random_state))
    return sample_dfs

def concat(dfs_list):
    return pd.concat(dfs_list, ignore_index=True)