"""
clustering_utils.py
====================
Modular utilities for evaluation, clustering (K-Means, Hierarchical, DBSCAN),
PCA visualization, and segment profiling.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA


def evaluate_kmeans_range(X_scaled: np.ndarray, k_range=range(2, 9), sample_size: int = 2500, random_state: int = 42) -> pd.DataFrame:
    """
    Evaluate K-Means over a range of cluster counts k.

    Parameters
    ----------
    X_scaled : np.ndarray
        Scaled feature matrix.
    k_range : range or list
        Range of k values to evaluate.
    sample_size : int
        Sample size for silhouette score computation to ensure memory efficiency.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Evaluation results with WCSS (Inertia), Silhouette, Calinski-Harabasz, and Davies-Bouldin metrics.
    """
    results = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X_scaled)
        wcss = km.inertia_
        sil = silhouette_score(X_scaled, labels, sample_size=sample_size, random_state=random_state)
        ch = calinski_harabasz_score(X_scaled, labels)
        db = davies_bouldin_score(X_scaled, labels)
        results.append({
            'k': k,
            'WCSS': wcss,
            'Silhouette': sil,
            'Calinski_Harabasz': ch,
            'Davies_Bouldin': db
        })
    return pd.DataFrame(results)


def fit_kmeans(X_scaled: np.ndarray, n_clusters: int = 4, random_state: int = 42):
    """
    Fit K-Means model with specified n_clusters.
    """
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = km.fit_predict(X_scaled)
    return km, labels


def fit_agglomerative(X_scaled: np.ndarray, n_clusters: int = 4, linkage: str = 'ward'):
    """
    Fit Agglomerative Hierarchical Clustering.
    """
    agg = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = agg.fit_predict(X_scaled)
    return agg, labels


def fit_dbscan(X_scaled: np.ndarray, eps: float = 0.5, min_samples: int = 15):
    """
    Fit DBSCAN clustering.
    """
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(X_scaled)
    return db, labels


def compute_cluster_profiles(df: pd.DataFrame, cluster_col: str = 'Cluster') -> pd.DataFrame:
    """
    Compute descriptive statistics for raw RFM metrics per cluster.
    """
    profile = df.groupby(cluster_col).agg(
        Customer_Count=('CustomerID', 'count'),
        Recency_Mean=('Recency', 'mean'),
        Recency_Median=('Recency', 'median'),
        Frequency_Mean=('Frequency', 'mean'),
        Frequency_Median=('Frequency', 'median'),
        Monetary_Mean=('Monetary', 'mean'),
        Monetary_Median=('Monetary', 'median'),
        Monetary_Total=('Monetary', 'sum')
    ).reset_index()

    total_rev = profile['Monetary_Total'].sum()
    profile['Revenue_Share_%'] = (profile['Monetary_Total'] / total_rev * 100).round(2)
    profile['Customer_Share_%'] = (profile['Customer_Count'] / profile['Customer_Count'].sum() * 100).round(2)

    return profile
