import gc

import numpy as np
from sklearn.cluster import AgglomerativeClustering


def clusters_hdbscan(points_set):
    import hdbscan

    clusterer = hdbscan.HDBSCAN(min_cluster_size=20)
    clusterer.fit(points_set)

    labels = clusterer.labels_.copy()

    del clusterer
    gc.collect()

    lbls, counts = np.unique(labels, return_counts=True)
    cluster_info = np.array(list(zip(lbls[1:], counts[1:])))
    cluster_info = cluster_info[cluster_info[:, 1].argsort()]
    clusters_labels = cluster_info[::-1][:, 0]
    labels[np.in1d(labels, clusters_labels, invert=True)] = -1
    labels = labels + 1  # Shift labels to start from 0

    return labels


def clusters_euclidean(points_set, distance_threshold=0.5, min_cluster_size=20):
    """
    Euclidean distance-based clustering using AgglomerativeClustering.

    Args:
        points_set (np.ndarray): (N, D) array of points.
        distance_threshold (float): Maximum distance threshold for clustering.
        min_cluster_size (int): Minimum cluster size. Smaller clusters will be removed.

    Returns:
        labels (np.ndarray): Cluster labels for each point (-1 for noise).
    """
    clusterer = AgglomerativeClustering(
        n_clusters=None, distance_threshold=distance_threshold, linkage="single"
    )
    clusterer.fit(points_set)

    labels = clusterer.labels_.copy()

    del clusterer
    gc.collect()

    # Count cluster sizes
    lbls, counts = np.unique(labels, return_counts=True)
    cluster_info = np.array(list(zip(lbls, counts)))
    cluster_info = cluster_info[cluster_info[:, 1].argsort()]  # sort by size

    # Keep only clusters larger than min_cluster_size
    clusters_labels = cluster_info[::-1][:, 0][cluster_info[::-1][:, 1] >= min_cluster_size]
    labels[np.in1d(labels, clusters_labels, invert=True)] = -1
    labels = labels + 1  # Shift labels to start from 0

    return labels
