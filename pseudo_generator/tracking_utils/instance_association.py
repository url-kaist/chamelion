import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from scipy.optimize import linear_sum_assignment as lsa
from sklearn.neighbors import BallTree, KDTree
from tracking_utils import tracking
from tracking_utils.kalman_filter import KalmanBoxTracker


def compute_associations_with_knn(previous_instances, current_instances):
    """compute the associations matrix and determine the instance associations"""
    p_n = len(previous_instances.keys())  # 101
    c_n = len(current_instances.keys())  # 98

    association_costs = np.zeros((p_n, c_n))  # 101 * 98
    prev_ids = []
    current_ids = []
    merge_to_previous = []
    prev_ids = list(previous_instances.keys())
    curr_ids = list(current_instances.keys())
    curr_centers = [
        list(current_instances.values())[i]["center"] for i in range(len(current_instances))
    ]
    if len(curr_centers) == 0:
        return association_costs, []
    tree = KDTree(curr_centers)
    k_neighbors = 5

    if len(curr_centers) < k_neighbors:
        k_neighbors = len(curr_centers)

    for i, (id1, v1) in enumerate(previous_instances.items()):
        prev_center = np.asarray(v1["center"]).reshape(1, -1)
        prev_dist, prev_ind = tree.query(prev_center, k=k_neighbors)
        prev_dist = prev_dist[0]
        prev_ind = prev_ind[0]
        association_costs[i] = 1e8
        prev_length = len(v1["frame"])

        for j in prev_ind:
            v2 = current_instances[curr_ids[j]]
            cost_3d = 1 - tracking.IoU(v2["bbox"], v1["bbox"])
            cost_center = tracking.euclidean_dist(v2["kalman_bbox"], v1["kalman_bbox"])
            cost_volume = tracking.volume(v2["kalman_bbox"], v1["kalman_bbox"])

            if cost_center > 2.0:
                continue

            # if prev_length < 5: #
            if cost_volume > 0.7:
                cost_volume = 1e8

            if cost_3d > 0.95:
                cost_3d = 1e8

            association_costs[i, j] = cost_3d + cost_center + cost_volume

    idx1, idx2 = lsa(association_costs)

    associations = []
    for i1, i2 in zip(idx1, idx2):
        if association_costs[i1][i2] < 1e8:
            associations.append((prev_ids[i1], curr_ids[i2]))

    return association_costs, associations
