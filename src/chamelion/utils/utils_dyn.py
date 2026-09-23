import numpy as np
import torch.nn.functional as F
from scipy.spatial import cKDTree

SCAN_TIMESTAMP = 1
MAP_TIMESTAMP = 0

STAT_LABEL = 0
PD_LABEL = 1
ND_LABEL = 2


def local_to_global(local_scan, trans_matrix):
    local_xyz = local_scan[:, :3]
    local_other = local_scan[:, 3:]

    homo_coords = np.hstack((local_xyz, np.ones((local_scan.shape[0], 1))))
    transformed_coords = np.dot(homo_coords, trans_matrix.T)
    transformed_local_xyz = transformed_coords[:, :3] / transformed_coords[:, 3][:, np.newaxis]
    transformed_local_scan = np.hstack((transformed_local_xyz, local_other))

    return transformed_local_scan.astype(np.float32)


def search_pose_pair_all_cover(poses_keyframes, num_distance_frame):
    """
    poses_keyframes : np.array, shape (N, 4, 4) (transformation matrices)
    num_distance_frame : int, minimum frame distance
    """

    pose_cKdTree = cKDTree(poses_keyframes[:, :3, 3])
    distances, indices = pose_cKdTree.query(poses_keyframes[:, :3, 3], k=100)

    spatial_mask = distances < 1.0

    temporal_mask = (
        np.abs(indices - np.arange(poses_keyframes.shape[0])[:, np.newaxis]) > num_distance_frame
    )

    indices_mask = temporal_mask & spatial_mask

    candidate_pairs = []

    for i, (row, dist_row, mask_row) in enumerate(zip(indices, distances, indices_mask)):
        valid_indices = row[mask_row]
        valid_distances = dist_row[mask_row]

        for idx, dist in zip(valid_indices, valid_distances):
            candidate_pairs.append((dist, i, idx))

    candidate_pairs.sort()

    assigned_i = set()
    filtered_pairs = {}
    expanded_pairs = {}

    for _, i, idx in candidate_pairs:
        if i not in assigned_i and idx not in assigned_i:
            if i < idx:
                filtered_pairs[i] = idx
                expanded_pairs[i] = idx
            else:
                filtered_pairs[idx] = i
                expanded_pairs[idx] = i

            assigned_i.add(i)
            assigned_i.add(idx)

    # find not matched indices !!!
    all_indices = set(range(poses_keyframes.shape[0]))
    unmatched_indices = all_indices - assigned_i

    map_list = list(filtered_pairs.keys())
    scans_list = list(filtered_pairs.values())

    for unmatch in unmatched_indices:
        map_min = np.argmin(np.abs(np.array(map_list) - unmatch))
        map_min_value = np.abs(map_list[map_min] - unmatch)
        value_min = np.argmin(np.abs(np.array(scans_list) - unmatch))
        scan_min_value = np.abs(scans_list[value_min] - unmatch)
        if map_min_value < scan_min_value:  # clasor to map
            expanded_pairs[unmatch] = expanded_pairs[map_list[map_min]]
            assigned_i.add(unmatch)
        else:  # close to scan
            expanded_pairs[unmatch] = map_list[value_min]
            assigned_i.add(unmatch)

    unmatched_indices = all_indices - assigned_i

    return filtered_pairs, expanded_pairs, list(unmatched_indices)


def get_dynamic_label(cls_output, scan_indices, map_indices, threshold=0.5):
    cls_output[scan_indices, ND_LABEL] = -float("inf")
    cls_output[map_indices, PD_LABEL] = -float("inf")
    cls_softmax = F.softmax(cls_output, dim=1)

    scan_label = cls_softmax[scan_indices, 1].cpu().data.numpy() > threshold
    map_label = cls_softmax[map_indices, 2].cpu().data.numpy() > threshold

    return scan_label, map_label
