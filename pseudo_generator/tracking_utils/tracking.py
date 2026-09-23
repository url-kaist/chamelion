#!/usr/bin/env python3
# Developed by Xieyuanli Chen
# Brief: some useful functions for 3D multiple object tracking
# This file is covered by the LICENSE file in the root of this project.

import math

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as Rot

# import rectangle_fitting


def box2corners2d(cls, bbox):
    """the coordinates for bottom corners"""
    bottom_center = np.array([bbox.x, bbox.y, bbox.z - bbox.h / 2])
    cos, sin = np.cos(bbox.o), np.sin(bbox.o)
    pc0 = np.array(
        [
            bbox.x + cos * bbox.l / 2 + sin * bbox.w / 2,
            bbox.y + sin * bbox.l / 2 - cos * bbox.w / 2,
            bbox.z - bbox.h / 2,
        ]
    )
    pc1 = np.array(
        [
            bbox.x + cos * bbox.l / 2 - sin * bbox.w / 2,
            bbox.y + sin * bbox.l / 2 + cos * bbox.w / 2,
            bbox.z - bbox.h / 2,
        ]
    )
    pc2 = 2 * bottom_center - pc0
    pc3 = 2 * bottom_center - pc1

    return [pc0.tolist(), pc1.tolist(), pc2.tolist(), pc3.tolist()]


def kalman_box_to_eight_point_with_orientation(kalman_bbox):
    """TODO: remove this function since there is no orientation applied"""
    # kalman_bbox = [cx, cy, cz, theta, l, w, h] to eight point = [x1,x2,y1,y2,z1,z2]
    theta = kalman_bbox[3]
    cx, cy, cz = kalman_bbox[0], kalman_bbox[1], kalman_bbox[2]
    l, w, h = kalman_bbox[4], kalman_bbox[5], kalman_bbox[6]

    bottom_center = np.array([cx, cy, cz - h / 2])
    cos_, sin_ = np.cos(theta), np.sin(theta)
    pc0 = np.array([cx + l / 2 * cos_ + w / 2 * sin_, cy + l / 2 * sin_ - w / 2 * cos_, cz - h / 2])
    pc1 = np.array([cx - l / 2 * cos_ - w / 2 * sin_, cy + l / 2 * sin_ + w / 2 * cos_, cz - h / 2])
    pc2 = 2 * bottom_center - pc0
    pc3 = 2 * bottom_center - pc1

    x1 = kalman_bbox[0] - kalman_bbox[4] / 2
    x2 = kalman_bbox[0] + kalman_bbox[4] / 2
    y1 = kalman_bbox[1] - kalman_bbox[5] / 2
    y2 = kalman_bbox[1] + kalman_bbox[5] / 2
    z1 = kalman_bbox[2] - kalman_bbox[6] / 2
    z2 = kalman_bbox[2] + kalman_bbox[6] / 2

    return [x1, y1, z1, x2, y2, z2]


def kalman_box_to_eight_point(kalman_bbox):
    """TODO: remove this function since there is no orientation applied"""
    # kalman_bbox = [cx, cy, cz, theta, l, w, h] to eight point = [x1,x2,y1,y2,z1,z2]
    x1 = kalman_bbox[0] - kalman_bbox[4] / 2
    x2 = kalman_bbox[0] + kalman_bbox[4] / 2
    y1 = kalman_bbox[1] - kalman_bbox[5] / 2
    y2 = kalman_bbox[1] + kalman_bbox[5] / 2
    z1 = kalman_bbox[2] - kalman_bbox[6] / 2
    z2 = kalman_bbox[2] + kalman_bbox[6] / 2

    return [x1, y1, z1, x2, y2, z2]


def get_bbox_from_points_with_orientation(points):
    x1 = np.min(points[:, 0])
    x2 = np.max(points[:, 0])
    y1 = np.min(points[:, 1])
    y2 = np.max(points[:, 1])
    z1 = np.min(points[:, 2])
    z2 = np.max(points[:, 2])


def get_bbox_from_points(
    points, estimate_orientation=False, region_growing=0.0, region_growing_z=0.0
):
    """
    Runs the loss on outputs of the model
    Input:
      points: instance points Nx3
    Return:
      3D bbox [x1,y1,z1,x2,y2,z2]
    """

    x1 = np.min(points[:, 0]) - region_growing
    x2 = np.max(points[:, 0]) + region_growing
    y1 = np.min(points[:, 1]) - region_growing
    y2 = np.max(points[:, 1]) + region_growing
    z1 = np.min(points[:, 2]) - region_growing_z
    z2 = np.max(points[:, 2]) + region_growing_z

    theta = 0
    if estimate_orientation:
        pca = np.cov(points[:, :2], y=None, rowvar=0, bias=1)  # this is covariance
        eig_val, eig_vect = np.linalg.eig(pca)
        theta = np.arccos(float(eig_vect[np.argmax(eig_val), 0]))

        if eig_val[0] >= eig_val[1]:
            big_ind = 0
            small_ind = 1
        else:
            big_ind = 1
            small_ind = 0

            a = math.sqrt(pca * eig_val[big_ind])
            b = math.sqrt(pca * eig_val[small_ind])
            theta = math.atan2(eig_vect[1, big_ind], eig_vect[0, big_ind])

    return [x1, y1, z1, x2, y2, z2], np.array(
        [
            x1 + (x2 - x1) / 2,
            y1 + (y2 - y1) / 2,
            z1 + (z2 - z1) / 2,
            theta,
            x2 - x1,
            y2 - y1,
            z2 - z1,
        ]
    )  # [xmin, ymin, zmin, xmax, ymax, zmax], cx, cy, cz, theta, l, w, h


def get_ellipsoid_from_points(points):
    mu = np.mean(points, axis=0)

    cov = np.cov(points.T)
    eigvals, eigvecs = np.linalg.eigh(cov)

    idx = np.sum(cov, axis=0).argsort()
    eigvals_temp = eigvals[idx]
    idx = eigvals_temp.argsort()
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]


def get_2d_bbox(points):
    x1 = np.min(points[0, :])
    x2 = np.max(points[0, :])
    y1 = np.min(points[1, :])
    y2 = np.max(points[1, :])

    return [x1, y1, x2, y2]


def IoU(bbox0, bbox1):
    """Runs the intersection over union of two bbox
    Inputs:
      bbox0 & bbox1 are two kalman bounding box
      kalman bounding box: x, y, z, theta, l, w, h
    Return: IoU
    """

    dim = int(len(bbox0) / 2)
    overlap = [
        max(0, min(bbox0[i + dim], bbox1[i + dim]) - max(bbox0[i], bbox1[i])) for i in range(dim)
    ]
    intersection = 1
    for i in range(dim):
        intersection = intersection * overlap[i]
    area0 = 1
    area1 = 1
    for i in range(dim):
        area0 *= bbox0[i + dim] - bbox0[i]
        area1 *= bbox1[i + dim] - bbox1[i]
    union = area0 + area1 - intersection
    if union == 0:
        return 0
    return intersection / union


def volume(bbox0, bbox1):
    """Check the changes of volumes.
    Inputs:
      bbox0 & bbox1 are two kalman bounding box
    Return:
      volumes change ratio
    """
    volume0 = bbox0[4] * bbox0[5] * bbox0[6]
    volume1 = bbox1[4] * bbox1[5] * bbox1[6]

    volume_ratio = 1
    if min(volume0, volume1) > 0:
        volume_ratio = np.abs(volume0 - volume1) / max(volume0, volume1)

    return volume_ratio


def hashkey_iou(hash1, hash2):
    hash1_len = len(hash1)
    hash2_len = len(hash2)

    intersection = np.intersect1d(hash1, hash2)
    if len(intersection) == 0:
        return {"ratio": 0, "hash1_len": hash1_len, "hash2_len": hash2_len}
    union = np.union1d(hash1, hash2)

    ratio = len(intersection) / len(union)
    return {"ratio": ratio, "hash1_len": hash1_len, "hash2_len": hash2_len}


def volume_v2(bbox0, bbox1_list):
    """Check the changes of volumes.
    Inputs:
      bbox0 & bbox1 are two kalman bounding box
    Return:
      volumes change ratio
    """
    volume0 = bbox0[4] * bbox0[5] * bbox0[6]

    volume1 = 0
    for bbox1 in bbox1_list:
        volume1 += bbox1[4] * bbox1[5] * bbox1[6]
    volume1 /= len(bbox1_list)

    volume_ratio = 1
    if min(volume0, volume1) > 0:
        volume_ratio = np.abs(volume0 - volume1) / max(volume0, volume1)

    return volume_ratio


def similarity_shape(bbox0, bbox1):
    """check the changes of bounding boxes' shape
    Inputs:
      bbox0 & bbox1 are two kalman bounding box
    Return:
      volumes change ratio
    """
    delta_l = np.abs(bbox0[4] - bbox1[4]) / max(bbox0[4], bbox1[4])
    delta_w = np.abs(bbox0[5] - bbox1[5]) / max(bbox0[5], bbox1[5])
    delta_h = np.abs(bbox0[6] - bbox1[6]) / max(bbox0[6], bbox1[6])
    similarity = np.exp(-(delta_l + delta_w + delta_h) / 3.0)

    return similarity


def bhattacharyya_gaussian_distance(cluster1, cluster2):
    """Estimate Bhattacharyya Distance (between Gaussian Distributions)
        https://en.wikipedia.org/wiki/Bhattacharyya_distance
    Inputs:
      distribution1: a sample gaussian distribution 1
      distribution2: a sample gaussian distribution 2
    Returns:
      Bhattacharyya distance
    """
    cov1 = cluster1["std"]
    cov2 = cluster2["std"]

    T = (1 / 4) * np.log(
        (1 / 4) * ((cov1 * cov1) / (cov2 + cov2) + (cov2 + cov2) / (cov1 * cov1) + 2)
    )

    return T


def get_median_center_from_points(points):
    x = np.median(points[:, 0])
    y = np.median(points[:, 1])
    z = np.median(points[:, 2])

    return [x, y, z]


def euclidean_dist(b1, b2):
    ret_sum = 0
    for i in range(3):
        ret_sum += (b1[i] - b2[i]) ** 2
    return np.sqrt(ret_sum)


def find_points_in_box(pointcloud, bbox, region_growing=0.0):
    """Find out all points inside this bounding box
    Input:
      pointcloud: point cloud
      bbox: EKF bounding box [x, y, z, theta, l, w, h]
    Returns:
      indexes of points inside the bounding box
    """
    [x_min, y_min, z_min, x_max, y_max, z_max] = [0, 0, 0, 0, 0, 0]
    if len(bbox) > 6:
        [x_min, y_min, z_min, x_max, y_max, z_max] = kalman_box_to_eight_point(bbox)

    elif len(bbox) == 4:
        cx, cy, cz = bbox[0], bbox[1], bbox[2]
        radius = bbox[3]
        [x_min, y_min, z_min, x_max, y_max, z_max] = [
            cx - radius,
            cy - radius,
            cz - radius,
            cx + radius,
            cy + radius,
            cz + radius,
        ]
    else:
        [x_min, y_min, z_min, x_max, y_max, z_max] = bbox

    points_is_in_bbox = (
        (pointcloud[:, 0] <= x_max + region_growing)
        & (pointcloud[:, 1] <= y_max + region_growing)
        & (pointcloud[:, 2] <= z_max + region_growing)
        & (pointcloud[:, 0] >= x_min - region_growing)
        & (pointcloud[:, 1] >= y_min - region_growing)
        & (pointcloud[:, 2] >= z_min - region_growing)
    )
    return np.argwhere(points_is_in_bbox).squeeze()


def is_wall_histogram(points, angle_threshold=10):
    angles = np.arctan2(points[:, 1], points[:, 0]) * 180 / np.pi

    hist, _ = np.histogram(angles, bins=36, range=(-180, 180))

    if max(hist) > len(points) * 0.7:
        return True
    return False


def find_points_in_box_with_yaw(pointcloud, bbox):
    """Find out all points inside this bounding box
    Inputs:
      pointcloud: point cloud
      bbox: EKF bounding box [x, y, z, theta, l, w, h]
    Returns:
      indexes of points inside the bounding box
    """
    translation = bbox[0:3]
    l, w, h = bbox[4], bbox[5], bbox[6]
    rotation = bbox[3]

    # Standard 3x3 rotation matrix around the Z axis
    rotation_matrix = np.array(
        [
            [np.cos(rotation), -np.sin(rotation), 0.0],
            [np.sin(rotation), np.cos(rotation), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    # convert point into local instance coordinates
    pcd_centered = pointcloud[:, :3] - np.tile(translation, (len(pointcloud), 1))
    pcd_rotated = np.dot(pcd_centered, rotation_matrix)

    # check points in box
    points_is_in_bbox = (
        (pcd_rotated[:, 0] <= l / 2.0)
        & (pcd_rotated[:, 1] <= w / 2.0)
        & (pcd_rotated[:, 2] <= h / 2.0)
        & (pcd_rotated[:, 0] >= -l / 2.0)
        & (pcd_rotated[:, 1] >= -w / 2.0)
        & (pcd_rotated[:, 2] >= -h / 2.0)
    )

    return np.argwhere(points_is_in_bbox).squeeze()


def box_center_to_corner(bbox):
    """create bounding boxes that can be used for open3d visualization
    Inputs:
      bbox: EKF bounding box [x, y, z, theta, l, w, h]
    Returns:
      corner_box: bounding box with 8 corner coordinates
    """
    translation = bbox[0:3]
    l, w, h = bbox[4], bbox[5], bbox[6]
    rotation = bbox[3]

    # Create a bounding box outline
    bounding_box = np.array(
        [
            [-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2],
            [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
            [-h / 2, -h / 2, -h / 2, -h / 2, h / 2, h / 2, h / 2, h / 2],
        ]
    )

    # Standard 3x3 rotation matrix around the Z axis
    rotation_matrix = np.array(
        [
            [np.cos(rotation), -np.sin(rotation), 0.0],
            [np.sin(rotation), np.cos(rotation), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    # Repeat the [x, y, z] eight times
    eight_points = np.tile(translation, (8, 1))

    # Translate the rotated bounding box by the
    # original center position to obtain the final box
    corner_box = np.dot(rotation_matrix, bounding_box) + eight_points.transpose()

    return corner_box.transpose()
