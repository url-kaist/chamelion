import argparse
import os
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import yaml
from datasets.datasets import *
from interactive_ui import (
    MAP_LEGEND,
    REMOVE_COLOR,
    map_display_copy,
    pick_points,
    review,
    stage,
)
from retry_config import create_retry_config
from tqdm import tqdm
from tracking_utils import tracking, utils
from tracking_utils.instance_association import *
from tracking_utils.kalman_filter import KalmanBoxTracker

STATIC_LABEL = 0
PD_LABEL = 1
ND_LABEL = 2

ANSI_COLOR_RED = "\x1b[31m"
ANSI_COLOR_GREEN = "\x1b[32m"
ANSI_COLOR_YELLOW = "\x1b[33m"
ANSI_COLOR_BLUE = "\x1b[34m"
ANSI_COLOR_MAGENTA = "\x1b[35m"
ANSI_COLOR_CYAN = "\x1b[36m"
ANSI_COLOR_RESET = "\x1b[0m"


class dataset_generator:
    def __init__(self, dataset, config):
        self.dataset = dataset
        self.config = config

        # poses
        self.poses = None

        # tracking parameters
        self.last_ins_id = 0  # global instance id
        self.tracked_instances = {}
        self.instance_associate_list = []
        self.instances_voting_results = {}
        self.rearrange_associates = {}
        self.raw_tracked_associates = {}
        self.inst_color_dict = {}
        self.new_label_lists = []

        self.high_dynamic_ids_list = []
        self.low_dynamic_ids_list = []
        self.low_dynamic_points_dict = {}

        # FLAG
        self.save_database = False
        self.save_dense_database = True
        self.save_static_map = True
        self.voxel_size = 0.1
        self.save_top_points = False
        self.save_static_scans = True
        self.manual_hd_remove = True
        self.save_high_dynamic_tracked = False

        # Staic map
        self.static_map = None
        self.static_above_map = None

    def clear_dict(self):
        self.last_ins_id = 0
        self.tracked_instances = {}
        self.instance_associate_list = []
        self.instances_voting_results = {}
        self.rearrange_associates = {}
        self.raw_tracked_associates = {}
        self.inst_color_dict = {}
        self.new_label_lists = []

        self.high_dynamic_ids_list = []
        self.low_dynamic_ids_list = []
        self.low_dynamic_points_dict = {}

    def rearrange_new_instances_ids(self, current_inst_raw):
        """rearrange current new instances ids based on the last maximum id"""
        current_inst_label = np.copy(current_inst_raw)
        unique_labels = np.unique(current_inst_raw)
        original_indices = np.arange(len(current_inst_label))

        for idx, unique_label in enumerate(unique_labels):
            if unique_label == 0:
                continue
            if np.sum(current_inst_raw == unique_label) < 5:
                current_inst_label[current_inst_raw == unique_label] = 0
                continue
            inds = original_indices[current_inst_raw == unique_label]

            self.last_ins_id += 1
            current_inst_label[inds] = self.last_ins_id
            self.rearrange_associates[unique_label] = self.last_ins_id  # raw id - new id
            self.raw_tracked_associates[self.last_ins_id] = self.last_ins_id  # new id - new -id

        return current_inst_label

    def make_new_instances(self, frame_idx, scan, inst_labels):
        unique_ids = np.unique(inst_labels)
        new_instances = {}

        for _id in unique_ids:
            if _id == 0:
                continue

            self.inst_color_dict[_id] = np.random.rand(3)
            mask = inst_labels == _id
            points = scan[mask]

            if len(points) < 5:
                continue

            bbox, kalman_bbox = tracking.get_bbox_from_points(points)

            tracker = KalmanBoxTracker(kalman_bbox, _id)
            center = tracking.get_median_center_from_points(points)
            kalman_bboxes = []
            kalman_bboxes.append(kalman_bbox.flatten().tolist())

            new_instances[_id] = {
                "life": 5,
                "center": center,
                "tracker": tracker,
                "bbox": bbox,
                "kalman_bbox": kalman_bbox,
                "counter": 1,
                "kalman_bboxes": kalman_bboxes,
                "frame": [frame_idx],
                "merged_ids": [],
            }

        return new_instances

    def associate_instances(self, frame_idx, new_instances, debug=False):
        if len(self.tracked_instances) > 0:
            for i in self.tracked_instances.keys():
                self.tracked_instances[i]["kalman_bbox"] = (
                    (self.tracked_instances[i]["tracker"].predict()).flatten().tolist()
                )
                self.tracked_instances[i]["bbox"] = tracking.kalman_box_to_eight_point(
                    self.tracked_instances[i]["kalman_bbox"]
                )
                self.tracked_instances[i]["center"] = self.tracked_instances[i]["kalman_bbox"][:3]

            association_costs, associations = compute_associations_with_knn(
                self.tracked_instances, new_instances
            )

            for (
                prev_id,
                new_id,
            ) in associations:
                self.tracked_instances[prev_id]["life"] += 1
                self.tracked_instances[prev_id]["tracker"].update(
                    new_instances[new_id]["kalman_bbox"], prev_id
                )
                self.tracked_instances[prev_id]["kalman_bbox"] = (
                    (self.tracked_instances[prev_id]["tracker"].get_state()).flatten().tolist()
                )
                self.tracked_instances[prev_id]["bbox"] = tracking.kalman_box_to_eight_point(
                    self.tracked_instances[prev_id]["kalman_bbox"]
                )
                self.tracked_instances[prev_id]["merged_ids"].append(new_id)
                self.tracked_instances[prev_id]["counter"] += 1
                self.tracked_instances[prev_id]["frame"].append(frame_idx)

                self.raw_tracked_associates[new_id] = prev_id

                del new_instances[new_id]
                del self.inst_color_dict[new_id]

    def update_tracked_instances(self):
        for i in self.tracked_instances.keys():
            self.tracked_instances[i]["kalman_bboxes"].append(
                self.tracked_instances[i]["kalman_bbox"][:7]
            )
            self.tracked_instances[i]["center"] = self.tracked_instances[i]["kalman_bbox"][:3]

    def add_new_instances(self, new_instances):
        for _id, instance in new_instances.items():
            if _id in self.tracked_instances.keys():
                continue
            self.tracked_instances[_id] = instance

    def remove_dead_instances(self):
        dont_track_ids = []
        for _id in self.tracked_instances.keys():
            if self.tracked_instances[_id]["life"] == 0:
                dont_track_ids.append(_id)
            else:
                self.tracked_instances[_id]["life"] -= 1

        for _id in dont_track_ids:
            label = 0
            if self.tracked_instances[_id]["counter"] > 5:
                kalman_bboxes = np.array(self.tracked_instances[_id]["kalman_bboxes"])
                exists_frames = np.array(self.tracked_instances[_id]["frame"])
                if tracking.euclidean_dist(kalman_bboxes[-1], kalman_bboxes[0]) > np.max(
                    kalman_bboxes[:, 4:7]
                ):
                    label = 1
            self.instances_voting_results[_id] = (
                label,
                self.tracked_instances[_id]["counter"],
            )

            del self.tracked_instances[_id]

        # Keep the generated instance ID monotonic. Rewinding it to the largest
        # currently alive track can reuse an ID that already belongs to a dead
        # track and corrupt the final HD/static vote table.

    def kill_remain_instances(self):
        for _id in self.tracked_instances.keys():
            label = 0
            if self.tracked_instances[_id]["counter"] > 5:
                kalman_bboxes = np.array(self.tracked_instances[_id]["kalman_bboxes"])
                exists_frames = np.array(self.tracked_instances[_id]["frame"])
                if tracking.euclidean_dist(kalman_bboxes[-1], kalman_bboxes[0]) > np.max(
                    kalman_bboxes[:, 4:7]
                ):
                    label = 1
            self.instances_voting_results[_id] = (
                label,
                self.tracked_instances[_id]["counter"],
            )

    def run_high_dynamic_search(self):
        poses = []

        stage(
            2,
            "Object Tracking",
            "Match objects across frames and identify moving tracks. No manual input is needed yet.",
        )

        # for i in tqdm(range(len(self.dataset))):
        for i in tqdm(range(len(self.dataset)), desc="Tracking objects", unit="frame"):
            scan, inst_raw, ground_label, pose = self.dataset[i]
            poses.append(pose)
            inst_label = self.rearrange_new_instances_ids(inst_raw)

            new_instances = self.make_new_instances(i, scan, inst_label)
            self.associate_instances(i, new_instances, debug=False)
            new_label = np.zeros_like(inst_raw)

            for _id in np.unique(inst_label):
                if _id == 0:
                    continue
                new_label[inst_label == _id] = self.raw_tracked_associates[_id]

            self.new_label_lists.append(new_label)
            self.update_tracked_instances()
            self.add_new_instances(new_instances)
            self.remove_dead_instances()

        self.kill_remain_instances()

        self.poses = poses

    def save_high_dynamic_tracked_for_fig(self):
        if self.save_high_dynamic_tracked:
            print(ANSI_COLOR_YELLOW + "Saving high dynamic tracked instances..." + ANSI_COLOR_RESET)
            high_dynamic_dir = os.path.join(
                self.config["load_dataset"]["pred_save_dir"].format(
                    seq=self.config["load_dataset"]["seq"]
                ),
                "high_dynamic_tracked",
            )
            if not os.path.exists(high_dynamic_dir):
                os.makedirs(high_dynamic_dir)

            high_dynamic_dict = {}

            for i in tqdm(range(len(self.dataset))):
                if i % 3 != 0:
                    continue
                scan, inst_raw, ground_label, pose = self.dataset[i]
                new_label = self.new_label_lists[i]

                for _id in np.unique(new_label):
                    if _id in self.high_dynamic_ids_list:
                        points = scan[new_label == _id]
                        if _id not in high_dynamic_dict.keys():
                            high_dynamic_dict[_id] = []
                        high_dynamic_dict[_id].append(points)

            for _id, points_list in high_dynamic_dict.items():
                points = np.concatenate(points_list, axis=0)
                o3d_points = o3d.geometry.PointCloud()
                o3d_points.points = o3d.utility.Vector3dVector(points)
                o3d.io.write_point_cloud(
                    os.path.join(high_dynamic_dir, "id_{0}.ply".format(_id)), o3d_points
                )

    def construct_object_database(self):
        print(ANSI_COLOR_YELLOW + "Constructing object database..." + ANSI_COLOR_RESET)

        for _id, result in self.instances_voting_results.items():
            if result[0] == 1:
                self.high_dynamic_ids_list.append(_id)
            elif result[0] == 0:
                if result[1] > 5:
                    self.low_dynamic_ids_list.append(_id)

        print(
            ANSI_COLOR_YELLOW + "Retained object tracks observed in at least 6 frames: ",
            len(self.low_dynamic_ids_list),
            ANSI_COLOR_RESET,
        )

        for i in tqdm(range(len(self.dataset)), desc="Collecting object tracks", unit="frame"):
            scan, inst_raw, ground_label, pose = self.dataset[i]
            new_label = self.new_label_lists[i]

            for _id in np.unique(new_label):
                if _id in self.low_dynamic_ids_list:
                    if _id not in self.low_dynamic_points_dict.keys():
                        self.low_dynamic_points_dict[_id] = []
                    self.low_dynamic_points_dict[_id].append(scan[new_label == _id])

        if self.save_database:
            print(ANSI_COLOR_YELLOW + "Saving object database..." + ANSI_COLOR_RESET)
            database_dir = os.path.join(
                self.config["load_dataset"]["pred_save_dir"].format(
                    seq=self.config["load_dataset"]["seq"]
                ),
                "object_database",
            )

            if not os.path.exists(database_dir):
                os.makedirs(database_dir)

            for _id, points in self.low_dynamic_points_dict.items():
                points = np.concatenate(points, axis=0)
                mean_point = np.mean(points, axis=0)
                mean_point[-1] = np.min(points[:, -1])
                points = points - mean_point
                l, w, h = np.max(points, axis=0) - np.min(points, axis=0)  # aabb box
                if l > 2.0 or w > 2.0 or h > 2.0:
                    continue
                o3d_points = o3d.geometry.PointCloud()
                o3d_points.points = o3d.utility.Vector3dVector(points)
                o3d.io.write_point_cloud(
                    os.path.join(database_dir, "id_{0}.ply".format(_id)), o3d_points
                )

    def interactive_hd_remove(self, iteration=10):
        stage(
            3,
            "Dynamic Object Removal",
            "Select objects incorrectly kept by tracking. Picking one point selects the entire object. Changes require confirmation.",
        )
        while True:
            ld_pcd = o3d.geometry.PointCloud()
            ld_id_list = []

            for _id, points in self.low_dynamic_points_dict.items():
                points = np.concatenate(points, axis=0)
                o3d_points = o3d.geometry.PointCloud()
                o3d_points.points = o3d.utility.Vector3dVector(points)

                o3d_points.paint_uniform_color([0.65, 0.69, 0.73])
                ld_pcd += o3d_points
                ld_id_list += [_id] * len(points)

            if not ld_id_list:
                print("No candidate objects remain.")
                return
            picked_points = pick_points(
                ld_pcd, "3/5 Remove moving objects | Gray: candidate objects"
            )

            if len(picked_points) == 0:
                if (
                    review([ld_pcd], "Finish removal?", "Gray: keep", {"C": "finish", "R": "retry"})
                    == "finish"
                ):
                    return
                continue

            picked_ld_ids = [ld_id_list[idx] for idx in picked_points]
            picked_ld_ids = list(set(picked_ld_ids))
            new_hd_colors = np.asarray(ld_pcd.colors)
            new_hd_colors[np.isin(ld_id_list, picked_ld_ids)] = REMOVE_COLOR
            ld_pcd.colors = o3d.utility.Vector3dVector(new_hd_colors)
            if (
                review(
                    [ld_pcd], f"Remove {len(picked_ld_ids)} objects?", "Red: remove | Gray: keep"
                )
                == "retry"
            ):
                continue

            self.high_dynamic_ids_list += picked_ld_ids
            for picked_ld_id in picked_ld_ids:
                self.low_dynamic_points_dict.pop(picked_ld_id)
            print(f"Confirmed removal of object IDs: {sorted(picked_ld_ids)}")

    def build_static_map(self):
        print(ANSI_COLOR_GREEN + "Building static map..." + ANSI_COLOR_RESET)

        static_points = []
        static_above_points = []
        static_points_color_dict = {}
        static_points_color = []

        for i in tqdm(range(len(self.dataset)), desc="Building static map and scans", unit="frame"):
            # for i in tqdm(range(0, 1000)):

            static_points_in_frame = []
            static_colors_in_frame = []
            scan, inst_raw, ground_label, pose = self.dataset[i]
            new_label = self.new_label_lists[i]

            ground_points = scan[ground_label == 1]

            static_points.append(ground_points)
            static_points_in_frame.append(ground_points)
            static_points_color.append(np.zeros((len(ground_points), 3)))
            static_colors_in_frame.append(np.zeros((len(ground_points), 3)))

            for _id in np.unique(new_label):
                if _id == 0:
                    if self.save_top_points:
                        top_points = scan[new_label == _id]
                        static_points.append(top_points)
                        static_points_in_frame.append(top_points)
                        static_points_color.append(np.zeros((len(top_points), 3)))
                        static_colors_in_frame.append(np.zeros((len(top_points), 3)))
                    continue
                elif _id in self.high_dynamic_ids_list:
                    continue
                elif self.instances_voting_results[_id][1] < 6:
                    continue
                elif len(scan[new_label == _id]) < 20:
                    continue

                static_points.append(scan[new_label == _id])
                static_points_in_frame.append(scan[new_label == _id])
                static_above_points.append(scan[new_label == _id])

                if _id not in static_points_color_dict.keys():
                    static_points_color_dict[_id] = [
                        232 / 255,
                        116 / 255,
                        97 / 255,
                    ]  # np.random.rand(3)
                static_points_color.append(
                    np.ones((len(scan[new_label == _id]), 3)) * static_points_color_dict[_id]
                )
                static_colors_in_frame.append(
                    np.ones((len(scan[new_label == _id]), 3)) * static_points_color_dict[_id]
                )

            if self.save_static_scans:
                static_o3d_points = o3d.geometry.PointCloud()
                static_o3d_points.points = o3d.utility.Vector3dVector(
                    np.concatenate(static_points_in_frame, axis=0)
                )
                static_o3d_points.colors = o3d.utility.Vector3dVector(
                    np.concatenate(static_colors_in_frame, axis=0)
                )
                save_scans_dir = os.path.join(
                    self.config["load_dataset"]["pred_save_dir"].format(
                        seq=self.config["load_dataset"]["seq"]
                    ),
                    "hd_removed_scans",
                )
                if not os.path.exists(save_scans_dir):
                    os.makedirs(save_scans_dir)
                # o3d.visualization.draw_geometries([static_o3d_points])
                o3d.io.write_point_cloud(
                    os.path.join(save_scans_dir, "{:06d}.pcd".format(i)),
                    static_o3d_points,
                )

        static_points = np.concatenate(static_points, axis=0)
        static_above_points = np.concatenate(static_above_points, axis=0)
        static_points_color = np.concatenate(static_points_color, axis=0)

        self.static_map = static_points
        self.static_above_map = static_above_points

        if self.save_static_map:
            static_o3d_points = o3d.geometry.PointCloud()
            static_o3d_points.points = o3d.utility.Vector3dVector(static_points)
            static_o3d_points.colors = o3d.utility.Vector3dVector(static_points_color)
            # voxelize
            static_o3d_points = static_o3d_points.voxel_down_sample(voxel_size=self.voxel_size)
            save_map_dir = os.path.join(
                self.config["load_dataset"]["pred_save_dir"].format(
                    seq=self.config["load_dataset"]["seq"]
                ),
                "hd_removed_map",
            )
            if not os.path.exists(save_map_dir):
                os.makedirs(save_map_dir)
            review(
                [map_display_copy(static_o3d_points)],
                "3/5 Static map ready",
                MAP_LEGEND,
                {"C": "save prepared map"},
            )
            o3d.io.write_point_cloud(
                os.path.join(save_map_dir, "hd_removed_map.pcd"), static_o3d_points
            )

    def construct_dense_object_database(self):
        if self.save_dense_database:
            print(ANSI_COLOR_YELLOW + "Saving dense object database..." + ANSI_COLOR_RESET)
            database_dir = os.path.join(
                self.config["load_dataset"]["pred_save_dir"].format(
                    seq=self.config["load_dataset"]["seq"]
                ),
                "dense_object_database",
            )

            if not os.path.exists(database_dir):
                os.makedirs(database_dir)

            for _id, points in self.low_dynamic_points_dict.items():
                points = np.concatenate(points, axis=0)

                if len(points) < 1000:
                    continue

                mean_point = np.mean(points, axis=0)
                mean_point[-1] = np.min(points[:, -1])

                points = points - mean_point
                l, w, h = np.max(points, axis=0) - np.min(points, axis=0)  # aabb box
                if l > 2.0 or w > 2.0 or h > 2.0:
                    continue

                o3d_points = o3d.geometry.PointCloud()
                o3d_points.points = o3d.utility.Vector3dVector(points)
                o3d.io.write_point_cloud(
                    os.path.join(database_dir, "id_{0}.ply".format(_id)), o3d_points
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Track objects and prepare a static map")
    parser.add_argument(
        "config", nargs="?", type=Path, default=Path(__file__).parent / "config/generator.yaml"
    )
    parser.add_argument(
        "--new-run-config",
        type=Path,
        help="Create a new config and fresh preparation/output paths; reuse existing labels",
    )
    args = parser.parse_args()
    config_filename = str(args.config)

    if yaml.__version__ >= "5.1":
        config = yaml.load(open(config_filename), Loader=yaml.FullLoader)
    else:
        config = yaml.load(open(config_filename))

    configured_preparation = Path(
        config["load_dataset"]["pred_save_dir"].format(seq=config["load_dataset"]["seq"])
    )
    if args.new_run_config is not None or configured_preparation.exists():
        requested_config = args.new_run_config or args.config.with_name(
            f"{args.config.stem}_retry.yaml"
        )
        config, actual_config = create_retry_config(config, args.config, requested_config)
        config_filename = str(actual_config)
        print(
            "Starting a fresh preparation run. Existing results and labels are preserved.",
            flush=True,
        )
        print(f"New configuration: {config_filename}", flush=True)
        print(f"New preparation: {config['load_dataset']['pred_save_dir']}", flush=True)

    # ? 1. Load Dataset
    if config["dataset_type"] == "kitti":
        dataset = SemanticKittiDataset(config["load_dataset"])

    elif config["dataset_type"] == "helimos":
        dataset = HeLiMOSDataset(config["load_dataset"])

    datagen = dataset_generator(dataset, config)
    preparation_dir = Path(
        config["load_dataset"]["pred_save_dir"].format(seq=config["load_dataset"]["seq"])
    )
    # Never merge a new preparation run into existing maps or object databases.
    preparation_dir.mkdir(parents=True, exist_ok=False)
    datagen.run_high_dynamic_search()
    datagen.construct_object_database()
    if datagen.manual_hd_remove:
        datagen.interactive_hd_remove()
        # datagen.interactive_hd_remove_w_crop()
    if datagen.save_high_dynamic_tracked:
        datagen.save_high_dynamic_tracked_for_fig()

    datagen.build_static_map()
    datagen.construct_dense_object_database()
    if not any((preparation_dir / "dense_object_database").glob("*.ply")):
        raise RuntimeError(
            "No eligible objects were saved. Do not run generation; review the input duration and object filters."
        )
    # datagen.run_low_dynamic_addtion_test()
    datagen.clear_dict()

    print(ANSI_COLOR_BLUE + "Process is done!" + ANSI_COLOR_RESET)
    print(f"Next: python3 pseudo_generator/generate_changes.py {config_filename}")
