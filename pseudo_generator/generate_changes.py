import argparse
import os
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import yaml
from interactive_ui import (
    ADD_COLOR,
    MAP_LEGEND,
    REMOVE_COLOR,
    display_copy,
    map_display_copy,
    pick_points,
    review,
    stage,
)
from public_output import prepare_output
from tqdm import tqdm

STATIC_LABEL = 0
PD_LABEL = 1
ND_LABEL = 2


class low_dynamic_object_database:
    def __init__(self, data_dir, dataset=None, output_dir=None):
        self.database_dir = os.path.join(data_dir, "dense_object_database")
        # load_dataset
        self.hd_removed_scans_dir = os.path.join(data_dir, "hd_removed_scans")
        self.hd_removed_map_dir = os.path.join(data_dir, "hd_removed_map")
        self.map_pcd_path = os.path.join(self.hd_removed_map_dir, "hd_removed_map.pcd")
        if not Path(self.map_pcd_path).is_file() or not Path(self.hd_removed_scans_dir).is_dir():
            raise FileNotFoundError("Run prepare_objects.py first: prepared map/scans are missing")
        if not Path(self.database_dir).is_dir() or not any(Path(self.database_dir).glob("*.ply")):
            raise ValueError("Prepared object database is empty; generation cannot proceed")

        # save_dataset
        self.save_scan_path = str(output_dir or data_dir)
        self.save_map_path = str(output_dir or data_dir)

        if not os.path.exists(self.save_scan_path):
            os.makedirs(self.save_scan_path)
        if not os.path.exists(self.save_map_path):
            os.makedirs(self.save_map_path)

        self.database = {}
        self.dataset = dataset

        self.positive_dynamics = []
        self.negative_dynamics = []

        self.used_id = []
        self.chunk_scan = {}

        self.octree_check_flag = True

        self.load_database()

    def clear_database(self):
        self.used_id = []
        self.positive_dynamics = []
        self.negative_dynamics = []
        self.chunk_scan = {}

    def load_database(self):
        object_files = sorted(path.name for path in Path(self.database_dir).glob("*.ply"))
        id_ = 0
        for object_file in object_files:
            object_pcd = o3d.io.read_point_cloud(os.path.join(self.database_dir, object_file))
            self.database[id_] = object_pcd
            id_ += 1

        self.database = dict(
            sorted(self.database.items(), key=lambda x: len(x[1].points), reverse=True)
        )

    # 1. Load the main point cloud map
    def load_point_cloud(self, file_path):
        pcd = o3d.io.read_point_cloud(file_path)
        # pcd.paint_uniform_color([0.7, 0.7, 0.7])  # Gray color for the map
        return pcd

    # 2. Generate a random object point cloud (e.g., a small cube)
    def generate_random_object():
        cube = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=1.0)
        cube.compute_vertex_normals()
        cube.paint_uniform_color([1, 0, 0])  # Red color for visibility
        object_pcd = cube.sample_points_uniformly(number_of_points=100)
        return object_pcd

    # 3. Main visualization function with picking
    def visualize_point_cloud_with_objects(self, chunk):
        # load map pcd
        pcd = self.load_point_cloud(self.map_pcd_path)
        # load scan pcd
        for i in chunk:
            # scan, _, _, pose = self.dataset[i]
            scan = o3d.io.read_point_cloud(
                os.path.join(self.hd_removed_scans_dir, "{0:06d}.pcd".format(i))
            )
            self.chunk_scan[i] = scan

        stage(
            4,
            "Change Placement",
            "Pick all locations, then press Q to place all objects at once. Objects and change types are assigned automatically (about 70% removed, the rest added). Press Q in the combined preview to save.",
        )
        background = map_display_copy(pcd)
        object_ids = list(self.database)
        while True:
            picked_points = pick_points(
                background, f"4/5 Place objects | Frames {chunk[0]}-{chunk[-1]} | {MAP_LEGEND}"
            )
            if not picked_points:
                if (
                    review(
                        [background],
                        "No locations selected",
                        "Nothing will be saved",
                        {"R": "retry"},
                    )
                    == "retry"
                ):
                    continue
            positive, negative = [], []
            removed_count = int(len(picked_points) * 0.7)
            types = np.random.permutation(
                [ND_LABEL] * removed_count + [PD_LABEL] * (len(picked_points) - removed_count)
            )
            available = []
            for index, point_idx in enumerate(picked_points):
                if not available:
                    available = list(np.random.permutation(object_ids))
                object_id = available.pop()
                added = types[index] == PD_LABEL
                candidate = display_copy(
                    self.database[object_id], ADD_COLOR if added else REMOVE_COLOR
                )
                candidate.translate(np.asarray(pcd.points)[point_idx])
                if added:
                    candidate = candidate.voxel_down_sample(voxel_size=0.1)
                (positive if added else negative).append(candidate)
            action = review(
                [background, *positive, *negative],
                f"Save submap? {len(positive)} added, {len(negative)} removed",
                f"{MAP_LEGEND} | Blue: added | Red: removed",
            )
            if action == "confirm":
                self.positive_dynamics, self.negative_dynamics = positive, negative
                return

    def is_cluster_free_space_by_point(self, cluster_points, pc2, threshold=0.5):  # Too slow!
        for point in cluster_points:
            distances = np.asarray(
                pc2.compute_point_cloud_distance(
                    o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector([point]))
                )
            )

            # If any point in the cluster is within the occupied threshold, return False
            if np.min(distances) <= threshold:
                return False  # Occupied space detected

        return True

    def is_cluster_free_space(self, cluster_points, pc2, threshold=0.5):
        # Convert cluster_points to a PointCloud object
        cluster_pc = o3d.geometry.PointCloud()
        cluster_pc.points = o3d.utility.Vector3dVector(cluster_points)

        # Compute distances from cluster points to pc2 points
        distances = np.asarray(pc2.compute_point_cloud_distance(cluster_pc))

        # Check if any distance is within the threshold
        if np.min(distances) <= threshold:
            return False  # Occupied space detected

        return True  # All points in the cluster are in free space

    def save_pd_nd_added_scans_and_map(self, cluster_id, chunk):
        stage(
            5,
            "Export",
            f"Saving submap_{cluster_id:03d}. Collision filtering may exclude a placed object from individual scans.",
        )
        submap = Path(self.save_map_path) / f"submap_{cluster_id:03d}"
        for name in ("scans", "scan_labels", "map_labels"):
            (submap / name).mkdir(parents=True, exist_ok=True)

        for chunk_idx in tqdm(chunk, desc=f"Export submap {cluster_id:03d}", unit="frame"):
            save_scan = self.chunk_scan[chunk_idx]
            save_scan.paint_uniform_color([0.7, 0.7, 0.7])
            scans_label = np.zeros((len(save_scan.points)), dtype=np.int32)

            for pd in self.positive_dynamics:
                if self.octree_check_flag:
                    if self.is_cluster_free_space(pd.points, save_scan, threshold=0.5):
                        continue
                    else:
                        save_scan += pd
                        scans_label = np.concatenate(
                            (
                                scans_label,
                                np.ones((len(pd.points) * PD_LABEL), dtype=np.int32),
                            )
                        )

                else:
                    save_scan += pd
                    scans_label = np.concatenate(
                        (
                            scans_label,
                            np.ones((len(pd.points) * PD_LABEL), dtype=np.int32),
                        )
                    )

            o3d.io.write_point_cloud(
                os.path.join(
                    self.save_scan_path,
                    "submap_{:03d}".format(cluster_id),
                    "scans",
                    "{0:06d}.pcd".format(chunk_idx),
                ),
                save_scan,
            )
            scans_label.tofile(
                os.path.join(
                    self.save_scan_path,
                    "submap_{:03d}".format(cluster_id),
                    "scan_labels",
                    "{0:06d}.label".format(chunk_idx),
                )
            )

        map_pcd = self.load_point_cloud(self.map_pcd_path)
        map_pcd.paint_uniform_color([0.7, 0.7, 0.7])

        for nd in self.negative_dynamics:
            map_pcd += nd
            # map_label = np.concatenate((map_label, np.ones((len(nd.points) * ND_LABEL), dtype=np.int32)))

        map_label = np.zeros((len(map_pcd.points)), dtype=np.int32)
        map_color = np.asarray(map_pcd.colors)
        map_color = np.round(map_color, 1)

        map_label[map_color[:, 0] == 0.0] = PD_LABEL
        map_label[map_color[:, 0] == 0.7] = 0
        map_label[map_color[:, 0] == 1.0] = ND_LABEL

        o3d.io.write_point_cloud(
            os.path.join(
                self.save_map_path,
                "submap_{:03d}".format(cluster_id),
                ".",
                "prior_map.pcd",
            ),
            map_pcd,
        )
        map_label.tofile(
            os.path.join(
                self.save_map_path,
                "submap_{:03d}".format(cluster_id),
                "map_labels",
                "static.label",
            )
        )

        print(f"Saved submap: {submap}", flush=True)


# Run the visualization with your point cloud file path

if __name__ == "__main__":
    from datasets.datasets import HeLiMOSDataset, SemanticKittiDataset

    parser = argparse.ArgumentParser(description="Interactively place pseudo change objects")
    parser.add_argument(
        "config", nargs="?", type=Path, default=Path(__file__).parent / "config/generator.yaml"
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Use a new final dataset root; preparation files are reused unchanged",
    )
    args = parser.parse_args()
    config_filename = args.config

    if yaml.__version__ >= "5.1":
        config = yaml.load(open(config_filename), Loader=yaml.FullLoader)
    else:
        config = yaml.load(open(config_filename))

    if args.output_root is not None:
        config["output_dataset"]["root"] = str(args.output_root)

    # ? 1. Load Dataset
    if config["dataset_type"] == "kitti":
        dataset = SemanticKittiDataset(config["load_dataset"])

    elif config["dataset_type"] == "helimos":
        dataset = HeLiMOSDataset(config["load_dataset"])

    data_len = list(range(len(dataset)))
    cluster_id = 0
    chunk_size = 100
    chunk_frame_idxs = [data_len[i : i + chunk_size] for i in range(0, len(data_len), chunk_size)]

    data_path = config["load_dataset"]["pred_save_dir"].format(seq=config["load_dataset"]["seq"])
    output_dir = prepare_output(config)
    ld_database = low_dynamic_object_database(data_path, dataset, output_dir)

    for chunk in chunk_frame_idxs:
        print(
            f"\nSubmap {cluster_id + 1}/{len(chunk_frame_idxs)} | Frames {chunk[0]}-{chunk[-1]}",
            flush=True,
        )
        ld_database.visualize_point_cloud_with_objects(chunk=chunk)
        ld_database.save_pd_nd_added_scans_and_map(cluster_id, chunk)
        ld_database.clear_database()
        cluster_id += 1
