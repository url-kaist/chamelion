"""English controls for the legacy Open3D viewer (no extra GUI dependencies)."""
import copy

import numpy as np
import open3d as o3d

MAP_COLOR = [0.65, 0.69, 0.73]
GROUND_COLOR = [0.45, 0.65, 0.48]
ADD_COLOR = [0.15, 0.65, 1.0]
REMOVE_COLOR = [1.0, 0.25, 0.18]
MAP_LEGEND = "Green: ground | Gray: non-ground"


def display_copy(cloud, color=MAP_COLOR):
    result = copy.deepcopy(cloud)
    result.paint_uniform_color(color)
    return result


def map_display_copy(cloud):
    """Recolor prepared-map ground/non-ground encoding for display only.

    Preparation stores black ground and salmon non-ground. Voxel averaging
    blends those colors; choose the nearest endpoint (majority in a voxel).
    This uses existing segmentation, not a height-based ground estimate.
    """
    result = copy.deepcopy(cloud)
    if not cloud.has_colors():
        print("Map has no ground color encoding; displaying it in gray.", flush=True)
        result.paint_uniform_color(MAP_COLOR)
        return result
    encoded = np.asarray(cloud.colors)
    nonground = np.array([232, 116, 97], dtype=float) / 255
    ground_mask = np.sum(encoded**2, axis=1) < np.sum((encoded - nonground) ** 2, axis=1)
    colors = np.tile(MAP_COLOR, (len(encoded), 1))
    colors[ground_mask] = GROUND_COLOR
    result.colors = o3d.utility.Vector3dVector(colors)
    return result


def stage(number, title, explanation):
    print(f"\n[{number}/5] {title}\n{explanation}", flush=True)


def setup(viewer, title, geometries):
    if not viewer.create_window(window_name=title, width=1280, height=720):
        raise RuntimeError("Could not open the viewer. Check the Docker display connection.")
    options = viewer.get_render_option()
    options.background_color = np.array([0.08, 0.10, 0.13])
    options.point_size = 3.0
    for geometry in geometries:
        viewer.add_geometry(geometry)


def pick_points(cloud, title):
    instructions = "Shift+Left: pick | Shift+Right: undo | Q: review (not save)"
    print(f"\n{title}\n{instructions}\nDrag: rotate | Wheel: zoom", flush=True)
    viewer = o3d.visualization.VisualizerWithEditing()
    setup(viewer, f"{title} | {instructions}", [cloud])
    try:
        viewer.run()
        return viewer.get_picked_points()
    finally:
        viewer.destroy_window()


def review(geometries, title, legend, actions=None):
    """Closing the window is cancellation, never implicit confirmation."""
    actions = actions or {"C": "confirm", "R": "retry"}
    # Q always advances, matching the picking window. X explicitly cancels.
    actions = {**actions, "Q": actions.get("C", actions.get("R", "confirm")), "X": "cancel"}
    controls = " | ".join(f"{key}: {value}" for key, value in actions.items())
    print(f"\n{title}\n{legend}\n{controls}", flush=True)
    viewer = o3d.visualization.VisualizerWithKeyCallback()
    setup(viewer, f"{title} | {legend} | {controls}", geometries)
    result = ["cancel"]

    def callback(action):
        def choose(vis):
            result[0] = action
            vis.close()
            return False

        return choose

    for key, action in actions.items():
        viewer.register_key_callback(ord(key), callback(action))
    try:
        viewer.run()
    finally:
        viewer.destroy_window()
    if result[0] == "cancel":
        raise RuntimeError(
            "Cancelled by user. Existing files were not deleted; unfinished output is retained."
        )
    return result[0]
