"""Open and close a small picking window without loading or writing datasets."""
import time

import open3d as o3d


def check():
    viewer = o3d.visualization.VisualizerWithEditing()
    if not viewer.create_window(
        window_name="Chamelion GUI check (closes automatically)", width=640, height=480
    ):
        raise RuntimeError("Could not open Open3D window. Check DISPLAY and XAUTHORITY.")
    try:
        viewer.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame())
        for _ in range(10):
            viewer.poll_events()
            viewer.update_renderer()
            time.sleep(0.05)
    finally:
        viewer.destroy_window()
    print("Open3D point-picking window: OK")


if __name__ == "__main__":
    check()
