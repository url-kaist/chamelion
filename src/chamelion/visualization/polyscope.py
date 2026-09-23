# MIT License
#
# Copyright (c) 2024 Benedikt Mersch, Luca Lobefaro, Ignazio Vizzo, Tiziano Guadagnino
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Polyscope point-cloud UI adapted from the archived Chamelion visualizer.

Retains point-cloud layers, metric point radii and local/global transforms.
No odometry, model, metric calculation or dataset writes live in this module.
"""
from pathlib import Path

import numpy as np

from chamelion.visualization.data import point_colors, save_frame, validate_frame


class ChamelionViewer:
    def __init__(self, frames, infer=None, output=None):
        self.frames, self.infer = frames, infer
        self.output = Path(output) if output is not None else None
        self.index = 0
        self.frame, self.name = frames.load(0)
        validate_frame(self.frame)
        self.mode = "Prediction" if "scan_pred" in self.frame else "Input"
        self.local = False
        self.scan_visible, self.map_visible = True, True
        self.radius = 0.06
        self.message = "Ready. Drag to rotate; scroll to zoom."
        self.ps = self.gui = None

    def select_frame(self, index):
        index = max(0, min(int(index), len(self.frames) - 1))
        frame, name = self.frames.load(index)
        validate_frame(frame)
        self.index, self.frame, self.name = index, frame, name
        if "scan_pred" not in frame:
            self.mode = "Input"
        self.message = (
            "Frame loaded. Run inference to see predictions." if self.infer else "Frame loaded."
        )

    def run_inference(self):
        if self.infer is None:
            raise ValueError("Result viewing does not run inference")
        result = self.infer(self.frame)
        validate_frame(result)
        self.frame, self.mode = result, "Prediction"
        self.message = "Inference complete. Raw framewise predictions; no temporal fusion."

    def save_current(self):
        if self.output is None or "scan_pred" not in self.frame:
            raise ValueError("Run inference before saving a prediction")
        path = self.output / f"frame_{self.index:06d}.npz"
        save_frame(path, self.frame)
        self.message = f"Saved {path.name}. Existing files are never replaced."

    def draw_frame(self):
        transform = (
            np.linalg.inv(self.frame["pose"]) if self.local and "pose" in self.frame else np.eye(4)
        )
        for name, enabled in (("scan", self.scan_visible), ("map", self.map_visible)):
            points = self.frame[f"{name}_xyz"]
            # Limit rendering only; inference, saved arrays and metrics use every point.
            indices = np.linspace(0, len(points) - 1, min(len(points), 250000), dtype=int)
            cloud = self.ps.register_point_cloud(name, points[indices], point_render_mode="quad")
            cloud.set_radius(self.radius, relative=False)
            cloud.add_color_quantity(
                "Class colors", point_colors(self.frame, name, self.mode)[indices], enabled=True
            )
            cloud.set_transform(transform)
            cloud.set_enabled(enabled)

    def _action(self, callback):
        try:
            callback()
            self.draw_frame()
        except Exception as error:
            self.message = f"{type(error).__name__}: {error}"

    def callback(self):
        gui = self.gui
        gui.TextUnformatted(
            "Chamelion | " + ("Interactive inference" if self.infer else "Evaluation results")
        )
        protocol = str(self.frame.get("protocol", "raw_framewise_global_coordinates"))
        gui.TextUnformatted(
            "Historical confidence filtering + accumulated map state"
            if protocol == "historical_b4d67f7"
            else "Raw framewise results - not the full paper inference protocol"
        )
        gui.TextUnformatted(f"{self.index + 1}/{len(self.frames)}: {self.name}")
        if gui.Button("Previous"):
            self._action(lambda: self.select_frame(self.index - 1))
        gui.SameLine()
        if gui.Button("Next"):
            self._action(lambda: self.select_frame(self.index + 1))
        changed, index = gui.SliderInt("Frame", self.index, 0, len(self.frames) - 1)
        if changed:
            self._action(lambda: self.select_frame(index))
        if self.infer:
            if gui.Button("Run inference on this frame"):
                self._action(self.run_inference)
            if gui.Button("Save prediction"):
                self._action(self.save_current)
        modes = ["Input"]
        if "scan_pred" in self.frame and "map_pred" in self.frame:
            modes.append("Prediction")
        if "scan_gt" in self.frame and "map_gt" in self.frame:
            modes.append("Ground truth")
            if "Prediction" in modes:
                modes.append("Errors")
        if self.mode not in modes:
            self.mode = modes[0]
        for mode in modes:
            if gui.Button(mode):
                self.mode = mode
                self.draw_frame()
            gui.SameLine()
        gui.NewLine()
        dirty, self.scan_visible = gui.Checkbox("Show scan", self.scan_visible)
        changed, self.map_visible = gui.Checkbox("Show prior map", self.map_visible)
        dirty |= changed
        if "pose" in self.frame:
            changed, self.local = gui.Checkbox("Sensor-local view", self.local)
            dirty |= changed
        changed, self.radius = gui.SliderFloat("Point radius (m)", self.radius, 0.01, 0.20)
        if dirty or changed:
            self.draw_frame()
        if gui.Button("Reset camera"):
            self.ps.reset_camera_to_home_view()
        if self.mode == "Errors":
            gui.TextUnformatted("Green: correct change | Red: false positive | Blue: missed change")
        else:
            gui.TextUnformatted("Blue: scan addition | Red: map removal | Gray: static/input")
        gui.TextUnformatted("Dark gray: ignored GT | Display capped at 250k points per layer")
        gui.TextUnformatted(self.message)

    def initialize(self):
        import polyscope as ps
        import polyscope.imgui as gui

        self.ps, self.gui = ps, gui
        ps.set_program_name("Chamelion Viewer")
        ps.set_use_prefs_file(False)
        ps.init()
        ps.set_ground_plane_mode("none")
        ps.set_background_color([0.06, 0.07, 0.09])
        ps.set_build_default_gui_panels(False)
        self.draw_frame()
        ps.set_user_callback(self.callback)

    def show(self):
        self.initialize()
        self.ps.show()
