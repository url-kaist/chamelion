<div align="center">

# Chamelion

### Reliable Change Detection for Long-Term LiDAR Mapping<br>in Transient Environments

**Seoyeon Jang · Alex Junho Lee · I Made Aswin Nahrendra · Hyun Myung**<br>
IEEE Robotics and Automation Letters, 2026

[![Paper](https://img.shields.io/badge/Paper-RA--L_2026-2563eb)](https://doi.org/10.1109/LRA.2026.3665079)
[![Project](https://img.shields.io/badge/Project-Website-0d9488)](https://chamelion-pages.github.io/)
[![Dataset](https://img.shields.io/badge/Dataset-Hugging_Face_(private)-eab308)](https://huggingface.co/datasets/se0yeon00/Const_pseudo_dataset)

[Setup](#-setup) · [Dataset](#-dataset) · [Training](#-training) · [Inference](#-inference)

<img src="docs/assets/chamelion_overview.png" alt="Chamelion overview: a prior map and LiDAR scans pass through a shared network with change-classification and confidence heads." width="960">

Detect added and removed objects by comparing LiDAR scans with a prior map.

</div>

## 🚀 Setup

Clone the repository:

```bash
git clone --branch main --recurse-submodules https://github.com/url-kaist/chamelion.git
cd chamelion
```

**Training requirements:** Linux, an NVIDIA GPU, and Docker with GPU support.

```bash
./scripts/build.sh
```

This builds **`chamelion:train-cu128`** for training and checkpoint evaluation.
Pseudo generation uses its own container, built below.

## 📦 Dataset

Choose how you want to prepare your data:

- **[Download the prepared pseudo dataset](#option-a--download-the-pseudo-dataset)** to start training with our supplied splits.
- **[Generate your own pseudo dataset](#option-b--generate-your-own-pseudo-dataset)** from LiDAR clouds and matching poses.

Once your dataset is ready, continue to [Training](#-training) or [Inference](#-inference).

### Option A — Download the pseudo dataset

**[Const_pseudo_dataset on Hugging Face ↗](https://huggingface.co/datasets/se0yeon00/Const_pseudo_dataset)**<br>
Prepared point clouds, poses and labels. Approximately **3.61 GiB**; no pseudo generation needed.

| Split | Submaps | Scans |
| --- | ---: | ---: |
| Train | 62 | 5,877 |
| Validation | 12 | 1,158 |
| Test (Const-1F, Lab) | 2 | 2,296 |

Access is currently restricted to authorized Hugging Face accounts.

#### Download

```bash
pip install huggingface_hub
hf auth login
```

Run this Python snippet with a **new destination** to preserve existing datasets:

```python
from pathlib import Path
from huggingface_hub import snapshot_download

destination = Path("/path/to/Const_pseudo_dataset")
if destination.exists():
    raise FileExistsError("Choose a new destination")

repo = "se0yeon00/Const_pseudo_dataset"
snapshot_download(repo, repo_type="dataset", local_dir=destination)
```

Keep `checksums.json`, `splits/` and `sequences/` together. Use the Chamelion
point-cloud loader, not the tabular `load_dataset()` API.

#### Prepare the dataset

The download is ready to use. No renaming or conversion is needed:

```text
Const_pseudo_dataset/
├── checksums.json
├── splits/                       # train.txt, val.txt, test.txt
└── sequences/sequence_000/
    ├── poses.txt
    └── submaps/submap_000/
        ├── prior_map.pcd
        ├── scans/                # 000000.pcd, …
        ├── scan_labels/          # 000000.label, …
        └── map_labels/static.label
```

Use `sequences/` as the dataset path in the commands below. Each submap has one
`map_labels/static.label`. The supplied train/validation splits are selected by
[`config/cham.yaml`](config/cham.yaml); test data is not used for training.

Each scan has a label file with the same filename stem. Labels are binary `int32`,
one per point in cloud order:

| Label | Meaning |
| --- | --- |
| `0` | Static |
| `1` | Added |
| `2` | Removed |
| `-1`, `251` | Ignored (`251` occurs in the test data) |

Maps and scans share a global coordinate frame. Each `poses.txt` row is a
row-major 3×4 local-to-global transform (12 numbers). Scan `000123.pcd` uses
pose row 123, counting from zero. Keep scan numbers and pose rows unchanged,
even when using only part of a sequence.

Ready to use the downloaded data? Skip Option B and continue to [Training](#-training).

<a id="-generate-your-own-data"></a>

### Option B — Generate your own pseudo dataset

Start with **LiDAR clouds + matching `poses.txt`**. The workflow uses the pinned
[TRAVEL](https://github.com/url-kaist/TRAVEL) submodule and includes human selection.

**First time? [Follow the screenshot walkthrough →](docs/pseudo_generation_visual_guide.md)**<br>
See what each window means, what to click, when to press Q, and which files are saved.

[![Example saved pseudo changes: blue added points and red removed points](docs/assets/pseudo-guide/05-saved-changes.png)](docs/pseudo_generation_visual_guide.md)


**1. Prepare the input**

```text
source_dataset/                  # Dataset root: pass this path to the launcher
└── sequences/                   # Collection of recording sessions
    └── my_sequence/             # One recording session; choose your own name
        ├── poses.txt            # One sensor-to-world pose per frame
        └── clouds/              # Raw clouds in sensor coordinates
            ├── 000000.bin
            ├── 000001.bin
            └── …
```

The default is float32 XYZI `.bin` clouds, numbered contiguously from zero.
Poses are row-major 3×4 transforms (12 numbers per row). `000000.bin` uses the
first row of `poses.txt`, `000001.bin` uses the second, and so on. Keep one pose
row per cloud; both must describe the same recording session.
For PCD input, put `000000.pcd`, `000001.pcd`, … in `clouds/` and set
`load_dataset.is_bin: false` in `/output/generator.yaml` before step 3.

You can place multiple sessions under `sequences/`, each with its own `poses.txt`
and `clouds/`. The launcher processes **one session at a time**. No input labels
are required.

**2. Build and open the GUI container**

Use a Linux graphical desktop with Docker and `xauth`:

```bash
bash scripts/build.sh pseudo
bash scripts/generate.sh /path/to/source_dataset my_sequence
```

The first argument is the folder **containing `sequences/`**, not the session
folder or `clouds/`. The second is the session folder name, here `my_sequence`.
Inside the container, that root is mounted at `/input` (read-only).
The launcher prints a new host output directory, mounted at `/output`.

**3. Run these commands inside the container, one at a time**

```bash
# Segment ground and cluster objects
python3 pseudo_generator/label_instances.py /output/generator.yaml

# Select dynamic objects to remove and review the static map
python3 pseudo_generator/prepare_objects.py /output/generator.yaml

# Pick locations, place objects, and save the pseudo dataset
python3 pseudo_generator/generate_changes.py /output/generator.yaml
```

| Action | Control |
| --- | --- |
| Select / undo a point | **Shift + left click** / **Shift + right click** |
| Continue or accept the preview | **Q** |
| Finish removal | Select no points, then **Q** in both windows |
| Place all selected objects | Pick all locations, then **Q** |
| Retry / cancel a review | **R** / **X** |

🟢 Ground · ⚪ Non-ground · 🔵 Added objects · 🔴 Removed objects

The generated dataset is saved under `/output/Const_pseudo_dataset/`:

```text
Const_pseudo_dataset/
└── sequences/sequence_000/       # Output sequence name, independent of input name
    ├── poses.txt
    └── submaps/submap_000/       # A local map built from a group of frames
        ├── prior_map.pcd
        ├── scans/               # Generated per-frame clouds in world coordinates
        ├── scan_labels/         # Matching per-point labels for each scan
        └── map_labels/static.label  # One shared map label file for this submap
```

A sequence can contain multiple submaps. Each submap pairs a prior map with
its scans and labels. Input `clouds/` holds raw recordings; output `scans/`
holds generated training examples. They are separate datasets.
Use the saved dataset's `sequences/` directory as the training input, not the
original clouds or intermediate preparation files.
For your generated data, create train/validation split files with one submap path
per line, such as `sequence_000/submaps/submap_000`. Set `training.train_split`
and `training.val_split` in `config/cham.yaml` to those files (paths are relative
to the config). Keep each source session in only one split. The bundled splits
refer to the downloaded dataset, not your new output.

## 🏋️ Training

Point the launcher at your downloaded or generated **`sequences` directory** and choose an
output directory for caches and checkpoints:

```bash
CHAMELION_DATASET_PATH=/path/to/Const_pseudo_dataset/sequences \
CHAMELION_OUTPUT_PATH=/path/to/training_output \
./scripts/train.sh
```

The default runs **50 epochs** using [`config/cham.yaml`](config/cham.yaml).
The dataset is mounted **read-only**; outputs go to the directory you selected.

## 🔎 Inference

Evaluate scan changes and the final map using **scan IoU** and **map PR, RR and F1**.
Inference settings are in [`config/inference.yaml`](config/inference.yaml).

### Pretrained weights

Download from **[se0yeon00/Chamelion on Hugging Face](https://huggingface.co/se0yeon00/Chamelion)**
(currently private; run `hf auth login` with an authorized account).

Run from the repository root:

```python
from huggingface_hub import hf_hub_download

for filename in ("chamelion_pretrained.pt", "inference.yaml"):
    hf_hub_download("se0yeon00/Chamelion", filename, local_dir="pretrained")
```

### Evaluate, then view saved results

```bash
CHAMELION_DATASET_PATH=/path/to/Const_pseudo_dataset/sequences \
CHAMELION_OUTPUT_PATH=/path/to/new_test_run \
bash scripts/evaluate.sh pretrained/chamelion_pretrained.pt \
  --inference-config pretrained/inference.yaml \
  --profile custom \
  --sequence sequence_005/submaps/submap_000
```

Results are saved under `new_test_run/results/`, including metrics, frame previews
and the final map (`final_map.npz` and `final_map.png`). Use `--save-all-frames`
to save every frame for viewing.

Build the optional viewer image once, then open the results on a Linux desktop:

```bash
bash scripts/build.sh viewer
bash scripts/view.sh evaluation /path/to/new_test_run/results
```

The Polyscope viewer switches between **Input**, **Prediction**, **Ground truth**
and **Errors**. Viewing saved results does not run the model or use CUDA.

### Interactive inference without labels

Provide a global-coordinate prior map, numeric PCD scans and their full pose table:

This tool shows **single-frame predictions**, without map accumulation.

```bash
bash scripts/view.sh inference /path/to/model.ckpt \
  /path/to/prior_map.pcd /path/to/scans /path/to/poses.txt global
```

Use `global` for the released dataset's scans, or `local` for sensor-coordinate
scans. No GT labels are required. Select a frame, click **Run inference on this
frame**, then **Save prediction**. The launcher prints the output directory.

## 📄 Citation

If you use Chamelion in your research, please cite:

```bibtex
@article{jang2026chamelion,
  title   = {Chamelion: Reliable Change Detection for Long-Term LiDAR Mapping in Transient Environments},
  author  = {Jang, Seoyeon and Lee, Alex Junho and Nahrendra, I Made Aswin and Myung, Hyun},
  journal = {IEEE Robotics and Automation Letters},
  year    = {2026},
  doi     = {10.1109/LRA.2026.3665079}
}
```

### Acknowledgments

We thank [MapMOS](https://github.com/PRBonn/MapMOS) and
[TRAVEL](https://github.com/url-kaist/TRAVEL) for their open-source code.

### License

Chamelion is distributed under the [GNU General Public License v3.0 or later](LICENSE)
(`GPL-3.0-or-later`). Third-party components retain their original license notices.
Datasets and pretrained weights have separate licensing terms.
