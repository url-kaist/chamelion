#!/usr/bin/env bash
set -euo pipefail
python3 - <<'PY'
import os
from pathlib import Path
import yaml
for mount in ('/input', '/workspace'):
    if not os.statvfs(mount).f_flag & os.ST_RDONLY:
        raise RuntimeError(f'{mount} must be read-only; refusing to start')
if os.statvfs('/output').f_flag & os.ST_RDONLY:
    raise RuntimeError('/output must be the new writable run directory')
print('Source and repository read-only mount checks: OK')
path = Path('/output/generator.yaml')
config = yaml.safe_load(Path('pseudo_generator/config/generator.yaml').read_text())
config['travel_config'] = '/workspace/pseudo_generator/config/travel_demo.yaml'
config['load_dataset'].update(dataset_root='/input', seq=os.environ['PSEUDO_SOURCE_SEQUENCE'],
    pred_inst_root='/output/instance_labels', pred_inst_sequence='sequence_000',
    pred_save_dir='/output/prepared/sequence_000')
config['output_dataset'].update(root='/output/Const_pseudo_dataset', sequence='sequence_000')
with path.open('x') as stream:
    yaml.safe_dump(config, stream, sort_keys=False)
import sys
sys.path.insert(0, '/workspace/pseudo_generator')
import travel_seg, trimesh, cv2, filterpy
import prepare_objects, generate_changes, label_instances
print('TRAVEL and interactive generation imports: OK')
PY
python3 pseudo_generator/gui_check.py
if [[ "${1:-}" == --check ]]; then exit 0; fi
printf '\n%s\n' \
  'Ready. Run these commands one at a time, in order.' \
  '1) python3 pseudo_generator/label_instances.py /output/generator.yaml' \
  '2) python3 pseudo_generator/prepare_objects.py /output/generator.yaml' \
  '3) python3 pseudo_generator/generate_changes.py /output/generator.yaml' \
  'Stages: Ground Segmentation / Object Tracking / Dynamic Object Removal / Change Placement / Export' \
  'Pick: Shift+Left click. Undo pick: Shift+Right click. Q opens the review.' \
  'Review: Q or C confirms, R retries, X cancels. Closing a review does not confirm.' \
  'Source: /input (read-only). New results: /output.' \
  'Type exit when finished. Results remain on the host.'
exec bash --noprofile --norc
