#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/disk1/workspace_jym/LCZ"
PYTHON="${PYTHON:-/home/jym/anaconda3/envs/mmseg_final/bin/python}"
GPUS=(${GPUS:-0 1 2 3})
TRAIN="$ROOT/train/train_lcz_graph_encoder_v7_ablation.py"
GRAPH_CACHE="$ROOT/work_dirs/graph_v7_node_edge_scale_interaction_oa/graph_cache"
LOG_DIR="$ROOT/work_dirs/ablation_v7_logs"
mkdir -p "$LOG_DIR"

COMMON=(
  --base_dir "$ROOT/data"
  --work_dir "$ROOT/work_dirs"
  --rs_norm_dir "$ROOT/data/Satellite/processed/norm"
  --building_shp "$ROOT/data/Building/AL_11_D010_20200502/AL_11_D010_20200502.shp"
  --story_col A10
  --split_dir "$ROOT/work_dirs/cnn_rs_baseline/splits"
  --graph_cache_dir "$GRAPH_CACHE"
  --patch_size 33
  --graph_context_m 500
  --global_scales_m 330 500 700
  --edge_mode hybrid
  --knn 8
  --radius_m 120
  --max_nodes 192
  --graph_hidden 96
  --graph_out 192
  --graph_layers 3
  --graph_heads 4
  --batch_size 96
  --epochs 180
  --patience 35
  --lr 5e-4
  --dropout 0.35
  --class_weight
  --monitor oa
  --seed 42
)

# The complete V7 result already exists in graph_v7_node_edge_scale_interaction_oa.
if [ "${#GPUS[@]}" -lt 4 ]; then
  echo "Need 4 GPU ids in GPUS, got: ${GPUS[*]}" >&2
  exit 1
fi

COMMON_ARGS="$(printf '%q ' "${COMMON[@]}")"
GPUS_STR="${GPUS[*]}"
export COMMON_ARGS GPUS_STR

"$PYTHON" - <<'PY'
import os
import shlex
import subprocess
from pathlib import Path

root = Path("/mnt/disk1/workspace_jym/LCZ")
python_bin = "/home/jym/anaconda3/envs/mmseg_final/bin/python"
train = root / "train" / "train_lcz_graph_encoder_v7_ablation.py"
log_dir = root / "work_dirs" / "ablation_v7_logs"
gpus = os.environ.get("GPUS_STR", "0 1 2 3").split()
common_args = shlex.split(os.environ["COMMON_ARGS"])

jobs = [
    ("no_salient_selection", "ablation_v7_no_salient", gpus[0]),
    ("no_ump_tokens", "ablation_v7_no_ump", gpus[1]),
    ("no_global_context", "ablation_v7_no_global", gpus[2]),
    ("no_edge_type", "ablation_v7_no_edge_type", gpus[3]),
]

procs = []
for ablation, exp_name, gpu in jobs:
    log_path = log_dir / f"{exp_name}.log"
    log_file = open(log_path, "w", encoding="utf-8")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    cmd = [python_bin, str(train), *common_args, "--ablation", ablation, "--exp_name", exp_name]
    print(f"=== Running {exp_name} on GPU {gpu} ===", flush=True)
    proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, env=env)
    procs.append((exp_name, gpu, proc, log_file))

failed = []
for exp_name, gpu, proc, log_file in procs:
    rc = proc.wait()
    log_file.close()
    if rc == 0:
        print(f"=== Finished {exp_name} on GPU {gpu} ===", flush=True)
    else:
        failed.append((exp_name, gpu, rc))
        print(f"=== FAILED {exp_name} on GPU {gpu} (exit {rc}) ===", flush=True)

if failed:
    raise SystemExit(1)
PY

"$PYTHON" "$ROOT/code/9_summarize_v7_ablation.py" \
  --run "Full V7=$ROOT/work_dirs/graph_v7_node_edge_scale_interaction_oa" \
  --run "w/o Salient selection=$ROOT/work_dirs/ablation_v7_no_salient" \
  --run "w/o UMP tokens=$ROOT/work_dirs/ablation_v7_no_ump" \
  --run "w/o Global context=$ROOT/work_dirs/ablation_v7_no_global" \
  --run "w/o Edge type=$ROOT/work_dirs/ablation_v7_no_edge_type" \
  --out_dir "$ROOT/work_dirs/analysis/v7_ablation"
