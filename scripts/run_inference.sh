#!/usr/bin/env bash
set -euo pipefail

# Example inference script for SignRAG (CSL-Daily).
# Adjust paths to your local/HF-downloaded assets.

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

MODEL_DIR=${MODEL_DIR:-./models/SignRAG-Qwen3-8B-CSL-Daily-PreinNews}
NON_LORA=${NON_LORA:-./models/non_lora_trainables.bin}
V_INIT=${V_INIT:-./models/mytran_CSL_News_512}
DATA_PATH=${DATA_PATH:-./data/CSL_Daily/label/rag_dep.test}
KEYPOINT_PATH=${KEYPOINT_PATH:-./data/CSL_Daily/pose_format}
OUTPUT_DIR=${OUTPUT_DIR:-./outputs/csl_daily}

NPROC_PER_NODE=${NPROC_PER_NODE:-4}

torchrun --nproc_per_node="$NPROC_PER_NODE" inference.py \
  --model_dir "$MODEL_DIR" \
  --non_lora_trainables "$NON_LORA" \
  --v_init_model "$V_INIT" \
  --data_path "$DATA_PATH" \
  --keypoint_path "$KEYPOINT_PATH" \
  --data_version CSL_Daily \
  --language zh \
  --output_dir "$OUTPUT_DIR" \
  --infer_batch_size 2 \
  --infer_num_workers 8 \
  --beam_size 5 \
  --max_new_tokens 80 \
  --torch_dtype bfloat16
