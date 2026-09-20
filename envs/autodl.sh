#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# AutoDL instance paths (source from run scripts or setup scripts).
#
# Clone behavior on AutoDL:
#   - system disk  (/root, miniconda, project code) is included in the clone;
#   - data disk     (/root/autodl-tmp) is copied ONLY if the "data disk"
#                    option is checked in the clone dialog.
#
# All data/model paths in this project should be read from this file so that
# a cloned instance can point DATA_ROOT elsewhere by exporting it beforehand:
#   export SWM_DATA_ROOT=/root/autodl-tmp && bash scripts/setup_instance.sh
# ---------------------------------------------------------------------------

export SWM_CODE_ROOT="${SWM_CODE_ROOT:-/root/sector_watermark}"
export SWM_DATA_ROOT="${SWM_DATA_ROOT:-/root/autodl-tmp}"
export SWM_MODEL_DIR="${SWM_MODEL_DIR:-${SWM_DATA_ROOT}/models}"

# Stable Diffusion 2.1-base (diffusers format, ModelScope mirror)
export SWM_SD_MODEL="${SWM_SD_MODEL:-${SWM_MODEL_DIR}/sd21base}"

# CLIP ViT-g-14 weights for image-text similarity (open_clip checkpoint)
export SWM_CLIP_DIR="${SWM_CLIP_DIR:-${SWM_MODEL_DIR}/ViT-g-14}"
export SWM_CLIP_MODEL="${SWM_CLIP_MODEL:-${SWM_CLIP_DIR}/open_clip_pytorch_model.bin}"

# COCO 2014 5k test split: prompts + 512x512 reference images for FID
export SWM_COCO_ROOT="${SWM_COCO_ROOT:-${SWM_DATA_ROOT}/data/coco5k}"
export SWM_COCO_META="${SWM_COCO_META:-${SWM_COCO_ROOT}/meta_data.json}"
export SWM_COCO_IMAGES="${SWM_COCO_IMAGES:-${SWM_COCO_ROOT}/images512}"

# Mirror and runtime settings
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
