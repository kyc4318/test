#!/usr/bin/env bash
# Session queue: controls -> rotation -> operators -> spectrum -> pareto.
#
# Deliberate differences from the upstream ``run_p0_queue.sh``:
#
#   * disjoint calib/test splits *inside* this instance's prompt set.
#     ``/root/autodl-tmp/data/coco5k/meta_data.json`` holds 1000 captions (the
#     images512 folder has 5000 files), so the upstream default
#     ``SplitPlan(test_offset=1000)`` indexes past the end and crashes.  Here
#     calib = 0..N-1 and test = 500..500+N-1, both in range and disjoint.
#   * adds the unseen-rotation-operator stage (``run_unseen_operators.py``),
#     which the upstream queue does not have.
#   * the operator stage now runs the **matched padding pairs** (same library,
#     interpolation, rotation matrix and output size; only the border rule
#     differs) so that the padding question is answered by a clean causal
#     ablation rather than by inference.
#
# Every stage is resumable: the scripts append to runs/<stage>/rows.jsonl and
# skip finished (image, attack, condition) triples, so re-running is safe.
#
# Error handling: ``set -euo pipefail`` plus a ``run`` that *returns* the
# Python exit code means a failing stage aborts the queue and does NOT write its
# ``*_done.txt`` marker.  (An earlier version wrote the marker unconditionally,
# so a failed bar stage looked finished.)
set -euo pipefail

cd /root/sector_watermark || exit 1
source /root/miniconda3/etc/profile.d/conda.sh
conda activate sector
mkdir -p logs runs results

N_CONTROLS="${N_CONTROLS:-50}"
N_ROTATION="${N_ROTATION:-30}"
N_OPS="${N_OPS:-15}"
N_SPECTRUM="${N_SPECTRUM:-25}"
N_PARETO="${N_PARETO:-50}"
N_QUALITY="${N_QUALITY:-200}"
N_PADDING="${N_PADDING:-25}"
N_VISUAL="${N_VISUAL:-4}"
ETAS_VISUAL="${ETAS_VISUAL:-0,5e3,1e4,2e4}"
CALIB_OFFSET="${CALIB_OFFSET:-0}"
TEST_OFFSET="${TEST_OFFSET:-500}"

DESIGN_B8="results/design_searched_realgeom.npz"
DESIGN_B16="results/design_B16_s0.npz"
CORE="clean,jpeg25,noise0.1,blur5,bright6,rot45,rot75,rot+noise0.05"

# Attack operators for the generalisation stage.  The split is deliberate:
#   * pil_bilinear  -- the operator used by every earlier run (reference cell)
#   * pil_bicubic   -- a different interpolation kernel, same library
#   * cv2_{linear,cubic,nearest}_{constant,reflect}
#                   -- three MATCHED PAIRS: identical library, interpolation,
#                      rotation matrix and output size; only the border rule
#                      differs.  constant leaves the zero-filled wedges,
#                      reflect does not -- this is the clean causal ablation for
#                      "does the synchronizer read the black wedges?".
OPS_ATTACK="${OPS_ATTACK:-pil_bilinear,pil_bicubic,cv2_linear_constant,cv2_linear_reflect,cv2_cubic_constant,cv2_cubic_reflect,cv2_nearest_constant,cv2_nearest_reflect}"

WRONGKEYS=""
for s in 1 2 3; do
  f="results/design_B8_s${s}.npz"
  [ -f "$f" ] && WRONGKEYS="${WRONGKEYS}${WRONGKEYS:+,}${f}"
done

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a logs/p0_session.log; }
run() {
  log "START $*"
  # `set -e` would otherwise abort on the pipeline before we can read the exit
  # code of the python process (PIPESTATUS), so disable it around the call.
  set +e
  python -u "$@" 2>&1 | tee -a logs/p0_session.log
  local rc=${PIPESTATUS[0]}
  set -e
  log "END   $* (rc=$rc)"
  return "$rc"
}

stage_controls() {
  run run_paper_controls.py --design "$DESIGN_B8" --N "$N_CONTROLS" \
      --cases "$CORE" --grid_step 2 \
      --calib_offset "$CALIB_OFFSET" --test_offset "$TEST_OFFSET" \
      --wrongkey_designs "$WRONGKEYS" --out_dir runs/p0_controls
  # rotation-operator control: the same run with the zero wedges cropped away
  run run_paper_controls.py --design "$DESIGN_B8" --N "$N_CONTROLS" \
      --cases "clean,rot45,rot75" --grid_step 2 \
      --calib_offset "$CALIB_OFFSET" --test_offset "$TEST_OFFSET" \
      --crop_black --out_dir runs/p0_controls
  date > logs/p0_controls_done.txt
}

stage_rotation() {
  run run_continuous_rotation.py --design "$DESIGN_B8" --N "$N_ROTATION" \
      --mode grid --angle_step 15 --sigmas 0 --grid_step 1 \
      --out_dir runs/p0_rotation
  run run_continuous_rotation.py --design "$DESIGN_B8" --N "$N_ROTATION" \
      --mode fixed --angles "7.3,22.5,37.3,58.7,102.5,168.9,-30,-75,-135" \
      --sigmas "0,0.05,0.1" --grid_step 1 --out_dir runs/p0_rotation
  run run_continuous_rotation.py --design "$DESIGN_B8" --N "$N_ROTATION" \
      --mode random --n_angles_per_image 8 --sigmas 0 --grid_step 1 \
      --out_dir runs/p0_rotation
  date > logs/p0_rotation_done.txt
}

stage_operators() {
  run run_unseen_operators.py --design "$DESIGN_B8" --N "$N_OPS" \
      --angles "37.3,58.7,102.5,-30" \
      --attack_ops "$OPS_ATTACK" \
      --decode_ops "tv_nearest_c31.5,tv_bilinear_c31.5,tv_bicubic_c31.5,tv_bilinear_c32" \
      --grid_step 2 --out_dir "${OPS_OUT:-runs/p0_operators_v2}"
  date > logs/p0_operators_done.txt
}

# The clean padding ablation: three matched pairs, nothing else.  Fresh out_dir
# on purpose -- the legacy runs/p0_operators has no fingerprint sidecar, and
# ResumeLog now refuses (rightly) to stamp a new configuration onto it.
stage_padding() {
  run run_unseen_operators.py --design "$DESIGN_B8" --N "$N_PADDING" \
      --angles "37.3,58.7,102.5,-30" \
      --attack_ops "cv2_linear_constant,cv2_linear_reflect,cv2_cubic_constant,cv2_cubic_reflect,cv2_nearest_constant,cv2_nearest_reflect" \
      --decode_ops "tv_nearest_c31.5" \
      --grid_step 2 --out_dir runs/p0_padding
  date > logs/p0_padding_done.txt
}

# Paired visual quality: same prompt + same latent, no watermark vs several eta,
# with amplified difference maps (real images for human inspection).
stage_visual() {
  run run_visual_quality.py --design "$DESIGN_B8" --N "$N_VISUAL" \
      --etas "$ETAS_VISUAL" --lpips \
      --out_dir runs/visual_quality --report_path results/visual_quality.md
  date > logs/p0_visual_done.txt
}

stage_spectrum() {
  run run_inversion_error_spectrum.py --design "$DESIGN_B8" \
      --N "$N_SPECTRUM" \
      --cases "clean,rot45,rot75,noise0.05,rot+noise0.05" \
      --grid_step 2 --out_dir runs/p0_spectrum
  date > logs/p0_spectrum_done.txt
}

stage_pareto() {
  DESIGNS="$DESIGN_B8"
  [ -f "$DESIGN_B16" ] && DESIGNS="$DESIGNS,$DESIGN_B16"
  run run_capacity_quality_pareto.py --designs "$DESIGNS" \
      --etas "5e3,1e4,2e4" --cases "clean,rot45,rot+noise0.05" \
      --N "$N_PARETO" --clip --lpips \
      --report_path results/paper_pareto.md --out_dir runs/pareto
  date > logs/p0_pareto_done.txt
}

case "${1:-all}" in
  controls)  stage_controls ;;
  rotation)  stage_rotation ;;
  operators) stage_operators ;;
  padding)   stage_padding ;;
  visual)    stage_visual ;;
  spectrum)  stage_spectrum ;;
  pareto)    stage_pareto ;;
  all)
    stage_controls
    stage_rotation
    stage_operators
    stage_padding
    stage_visual
    stage_spectrum
    stage_pareto
    ;;
  *) echo "unknown stage: $1" >&2; exit 2 ;;
esac

log "STAGE ${1:-all} DONE"
