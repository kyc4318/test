#!/usr/bin/env bash
# P0 evidence-gate queue (GPU).  Each stage is independent and resumable:
# the python scripts append to runs/<stage>/rows.jsonl and skip finished
# (image, attack, condition) triples, so an interrupted stage can be restarted
# with the same command.
#
#   bash run_p0_queue.sh controls   # angle-search FPR / wrong-key / leak
#   bash run_p0_queue.sh rotation   # continuous + negative angles
#   bash run_p0_queue.sh spectrum   # inversion-error decomposition
#   bash run_p0_queue.sh identity   # key-registry benchmark (ours included)
#   bash run_p0_queue.sh pareto     # capacity x energy quality/robustness
#   bash run_p0_queue.sh quality    # CLIP + paired distortion + FID vs COCO GT
#   bash run_p0_queue.sh tables     # rebuild the markdown tables
#   bash run_p0_queue.sh all        # controls -> rotation -> spectrum -> ...
#
# Every stage appends to logs/p0_queue.log and writes logs/p0_<stage>_done.txt.
set -u

cd /root/sector_watermark || exit 1
source /root/miniconda3/etc/profile.d/conda.sh
conda activate sector
mkdir -p logs runs results

STAGE="${1:-all}"

# scales are overridable from the environment, e.g. P0_N_CONTROLS=20
P0_N_CONTROLS="${P0_N_CONTROLS:-50}"
P0_N_ROTATION="${P0_N_ROTATION:-30}"
P0_N_SPECTRUM="${P0_N_SPECTRUM:-25}"
P0_N_IDENTITY="${P0_N_IDENTITY:-20}"
P0_N_KEYS="${P0_N_KEYS:-2048}"
P0_N_PARETO="${P0_N_PARETO:-50}"
P0_N_QUALITY="${P0_N_QUALITY:-200}"

DESIGN_B8="${DESIGN_B8:-results/design_searched_realgeom.npz}"
DESIGN_B16="${DESIGN_B16:-results/design_B16_s0.npz}"
COCO_GT=/root/autodl-tmp/data/coco5k/images512

CORE="clean,jpeg25,noise0.1,blur5,bright6,rot45,rot75,rot+noise0.05"

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a logs/p0_queue.log; }
run() { log "START $*"; python -u "$@" 2>&1 | tee -a logs/p0_queue.log; \
        log "END   $* (rc=${PIPESTATUS[0]})"; }

# wrong-key codebooks that actually exist on this machine
WRONGKEYS=""
for s in 1 2 3; do
  f="results/design_B8_s${s}.npz"
  [ -f "$f" ] && WRONGKEYS="${WRONGKEYS}${WRONGKEYS:+,}${f}"
done

stage_controls() {
  run run_paper_controls.py --design "$DESIGN_B8" --N "$P0_N_CONTROLS" \
      --cases "$CORE" --grid_step 2 --wrongkey_designs "$WRONGKEYS" \
      --out_dir runs/p0_controls
  # rotation-operator control: same run with the zero wedges cropped away
  run run_paper_controls.py --design "$DESIGN_B8" --N "$P0_N_CONTROLS" \
      --cases "clean,rot45,rot75" --grid_step 2 --crop_black \
      --out_dir runs/p0_controls
  date > logs/p0_controls_done.txt
}

stage_rotation() {
  run run_continuous_rotation.py --design "$DESIGN_B8" --N "$P0_N_ROTATION" \
      --mode grid --angle_step 15 --sigmas 0 --grid_step 1 \
      --out_dir runs/p0_rotation
  run run_continuous_rotation.py --design "$DESIGN_B8" --N "$P0_N_ROTATION" \
      --mode fixed --angles "7.3,22.5,37.3,58.7,102.5,168.9,-30,-75,-135" \
      --sigmas "0,0.05,0.1" --grid_step 1 --out_dir runs/p0_rotation
  run run_continuous_rotation.py --design "$DESIGN_B8" --N "$P0_N_ROTATION" \
      --mode random --n_angles_per_image 8 --sigmas 0 --grid_step 1 \
      --out_dir runs/p0_rotation
  date > logs/p0_rotation_done.txt
}

stage_spectrum() {
  run run_inversion_error_spectrum.py --design "$DESIGN_B8" \
      --N "$P0_N_SPECTRUM" \
      --cases "clean,rot45,rot75,noise0.05,rot+noise0.05" \
      --grid_step 2 --out_dir runs/p0_spectrum
  date > logs/p0_spectrum_done.txt
}

stage_identity() {
  run run_identity_benchmark.py --design "$DESIGN_B8" --N "$P0_N_IDENTITY" \
      --n_keys "$P0_N_KEYS" --methods ours,sfw_hstr,sfw_hsqr \
      --cases "clean,rot45,rot75" --out_dir runs/identity
  date > logs/p0_identity_done.txt
}

stage_pareto() {
  DESIGNS="$DESIGN_B8"
  [ -f "$DESIGN_B16" ] && DESIGNS="$DESIGNS,$DESIGN_B16"
  run run_capacity_quality_pareto.py --designs "$DESIGNS" \
      --etas "5e3,1e4,2e4" --cases "clean,rot45,rot+noise0.05" \
      --N "$P0_N_PARETO" --out_dir runs/pareto
  date > logs/p0_pareto_done.txt
}

stage_quality() {
  run run_quality_gt.py --N "$P0_N_QUALITY" \
      --methods "no_wm,ours,gs256,gs8,sfw_hsqr" --out_dir runs/quality_gt
  for cfg in no_wm ours gs256 gs8 sfw_hsqr; do
    log "FID $cfg vs COCO GT"
    python run_fid.py "runs/quality_gt/images/$cfg" "$COCO_GT" \
        --json_out "runs/quality_gt/fid_${cfg}.json" \
        2>&1 | tee -a logs/p0_queue.log
  done
  date > logs/p0_quality_done.txt
}

stage_tables() {
  python make_paper_tables.py 2>&1 | tee -a logs/p0_queue.log
  python add_confidence_intervals.py 2>&1 | tee -a logs/p0_queue.log
  date > logs/p0_tables_done.txt
}

case "$STAGE" in
  controls) stage_controls ;;
  rotation) stage_rotation ;;
  spectrum) stage_spectrum ;;
  identity) stage_identity ;;
  pareto)   stage_pareto ;;
  quality)  stage_quality ;;
  tables)   stage_tables ;;
  all)
    stage_controls
    stage_rotation
    stage_spectrum
    stage_identity
    stage_pareto
    stage_tables
    ;;
  *)
    echo "unknown stage: $STAGE" >&2
    exit 2
    ;;
esac

log "STAGE $STAGE DONE"
