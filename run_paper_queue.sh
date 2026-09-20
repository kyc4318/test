#!/usr/bin/env bash
# Public-baseline comparison queue (open-source methods only).
#
#   bash run_paper_queue.sh core     # N=50, 5 methods, signal+rotation+composed
#   bash run_paper_queue.sh extra    # N=50, crops and extra signal attacks
#   bash run_paper_queue.sh regen    # N=30, VAE-B / VAE-C / diffusion regeneration
#   bash run_paper_queue.sh quality  # N=500 generation quality (FID vs COCO GT)
#   bash run_paper_queue.sh tables   # rebuild results/paper_main_tables.md
#
# Each stage appends to logs/paper_queue.log and writes logs/paper_<stage>_done.txt
set -u

cd /root/sector_watermark || exit 1
source /root/miniconda3/etc/profile.d/conda.sh
conda activate sector
mkdir -p logs runs results

STAGE="${1:-core}"
# headline table first (fast), then the wide attack set
CORE="clean,jpeg25,noise0.1,blur5,bright6,rot45,rot75,rot+noise0.05"
WIDE="signal,signal_extra,rotation,composed,crop"
EXTRA="crop"
REGEN="clean,regen"
COCO_GT=/root/autodl-tmp/data/coco5k/images512

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a logs/paper_queue.log; }
run() { log "START $*"; python "$@" 2>&1 | tee -a logs/paper_queue.log; \
        log "END   $* (rc=${PIPESTATUS[0]})"; }

case "$STAGE" in
  core)
    run run_paper_compare.py --method ours      --N 50 --cases "$CORE" --roc 1
    run run_paper_compare.py --method gs256     --N 50 --cases "$CORE" --roc 1
    # GS-8: the official one-bit FPR calibration needs user_number=1, fpr=0.01
    # (larger settings leave tau_onebit unset and eval_watermark raises).
    run run_paper_compare.py --method gs8       --N 50 --cases "$CORE" --roc 1 \
        --user_number 1 --fpr 0.01
    run run_paper_compare.py --method sfw_hstr  --N 50 --cases "$CORE" --roc 1
    run run_paper_compare.py --method sfw_hsqr  --N 50 --cases "$CORE" --roc 1
    ;;
  wide)
    run run_paper_compare.py --method ours      --N 50 --cases "$WIDE" --roc 1
    run run_paper_compare.py --method gs256     --N 50 --cases "$WIDE" --roc 1
    run run_paper_compare.py --method gs8       --N 50 --cases "$WIDE" --roc 1 \
        --user_number 1 --fpr 0.01
    run run_paper_compare.py --method sfw_hstr  --N 50 --cases "$WIDE" --roc 1
    run run_paper_compare.py --method sfw_hsqr  --N 50 --cases "$WIDE" --roc 1
    ;;
  gs8)
    # GS-8 alone: the official one-bit calibration needs user_number=1, fpr=0.01
    run run_paper_compare.py --method gs8 --N 50 --cases "$CORE" --roc 1 \
        --user_number 1 --fpr 0.01
    ;;
  extra)
    run run_paper_compare.py --method ours      --N 50 --cases "$EXTRA" --roc 1
    run run_paper_compare.py --method gs256     --N 50 --cases "$EXTRA" --roc 1
    run run_paper_compare.py --method sfw_hsqr  --N 50 --cases "$EXTRA" --roc 1
    ;;
  regen)
    run run_paper_compare.py --method ours      --N 30 --cases "$REGEN" --roc 1
    run run_paper_compare.py --method gs256     --N 30 --cases "$REGEN" --roc 1
    run run_paper_compare.py --method sfw_hsqr  --N 30 --cases "$REGEN" --roc 1
    ;;
  quality)
    run run_quality_gt.py --N 500 --methods no_wm,ours,gs256,gs8,sfw_hsqr
    for cfg in no_wm ours gs256 gs8 sfw_hsqr; do
      log "FID $cfg vs COCO GT"
      python run_fid.py "runs/quality_gt/images/$cfg" "$COCO_GT" \
        2>&1 | tee -a logs/paper_queue.log
    done
    ;;
  tables)
    python make_paper_tables.py; exit 0
    ;;
  *)
    echo "unknown stage: $STAGE" >&2; exit 2
    ;;
esac

python make_paper_tables.py 2>&1 | tee -a logs/paper_queue.log
date > "logs/paper_${STAGE}_done.txt"
log "STAGE $STAGE DONE"
