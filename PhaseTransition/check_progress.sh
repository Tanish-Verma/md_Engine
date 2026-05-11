#!/bin/bash
# ~/md_Engine/PhaseTransition/check_progress.sh
# Quick summary of done / running / failed jobs

N_CELLS=4
OUTDIR="/home/co24btech11023/md_Engine/PhaseTransition/data_${N_CELLS}"
LOGDIR="/home/co24btech11023/md_Engine/PhaseTransition/logs"

TOTAL=1681  # 41 rho values x 41 T* values = 1681 total points
DONE=$(ls "$OUTDIR"/*.h5 2>/dev/null | wc -l)
RUNNING=$(squeue -u co24btech11023 --noheader 2>/dev/null | wc -l)
FAILED=$(grep -l "Error\|Traceback\|FAILED" "$LOGDIR"/*.log 2>/dev/null | wc -l)

echo "================================"
echo "  Phase Diagram Progress"
echo "================================"
echo "  Total points : $TOTAL"
echo "  Completed    : $DONE"
echo "  Running now  : $RUNNING"
echo "  Log errors   : $FAILED"
echo "  Remaining    : $((TOTAL - DONE))"
echo "================================"

if [ "$FAILED" -gt 0 ]; then
    echo ""
    echo "Failed logs:"
    grep -l "Error\|Traceback\|FAILED" "$LOGDIR"/*.log 2>/dev/null
fi