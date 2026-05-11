#!/bin/bash
# ~/md_Engine/PhaseTransition/submit_all.sh
#
# Usage: bash submit_all.sh
# Submits 210 SLURM jobs (one per rho, T* combination)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="/home/co24btech11023/md_Engine/.venv/bin/python3"
RUN_SCRIPT="$SCRIPT_DIR/run_one_point.py"
N_CELLS=4
OUTDIR="/home/co24btech11023/md_Engine/PhaseTransition/data_${N_CELLS}"
LOGDIR="/home/co24btech11023/md_Engine/PhaseTransition/logs"

mkdir -p "$OUTDIR"
mkdir -p "$LOGDIR"

# Grid parameters
RHO_VALUES=($("$PYTHON" -c "
import numpy as np
vals = np.linspace(0.1, 2.0, 41)
print(' '.join(f'{v:.4f}' for v in vals))
"))

# 21 T* values from 0.5 to 2.0 inclusive
# Generate with python to avoid bash float issues
T_STAR_VALUES=($("$PYTHON" -c "
import numpy as np
vals = np.linspace(0.5, 2.0, 41)
print(' '.join(f'{v:.4f}' for v in vals))
"))

echo "T* values: ${T_STAR_VALUES[@]}"
echo "Submitting ${#RHO_VALUES[@]} x ${#T_STAR_VALUES[@]} = $((${#RHO_VALUES[@]} * ${#T_STAR_VALUES[@]})) jobs"
echo ""

JOB_COUNT=0

for rho in "${RHO_VALUES[@]}"; do
    for T_star in "${T_STAR_VALUES[@]}"; do

        rho_str=$(printf "%.4f" "$rho")
        T_str=$(printf "%.4f" "$T_star")
        job_name="md_rho${rho_str}_T${T_str}"
        log_file="$LOGDIR/${job_name}.log"

        sbatch \
            --job-name="$job_name" \
            --partition=LocalQ \
            --ntasks=1 \
            --cpus-per-task=1 \
            --mem=4G \
            --time=UNLIMITED \
            --output="$log_file" \
            --error="$log_file" \
            --wrap="$PYTHON $RUN_SCRIPT --rho $rho --T_star $T_star --outdir $OUTDIR"

        JOB_COUNT=$((JOB_COUNT + 1))
    done
done

echo ""
echo "Submitted $JOB_COUNT jobs."
echo "Monitor with: watch -n 5 squeue -u co24btech11023"
echo "Results go to: $OUTDIR"
echo "Logs go to:    $LOGDIR"