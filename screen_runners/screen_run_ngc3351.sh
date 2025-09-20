#!/bin/bash

# Define your target (galaxy name or object)
TARGET="NGC3351"

# Screen session name
SESSION_NAME="sitelle_${TARGET}_run"

# Conda environment name
CONDA_ENV="/home/habjan/.conda/envs/orb3" 

# Number of CPUs to allow
NUM_CPUS=4

# Path to script (no target here)
SCRIPT_PATH="/home/habjan/SITELLE/sitelle_metallicities/runner.py"

# Log file path
LOG_DIR="/home/habjan/SITELLE/sandbox_notebooks/job_logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/${TARGET}_run.log"

# Start the screen session
screen -dmS "$SESSION_NAME" bash -c "
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate $CONDA_ENV
    export OMP_NUM_THREADS=$NUM_CPUS
    export MKL_NUM_THREADS=$NUM_CPUS
    export NUMEXPR_NUM_THREADS=$NUM_CPUS
    python $SCRIPT_PATH $TARGET > $LOG_FILE 2>&1
"