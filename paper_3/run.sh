#!/bin/bash

# ===================================================================
# --- Set default command here ---
DEFAULT_COMMAND="python paper_3_test.py"
# ===================================================================

# Free GPU threshold (MiB).
THRESHOLD=100

# Find first 'free' GPU.
FREE_GPU_ID=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' -v thres="$THRESHOLD" '$2 < thres {print $1; exit}')
if [ -z "$FREE_GPU_ID" ]; then
    echo "Error: No free GPU available."
    exit 1
fi

export CUDA_VISIBLE_DEVICES=$FREE_GPU_ID
echo "Free GPU found, CUDA_VISIBLE_DEVICES=$FREE_GPU_ID"
eval $DEFAULT_COMMAND
