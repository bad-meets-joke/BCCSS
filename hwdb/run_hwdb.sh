set -ex

GPU_ID=7

OUTF='hwdb500'
SEENSIZE=500

BACKBONE='resnetlike'   # Ablation 
FEADIM=256
# BACKBONE='resnetlike-v3'  # SoTA
# FEADIM=512

SAMPLINGSIZE=1000


python hwdb/main.py \
--gpu_id ${GPU_ID} \
--outf ${OUTF} \
--randomseed 2023 \
--seenSize ${SEENSIZE} \
--samplingSize ${SAMPLINGSIZE} \
--input_size ${INPUT_SIZE} \
--backbone ${BACKBONE} \
--feaDim ${FEADIM} \
--lr 0.001 \
--epochs 10 \
--workers 24 