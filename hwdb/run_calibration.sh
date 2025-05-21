set -ex

# Environment
GPU_ID=5

# --------------------Recognizer------------------
OUTF='hwdb500'     #  500
SEENSIZE=500

BACKBONE='resnetlike'       # Ablation Study
FEADIM=256
# BACKBONE='resnetlike-v3'  # SoTA
# FEADIM=512

CHECKPOINT='checkpoint10'

TESTBATCHSIZE=256


# -------------------Generator-------------------
SYN_PATH='./data_syn/test_350epo_syn_unseen_1000shot.hdf5'
SYN_KSHOT=1000


#-------------------Calibration-----------------
CAL_MU='intp'
CAL_COV='RDA'
INTERPOLATION1=0.5
ALPHA=0.8
BETA=0.2


python hwdb/main_adjust_proto.py \
--gpu_id ${GPU_ID} \
--randomseed 2023 \
--outf ${OUTF} \
--checkpoint ${CHECKPOINT} \
--test \
--seenSize ${SEENSIZE} \
--input_size ${INPUT_SIZE} \
--feaDim ${FEADIM} \
--samplingSize 500 \
--backbone ${BACKBONE} \
--lr 0.001 \
--workers 16 \
--syn_path ${SYN_PATH} \
--syn_kshot ${SYN_KSHOT} \
--testBatchSize ${TESTBATCHSIZE} \
--bayes \
--cal_mu ${CAL_MU} \
--cal_cov ${CAL_COV} \
--interpolation1 ${INTERPOLATION1} \
--alpha ${ALPHA} \
--beta ${BETA} 
