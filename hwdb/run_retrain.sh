set -ex


# Environment
GPU_ID=4


# =======================Retrain============================
# RETRAIN='u'
RETRAIN='s+u'

# RETRAIN_MODEL='softmax'
RETRAIN_MODEL='cmpl'


# Recognizer
# =============================================================    
# OUTF='2024-03-16_01_hw500_retrain_U_softmax' # 500
# OUTF='2024-03-16_01_hw500_retrain_U_cmpl'   
# OUTF='2024-03-23_01_hw500_retrain_S+U_softmax'
# OUTF='2024-03-23_01_hw500_retrain_S+U_cmpl'
# SEENSIZE=500

# OUTF='2024-03-16_01_hw1000_retrain_U_softmax'  # 1000
# OUTF='2024-03-16_01_hw1000_retrain_U_cmpl'
# OUTF='2024-03-23_02_hw1000_retrain_S+U_softmax'       
# OUTF='2024-03-23_02_hw1000_retrain_S+U_cmpl'       
# SEENSIZE=1000


# OUTF='2024-03-23_03_hw1500_retrain_S+U_softmax'   # 1500
OUTF='2024-03-23_03_hw1500_retrain_S+U_cmpl'    
SEENSIZE=1500


BACKBONE='resnetlike'
FEADIM=256
# ==============================================================


INPUT_SIZE=64
LR=0.001


# 损失相关的超参不影响分类结果
DISTANCE_PRN='euclidean'           # 默认, 非标准DCE
# DISTANCE_PRN='euclidean_square'  # 标准DCE
SCALE_CHOICE='learnable'           # DCE的尺度系数的类型（可学 or 常量）. 默认可学
# SCALE_CHOICE='constant'          
SCALE_WEIGHT=1.0                   # DCE的尺度系数为常量时的值   
LAMBDA_PL=0.                       # PL权重
# LAMBDA_PL=0.002       


# -------------------Generator-----------------
SYN_KSHOT=1000
# --------BicycleGAN--------
# SYN_PATH='../interested_methods/BicycleGAN-master/results/prn2hw/2023-04-24_02/val_60epo_syn_unseen_1000shot.hdf5'     # 500
# SYN_PATH='../interested_methods/BicycleGAN-master/results/prn2hw/2023-04-28_01_1000/val_60epo_syn_unseen_1000shot.hdf5'  # 1000

# --------Palette-----------
# 500, 350 epo
# SYN_PATH='../interested_methods/Palette-Image-to-Image-Diffusion-Models-main/experiments/train_prn2hw_240229_225130/test_350epo_syn_unseen_1000shot.hdf5'

# 1000, 350+100 epo
# SYN_PATH='../interested_methods/Palette-Image-to-Image-Diffusion-Models-main/experiments/train_prn2hw_240309_001929/test_450epo_syn_unseen_1000shot.hdf5'

# 1500, 350+70 epo
SYN_PATH='../interested_methods/Palette-Image-to-Image-Diffusion-Models-main/experiments/train_prn2hw_240309_002012/test_420epo_syn_unseen_1000shot.hdf5'




python hwdb/main_retrain.py \
--gpu_id ${GPU_ID} \
--randomseed 2023 \
--outf ${OUTF} \
--test \
--seenSize ${SEENSIZE} \
--input_size ${INPUT_SIZE} \
--feaDim ${FEADIM} \
--samplingSize 1000 \
--backbone ${BACKBONE} \
--distance_prn ${DISTANCE_PRN} \
--scale_choice ${SCALE_CHOICE} \
--scale_weight ${SCALE_WEIGHT} \
--lambda_pl ${LAMBDA_PL} \
--lr ${LR} \
`# --epochs 20 ` \
--workers 16 \
--syn_path ${SYN_PATH} \
--syn_kshot ${SYN_KSHOT} \
--retrain ${RETRAIN} \
--retrain_model ${RETRAIN_MODEL}
