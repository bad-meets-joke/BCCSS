set -ex

# Environment
GPU_ID=1

# Recognizer
# OUTF='2023-11-16_01_hwdb_500' 
# OUTF='2023-11-16_02_hwdb_500' 
# OUTF='2024-01-23_01_hwdb_500_CMPL_ndim=256' 
# OUTF='2023-12-08_01_hwdb_500_dce_LearnableScale' 
# OUTF='2023-12-08_02_hwdb_500_dce_ConstantScale=1' 
# OUTF='2024-01-25_02_hwdb_500_CMPL_ndim=128_DSBN'
# OUTF='2023-04-28_01_hwdb_1000' 
# OUTF='2023-04-28_02_hwdb_1500' 
# OUTF='2023-04-28_03_hwdb_2000' 
# OUTF='2023-04-28_04_hwdb_2755' 

# OUTF='2024-02-19_01_hw500_resnetlikeV2dsbn_64x64' 
# OUTF='2024-02-19_01_hw1000_resnetlikeV2dsbn_64x64' 
# OUTF='2024-02-19_01_hw500_densenet44dsbn_32x32' 
# OUTF='2024-02-22_01_hw500_resnetlikeV2dsbn-dim256'
# OUTF='2024-02-22_01_hw1000_resnetlikeV2dsbn'
# OUTF='2024-02-24_02_hw1000_resnetlikeV4dsbn'
# OUTF='2024-02-25_01_hw1000_resnetlikeV2dsbn_stdDCE-scale1-PL2e-3'
# OUTF='2024-02-25_02_hw1000_resnetlikeV2dsbn_stdDCE-scalelearn'
# OUTF='2024-02-26_01_hw1000_resnetlikeV3dsbn-dim256'
# OUTF='2024-02-26_02_hw1000_resnetlikeV4dsbn-dim256'
# OUTF='2024-03-02_01_hw1500_resnetlikeV3dsbn'
# OUTF='2024-03-02_02_hw1500_resnetlikeV3dsbn-dim256'
# OUTF='2024-03-02_03_hw1500_resnetlikeV4dsbn'

# ============================Sota============================
# OUTF='2024-02-24_01_hw1000_resnetlikeV3dsbn'        # 1000
# OUTF='2024-03-02_01_hw1500_resnetlikeV3dsbn'        # 1500
# OUTF='2024-03-08_01_hw2000_resnetlikV3dsbn-dim512'  # 2000
# OUTF='2024-03-08_01_hw2755_resnetlikeV3dsbn-dim512' # 2755
# BACKBONE='resnetlike-v3'
# FEADIM=512 
# SEENSIZE=500
# SEENSIZE=1000
# SEENSIZE=1500
# SEENSIZE=2000
# SEENSIZE=2755
# ============================================================


# ====================Bayesian vs Prototype====================
# Bayesian vs prototype
# OUTF='2024-01-25_01_hwdb_500_CMPL_ndim=256_DSBN'     #  500
# OUTF='2024-03-02_04_hw1000_resnetlikedsbn-dim256'    # 1000
# BACKBONE='resnetlike'
# FEADIM=256
# SEENSIZE=500
# SEENSIZE=1000
# ==============================================================


CHECKPOINT='checkpoint10'
# SEENSIZE=500
SEENSIZE=1000
# SEENSIZE=1500
# SEENSIZE=2000
# SEENSIZE=2755


INPUT_SIZE=64
BACKBONE='resnetlike'
# BACKBONE='resnetlike-v2'
# BACKBONE='resnetlike-v3'
# BACKBONE='resnetlike-v4'
# CHECKPOINT='checkpoint5'
# FEADIM=512  # 512, 256, 128
FEADIM=256
# FEADIM=128

# 损失相关的超参不影响分类结果
DISTANCE_PRN='euclidean'           # 默认, 非标准DCE
# DISTANCE_PRN='euclidean_square'  # 标准DCE
SCALE_CHOICE='learnable'           # DCE的尺度系数的类型（可学 or 常量）. 默认可学
# SCALE_CHOICE='constant'          
SCALE_WEIGHT=1.0                   # DCE的尺度系数为常量时的值   
LAMBDA_PL=0.                       # PL权重
# LAMBDA_PL=0.002       


# Generator
# SYN_PATH=''../interested_methods/BicycleGAN-master/results/prn2hw/2023-04-28_01_1000/val_60epo_syn_unseen_30shot.hdf5''
# SYN_PATH=''../interested_methods/BicycleGAN-master/results/prn2hw/2023-04-28_02_1500/val_60epo_syn_unseen_30shot.hdf5''
# SYN_PATH=''../interested_methods/BicycleGAN-master/results/prn2hw/2023-04-28_03_2000/val_60epo_syn_unseen_30shot.hdf5''
# SYN_PATH=''../interested_methods/BicycleGAN-master/results/prn2hw/2023-04-28_04_2755/val_60epo_syn_unseen_30shot.hdf5''

# SYN_PATH='../interested_methods/BicycleGAN-master/results/prn2hw/2023-04-24_02/val_60epo_syn_unseen_1000shot.hdf5'     # 500
SYN_PATH='../interested_methods/BicycleGAN-master/results/prn2hw/2023-04-28_01_1000/val_60epo_syn_unseen_1000shot.hdf5'  # 1000

# SYN_PATH=''../interested_methods/BicycleGAN-master/results/prn2hw/2024-02-08_01_500_rbe+sn/val_60epo_syn_unseen_1000shot.hdf5''
# SYN_PATH=''../interested_methods/BicycleGAN-master/results/prn2hw/2024-02-19_02_500_rbe+sn_5e0_OnlyUpdateG/val_60epo_syn_unseen_1000shot.hdf5''

# 扩散模型
# SYN_PATH='../interested_methods/Palette-Image-to-Image-Diffusion-Models-main/experiments/train_prn2hw_240229_225130/test_350epo_syn_unseen_1000shot.hdf5'

SYN_KSHOT=1000


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
--distance_prn ${DISTANCE_PRN} \
--scale_choice ${SCALE_CHOICE} \
--scale_weight ${SCALE_WEIGHT} \
--lambda_pl ${LAMBDA_PL} \
--lr 0.001 \
`# --epochs 20 ` \
--workers 16 \
--syn_path ${SYN_PATH} \
--syn_kshot ${SYN_KSHOT} \
--bayes \
--testBatchSize 64 \
`# --testBatchSize 24` \
`# --testBatchSize 48` \
# --mqdf \
# --bias_whole_img 0.7 \
# --jigsaw_weight 0.9 \
# --jigsaw_n_classes 100 \
# --mixup_alpha 3 \
# --mixup_ratio 0.5 \
# --lambda3 1e-4 \
# --lambda4 1e-4 \
# --reconstruction
# --test
